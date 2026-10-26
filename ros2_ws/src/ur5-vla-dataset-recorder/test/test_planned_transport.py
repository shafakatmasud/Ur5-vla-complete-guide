"""Checked assisted transport must not run discontinuous, stale or unready paths."""
import json
from collections import deque
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'scripts'))
from bowl_release import BowlRelease
from planned_transport import (route_waypoints,dense_joint_path,PlannedBowlTransport,
                               refined_joint_path,interpolate_pose,PathKnotGap)
from test_execution_profile import harness,Message

ORDER=['j'+str(i) for i in range(6)]
LIMITS={k:(-3.,3.) for k in ORDER}


@pytest.fixture
def bowl(tmp_path):
    path=tmp_path/'bowl.json'
    path.write_text(json.dumps({'frame':'base','tip':'tool0_controller','position_m':[.58,.04,.38],
        'quaternion_xyzw':[0.,0.,0.,1.],'release_zone':{'horizontal_radius_m':.04,
        'below_reference_m':.03,'above_reference_m':.12,'orientation_tolerance_deg':30.}}))
    return BowlRelease(path)


def test_route_lifts_vertically_before_moving_over_bowl(bowl):
    route=route_waypoints([.36,.16,.26],[0,0,0,1],bowl)
    assert route[0][:2]==(.36,.16) and route[0][2]>=.46
    assert route[1][:2]==bowl.position[:2] and route[1][2]==route[0][2]
    assert route[2][:2]==bowl.position[:2] and route[2][2]==pytest.approx(.405)
    assert route[2][3:]==bowl.orientation


def test_excessive_clearance_is_rejected(bowl):
    with pytest.raises(ValueError,match='height envelope'):
        route_waypoints([.4,.1,.59],[0,0,0,1],bowl)


@pytest.mark.parametrize('path',[[[float('nan')]*6],[[4.]*6],[[.1]*6],[[0.]*6,[.16]*6]])
def test_invalid_or_discontinuous_joint_route_is_rejected(path):
    with pytest.raises(ValueError):dense_joint_path(path,[0.]*6,ORDER,LIMITS)


def test_playback_interpolation_respects_small_joint_steps():
    route=dense_joint_path([[0.]*6,[.02]*6,[.04]*6],[0.]*6,ORDER,LIMITS)
    previous=[0.]*6
    for q in route:
        assert max(abs(a-b) for a,b in zip(previous,q))<=.002+1e-12
        previous=q
    assert route[-1]==[.04]*6


def test_reported_sparse_knot_gap_is_replanned_as_smaller_cartesian_legs():
    calls=[]
    def solve(start,goal):
        calls.append(goal[0])
        return [start,[goal[0],0.,0.,0.,0.,0.]]
    pose_of=lambda q:[q[0],0.,.3,0.,0.,0.,1.]
    path,count=refined_joint_path([0.]*6,[.155182,0.,.3,0.,0.,0.,1.],solve,pose_of,ORDER,LIMITS)
    assert count==1 and len(calls)==3 and path[-1][0]==pytest.approx(.155182)
    previous=[0.]*6
    for q in path:
        assert max(abs(a-b) for a,b in zip(q,previous))<=.002+1e-12
        previous=q


def test_initial_discontinuity_cannot_be_hidden_by_refinement():
    calls=[]
    def solve(start,goal):
        calls.append(goal)
        return [[.02]*6]
    with pytest.raises(PathKnotGap):
        refined_joint_path([0.]*6,[.2,0.,.3,0.,0.,0.,1.],solve,lambda q:[0.,0.,.3,0.,0.,0.,1.],ORDER,LIMITS)
    assert len(calls)==1


def test_persistent_branch_jump_still_fails_after_bounded_refinement():
    with pytest.raises(PathKnotGap):
        refined_joint_path([0.]*6,[.2,0.,.3,0.,0.,0.,1.],
                           lambda start,goal:[start,[1.]*6],
                           lambda q:[q[0],0.,.3,0.,0.,0.,1.],ORDER,LIMITS)


def test_midpoint_handles_equivalent_opposite_quaternion_signs():
    pose=interpolate_pose([0.,0.,.3,0.,0.,0.,1.],[.2,0.,.3,0.,0.,0.,-1.])
    assert pose==[.1,0.,.3,0.,0.,0.,1.]


@pytest.fixture
def transport(tmp_path):
    planner=SimpleNamespace(inside_release_region=lambda q:False)
    logger=SimpleNamespace(info=lambda _:None)
    h=PlannedBowlTransport(planner,logger,tmp_path)
    h.observe(1.,0.)
    for t in [.1,.2,.3,.4,.5,.61]:h.observe(0.,t)
    assert h.stage=='requested'
    h.begin([0.]*6+[0.],.7)
    return h


def test_closed_at_start_cannot_trigger_assisted_route(tmp_path):
    h=PlannedBowlTransport(None,None,tmp_path)
    for t in [0,.1,.2,.3,.4,.5,.6]:h.observe(0.,t)
    assert not h.active


def test_handoff_holds_target_before_planning(transport):
    transport.closed_at=.8
    action=transport.tick([0.]*6,.8,.8)
    assert action==[0.]*7 and transport.stage=='settling' and transport.worker is None


@pytest.mark.parametrize('joint_age,grip_age',[(.11,0.),(0.,.21)])
def test_stale_feedback_cannot_drive_assisted_motion(transport,joint_age,grip_age):
    transport.closed_at=1.-grip_age
    with pytest.raises(ValueError,match='Stale'):
        transport.tick([0.]*6,1.-joint_age,1.)


def test_unexpected_open_outside_bowl_is_a_fault(transport):
    transport.stage='executing';transport.observe(1.,1.)
    with pytest.raises(ValueError,match='Unexpected gripper OPEN'):
        transport.tick([0.]*6,1.,1.)


def test_verified_arrival_open_holds_and_never_resumes_policy(transport):
    transport.stage='executing';transport.planner.inside_release_region=lambda q:True
    transport.observe(1.,1.)
    action=transport.tick([0.]*6,1.,1.)
    assert transport.stage=='complete' and transport.active and action==[0.]*6+[1.]
    transport.observe(1.,1.1)
    assert transport.tick([0.]*6,1.1,1.1)==action


def test_playback_waits_for_measured_joints_and_bounds_stall(transport):
    transport.stage='executing';transport.path=[[.03]*6];transport.stage_started=1.
    transport.closed_at=1.
    assert transport.tick([0.]*6,1.,1.)==[0.]*7
    assert transport.index==0
    transport.closed_at=12.
    with pytest.raises(TimeoutError,match='stopped following'):
        transport.tick([0.]*6,12.,12.)


def test_handoff_invalidates_policy_queue_and_publishes_only_helper_target():
    sent=[];begun=[]
    planner=SimpleNamespace(stage='requested',begin=lambda action,now:begun.append(list(action)),
                            tick=lambda *args:[.001]*6+[0.])
    h=harness('policy_executor.py','PolicyExecutor',{'publish_planned_transport'},
              dict(time=SimpleNamespace(monotonic=lambda:1.),Float64MultiArray=Message))
    h.planned_transport=planner;h.buffer=SimpleNamespace(snapshot=lambda *args:None)
    h.queue_lock=threading.Lock();h.action_queue=deque([[1.]*7]);h.raw_action_queue=deque([[1.]*7])
    h.inference_generation=4;h.last_action=[0.]*7
    h.publisher=SimpleNamespace(publish=lambda m:sent.append(m.data))
    h.get_logger=lambda:SimpleNamespace(error=lambda msg:pytest.fail(msg))
    h.publish_planned_transport([0.]*6,1.)
    assert begun==[[0.]*7] and h.inference_generation==5
    assert not h.action_queue and not h.raw_action_queue
    assert sent==[[.001]*6+[0.]]
