"""Smooth approach-to-carry transition with continuous mode preserved."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'scripts'))
from approach_offset import ApproachOffsetSchedule, CarryLiftSchedule


def close(s, start=1.):
    s.observe(1., start-.1)
    for i in range(8):
        s.observe(0., start+i*.1)


def test_full_offset_before_grasp_and_smooth_monotone_fade_to_raw_carry():
    s = ApproachOffsetSchedule()
    assert s.scales(10., 10, .1) == [1.]*10
    close(s)
    t = s.release_started
    assert t is not None
    scales = s.scales(t, 31, .1)
    assert scales[0] == 1. and scales[-1] == 0.
    assert all(a >= b for a, b in zip(scales, scales[1:]))
    assert scales[10] == pytest.approx(.5)
    assert all(v == 0 for v in s.scales(t+3, 10, .1))
    s.observe(1., t+4)
    assert s.scales(t+4, 10, .1) == [0.]*10  # One episode, no re-offset on release.


def test_initial_closed_gripper_does_not_start_carry_transition():
    s = ApproachOffsetSchedule()
    for i in range(30):s.observe(0., i*.1)
    assert s.release_started is None


def test_reopening_or_missing_confirmation_resets_close_hold():
    s = ApproachOffsetSchedule()
    s.observe(1., 0.)
    s.observe(0., .1)
    s.observe(0., .2)
    s.observe(1., .3)
    s.observe(0., .4)
    s.observe(0., 1.)  # A gap is not evidence of continuously fresh closed IO.
    assert s.release_started is None
    for i in range(1, 8):s.observe(0., 1.+i*.1)
    assert s.release_started is not None


def test_planner_applies_fade_and_still_validates_raw_carry(monkeypatch):
    from trajectory_offset import TrajectoryOffsetPlanner
    class Request:
        def __init__(self):
            self.robot_state = SimpleNamespace(is_diff=False, joint_state=SimpleNamespace())
    # Replace only the ROS service type; execute the production planner method.
    monkeypatch.setitem(sys.modules, 'moveit_msgs', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'moveit_msgs.srv', SimpleNamespace(GetStateValidity=SimpleNamespace(Request=Request)))
    planner = TrajectoryOffsetPlanner.__new__(TrajectoryOffsetPlanner)
    planner.frame, planner.order, planner.vector = 'base', list('abcdef'), [.0013, -.0003, 0.]
    vectors, checked = [], []
    def translate(q, vector, carry_height=0.):
        vectors.append(vector)
        return [q[0]+vector[0], q[1]+vector[1], *q[2:]]
    planner.model = SimpleNamespace(translate=translate)
    class Future:
        def result(self):return SimpleNamespace(valid=True)
        def done(self):return True
        def cancelled(self):return False
        def add_done_callback(self, cb):cb(self)
    def validate(request):
        checked.append(request.robot_state.joint_state.position)
        return Future()
    planner.validity = SimpleNamespace(service_is_ready=lambda: True, call_async=validate)
    actions = [[.01]*6+[0.]]*3
    result, report = planner.plan_actions(actions, [0.]*6, scales=[1., .5, 0.])
    assert vectors[0] == planner.vector
    assert vectors[1] == [v*.5 for v in planner.vector]
    assert vectors[-1] == [0.]*3
    assert result[-1] == actions[-1] and len(checked) >= 3
    assert report['offset_scale_end'] == 0.
    # Default continuous mode still applies the entire correction to all rows.
    vectors.clear()
    planner.plan_actions(actions, [0.]*6)
    assert vectors == [planner.vector]*3
    with pytest.raises(ValueError, match='offset scales'):
        planner.plan_actions(actions, [0.]*6, scales=[1., -1., 0.])


def test_executor_confirmation_drives_fade_and_passes_scales_to_planner():
    from collections import deque
    import threading
    from test_execution_profile import harness
    clock = [0.]
    seen = []
    h = harness('policy_executor.py', 'PolicyExecutor', {'gripper_confirmation', 'infer'},
                dict(time=SimpleNamespace(monotonic=lambda: clock[0]), ACTION_PERIOD=.05,
                     request_prediction=lambda *args: ([[.01]*6+[0.]]*50, .1),
                     SERVER_HOST='localhost', SERVER_PORT=8765, TASK='test'))
    h.approach_schedule = ApproachOffsetSchedule()
    h.carry_lift = None
    h.buffer = SimpleNamespace(update_gripper=lambda value, now: None)
    h.get_logger = lambda: SimpleNamespace(info=lambda msg: None, error=lambda msg: None)
    h.gripper_confirmation(SimpleNamespace(data=1.))
    for i in range(1, 9):
        clock[0] = i*.1
        h.gripper_confirmation(SimpleNamespace(data=0.))
    assert h.approach_schedule.release_started is not None
    clock[0] += 3.
    h.speed_scale, h.actions_per_prediction, h.blend_actions = .5, 10, 5
    h.queue_lock = threading.Lock()
    h.action_queue, h.raw_action_queue = deque(), deque()
    h.last_raw_action = h.last_action = None
    h.grasp_assist = None
    h.offset_fault = False
    h.inference_running = True
    h.inference_generation = 0
    h.measured_joints = [0.]*6
    h.measured_joints_at = clock[0]
    def plan(actions, reference, scales):
        seen.append(scales)
        return actions, {}
    h.trajectory_offset = SimpleNamespace(plan_actions=plan)
    h.infer([0.]*7, [], 0)
    assert not h.offset_fault and len(h.action_queue) == 10
    assert seen == [[0.]*10]


def test_carry_lift_stays_zero_until_stable_close_then_ramps_up():
    s=CarryLiftSchedule(.025)
    assert s.heights(0.,50,.1)==[0.]*50
    s.observe(1.,0.)
    s.observe(0.,.1)
    s.observe(0.,.2)
    assert s.heights(.2,50,.1)==[0.]*50
    for i in range(3,9):s.observe(0.,i*.1)
    start=s.release_started
    assert start is not None
    heights=s.heights(start,31,.1)
    assert heights[0]==0. and heights[-1]==.025
    assert all(a<=b for a,b in zip(heights,heights[1:]))
    s.observe(1.,4.)
    assert s.heights(5.,2,.1)==[.025,.025]


@pytest.mark.parametrize('height',[-.001,.031,float('nan')])
def test_invalid_carry_lifts_rejected(height):
    with pytest.raises(ValueError):CarryLiftSchedule(height)


def test_carry_ramp_starts_at_committed_height_even_if_close_happened_long_ago():
    s=CarryLiftSchedule(.025)
    close(s)
    # Wall time may advance while an already planned zero-lift chunk plays.
    h=s.next_heights(0.,50,.1)
    assert h[0]==0. and h[-1]==.025
    assert max(b-a for a,b in zip(h,h[1:])) < .002
    assert s.next_heights(h[-1],50,.1)==[.025]*50
