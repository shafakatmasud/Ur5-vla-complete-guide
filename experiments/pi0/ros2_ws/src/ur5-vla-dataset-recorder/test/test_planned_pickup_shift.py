import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'scripts'))
from planned_pickup_shift import PlannedPickupShift,load_shift,load_unrecorded_repeat
from grasp_offset import load_offset


def helper(tmp_path):
    return PlannedPickupShift(SimpleNamespace(),tmp_path)


def test_begin_requires_fresh_open_and_old_open_target(tmp_path):
    p=helper(tmp_path)
    for old,confirmed,age in [(None,1.,0.),([0.]*7,1.,0.),([0.]*6+[1.],0.,0.),([0.]*6+[1.],1.,.3)]:
        with pytest.raises(ValueError):p.begin(old,confirmed,1.-age,1.)


def test_no_close_during_execution_and_requires_settled_arrival(tmp_path):
    p=helper(tmp_path);p.begin([0.]*6+[1.],1.,1.,1.)
    p.stage='executing';p.path=[[0.]*6,[.002]*6]
    assert p.tick([0.]*6,1.1,1.,1.1,1.1)[6]==1.
    assert p.tick([0.]*6,1.2,1.,1.2,1.2)[6]==1.
    assert p.stage=='arrival'
    assert p.tick([0.]*6,1.3,1.,1.3,1.3)[6]==1.
    assert p.tick([.002]*6,1.4,1.,1.4,1.4)[6]==1.
    assert p.tick([.002]*6,1.71,1.,1.71,1.71)[6]==0.
    assert p.tick([.002]*6,1.8,0.,1.8,1.8)[6]==0.
    assert p.stage=='closed'


def test_gripper_close_cannot_interrupt_open_path(tmp_path):
    p=helper(tmp_path);p.begin([0.]*6+[1.],1.,1.,1.)
    with pytest.raises(ValueError,match='closed before'):p.tick([0.]*6,1.1,0.,1.1,1.1)


def test_stale_feedback_and_plan_failure_block(tmp_path):
    p=helper(tmp_path);p.begin([0.]*6+[1.],1.,1.,1.)
    with pytest.raises(ValueError,match='Stale'):p.tick([0.]*6,1.,1.,1.2,1.2)
    p.error='collision';
    with pytest.raises(ValueError,match='collision'):p.tick([0.]*6,1.2,1.,1.2,1.2)


def shift_file(tmp_path):
    p=tmp_path/'shift.json'
    p.write_text(json.dumps({'frame':'base','allowed_attempts':[12,13],'additional_base_vector_m':[.02,.02,0.]}))
    return p


def test_separate_mode_does_not_relax_small_offset_limit(tmp_path):
    p=shift_file(tmp_path)
    assert load_shift(p,12,False)==[.02,.02,0.]
    assert load_shift(p,13,False)==[.02,.02,0.]
    p.write_text(json.dumps({'offset_frame':'base','offset_m':[.02,.02,0.]}))
    with pytest.raises(ValueError,match='7 mm'):load_offset(p)


@pytest.mark.parametrize('attempt',[None,11,14])
def test_out_of_scope_attempt_rejected(tmp_path,attempt):
    with pytest.raises(ValueError,match='12 and 13'):load_shift(shift_file(tmp_path),attempt,False)


@pytest.mark.parametrize('vec',[[.05,.05,0.],[.02,.02,.002],[float('nan'),0.,0.]])
def test_unbounded_or_nonplanar_shift_rejected(tmp_path,vec):
    p=shift_file(tmp_path);d=json.loads(p.read_text());d['additional_base_vector_m']=vec;p.write_text(json.dumps(d))
    with pytest.raises(ValueError):load_shift(p,12,False)


def test_active_recording_identity_is_required(tmp_path):
    p=shift_file(tmp_path);d=json.loads(p.read_text());d['session_root']=str(tmp_path);p.write_text(json.dumps(d))
    (tmp_path/'status.json').write_text(json.dumps({'state':'waiting','attempt':None}))
    with pytest.raises(ValueError,match='Start recording'):load_shift(p,12)


def test_first_policy_close_is_intercepted_without_publishing(tmp_path):
    from test_execution_profile import harness,Message
    from collections import deque
    import threading
    h=harness('policy_executor.py','PolicyExecutor',{'publish_action'},
              dict(time=SimpleNamespace(monotonic=lambda:1.),Float64MultiArray=Message))
    h.offset_fault=False;h.execution_started=True;h.carry_task_pending=False
    h.measured_joints=[0.]*6;h.measured_joints_at=1.;h.measured_sample=(h.measured_joints,1.)
    h.trajectory_offset=None;h.grasp_assist=None;h.planned_transport=None
    h.queue_lock=threading.Lock();h.action_queue=deque([[.01]*6+[0.]])
    h.raw_action_queue=deque([[.01]*6+[0.]])
    h.inference_generation=0;h.last_action=[0.]*6+[1.]
    h.confirmed_gripper=1.;h.confirmed_gripper_at=1.
    h.pickup_shift=helper(tmp_path)
    h.publisher=SimpleNamespace(publish=lambda _:pytest.fail('CLOSE must not be published before shift'))
    h.publish_action()
    assert h.pickup_shift.stage=='settling'
    assert not h.action_queue and not h.raw_action_queue
    assert h.inference_generation==1 and h.last_action[-1]==1.


def test_unrecorded_repeat_does_not_change_formal_trial(tmp_path):
    p=shift_file(tmp_path);d=json.loads(p.read_text());d['session_root']=str(tmp_path);p.write_text(json.dumps(d))
    (tmp_path/'status.json').write_text(json.dumps({'state':'waiting'}))
    trial=tmp_path/'cross_01/attempt_13/trial.json';trial.parent.mkdir(parents=True)
    trial.write_text(json.dumps({'status':'finished','operator_outcome':'pass'}));before=trial.read_bytes()
    assert load_unrecorded_repeat(p,13)==[.02,.02,0.]
    assert trial.read_bytes()==before
    assert not (trial.parent/'configuration').exists()
    with pytest.raises(ValueError,match='13'):load_unrecorded_repeat(p,12)
    (tmp_path/'status.json').write_text(json.dumps({'state':'recording'}))
    with pytest.raises(ValueError,match='Finish'):load_unrecorded_repeat(p,13)


def test_unrecorded_repeat_requires_finished_formal_trial(tmp_path):
    p=shift_file(tmp_path);d=json.loads(p.read_text());d['session_root']=str(tmp_path);p.write_text(json.dumps(d))
    (tmp_path/'status.json').write_text(json.dumps({'state':'waiting'}))
    trial=tmp_path/'cross_01/attempt_13/trial.json';trial.parent.mkdir(parents=True)
    trial.write_text(json.dumps({'status':'recording'}))
    with pytest.raises(ValueError,match='finished'):load_unrecorded_repeat(p,13)
