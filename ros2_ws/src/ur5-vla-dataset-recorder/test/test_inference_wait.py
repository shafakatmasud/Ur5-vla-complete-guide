"""Bounded keepalive during real executor queue gaps, without ROS or motion."""
from collections import deque
import threading
from types import SimpleNamespace

import pytest

from test_execution_profile import harness, Message


def executor():
    clock = [10.]
    h = harness('policy_executor.py', 'PolicyExecutor',
                {'publish_action', 'publish_inference_wait', 'gripper_confirmation'},
                dict(time=SimpleNamespace(monotonic=lambda: clock[0]),
                     Float64MultiArray=Message, MAX_INFERENCE_WAIT=2.))
    h.queue_lock = threading.Lock()
    h.action_queue = deque()
    h.raw_action_queue = deque()
    h.last_action = [.01]*6+[1.]
    h.last_raw_action = [.009]*6+[1.]
    h.inference_wait_started = None
    h.inference_running = True
    h.inference_generation = 0
    h.offset_fault = False
    h.execution_started = True
    h.trajectory_offset = object()  # Corrected targets must not be offset again.
    h.grasp_assist = None
    h.carry_lift = h.approach_schedule = None
    h.measured_joints = [0.]*6
    h.measured_joints_at = clock[0]
    snapshots, sent = [], []
    h.buffer = SimpleNamespace(snapshot=lambda *args: snapshots.append(args))
    h.publisher = SimpleNamespace(publish=lambda msg: sent.append(list(msg.data)))
    h.get_logger = lambda: SimpleNamespace(error=lambda msg: None, info=lambda msg: None)
    return h, clock, sent, snapshots


def test_normal_gap_repeats_exact_corrected_target_and_resumes_new_chunk():
    from test_control_engine_startup import engine, measured
    e, engine_clock, arm, _ = engine()
    h, clock, sent, snapshots = executor()
    measured(e, 0.)
    # A 0.9-second queue gap exceeds the engine watchdog, but periodic bounded
    # repeats keep it healthy while observations and inference remain live.
    for i in range(10):
        clock[0] = engine_clock[0] = 10.+i*.1
        h.measured_joints_at = clock[0]
        h.publish_action()
        measured(e, 0.)
        e.command_callback(SimpleNamespace(data=sent[-1]))
        e.publish_smoothed_arm()
        assert e.fault is None
    assert sent == [[.01]*6+[1.]]*10
    assert len(snapshots) == 10
    assert h.last_raw_action == [.009]*6+[1.]
    h.action_queue.append([.02]*6+[0.])
    h.raw_action_queue.append([.019]*6+[0.])
    h.publish_action()
    assert sent[-1] == [.02]*6+[0.]
    assert h.inference_wait_started is None


def test_wait_budget_latches_and_cannot_be_extended_by_ongoing_inference():
    h, clock, sent, _ = executor()
    h.publish_action()
    clock[0] += 2.01
    h.measured_joints_at = clock[0]
    h.publish_action()
    assert h.offset_fault and h.inference_generation == 1 and len(sent) == 1
    h.publish_action()
    assert len(sent) == 1


@pytest.mark.parametrize('failure', ['joints', 'camera', 'fault', 'no_action', 'no_inference'])
def test_no_keepalive_for_stale_inputs_fault_or_inactive_inference(failure):
    h, clock, sent, _ = executor()
    if failure == 'joints':
        h.measured_joints_at -= .11
    elif failure == 'camera':
        def stale(*args):
            raise ValueError('stale camera')
        h.buffer.snapshot = stale
    elif failure == 'fault':
        h.offset_fault = True
    elif failure == 'no_action':
        h.last_action = None
    else:
        h.inference_running = False
    h.publish_action()
    assert not sent
    if failure in ('joints', 'camera'):
        assert h.offset_fault


def fresh_except_gripper(buffer,now):
    buffer.update_joints([0.]*6,now)
    buffer.update_servo(0,now)
    for key in ('observation.images.base','observation.images.wrist'):
        buffer.update_image(key,1,1,'rgb8',3,bytes(3),now)


def test_live_engine_gripper_heartbeats_keep_actual_buffer_fresh_during_wait():
    from ur5_vla_dataset_recorder.policy_executor import ShadowObservationBuffer
    h,clock,sent,_=executor()
    h.buffer=ShadowObservationBuffer()
    for i in range(30):
        clock[0]=10.+i*.05
        h.measured_joints_at=clock[0]
        fresh_except_gripper(h.buffer,clock[0])
        h.gripper_confirmation(SimpleNamespace(data=0.))
        h.publish_inference_wait()
    assert not h.offset_fault and len(sent)==30
    _,_,ages=h.buffer.snapshot(clock[0],.5,1.,.5)
    assert ages['gripper_state']==0.


def test_missing_engine_heartbeat_still_blocks_even_with_live_cameras():
    from ur5_vla_dataset_recorder.policy_executor import ShadowObservationBuffer
    h,clock,sent,_=executor()
    h.buffer=ShadowObservationBuffer()
    h.gripper_confirmation(SimpleNamespace(data=0.))
    clock[0]+=1.1
    h.measured_joints_at=clock[0]
    fresh_except_gripper(h.buffer,clock[0])
    h.publish_inference_wait()
    assert h.offset_fault and not sent


@pytest.mark.parametrize('fresh_sample',[True,False])
def test_wait_uses_coherent_joint_sample_instead_of_separate_legacy_fields(fresh_sample):
    h,clock,sent,_=executor()
    h.measured_sample=([0.]*6,clock[0] if fresh_sample else clock[0]-.11)
    h.measured_joints_at=clock[0]-.5 if fresh_sample else clock[0]
    h.publish_inference_wait()
    assert bool(sent)==fresh_sample
    assert h.offset_fault != fresh_sample
