"""Placement instruction changes invalidate pickup actions without moving the arm."""
from collections import deque
from types import SimpleNamespace
import threading
import time

from test_execution_profile import harness
from approach_offset import ApproachOffsetSchedule


def executor(predict):
    h = harness('policy_executor.py', 'PolicyExecutor', {'apply_carry_task', 'infer'},
                dict(time=time, request_prediction=predict,
                     SERVER_HOST='localhost', SERVER_PORT=8765, TASK='pickup blue'))
    h.queue_lock = threading.Lock()
    h.carry_task = 'Place the cube in the bowl'
    h.carry_task_pending = True
    h.action_queue = deque([[.01]*6+[0.]])
    h.raw_action_queue = deque([[0.]*7])
    h.last_action = [.005]*6+[0.]
    h.last_raw_action = [0.]*7
    h.inference_generation = 3
    h.inference_running = True
    h.inference_wait_started = None
    h.offset_fault = False
    h.grasp_assist = h.trajectory_offset = h.approach_schedule = None
    h.actions_per_prediction = 10
    h.blend_actions = 5
    h.get_logger = lambda: SimpleNamespace(info=lambda _: None, error=lambda _: None)
    return h


def test_instruction_switch_holds_last_target_and_discards_queued_pickup():
    sent = []
    h = executor(lambda *args: (sent.append(args[4]) or [[.01]*6+[0.]]*50, 0.))
    hold = list(h.last_action)
    h.apply_carry_task()
    assert h.task == h.carry_task and h.inference_generation == 4
    assert not h.action_queue and not h.raw_action_queue
    assert h.last_action == hold and h.inference_running
    h.infer([0.]*7, [], 4)
    assert sent == ['Place the cube in the bowl']
    assert len(h.action_queue) == 10 and not h.offset_fault


def test_old_pickup_worker_cannot_commit_after_switch():
    h = executor(lambda *args: ([[.01]*6+[0.]]*50, 0.))
    h.apply_carry_task()
    h.infer([0.]*7, [], 3)
    assert not h.action_queue and not h.inference_running and not h.offset_fault


def test_error_from_invalidated_worker_does_not_latch_fault():
    def fail(*args):
        raise TimeoutError('obsolete pickup response')
    h = executor(fail)
    h.trajectory_offset = SimpleNamespace()
    h.apply_carry_task()
    h.infer([0.]*7, [], 3)
    assert not h.offset_fault and not h.action_queue and not h.inference_running


def test_instruction_transition_requires_open_then_stable_closed_feedback():
    s = ApproachOffsetSchedule(close_hold=.5)
    assert not s.observe(0., 0.)
    assert not s.observe(1., .1)
    assert not s.observe(0., .2)
    for now in [.3, .4, .5, .6]:
        assert not s.observe(0., now)
    assert s.observe(0., .71)
    assert not s.observe(1., .8)
    assert not s.observe(0., .9)


def test_inference_waits_for_action_timer_to_apply_instruction_switch():
    h = harness('policy_executor.py', 'PolicyExecutor', {'schedule_inference'}, {})
    h.offset_fault = False
    h.carry_task_pending = True
    h.schedule_inference()  # No worker or ROS call may start before the switch.


def test_feedback_callback_requests_switch_only_after_stable_close():
    clock = [0.]
    h = harness('policy_executor.py', 'PolicyExecutor', {'gripper_confirmation'},
                dict(time=SimpleNamespace(monotonic=lambda: clock[0])))
    h.buffer = SimpleNamespace(update_gripper=lambda *args: None)
    h.carry_task_schedule = ApproachOffsetSchedule(close_hold=.5)
    h.carry_task_pending = False
    h.carry_lift = h.approach_schedule = None
    h.gripper_confirmation(SimpleNamespace(data=1.))
    for value in [.1, .2, .3, .4, .5]:
        clock[0] = value
        h.gripper_confirmation(SimpleNamespace(data=0.))
        assert not h.carry_task_pending
    clock[0] = .61
    h.gripper_confirmation(SimpleNamespace(data=0.))
    assert h.carry_task_pending
