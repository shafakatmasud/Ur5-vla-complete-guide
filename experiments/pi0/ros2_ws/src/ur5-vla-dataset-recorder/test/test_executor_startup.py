"""Replay subscription/timer order with real observation validation and callbacks."""
from collections import deque
import threading
from types import SimpleNamespace

import pytest

from test_execution_profile import harness, Message
from ur5_vla_dataset_recorder.core import UR5_JOINT_ORDER, reorder_joint_positions
from ur5_vla_dataset_recorder.policy_executor import ShadowObservationBuffer


@pytest.mark.parametrize('arrival_order', [
    ('joints', 'base', 'wrist', 'servo'),
    ('servo', 'wrist', 'base', 'joints'),
    ('base', 'joints', 'servo', 'wrist'),
    ('wrist', 'servo', 'joints', 'base'),
])
def test_delayed_startup_then_inference_then_feedback_loss(arrival_order):
    clock = [10.]
    pending = []
    class DeferredThread:
        def __init__(self, target, args, daemon):
            self.run = lambda: target(*args)
        def start(self):
            pending.append(self.run)
    h = harness('policy_executor.py', 'PolicyExecutor',
                {'image', 'joints', 'schedule_inference', 'infer', 'publish_action', 'publish_inference_wait'},
                dict(time=SimpleNamespace(monotonic=lambda: clock[0]),
                     threading=SimpleNamespace(Thread=DeferredThread),
                     Float64MultiArray=Message, MAX_INFERENCE_WAIT=2.,
                     UR5_JOINT_ORDER=UR5_JOINT_ORDER,
                     reorder_joint_positions=reorder_joint_positions,
                     request_prediction=lambda *args: ([[.01]*6+[1.]]*50, .6),
                     SERVER_HOST='localhost', SERVER_PORT=8765, TASK='test'))
    h.buffer = ShadowObservationBuffer()
    h.gripper = 1.
    h.action_queue, h.raw_action_queue = deque(), deque()
    h.queue_lock = threading.Lock()
    h.last_action = h.last_raw_action = None
    h.inference_wait_started = None
    h.inference_running = h.execution_started = h.offset_fault = False
    h.inference_generation = 0
    h.measured_joints, h.measured_joints_at = None, 0.
    h.confirmed_gripper, h.confirmed_gripper_at = None, 0.
    h.approach_schedule = None
    h.carry_lift = None
    h.grasp_assist = None
    h.trajectory_offset = SimpleNamespace(plan_actions=lambda actions, reference: (actions, {}))
    h.actions_per_prediction, h.blend_actions, h.prefetch_actions = 10, 5, 2
    h.get_logger = lambda: SimpleNamespace(error=lambda msg: None, warning=lambda msg: None, info=lambda msg: None)
    sent = []
    h.publisher = SimpleNamespace(publish=lambda msg: sent.append(list(msg.data)))
    image = SimpleNamespace(height=480, width=640, encoding='rgb8', step=1920,
                            data=bytes(480*640*3))
    def deliver(key):
        # Simulate the independent engine IO heartbeat during startup arrivals.
        h.buffer.update_gripper(1.,clock[0])
        if key == 'joints':
            h.joints(SimpleNamespace(name=list(UR5_JOINT_ORDER), position=[0.]*6))
        elif key == 'servo':
            h.buffer.update_servo(0, clock[0])
        else:
            h.image(f'observation.images.{key}', image)
    # Exactly the reported order: inference and action timers run repeatedly
    # while cameras, joint state and Servo have not produced initial samples.
    for _ in range(20):
        h.schedule_inference()
        h.publish_action()
        clock[0] += .05
        assert not h.offset_fault and not sent and not pending
    arrived = []
    for key in arrival_order:
        arrived.append(key)
        for available in arrived:
            deliver(available)
        h.schedule_inference()
        h.publish_action()
        assert not h.offset_fault and not sent
        if len(arrived) < 4:
            assert not pending
        clock[0] += .01
    assert len(pending) == 1
    pending.pop()()
    assert len(h.action_queue) == 10 and not h.offset_fault
    h.publish_action()
    assert h.execution_started and sent == [[.01]*6+[1.]]
    clock[0] += .11
    h.publish_action()
    assert h.offset_fault and len(sent) == 1


def test_startup_prediction_is_discarded_when_its_inputs_expire():
    from test_inference_wait import executor
    h, clock, sent, _ = executor()
    h.execution_started = False
    h.last_action = None
    h.measured_joints = None
    h.action_queue.append([.01]*6+[1.])
    h.raw_action_queue.append([.009]*6+[1.])
    h.publish_action()
    assert not h.offset_fault and not sent
    assert not h.action_queue and not h.raw_action_queue
    assert h.inference_generation == 1
    h.measured_joints = [0.]*6
    h.measured_joints_at = clock[0]
    h.action_queue.append([.02]*6+[1.])
    h.raw_action_queue.append([.019]*6+[1.])
    h.publish_action()
    assert h.execution_started and sent == [[.02]*6+[1.]]
