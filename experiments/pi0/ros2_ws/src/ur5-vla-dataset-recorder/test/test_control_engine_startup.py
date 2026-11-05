"""Exercise actual control callbacks without ROS services or physical outputs."""
import ast
import math
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts'))
from execution_profile import StableGripperCommand


def engine():
    tree = ast.parse((ROOT / 'scripts/control_engine.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ControlEngine')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name != '__init__']
    module = ast.Module(body=[ast.ClassDef(name='Harness', bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
    ast.fix_missing_locations(module)
    clock = [10.]
    names = dict(json=json, math=math, time=SimpleNamespace(monotonic=lambda: clock[0]),
                 reorder_joint_positions=lambda names, positions, order: positions,
                 UR5_JOINT_ORDER=[], Float64MultiArray=SimpleNamespace,
                 SetIO=SimpleNamespace(Request=SimpleNamespace), Float64=SimpleNamespace,
                 JOINT_TIMEOUT=.2, COMMAND_TIMEOUT=.5, MAX_TRACKING_ERROR=.05,
                 MAX_INITIAL_ERROR=.1, MAX_TARGET_STEP=.15)
    names['rclpy'] = SimpleNamespace(time=SimpleNamespace(Time=lambda: None))
    exec(compile(module, 'control_engine_callbacks', 'exec'), names)
    h = names['Harness']()
    h.target = h.command = h.filtered_target = h.last_raw_target = None
    h.measured = None
    h.measured_at = h.last_command_at = h.last_tick = 0.
    h.velocity = [0.]*6
    h.fault = None
    h.fault_kind = None
    h.startup_hold = False
    h.servo_stopped = True
    h.servo_stop_future = None
    h.confirmed_gripper = None
    h.last_gripper = None
    # Existing guard/IO tests isolate their concern; stability is covered below.
    h.gripper_filter = StableGripperCommand(stable_s=0.)
    h.bowl_release = None
    h.last_bowl_diagnostics = 0.
    h.gripper_busy = h.dry_run = False
    h.max_velocity, h.max_acceleration, h.filter_time_constant = .05, .15, .56
    h.output_count = 0
    h.last_motion_diagnostics = 0.
    h.count_publishers = lambda _: 1
    h.get_node_names = lambda: ["control_engine"]
    h.get_logger = lambda: SimpleNamespace(error=lambda msg: None, warning=lambda msg: None, info=lambda msg: None)
    sent, gripper = [], []
    h.arm_publisher = SimpleNamespace(publish=lambda msg: sent.append(list(msg.data)))
    h.start_gripper = lambda value: gripper.append(value)
    return h, clock, sent, gripper


def measured(h, value):
    h.joint_state_callback(SimpleNamespace(name=[], position=[value]*6))


def command(h, value, gripper=1.):
    h.command_callback(SimpleNamespace(data=[value]*6+[gripper]))


def test_gripper_rejects_brief_reversals_but_allows_sustained_close_and_release():
    h, clock, _, writes = engine()
    h.gripper_filter = StableGripperCommand()
    h.last_gripper = h.confirmed_gripper = 1.
    def tick(value):
        measured(h, 0.)
        command(h, .01, value)
        clock[0] += .1
    tick(0.); tick(1.)
    tick(0.); tick(1.)
    assert writes == []
    tick(0.); tick(0.); tick(0.)
    assert writes == [0.]
    h.last_gripper = h.confirmed_gripper = 0.
    tick(1.); tick(0.)
    assert writes == [0.]
    tick(1.); tick(1.); tick(1.)
    assert writes == [0., 1.]


def test_gripper_stability_does_not_accumulate_across_missing_commands():
    f = StableGripperCommand()
    assert f.update(0., 1., 1.) == 1.
    assert f.update(0., 2., 1.) == 1.
    assert f.update(0., 2.1, 1.) == 1.
    assert f.update(0., 2.2, 1.) == 0.
    f.reset()
    assert f.update(0., 3., 1.) == 1.


def test_bowl_release_requests_open_once_and_blocks_policy_reclose():
    h, clock, _, writes = engine()
    h.last_gripper = h.confirmed_gripper = 0.
    g = SimpleNamespace(released=False, diagnostics={})
    def arrival(*args):
        g.released = True
        return True
    g.update = arrival
    h.bowl_release = g
    h.release_tf = SimpleNamespace(lookup_transform=lambda *args: SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=10,nanosec=0)),
        transform=SimpleNamespace(translation=SimpleNamespace(x=0.,y=0.,z=0.),
                                  rotation=SimpleNamespace(x=0.,y=0.,z=0.,w=1.))))
    h.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=10_000_000_000))
    h.update_bowl_release(clock[0])
    h.update_bowl_release(clock[0])
    assert writes == [1.]
    h.last_gripper = h.confirmed_gripper = 1.
    measured(h,0.)
    command(h,.01,0.)
    assert writes == [1.]


def test_dry_run_bowl_arrival_never_writes_io():
    h, _, _, writes = engine()
    h.dry_run = True
    h.bowl_release = SimpleNamespace(released=False, diagnostics={}, update=lambda *args: True)
    h.release_tf = SimpleNamespace(lookup_transform=lambda *args: SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=10,nanosec=0)),
        transform=SimpleNamespace(translation=SimpleNamespace(x=0.,y=0.,z=0.),
                                  rotation=SimpleNamespace(x=0.,y=0.,z=0.,w=1.))))
    h.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=10_000_000_000))
    h.update_bowl_release(10.)
    assert not writes


def test_transport_hold_rejects_sustained_policy_open_until_release():
    h, clock, _, writes = engine()
    h.last_gripper = h.confirmed_gripper = 0.
    h.bowl_release = SimpleNamespace(released=False,transport_hold=True)
    for _ in range(20):
        measured(h,0.)
        command(h,.01,1.)
        clock[0] += .1
    assert not writes


def test_manual_reposition_before_policy_uses_latest_pose_and_first_output_holds():
    h, clock, sent, _ = engine()
    measured(h, 0.)  # Pose at engine startup, now obsolete.
    clock[0] += 30.
    measured(h, 1.)
    command(h, 1.04)
    h.publish_smoothed_arm()
    assert sent == [[1.]*6]
    clock[0] += .008
    measured(h, 1.)
    h.publish_smoothed_arm()
    assert 0 < sent[-1][0]-1. <= .15*.008**2+1e-12


@pytest.mark.parametrize('value', [.11, float('nan'), float('inf')])
def test_invalid_first_target_never_moves_arm_or_gripper(value):
    h, _, sent, gripper = engine()
    measured(h, 0.)
    command(h, value, 0.)
    h.publish_smoothed_arm()
    assert h.fault and not sent and not gripper


def test_servo_ack_and_fresh_feedback_required_before_accepting_commands():
    h, clock, sent, gripper = engine()
    measured(h, 0.)
    h.servo_stopped = False
    command(h, .01)
    assert h.target is None and not gripper
    h.servo_stopped = True
    clock[0] += .21
    command(h, .01)
    assert h.target is None and not gripper
    measured(h, 0.)
    command(h, .01)
    h.publish_smoothed_arm()
    assert sent == [[0.]*6]


@pytest.mark.parametrize('failure', ['feedback', 'timeout', 'tracking', 'publisher'])
def test_active_failure_latches_and_cannot_resume_after_manual_reposition(failure):
    h, clock, sent, gripper = engine()
    measured(h, 0.)
    command(h, .01)
    h.publish_smoothed_arm()
    sent.clear()
    if failure == 'feedback':
        clock[0] += .21
    elif failure == 'timeout':
        clock[0] += .51
        measured(h, 0.)
    elif failure == 'tracking':
        measured(h, .06)
    else:
        h.count_publishers = lambda _: 2
    h.publish_smoothed_arm()
    assert h.fault and not sent
    h.count_publishers = lambda _: 1
    measured(h, 1.)
    command(h, 1.01, 0.)
    h.publish_smoothed_arm()
    assert not sent and gripper == [1.]


def test_delayed_new_command_cannot_bypass_timeout_between_timer_callbacks():
    h, clock, _, gripper = engine()
    measured(h, 0.)
    command(h, .01)
    clock[0] += .51
    measured(h, 1.)
    command(h, 1.01, 0.)
    assert h.fault and gripper == [1.]


def test_dry_run_never_calls_servo_service():
    h, _, _, _ = engine()
    h.dry_run = True
    h.servo_stop_client = SimpleNamespace(service_is_ready=lambda: pytest.fail('dry-run called Servo'))
    h.ensure_servo_stopped()


def test_gripper_first_write_failure_unblocks_transaction_and_latches_fault():
    h, _, _, _ = engine()
    h.gripper_busy = True
    h.io_finished(SimpleNamespace(result=lambda: SimpleNamespace(success=False)), [(0, 0.), (1, 1.)], 0, 1.)
    assert not h.gripper_busy and h.fault and h.last_gripper is None



def test_automatic_gripper_serializes_close_and_open_and_confirms_only_completed_io():
    h, _, _, _ = engine()
    measured(h, 0.)
    h.start_gripper = type(h).start_gripper.__get__(h)
    writes, futures, acknowledgments = [], [], []
    class Future:
        def add_done_callback(self, callback): self.callback = callback
        def result(self): return SimpleNamespace(success=True)
        def finish(self): self.callback(self)
    def write(request):
        writes.append((request.pin, request.state))
        future = Future(); futures.append(future)
        return future
    h.io_client = SimpleNamespace(service_is_ready=lambda: True, call_async=write)
    h.gripper_state_publisher = SimpleNamespace(publish=lambda msg: acknowledgments.append(msg.data))
    command(h, .01, 0.)
    assert writes == [(1, 0.)] and h.gripper_busy and not acknowledgments
    # A later policy target cannot interleave another output pair.
    command(h, .01, 1.)
    assert len(writes) == 1
    futures[0].finish()
    assert writes == [(1, 0.), (0, 1.)] and not acknowledgments
    futures[1].finish()
    assert acknowledgments == [0.] and not h.gripper_busy
    command(h, .01, 1.)
    futures[2].finish()
    futures[3].finish()
    assert writes == [(1, 0.), (0, 1.), (0, 0.), (1, 1.)]
    assert acknowledgments == [0., 1.] and not h.gripper_busy
