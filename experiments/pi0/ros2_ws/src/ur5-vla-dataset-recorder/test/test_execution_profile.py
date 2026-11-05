"""Verify slower motion limits and shorter, fresh policy execution horizons."""
import ast
from collections import deque
import math
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts'))
from execution_profile import validated_speed, validated_horizon


@pytest.mark.parametrize('speed', [0, -.1, 1.01, float('nan'), float('inf')])
def test_invalid_speed_is_rejected(speed):
    with pytest.raises(ValueError):
        validated_speed(speed)


@pytest.mark.parametrize('horizon', [0, -1, 51])
def test_invalid_horizon_is_rejected(horizon):
    with pytest.raises(ValueError):
        validated_horizon(horizon)


def harness(script, cls_name, methods, namespace):
    tree = ast.parse((ROOT / 'scripts' / script).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    selected = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    module = ast.Module(body=[ast.ClassDef(name='Harness', bases=[], keywords=[], body=selected, decorator_list=[])], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, 'execution_methods', 'exec'), namespace)
    instance = namespace['Harness']()
    instance.task = namespace.get('TASK', 'test')
    return instance


class Message:
    pass


def test_half_speed_limits_actual_command_velocity_and_acceleration():
    def run(scale):
        clock = [0.]
        h = harness('control_engine.py', 'ControlEngine', {'publish_smoothed_arm'},
                    dict(math=math, time=SimpleNamespace(monotonic=lambda: clock[0]), Float64MultiArray=Message, JOINT_TIMEOUT=.2, COMMAND_TIMEOUT=.5, MAX_TRACKING_ERROR=.05))
        h.fault = None
        h.update_bowl_release = lambda now: None
        h.startup_hold = False
        h.measured = [0.]*6
        h.measured_at = h.last_command_at = 0.
        h.target = [1.]*6
        h.command = [0.]*6
        h.filtered_target = [0.]*6
        h.velocity = [0.]*6
        h.last_tick = 0.
        h.max_velocity = .1*scale
        h.max_acceleration = .3*scale
        h.filter_time_constant = .28/scale
        h.count_publishers = lambda _: 1
        h.output_count = 0
        h.last_motion_diagnostics = 0.
        h.get_logger = lambda: SimpleNamespace(info=lambda msg: None)
        h.dry_run = False
        sent = []
        h.arm_publisher = SimpleNamespace(publish=lambda msg: sent.append(msg.data))
        positions = [0.]
        previous_speed = 0.
        for _ in range(500):
            clock[0] += .008
            h.measured = list(h.command)
            h.measured_at = h.last_command_at = clock[0]
            h.publish_smoothed_arm()
            speed = (sent[-1][0]-positions[-1])/.008
            assert abs(speed) <= h.max_velocity+1e-9
            assert abs(speed-previous_speed)/.008 <= h.max_acceleration+1e-7
            positions.append(sent[-1][0])
            previous_speed = speed
        return positions[-1]
    assert run(.5) < .6*run(1.)


@pytest.mark.parametrize("horizon", [10, 50])
@pytest.mark.parametrize('task', ['Pick up the orange cube and place it in the bowl',
                                  'Pick up the blue cube and place it in the bowl'])
def test_selected_horizon_preserves_action_order(horizon, task):
    chunk = [[float(i)]*6+[float(i<5)] for i in range(50)]
    requested_tasks = []
    def predict(*args):
        requested_tasks.append(args[4])
        return chunk, 0.
    h = harness('policy_executor.py', 'PolicyExecutor', {'infer'},
                dict(time=time, request_prediction=predict,
                     SERVER_HOST='localhost', SERVER_PORT=8765, TASK=task))
    h.queue_lock = threading.Lock()
    h.action_queue = deque()
    h.last_action = None
    h.approach_schedule = None
    h.grasp_assist = h.trajectory_offset = None
    h.offset_fault = False
    h.inference_generation = 0
    h.inference_running = True
    h.actions_per_prediction = horizon
    h.blend_actions = 5
    h.get_logger = lambda: SimpleNamespace(info=lambda msg: None, error=lambda msg: None)
    h.infer([0.]*7, [], 0)
    assert list(h.action_queue) == chunk[:horizon]
    assert not h.inference_running
    assert requested_tasks == [task]


def test_carry_lift_default_matches_supported_execution_mode_and_allows_opt_out():
    from execution_profile import resolve_carry_lift_mm
    assert resolve_carry_lift_mm(None,'continuous',True)==25.
    assert resolve_carry_lift_mm(0.,'continuous',True)==0.
    assert resolve_carry_lift_mm(None,'approach',True)==0.
    assert resolve_carry_lift_mm(None,'continuous',False)==0.
    assert resolve_carry_lift_mm(20.,'continuous',True)==20.


@pytest.mark.parametrize('value,mode,has_file',[(31.,'continuous',True),
    (float('nan'),'continuous',True),(25.,'approach',True),(25.,'continuous',False)])
def test_unsupported_carry_lift_configuration_is_rejected(value,mode,has_file):
    from execution_profile import resolve_carry_lift_mm
    with pytest.raises(ValueError):resolve_carry_lift_mm(value,mode,has_file)
