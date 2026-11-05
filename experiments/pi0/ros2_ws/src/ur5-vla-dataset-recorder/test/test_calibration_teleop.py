import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'scripts'))
from calibration_teleop import joystick_twist


@pytest.mark.parametrize('enabled,trigger', [(False, 1), (True, 0), (False, 0)])
def test_deadman_and_enable_both_required(enabled, trigger):
    assert joystick_twist([1, 1, 1], [trigger, 0, 0, 1, 0, 1], enabled) == ([0., 0., 0.], [0., 0., 0.])


def test_tilt_mode_disables_all_translation():
    linear, angular = joystick_twist([1, 1, 1], [1, 0, 0, 1, 0, 1], True)
    assert linear == [0., 0., 0.]
    assert angular == [-.1, .1, .1]


def test_translation_mode_speed_and_axes():
    assert joystick_twist([1, 1, 1], [1, 0, 0, 0, 0, 1], True) == ([-.015, .015, .015], [0., 0., .1])


def test_gripper_and_reset_buttons_do_not_generate_motion():
    assert joystick_twist([0, 0, 0], [1, 1, 1, 0, 0, 0, 1], True) == ([0., 0., 0.], [0., 0., 0.])


@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
def test_nonfinite_input_generates_no_motion(value):
    assert joystick_twist([1, value, 1], [1, 0, 0, 0], True) == ([0., 0., 0.], [0., 0., 0.])


def test_missing_and_excessive_axes():
    assert joystick_twist([], [], True) == ([0., 0., 0.], [0., 0., 0.])
    assert joystick_twist([10, -10, 10], [1], True) == ([.015, .015, 0.], [0., 0., .1])
