"""Regression: successful save followed by an idle retry uses valid ROS logging."""
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import rclpy.logging
from std_srvs.srv import Trigger
from ur5_vla_dataset_recorder.recorder_node import DatasetRecorder


def test_save_then_idle_retry_does_not_change_logging_callsite_severity(tmp_path):
    logger = rclpy.logging.get_logger("recorder_save_regression")
    errors = []
    proxy = SimpleNamespace(info=logger.info, warning=logger.warning, error=errors.append)
    state = SimpleNamespace(
        episode_active=True, episode_stopped=False,
        joystick_control_lock=threading.Lock(), joystick_control_busy=True,
        get_logger=lambda: proxy,
    )
    def stop(*args):
        state.episode_active = False
        state.episode_stopped = True
        return Trigger.Response(success=True, message="stopped")
    def save(*args):
        state.episode_stopped = False
        return Trigger.Response(success=True, message="saved")
    state.stop_episode = Mock(side_effect=stop)
    state.save_success = Mock(side_effect=save)
    DatasetRecorder._run_joystick_action(state, "save_success")
    state.joystick_control_busy = True
    DatasetRecorder._run_joystick_action(state, "save_success")
    assert not errors
    assert not state.joystick_control_busy
    state.stop_episode.assert_called_once()
    state.save_success.assert_called_once()
