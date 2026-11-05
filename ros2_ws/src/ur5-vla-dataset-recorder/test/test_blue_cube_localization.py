"""Depth/ambiguity and measured-extrinsic regression checks without ROS."""
import sys
from pathlib import Path
import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'scripts'))
from blue_cube_localization import detect_blue
from fit_camera_extrinsics import fit


def scene():
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.rectangle(rgb, (60, 40), (90, 70), (255, 0, 0), -1)
    return rgb, np.full((120, 160), 500, dtype=np.uint16)


K = [100, 0, 80, 0, 100, 60, 0, 0, 1]


def test_metric_projection_and_no_actuators():
    result = detect_blue(*scene(), '16UC1', K)
    assert result['status'] == 'unique_candidate'
    assert result['candidate']['surface_xyz_camera_m'] == pytest.approx([-.025, -.025, .5])
    assert not result['robot_motion_enabled']


def test_float_depth_matches_mm():
    rgb, depth = scene()
    assert detect_blue(rgb, depth.astype(np.float32) * .001, '32FC1', K)['candidate']['surface_xyz_camera_m'] == pytest.approx([-.025, -.025, .5])


def test_multiple_blue_objects_ambiguous():
    rgb, depth = scene()
    cv2.rectangle(rgb, (10, 10), (35, 35), (255, 0, 0), -1)
    assert detect_blue(rgb, depth, '16UC1', K)['status'] == 'ambiguous'


def test_clipped_blue_is_rejected():
    rgb, depth = scene()
    rgb[:] = 0
    cv2.rectangle(rgb, (140, 40), (159, 70), (255, 0, 0), -1)
    assert detect_blue(rgb, depth, '16UC1', K)['status'] == 'no_candidate'


@pytest.mark.parametrize('value', [0, 65535])
def test_invalid_depth_rejected(value):
    rgb, depth = scene()
    depth[:] = value
    assert detect_blue(rgb, depth, '16UC1', K)['candidate'] is None


def test_unaligned_depth_rejected():
    rgb, depth = scene()
    with pytest.raises(ValueError, match='dimensions'):
        detect_blue(rgb, depth[:10], '16UC1', K)


def calibration():
    a = np.array([[0, 0, .4], [.1, 0, .4], [0, .1, .4], [.1, .1, .4]])
    v = a + [.02, .02, .03]
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]])
    shift = [.3, -.2, .1]
    return dict(camera_serial='test', camera_namespace='base', source_frame='optical', target_frame='base',
                camera_points_m=a.tolist(), robot_points_m=(a @ rotation.T + shift).tolist(),
                validation_camera_points_m=v.tolist(), validation_robot_points_m=(v @ rotation.T + shift).tolist())


def test_extrinsic_direction_with_heldout_points():
    result = fit(calibration())
    assert result['point_validation_passed']
    matrix = np.array(result['matrix_camera_to_base'])
    assert matrix[:3, 3] == pytest.approx([.3, -.2, .1])
    assert np.linalg.det(matrix[:3, :3]) == pytest.approx(1.)
    assert not result['robot_motion_enabled']


def test_incorrect_validation_fails():
    doc = calibration()
    doc['validation_robot_points_m'][0][0] += .02
    assert not fit(doc)['point_validation_passed']


def test_reused_calibration_points_rejected():
    doc = calibration()
    doc['validation_camera_points_m'] = doc['camera_points_m']
    with pytest.raises(ValueError, match='held out'):
        fit(doc)


def test_collinear_calibration_rejected():
    doc = calibration()
    doc['camera_points_m'] = [[i * .1, 0, .4] for i in range(4)]
    with pytest.raises(ValueError, match='spread'):
        fit(doc)
