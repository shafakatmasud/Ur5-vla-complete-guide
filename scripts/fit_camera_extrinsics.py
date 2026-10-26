"""Fit explicitly measured corresponding 3D points; never move the robot.

Input JSON: camera_serial, camera_namespace, source_frame, target_frame='base',
camera_points_m, robot_points_m, validation_camera_points_m,
validation_robot_points_m. Each pair must identify the SAME physical point.
Robot flange positions paired with cube surfaces are NOT correspondences.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def points(value):
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) < 4 or not np.isfinite(array).all():
        raise ValueError('Need at least four finite measured XYZ points per set')
    if np.linalg.svd(array - array.mean(axis=0), compute_uv=False)[1] < .02:
        raise ValueError('Points need two-dimensional spread of at least 20 mm')
    return array


def fit(document, tolerance_m=.005):
    if not 0 < tolerance_m <= .005:
        raise ValueError('Validation tolerance must be positive and at most 5 mm')
    for key in ('camera_serial', 'camera_namespace', 'source_frame'):
        if not isinstance(document.get(key), str) or not document[key].strip():
            raise ValueError(f'Missing {key}')
    if document.get('target_frame') != 'base':
        raise ValueError('Target frame must explicitly be base')
    a, b, va, vb = [points(document[key]) for key in (
        'camera_points_m', 'robot_points_m', 'validation_camera_points_m', 'validation_robot_points_m')]
    if a.shape != b.shape or va.shape != vb.shape:
        raise ValueError('Corresponding point counts differ')
    if np.min(np.linalg.norm(va[:, None, :] - a[None, :, :], axis=2)) < .01:
        raise ValueError('Validation points must be held out at distinct positions, at least 10 mm from fit points')
    u, _, vt = np.linalg.svd((a - a.mean(axis=0)).T @ (b - b.mean(axis=0)))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    translation = b.mean(axis=0) - rotation @ a.mean(axis=0)
    fit_error = np.linalg.norm(a @ rotation.T + translation - b, axis=1)
    validation_error = np.linalg.norm(va @ rotation.T + translation - vb, axis=1)
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, translation
    return {key: document[key] for key in ('camera_serial', 'camera_namespace', 'source_frame', 'target_frame')} | {
        'matrix_camera_to_base': transform.tolist(), 'fit_errors_m': fit_error.tolist(),
        'held_out_errors_m': validation_error.tolist(), 'tolerance_m': tolerance_m,
        'point_validation_passed': bool(max(fit_error.max(), validation_error.max()) <= tolerance_m),
        'robot_motion_enabled': False,
        'remaining_requirements': ['Independently verify camera mounting and axis directions',
                                   'Calibrate gripper grasp centre relative to tool0',
                                   'Validate table/cube geometry and collision-aware pickup path'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('measurements', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = fit(json.loads(args.measurements.read_text()))
    if args.output.exists():
        parser.error('Output already exists; use a new calibration filename')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('Point validation passed' if result['point_validation_passed'] else 'Point validation FAILED')
    print('Read-only calibration artifact; robot motion remains disabled')


if __name__ == '__main__':
    main()
