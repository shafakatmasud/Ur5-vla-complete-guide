"""Read-only checkerboard preview and camera/robot pose snapshots.

No publishers or service clients. Press s to save a stationary sample, q to quit.
Saved corner ordering is NOT physically oriented; resolve board orientation
consistently before a hand-eye solve. No calibration is produced by this tool.
"""
import argparse
import json
import time
import threading
from pathlib import Path

import cv2
import numpy as np


def board_observation(image, k, distortion):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(gray, (8, 6), cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        return {'detected': False}, image.copy()
    obj = np.zeros((48, 3), dtype=np.float32)
    obj[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2) * .02
    matrix = np.asarray(k, dtype=float).reshape(3, 3)
    dist = np.asarray(distortion, dtype=float)
    ok, rvec, tvec = cv2.solvePnP(obj, corners, matrix, dist)
    if not ok or tvec[2, 0] <= 0:
        return {'detected': False, 'reason': 'Invalid board pose'}, image.copy()
    projected, _ = cv2.projectPoints(obj, rvec, tvec, matrix, dist)
    residual = np.linalg.norm(projected.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)
    annotated = image.copy()
    cv2.drawChessboardCorners(annotated, (8, 6), corners, True)
    return {'detected': True, 'corner_count': 48, 'corners_uv': corners.reshape(-1, 2).tolist(),
            'rotation_vector': rvec.ravel().tolist(), 'translation_camera_m': tvec.ravel().tolist(),
            'reprojection_rms_px': float(np.sqrt(np.mean(residual ** 2))),
            'orientation_verified': False}, annotated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=30.)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--camera', choices=('both', 'wrist'), default='both')
    parser.add_argument('--auto-save', action='store_true', help='Save distinct stationary views, up to 20 total samples')
    parser.add_argument('--capture-once', action='store_true', help='Save first paired detection; initial reference only')
    args = parser.parse_args()
    cameras = ('wrist',) if args.camera == 'wrist' else ('base', 'wrist')
    if not 0 < args.duration <= 3600:
        parser.error('Duration must be greater than zero and at most 3600 seconds')
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, CameraInfo
    from cv_bridge import CvBridge
    from tf2_ros import Buffer, TransformListener
    from rclpy.time import Time
    from rclpy.duration import Duration
    from rclpy.executors import SingleThreadedExecutor

    rclpy.init()
    node = Node('checkerboard_capture_readonly')
    tf = Buffer()
    listener = TransformListener(tf, node)
    bridge = CvBridge()
    latest = {}
    subscriptions = []
    for cam in cameras:
        for key, suffix, typ in [('image', 'image_raw', Image), ('info', 'camera_info', CameraInfo)]:
            def receive(msg, c=cam, k=key):
                latest[c, k] = (msg, time.monotonic())
            subscriptions.append(node.create_subscription(typ, f'/{cam}/camera/color/{suffix}', receive, qos_profile_sensor_data))
    args.output.mkdir(parents=True, exist_ok=True)
    saved_poses = []
    for path in sorted(args.output.glob('sample_*/observation.json')):
        prior = json.loads(path.read_text())['cameras']['wrist']
        if 'base_from_tool0_at_image_stamp' in prior:
            saved_poses.append(prior['base_from_tool0_at_image_stamp'])
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    deadline = time.monotonic() + args.duration
    next_check = 0.
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            time.sleep(.02)
            if time.monotonic() < next_check:
                continue
            next_check = time.monotonic() + 1.
            result = {'wall_time_s': time.time(), 'robot_motion_enabled': False,
                      'calibration_complete': False, 'cameras': {}}
            raw = {}
            for cam in cameras:
                try:
                    msg, arrival = latest[cam, 'image']
                    info, info_arrival = latest[cam, 'info']
                    if max(time.monotonic() - arrival, time.monotonic() - info_arrival) > .5:
                        raise ValueError('Stale image or intrinsics')
                    if msg.header.frame_id != info.header.frame_id or (msg.width, msg.height) != (info.width, info.height):
                        raise ValueError('Image and intrinsics mismatch')
                    if info.distortion_model not in ('plumb_bob', ''):
                        raise ValueError('Unsupported camera distortion model')
                    image = bridge.imgmsg_to_cv2(msg, 'bgr8')
                    raw[cam] = image
                    observation, annotated = board_observation(image, info.k, info.d)
                    observation.update(camera_topic=f'/{cam}/camera/color/image_raw', source_frame=msg.header.frame_id,
                                       stamp_s=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                                       intrinsic_matrix=list(info.k), distortion=list(info.d))
                    if cam == 'wrist':
                        try:
                            pose = tf.lookup_transform('base', 'tool0_controller', Time.from_msg(msg.header.stamp), timeout=Duration(seconds=1.5))
                            t, q = pose.transform.translation, pose.transform.rotation
                            observation['base_from_tool0_at_image_stamp'] = {'xyz_m': [t.x, t.y, t.z], 'quaternion_xyzw': [q.x, q.y, q.z, q.w]}
                            earlier = tf.lookup_transform('base', 'tool0_controller',
                                Time.from_msg(msg.header.stamp) - Duration(seconds=.5))
                            et, eq = earlier.transform.translation, earlier.transform.rotation
                            drift = float(np.linalg.norm(np.array([t.x-et.x, t.y-et.y, t.z-et.z])))
                            qa = np.array([q.x, q.y, q.z, q.w]); qb = np.array([eq.x, eq.y, eq.z, eq.w])
                            angle = float(2 * np.arccos(np.clip(abs(np.dot(qa, qb) / (np.linalg.norm(qa) * np.linalg.norm(qb))), 0, 1)))
                            observation['stationarity'] = {'interval_s': .5, 'translation_m': drift,
                                'rotation_rad': angle, 'passed': bool(drift < .0005 and angle < np.deg2rad(.15))}
                        except Exception as exc:
                            observation['robot_pose_error'] = str(exc)
                    result['cameras'][cam] = observation
                    cv2.putText(annotated, '48 corners' if observation['detected'] else 'BOARD NOT DETECTED', (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
                    cv2.imwrite(str(args.output / f'{cam}_latest.jpg'), annotated)
                    if args.show:
                        cv2.imshow(f'Calibration / {cam} / READ ONLY: s=save q=quit', annotated)
                except (KeyError, ValueError, cv2.error) as exc:
                    result['cameras'][cam] = {'detected': False, 'reason': str(exc)}
            (args.output / 'latest.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
            key = cv2.waitKey(1) & 255 if args.show else -1
            if key == ord('q'):
                break
            paired = (all(result['cameras'][cam].get('detected', False)
                          and result['cameras'][cam].get('reprojection_rms_px', 999) < 1. for cam in cameras)
                      and 'base_from_tool0_at_image_stamp' in result['cameras']['wrist']
                      and result['cameras']['wrist'].get('stationarity', {}).get('passed', False))
            pose = result['cameras'].get('wrist', {}).get('base_from_tool0_at_image_stamp')
            distinct = pose is not None
            if distinct:
                for previous in saved_poses:
                    distance = np.linalg.norm(np.asarray(pose['xyz_m']) - previous['xyz_m'])
                    qa, qb = np.asarray(pose['quaternion_xyzw']), np.asarray(previous['quaternion_xyzw'])
                    angle = 2 * np.arccos(np.clip(abs(np.dot(qa, qb) / (np.linalg.norm(qa) * np.linalg.norm(qb))), 0, 1))
                    if distance < .015 and angle < np.deg2rad(8):
                        distinct = False
                        break
            if paired and distinct and (args.capture_once or key == ord('s') or (args.auto_save and len(saved_poses) < 20)):
                sample = args.output / ('sample_' + str(time.time_ns()))
                sample.mkdir()
                (sample / 'observation.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
                for cam, image in raw.items():
                    cv2.imwrite(str(sample / f'{cam}_raw.png'), image)
                print(f'Saved stationary reference (board orientation unverified): {sample}', flush=True)
                saved_poses.append(pose)
                print(f'Total distinct samples: {len(saved_poses)} / 20', flush=True)
                if args.capture_once:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        spin_thread.join(timeout=2)
        node.destroy_node()
        rclpy.shutdown()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
