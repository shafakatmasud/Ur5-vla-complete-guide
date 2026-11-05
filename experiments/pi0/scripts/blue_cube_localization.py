"""Read-only blue RGB-D localization. No robot commands or target publishers.

Points describe the visible blue surface, not a calibrated finger grasp centre.
Do not pass these points to an executor without extrinsic/TCP validation.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def detect_blue(color, depth, encoding, k, distortion=(), roi=None):
    """Return a unique, unclipped blue candidate with robust surface depth."""
    if color.shape[:2] != depth.shape or depth.ndim != 2:
        raise ValueError('Aligned color/depth dimensions differ')
    if encoding not in ('16UC1', '32FC1'):
        raise ValueError('Unsupported depth encoding')
    matrix = np.asarray(k, dtype=float).reshape(3, 3)
    if not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError('Invalid camera intrinsics')
    h, w = depth.shape
    x0, y0, x1, y1 = (0, 0, w, h) if roi is None else roi
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError('ROI must be inside image')
    mask = cv2.inRange(cv2.cvtColor(color, cv2.COLOR_BGR2HSV),
                       np.array([95, 90, 45]), np.array([135, 255, 255]))
    allowed = np.zeros_like(mask)
    allowed[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    contours, _ = cv2.findContours(allowed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, bw, bh = cv2.boundingRect(contour)
        if area < 120 or not .45 <= bw / bh <= 2.2 or area / (bw * bh) < .55:
            continue
        # Clipped cubes and blue edge clutter cannot supply a reliable centroid.
        if x <= x0 + 2 or y <= y0 + 2 or x + bw >= x1 - 2 or y + bh >= y1 - 2:
            continue
        region = np.zeros_like(mask)
        cv2.drawContours(region, [contour], -1, 255, -1)
        region = cv2.erode(region, np.ones((5, 5), np.uint8))
        ys, xs = np.where((region > 0) & (mask > 0))
        if len(xs) < 30:
            continue
        z = depth[ys, xs].astype(float) * (.001 if encoding == '16UC1' else 1.)
        valid = np.isfinite(z) & (z > .1) & (z < 2.)
        if np.mean(valid) < .7:
            continue
        median = float(np.median(z[valid]))
        spread = float(np.percentile(z[valid], 90) - np.percentile(z[valid], 10))
        if spread > .06:
            continue
        inliers = valid & (np.abs(z - median) < .015)
        if np.count_nonzero(inliers) < 30:
            continue
        # Distortion-aware projection of the robust interior pixel centre.
        pixel = np.array([[[np.median(xs[inliers]), np.median(ys[inliers])]]], dtype=float)
        ray = cv2.undistortPoints(pixel, matrix, np.asarray(distortion, dtype=float) if len(distortion) else None)[0, 0]
        candidates.append({'bbox_xywh': [x, y, bw, bh], 'pixel_uv': pixel[0, 0].tolist(),
                           'surface_xyz_camera_m': [float(ray[0] * median), float(ray[1] * median), median],
                           'depth_spread_m': spread, 'area_px': area})
    return {'status': 'unique_candidate' if len(candidates) == 1 else 'ambiguous' if candidates else 'no_candidate',
            'candidate': candidates[0] if len(candidates) == 1 else None, 'candidates': candidates,
            'robot_motion_enabled': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=30.)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--base-roi', type=int, nargs=4, metavar=('X0', 'Y0', 'X1', 'Y1'))
    args = parser.parse_args()
    if not 0 < args.duration <= 3600:
        parser.error('--duration must be between 0 and 3600 seconds')
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, CameraInfo
    from cv_bridge import CvBridge
    rclpy.init()
    node = Node('blue_cube_localization_readonly')
    bridge = CvBridge()
    latest = {}
    subscriptions = []
    for camera in ('base', 'wrist'):
        for key, suffix, kind in [('color', 'color/image_raw', Image), ('depth', 'aligned_depth_to_color/image_raw', Image), ('info', 'color/camera_info', CameraInfo)]:
            def receive(message, c=camera, k=key):
                latest[c, k] = (message, time.monotonic())
            subscriptions.append(node.create_subscription(kind, f'/{camera}/camera/{suffix}', receive, qos_profile_sensor_data))
    args.output.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.duration
    next_save = 0.
    try:
        with (args.output / 'observations.jsonl').open('a') as stream:
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=.02)
                now = time.monotonic()
                if now < next_save:
                    continue
                next_save = now + .5
                report = {'wall_time_s': time.time(), 'robot_motion_enabled': False, 'cameras': {}}
                for camera in ('base', 'wrist'):
                    try:
                        entries = [latest[camera, key] for key in ('color', 'depth', 'info')]
                        if any(now - arrival > .5 for _, arrival in entries):
                            raise ValueError('Stale camera inputs')
                        color_msg, depth_msg, info = [entry[0] for entry in entries]
                        stamp = lambda m: m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
                        if abs(stamp(color_msg) - stamp(depth_msg)) > .05:
                            raise ValueError('Color/depth source stamps differ by more than 50 ms')
                        if not color_msg.header.frame_id == depth_msg.header.frame_id == info.header.frame_id:
                            raise ValueError('Color/depth/intrinsics frames differ')
                        if info.distortion_model not in ('plumb_bob', ''):
                            raise ValueError('Unsupported distortion model')
                        color = bridge.imgmsg_to_cv2(color_msg, 'bgr8')
                        depth = bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
                        if (info.width, info.height) != (color.shape[1], color.shape[0]):
                            raise ValueError('CameraInfo dimensions differ')
                        result = detect_blue(color, depth, depth_msg.encoding, info.k, info.d,
                                             args.base_roi if camera == 'base' else None)
                        result.update(source_frame=info.header.frame_id, source_stamp_s=stamp(color_msg),
                                      calibration_status='UNAVAILABLE: no validated camera-to-base or grasp TCP calibration')
                        for candidate in result['candidates']:
                            x, y, w, h = candidate['bbox_xywh']
                            cv2.rectangle(color, (x, y), (x + w, y + h), (0, 255, 255), 2)
                        cv2.putText(color, result['status'], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
                        cv2.putText(color, 'READ ONLY - ROBOT TARGET BLOCKED', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
                        cv2.imwrite(str(args.output / f'{camera}_latest.jpg'), color)
                        if args.show:
                            cv2.imshow(f'Blue cube / {camera} / READ ONLY', color)
                        report['cameras'][camera] = result
                    except (KeyError, ValueError, cv2.error) as exc:
                        report['cameras'][camera] = {'status': 'blocked', 'reason': str(exc)}
                stream.write(json.dumps(report, allow_nan=False) + '\n')
                stream.flush()
                (args.output / 'latest.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
                if args.show and cv2.waitKey(1) & 0xff == ord('q'):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
