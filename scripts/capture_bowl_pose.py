#!/usr/bin/python3
"""Record a stationary operator-taught bowl pose from TF. Publishes no commands."""
import argparse
import json
import math
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new calibration filename')
    import rclpy
    from rclpy.node import Node
    from tf2_ros import Buffer, TransformListener
    rclpy.init()
    node = Node('capture_operator_taught_bowl_pose')
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    positions = []
    quaternions = []
    last_stamp = None
    deadline = time.monotonic() + 10
    try:
        while time.monotonic() < deadline and len(positions) < 30:
            rclpy.spin_once(node, timeout_sec=.05)
            try:
                transform = buffer.lookup_transform('base', 'tool0_controller', rclpy.time.Time())
            except Exception:
                continue
            stamp = transform.header.stamp
            stamp_ns = stamp.sec * 10**9 + stamp.nanosec
            age = (node.get_clock().now().nanoseconds - stamp_ns) / 1e9
            if not 0 <= age <= .2 or stamp_ns == last_stamp:
                continue
            last_stamp = stamp_ns
            t = transform.transform.translation
            q = transform.transform.rotation
            positions.append([t.x, t.y, t.z])
            quaternions.append([q.x, q.y, q.z, q.w])
        if len(positions) < 30:
            raise RuntimeError('Not enough fresh base -> tool0_controller TF samples')
        reference_q = quaternions[-1]
        if any(math.dist(p, positions[-1]) > .001 for p in positions):
            raise RuntimeError('Arm moved more than 1 mm during sampling; teach again while stationary')
        if any(2 * math.acos(min(1, abs(sum(a*b for a,b in zip(q, reference_q))))) > math.radians(1)
               for q in quaternions):
            raise RuntimeError('Arm orientation changed more than 1 degree')
        value = dict(frame='base', tip='tool0_controller', position_m=positions[-1],
                     quaternion_xyzw=reference_q, captured_unix_s=time.time(),
                     qualification='Operator-taught reference; bowl dimensions and route clearance require inspection')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            stream.write(json.dumps(value, indent=2) + '\n')
        print(args.output.resolve())
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
