#!/usr/bin/python3
"""Phased executor adapter using the shared startup, feedback and IO guards."""

from __future__ import annotations

import argparse

from control_engine_lock import acquire_engine_lock

import rclpy

from control_engine import ControlEngine
from execution_profile import DEFAULT_SPEED_SCALE, validated_speed


class PhasedControlEngine(ControlEngine):
    def __init__(self, dry_run=False, validation_count=0, initial_gripper=1.0,
                 speed_scale=DEFAULT_SPEED_SCALE):
        # Compatibility argument only: acknowledgments require successful IO.
        super().__init__(dry_run=dry_run, validation_count=validation_count,
                         speed_scale=speed_scale)
        self.get_logger().info("phased engine uses shared startup and feedback guards")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validation-count", type=int, default=0)
    parser.add_argument(
        "--initial-gripper",
        choices=("open", "closed"),
        default="open",
        help="compatibility flag; gripper acknowledgments require successful IO",
    )
    parser.add_argument("--speed-scale", type=validated_speed, default=DEFAULT_SPEED_SCALE)
    args, ros_args = parser.parse_known_args()
    initial = 1.0 if args.initial_gripper == "open" else 0.0
    try:
        engine_lock = acquire_engine_lock()
    except RuntimeError as error:
        parser.exit(2, f"{error}\n")
    rclpy.init(args=ros_args)
    node = PhasedControlEngine(
        dry_run=args.dry_run,
        validation_count=args.validation_count,
        initial_gripper=initial,
        speed_scale=args.speed_scale,
    )
    try:
        while rclpy.ok() and not node.validation_done:
            rclpy.spin_once(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        engine_lock.close()


if __name__ == "__main__":
    main()
