#!/usr/bin/python3
"""Smooth seven-value abstract commands for the UR5 arm and gripper."""

import argparse

from control_engine_lock import acquire_engine_lock
import math
import json
import sys
import time

from execution_profile import DEFAULT_SPEED_SCALE, StableGripperCommand, validated_speed
from project_paths import PACKAGE

sys.path.insert(0, str(PACKAGE))

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray
from std_srvs.srv import Trigger
from ur_msgs.srv import SetIO

from ur5_vla_dataset_recorder.core import UR5_JOINT_ORDER, reorder_joint_positions


OUTPUT_HZ = 125.0
FILTER_TIME_CONSTANT = 0.28
MAX_VELOCITY = 0.10       # rad/s per joint
MAX_ACCELERATION = 0.30   # rad/s^2 per joint
MAX_TARGET_STEP = 0.15    # rad between consecutive 20 Hz policy targets
MAX_INITIAL_ERROR = 0.10  # rad from fresh measured pose to first target
JOINT_TIMEOUT = 0.20
COMMAND_TIMEOUT = 0.50
MAX_TRACKING_ERROR = 0.05  # rad commanded versus measured, per joint


class ControlEngine(Node):
    def __init__(self, dry_run=False, validation_count=0, speed_scale=DEFAULT_SPEED_SCALE,
                 bowl_release_file=None):
        super().__init__("control_engine")
        scale = validated_speed(speed_scale)
        self.max_velocity = MAX_VELOCITY * scale
        self.max_acceleration = MAX_ACCELERATION * scale
        self.filter_time_constant = FILTER_TIME_CONSTANT / scale
        self.measured = None
        self.measured_at = 0.0
        self.last_command_at = 0.0
        self.startup_hold = False
        self.fault = None
        self.fault_kind = None
        self.servo_stopped = dry_run
        self.servo_stop_future = None
        self.dry_run = dry_run
        self.validation_count = validation_count
        self.output_count = 0
        self.validation_done = False
        self.last_source_warning = 0.0
        self.last_motion_diagnostics = 0.0
        self.target = None
        self.filtered_target = None
        self.command = None
        self.velocity = [0.0] * 6
        self.last_raw_target = None
        self.last_tick = time.monotonic()
        controller_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.arm_publisher = self.create_publisher(
            Float64MultiArray,
            "/forward_position_controller/commands",
            controller_qos,
        )
        self.create_subscription(
            Float64MultiArray,
            "/control_abstraction_layer",
            self.command_callback,
            1,
        )
        self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            qos_profile_sensor_data,
        )
        self.io_client = self.create_client(
            SetIO, "/io_and_status_controller/set_io"
        )
        self.servo_stop_client = self.create_client(
            Trigger, "/servo_node/stop_servo"
        )
        self.last_gripper = None
        self.gripper_filter = StableGripperCommand()
        self.bowl_release = None
        self.last_bowl_diagnostics = 0.
        if bowl_release_file is not None:
            from bowl_release import BowlRelease
            from tf2_ros import Buffer, TransformListener
            self.bowl_release = BowlRelease(bowl_release_file)
            self.release_tf = Buffer()
            self.release_listener = TransformListener(self.release_tf, self)
        self.gripper_busy = False
        self.confirmed_gripper = None
        self.gripper_state_publisher = self.create_publisher(
            Float64, "/control_engine/gripper_state",
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.create_service(Trigger, "/control_engine/status", self.status_callback)
        self.create_service(Trigger, "/control_engine/reset", self.reset_callback)
        self.create_service(Trigger, "/control_engine/prepare_episode", self.prepare_episode_callback)
        self.create_timer(0.05, self.publish_gripper_confirmation)
        self.create_timer(1.0 / OUTPUT_HZ, self.publish_smoothed_arm)
        self.create_timer(1.0, self.ensure_servo_stopped)
        mode = "DRY RUN" if dry_run else "LIVE"
        self.get_logger().info(f"{mode}: smoothed arm output at {OUTPUT_HZ:.0f} Hz")

    def status_callback(self, request, response):
        fresh = self.measured is not None and time.monotonic()-self.measured_at <= JOINT_TIMEOUT
        response.success = True
        response.message = json.dumps({
            "fault": self.fault, "fault_kind": self.fault_kind, "gripper_control": "policy",
            "teleop_mapper_running": "logitech_servo_mapper" in self.get_node_names(),
            "confirmed_gripper": self.confirmed_gripper,
            "fresh_joints": fresh, "servo_stopped": self.servo_stopped,
            "gripper_busy": self.gripper_busy,
            "policy_publishers": self.count_publishers("/control_abstraction_layer"),
            "active_target": self.target is not None,
            "gripper_stability_s": self.gripper_filter.stable_s,
            "gripper_pending": self.gripper_filter.candidate,
            "bowl_release_enabled": self.bowl_release is not None,
            "bowl_release_armed": self.bowl_release.armed if self.bowl_release else False,
            "bowl_release_completed": self.bowl_release.released if self.bowl_release else False,
            "bowl_transport_hold": self.bowl_release.transport_hold if self.bowl_release else False,
            "bowl_diagnostics": self.bowl_release.diagnostics if self.bowl_release else None,
        })
        return response

    def reset_callback(self, request, response):
        now = time.monotonic()
        if self.count_publishers("/control_abstraction_layer") != 0:
            response.success, response.message = False, "Stop the policy executor before reset"
        elif self.gripper_busy:
            response.success, response.message = False, "Wait for gripper IO to finish"
        elif not self.servo_stopped or self.measured is None or now-self.measured_at > JOINT_TIMEOUT:
            response.success, response.message = False, "Reset requires Servo stop acknowledgment and fresh measured joints"
        else:
            self.fault = None
            self.fault_kind = None
            self.target = self.last_raw_target = None
            self.command = list(self.measured)
            self.filtered_target = list(self.measured)
            self.velocity = [0.0]*6
            self.last_tick = now
            self.last_command_at = 0.
            self.startup_hold = False
            self.last_gripper = self.confirmed_gripper
            self.gripper_filter.reset()
            if self.bowl_release is not None:
                self.bowl_release.reset()
            response.success, response.message = True, "Software execution state reset; automatic gripper control ready; no actuator command sent"
            self.get_logger().info(response.message)
        return response

    def prepare_episode_callback(self, request, response):
        # This explicit new-episode operation opens the gripper. It is distinct
        # from reset, which deliberately sends no actuator commands.
        if "logitech_servo_mapper" in self.get_node_names():
            response.success, response.message = False, "Stop teleop mapper before preparing a policy episode"
            return response
        if self.fault is not None and self.fault_kind != "source_disconnected":
            response.success, response.message = False, "Inspect and reset the execution fault before preparing a new episode"
            return response
        response = self.reset_callback(request, response)
        if not response.success:
            return response
        self.start_gripper(1.0)
        if self.fault is not None:
            response.success, response.message = False, self.fault
        else:
            response.message = "New pickup episode requested: opening gripper; wait for successful IO confirmation before inference"
        return response

    def ensure_servo_stopped(self):
        # Dry-run never calls services or publishes actuator commands.
        if self.dry_run or self.servo_stopped or self.servo_stop_future is not None:
            return
        if self.servo_stop_client.service_is_ready():
            self.servo_stop_future = self.servo_stop_client.call_async(Trigger.Request())
            self.servo_stop_future.add_done_callback(self.servo_stop_finished)

    def servo_stop_finished(self, future):
        self.servo_stop_future = None
        try:
            if not future.result().success:
                raise RuntimeError("Servo rejected stop request")
            self.servo_stopped = True
            self.get_logger().info("Servo stop acknowledged; waiting for fresh policy target")
        except Exception as error:
            self.get_logger().error(f"Arm startup blocked: {error}")

    def block_execution(self, reason, kind="execution"):
        if self.fault is None:
            self.get_logger().error(f"EXECUTION BLOCKED; inspect cause before reset or fresh-episode preparation: {reason}")
        self.fault = reason
        self.fault_kind = kind
        self.target = None
        self.last_raw_target = None
        self.velocity = [0.0] * 6

    def joint_state_callback(self, message):
        try:
            measured = [float(value) for value in reorder_joint_positions(
                message.name, message.position, UR5_JOINT_ORDER
            )]
            if not all(math.isfinite(value) for value in measured):
                raise ValueError("non-finite joint state")
        except Exception as error:
            self.get_logger().warning(f"Waiting for ordered joint state: {error}")
            return
        self.measured = measured
        self.measured_at = time.monotonic()
        # Follow manual repositioning while idle, never the pose at node creation.
        if self.target is None:
            self.command = list(measured)
            self.filtered_target = list(measured)

    def command_callback(self, message):
        if self.fault is not None:
            return
        now = time.monotonic()
        if len(message.data) != 7 or not all(math.isfinite(float(v)) for v in message.data):
            self.block_execution("expected seven finite arm/gripper values")
            return
        if not self.servo_stopped or self.measured is None or now - self.measured_at > JOINT_TIMEOUT:
            self.get_logger().warning("Startup waiting for Servo stop acknowledgment and fresh joint state")
            return
        if self.count_publishers("/control_abstraction_layer") != 1:
            self.block_execution("expected exactly one abstraction publisher")
            return
        if self.target is not None and now - self.last_command_at > COMMAND_TIMEOUT:
            self.block_execution("policy command timed out")
            return
        if self.target is not None and max(abs(a-b) for a, b in zip(self.command, self.measured)) > MAX_TRACKING_ERROR:
            self.block_execution("commanded/measured joint error exceeds 0.05 rad")
            return
        candidate = [float(value) for value in message.data[:6]]
        starting = self.target is None
        reference = self.measured if starting else self.last_raw_target
        limit = MAX_INITIAL_ERROR if starting else MAX_TARGET_STEP
        largest_step = max(abs(a - b) for a, b in zip(candidate, reference))
        if largest_step > limit:
            self.block_execution(f"arm target jump {largest_step:.4f} rad (limit {limit:.4f})")
            return
        if starting:
            self.command = list(self.measured)
            self.filtered_target = list(self.measured)
            self.velocity = [0.0] * 6
            self.last_tick = now
            self.startup_hold = True
        self.last_raw_target = candidate
        self.target = candidate
        self.last_command_at = now
        # Gripper handling occurs only after all arm/startup checks pass.
        gripper = 1.0 if float(message.data[6]) >= 0.5 else 0.0
        if self.bowl_release is not None and self.bowl_release.released:
            gripper = 1.0  # One pickup/place episode; rearm only on explicit reset.
        elif self.bowl_release is not None and self.bowl_release.transport_hold:
            gripper = 0.0  # Preserve first close during transport; IO is not contact sensing.
        gripper = self.gripper_filter.update(gripper, now, self.last_gripper)
        if gripper is not None and gripper != self.last_gripper and not self.gripper_busy:
            if self.dry_run:
                self.last_gripper = gripper
                self.get_logger().info(f"DRY RUN gripper {'OPEN' if gripper else 'CLOSED'}")
            else:
                self.start_gripper(gripper)

    def update_bowl_release(self, now):
        if self.bowl_release is None or self.bowl_release.released:
            return
        try:
            tf = self.release_tf.lookup_transform('base', 'tool0_controller', rclpy.time.Time())
            stamp = tf.header.stamp.sec + tf.header.stamp.nanosec/1e9
            age = self.get_clock().now().nanoseconds/1e9-stamp
            p, q = tf.transform.translation, tf.transform.rotation
            release = self.bowl_release.update(now, stamp, age, [p.x,p.y,p.z],
                                               [q.x,q.y,q.z,q.w], self.confirmed_gripper,
                                               self.gripper_busy)
        except Exception as error:
            # Missing/invalid TF must never become evidence of bowl arrival.
            self.bowl_release.previous = self.bowl_release.inside_since = None
            self.bowl_release.closed_since = None
            self.bowl_release.armed = False
            self.bowl_release.diagnostics = {'reason': f'missing/invalid TF: {error}'}
            return
        if now-self.last_bowl_diagnostics >= 2.:
            self.last_bowl_diagnostics = now
            self.get_logger().info(f'BOWL DIAGNOSTICS: {json.dumps(self.bowl_release.diagnostics)}')
        if release:
            if self.dry_run:
                self.get_logger().info('DRY RUN: taught bowl release reached; no IO sent')
            else:
                self.get_logger().info('BOWL RELEASE: measured pose inside calibrated release region, slow motion and 0.3 s dwell; requesting OPEN, held until episode reset')
                self.start_gripper(1.0)

    def publish_smoothed_arm(self):
        if self.target is None or self.command is None:
            return
        now = time.monotonic()
        if self.fault is not None:
            return
        if (self.measured is None or now - self.measured_at > JOINT_TIMEOUT):
            self.block_execution("joint feedback stale")
            return
        if now - self.last_command_at > COMMAND_TIMEOUT:
            self.block_execution("policy command timed out")
            return
        source_count = self.count_publishers("/control_abstraction_layer")
        if source_count != 1:
            self.block_execution(
                f"policy publisher disconnected or competing publisher present (count={source_count})",
                kind="source_disconnected" if source_count == 0 else "source_conflict",
            )
            return
        if max(abs(a-b) for a, b in zip(self.command, self.measured)) > MAX_TRACKING_ERROR:
            self.block_execution("commanded/measured joint error exceeds 0.05 rad")
            return
        if self.startup_hold:
            # First arm message is exactly the latest measured pose, not an old
            # filter state or the first neural-network prediction.
            self.startup_hold = False
            self.command = list(self.measured)
            self.filtered_target = list(self.measured)
            self.velocity = [0.0] * 6
            self.last_tick = now
            if not self.dry_run:
                arm = Float64MultiArray()
                arm.data = list(self.command)
                self.arm_publisher.publish(arm)
            return
        self.update_bowl_release(now)
        if self.fault is not None:
            return
        dt = min(max(now - self.last_tick, 0.001), 0.02)
        self.last_tick = now
        alpha = 1.0 - math.exp(-dt / self.filter_time_constant)
        output = []
        new_velocity = []
        for index in range(6):
            filtered = self.filtered_target[index] + alpha * (
                self.target[index] - self.filtered_target[index]
            )
            self.filtered_target[index] = filtered
            desired_velocity = max(
                -self.max_velocity,
                min(self.max_velocity, (filtered - self.command[index]) / dt),
            )
            dv = self.max_acceleration * dt
            velocity = max(
                self.velocity[index] - dv,
                min(self.velocity[index] + dv, desired_velocity),
            )
            new_velocity.append(velocity)
            output.append(self.command[index] + velocity * dt)
        self.velocity = new_velocity
        self.command = output
        self.output_count += 1
        if self.dry_run:
            self.get_logger().info(
                f"DRY {self.output_count:03d}: "
                + ", ".join(f"{value:.6f}" for value in output)
            )
            if self.validation_count and self.output_count >= self.validation_count:
                self.get_logger().info(
                    f"DRY RUN COMPLETE: {self.output_count} filtered commands; none published"
                )
                self.validation_done = True
            return
        arm = Float64MultiArray()
        arm.data = output
        self.arm_publisher.publish(arm)
        if now - self.last_motion_diagnostics >= 2.0:
            self.last_motion_diagnostics = now
            target_gap = max(abs(a-b) for a, b in zip(self.target, self.measured))
            tracking_gap = max(abs(a-b) for a, b in zip(self.command, self.measured))
            speed = max(abs(v) for v in self.velocity)
            self.get_logger().info(
                f"CONTROL DIAGNOSTICS: target_vs_measured={target_gap:.5f} rad; "
                f"command_vs_measured={tracking_gap:.5f} rad; "
                f"command_speed={speed:.5f} rad/s; "
                "gripper_control=policy"
            )

    def start_gripper(self, value):
        if not self.io_client.service_is_ready():
            self.block_execution("SetIO service is unavailable")
            return
        self.gripper_busy = True
        writes = (
            ((0, 0.0), (1, 1.0))
            if value == 1.0
            else ((1, 0.0), (0, 1.0))
        )
        self.send_io(writes, 0, value)

    def publish_gripper_confirmation(self):
        # This is confirmed command history, never measured finger position.
        # Do not publish a guessed initial state before a successful IO pair.
        if self.confirmed_gripper is None:
            return
        message = Float64()
        message.data = self.confirmed_gripper
        self.gripper_state_publisher.publish(message)

    def send_io(self, writes, index, gripper_value):
        pin, state = writes[index]
        request = SetIO.Request()
        request.fun = 1
        request.pin = pin
        request.state = state
        future = self.io_client.call_async(request)
        future.add_done_callback(
            lambda result: self.io_finished(
                result, writes, index, gripper_value
            )
        )

    def io_finished(self, future, writes, index, gripper_value):
        try:
            if not future.result().success:
                raise RuntimeError(f"SetIO rejected pin {writes[index][0]}")
            if index == 0:
                self.send_io(writes, 1, gripper_value)
                return
            self.last_gripper = float(gripper_value)
            self.confirmed_gripper = float(gripper_value)
            self.publish_gripper_confirmation()
            state = "OPEN" if gripper_value == 1.0 else "CLOSED"
            self.get_logger().info(f"Gripper changed to {state}")
        except Exception as error:
            self.gripper_busy = False
            self.block_execution(f"Gripper operation failed: {error}")
        finally:
            if index == 1:
                self.gripper_busy = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validation-count", type=int, default=0)
    parser.add_argument("--speed-scale", type=validated_speed, default=DEFAULT_SPEED_SCALE)
    parser.add_argument("--bowl-release-file", help="Opt-in operator-taught fixed-bowl release calibration JSON")
    args, ros_args = parser.parse_known_args()
    try:
        engine_lock = acquire_engine_lock()
    except RuntimeError as error:
        parser.exit(2, f"{error}\n")
    rclpy.init(args=ros_args)
    node = ControlEngine(args.dry_run, args.validation_count, args.speed_scale,
                         bowl_release_file=args.bowl_release_file)
    try:
        while rclpy.ok() and not node.validation_done:
            rclpy.spin_once(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        engine_lock.close()


if __name__ == "__main__":
    main()
