#!/usr/bin/python3
"""Run an absolute-action SmolVLA policy and publish seven abstract values."""

import argparse
import json
import math
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from execution_profile import (DEFAULT_SPEED_SCALE, DEFAULT_ACTIONS_PER_PREDICTION,
                               validated_speed, validated_horizon, resolve_carry_lift_mm)
from project_paths import MODELS, PACKAGE, inference_environment

sys.path.insert(0, str(PACKAGE))

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64, Float64MultiArray, Int8
from std_srvs.srv import Trigger

from ur5_vla_dataset_recorder.core import UR5_JOINT_ORDER, reorder_joint_positions
from ur5_vla_dataset_recorder.policy_executor import ShadowObservationBuffer
from ur5_vla_dataset_recorder.shadow_protocol import request_prediction


# Final 110-episode, 18,000-step SmolVLA checkpoint.
MODEL_PATH = Path(
    MODELS / "checkpoints" /
    "ur5_cube_110ep_v3_smolvla_a100_b128_18000_steps_20261003_180950" /
    "pretrained_model"
)

TASK = "Pick up the orange cube and place it in the bowl"
INITIAL_GRIPPER = 1.0
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8765
ACTION_PERIOD = 0.05  # 20 Hz, matching the recorded demonstrations.
PREFETCH_ACTIONS = 12
BLEND_ACTIONS = 10
MAX_INFERENCE_WAIT = 2.0  # seconds after the action queue drains
REPOSITORY = PACKAGE


class PolicyExecutor(Node):
    def __init__(self, grasp_offset_file=None, offset_mode="continuous",
                 speed_scale=DEFAULT_SPEED_SCALE,
                 actions_per_prediction=DEFAULT_ACTIONS_PER_PREDICTION, robot_description=None,
                 carry_lift_mm=0., task=TASK, carry_task=None,
                 planned_transport_file=None, pickup_shift_file=None, experiment_attempt=None,
                 unrecorded_repeat=False):
        super().__init__("bypass_policy_executor")
        from command_diagnostics import QueuedCommandLogger
        self.command_logger = QueuedCommandLogger()
        self.diagnostic_timer = self.create_timer(.05, lambda: self.command_logger.flush(self.get_logger()))
        self.task = task.strip()
        if not self.task:
            raise ValueError('Task instruction must not be empty')
        self.get_logger().info(f'POLICY TASK: {self.task}')
        self.carry_task = carry_task.strip() if carry_task is not None else None
        if self.carry_task == '':
            raise ValueError('Carry instruction must not be empty')
        self.carry_task_pending = False
        self.carry_task_schedule = None
        if self.carry_task is not None:
            from approach_offset import ApproachOffsetSchedule
            self.carry_task_schedule = ApproachOffsetSchedule(close_hold=.5)
            self.get_logger().info(f'CARRY TASK CONFIGURED: {self.carry_task}; switch once after stable closed IO, which does not prove object contact')
        self.speed_scale = validated_speed(speed_scale)
        self.actions_per_prediction = validated_horizon(actions_per_prediction)
        self.blend_actions = min(5, self.actions_per_prediction)
        self.prefetch_actions = min(12, self.actions_per_prediction // 4)
        self.buffer = ShadowObservationBuffer()
        self.gripper = INITIAL_GRIPPER
        self.inference_running = False
        self.action_queue = deque()
        self.raw_action_queue = deque()
        self.last_action = None
        self.last_raw_action = None
        self.inference_wait_started = None
        self.execution_started = False
        self.queue_lock = threading.Lock()
        self.planned_transport = None
        self.pickup_shift = None
        from planning_service_node import PlanningServiceNode
        self.planning_node = PlanningServiceNode()
        if pickup_shift_file is not None:
            if planned_transport_file is None:
                raise ValueError('Temporary pickup shift requires planned bowl transport')
            from planned_pickup_shift import load_shift, load_unrecorded_repeat, record_intervention, PickupShiftPlanner, PlannedPickupShift
            vector = (load_unrecorded_repeat(pickup_shift_file, experiment_attempt) if unrecorded_repeat
                      else load_shift(pickup_shift_file, experiment_attempt))
            self.pickup_shift = PlannedPickupShift(
                PickupShiftPlanner(self.planning_node, UR5_JOINT_ORDER, planned_transport_file, robot_description, vector),
                MODELS/'evaluation/blue_cube_tests/transport_plans')
            if not unrecorded_repeat:
                record_intervention(pickup_shift_file, experiment_attempt)
            else:
                self.get_logger().warning('UNRECORDED REPEAT: attempt-13 condition only; formal result and trial configuration are unchanged')
            self.get_logger().warning(f'TEMPORARY PLANNED PICKUP SHIFT: attempt={experiment_attempt}, additional base metres={vector}; reposition OPEN before first close, then planned bowl transport')
        if planned_transport_file is not None:
            if carry_task is not None or carry_lift_mm:
                raise ValueError('Planned transport supplies its own carry route; omit carry-task and policy carry lift')
            from planned_transport import BowlTransportPlanner, PlannedBowlTransport
            planner = BowlTransportPlanner(self.planning_node, UR5_JOINT_ORDER, planned_transport_file, robot_description)
            self.planned_transport = PlannedBowlTransport(
                planner, self.command_logger, MODELS/'evaluation/blue_cube_tests/transport_plans')
            self.get_logger().warning('ASSISTED TRANSPORT ENABLED: policy pickup, then checked Cartesian lift/carry to calibrated bowl; CLOSED IO does not prove a held object')
        self.inference_generation = 0
        self.measured_joints = None
        self.measured_joints_at = 0.0
        self.measured_sample = (None, 0.)
        # Sensor callbacks must never queue behind policy timers/MoveIt clients.
        # This node is spun by its own executor thread; it has no command publisher.
        self.feedback_node = Node('bypass_policy_observations')
        self.action_callback_group = MutuallyExclusiveCallbackGroup()
        self.grasp_assist = None
        self.trajectory_offset = None
        self.approach_schedule = None
        self.carry_lift = None
        self.planned_carry_height = 0.
        if carry_lift_mm:
            if grasp_offset_file is None or offset_mode != 'continuous':
                raise ValueError('carry lift requires --grasp-offset-file and --offset-mode continuous')
            from approach_offset import CarryLiftSchedule
            self.carry_lift = CarryLiftSchedule(float(carry_lift_mm)/1000.)
            self.get_logger().info(f'CARRY LIFT CONFIGURED: {carry_lift_mm:.1f} mm after stable closed IO; ramp from committed queue height, compensated joint observations')
        self.offset_fault = False
        self.confirmed_gripper = None
        self.confirmed_gripper_at = 0.0
        self.feedback_node.create_subscription(
            Float64, "/control_engine/gripper_state", self.gripper_confirmation,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        if grasp_offset_file is not None:
            if offset_mode in ("continuous", "approach"):
                from grasp_offset import load_offset
                from trajectory_offset import TrajectoryOffsetPlanner
                if offset_mode == "approach":
                    from approach_offset import ApproachOffsetSchedule
                    self.approach_schedule = ApproachOffsetSchedule()
                frame, vector = load_offset(grasp_offset_file)
                self.trajectory_offset = TrajectoryOffsetPlanner(self.planning_node, UR5_JOINT_ORDER, frame, vector, robot_description=robot_description)
                if self.approach_schedule is None:
                    self.get_logger().warning(f"CONTINUOUS OFFSET ENABLED: {vector} metres in {frame}; every arm target, including carry and placement")
                else:
                    self.get_logger().warning(f"APPROACH OFFSET ENABLED: {vector} metres in {frame}; fade over 2 seconds after 0.5 seconds of confirmed closed IO")
            else:
                from grasp_offset import GraspOffsetAssist
                self.grasp_assist = GraspOffsetAssist(self, UR5_JOINT_ORDER, grasp_offset_file)
        self.publisher = self.create_publisher(
            Float64MultiArray, "/control_abstraction_layer", 10
        )
        self.feedback_node.create_subscription(
            Image,
            "/base/camera/color/image_raw",
            lambda message: self.image("observation.images.base", message),
            qos_profile_sensor_data,
        )
        self.feedback_node.create_subscription(
            Image,
            "/wrist/camera/color/image_raw",
            lambda message: self.image("observation.images.wrist", message),
            qos_profile_sensor_data,
        )
        self.feedback_node.create_subscription(
            JointState, "/joint_states", self.joints, qos_profile_sensor_data,
        )
        self.feedback_node.create_subscription(Float64, "/gripper/state", self.gripper_state, 10)
        self.feedback_node.create_subscription(
            Int8,
            "/servo_node/status",
            lambda message: self.buffer.update_servo(
                message.data, time.monotonic()
            ),
            10,
        )
        self.inference_timer = self.create_timer(0.05, self.schedule_inference)
        # A latched engine fault rejects arm targets even when inference keeps
        # succeeding. Observe it asynchronously and discard pending predictions.
        self.engine_status_client = self.feedback_node.create_client(Trigger, '/control_engine/status')
        self.engine_status_future = None
        self.engine_status_timer = self.feedback_node.create_timer(.2, self.check_engine_status)
        # An independent callback group still shares the main executor's worker
        # queue. MoveIt service bursts can delay it beyond the engine watchdog.
        # Give the existing checked publication callback its own executor/node;
        # this is not a heartbeat and never bypasses its feedback/fault checks.
        self.command_node = Node('bypass_policy_commands')
        self.action_timer = self.command_node.create_timer(
            ACTION_PERIOD / self.speed_scale, self.publish_action,
            callback_group=self.action_callback_group)

    def check_engine_status(self):
        if self.offset_fault:
            return
        if self.engine_status_future is not None:
            if not self.engine_status_future.done():
                return
            future, self.engine_status_future = self.engine_status_future, None
            try:
                response = future.result()
                if response is None or not response.success:
                    return
                state = json.loads(response.message)
            except Exception as error:
                self.command_logger.warning(f'Engine status response unavailable: {error}')
                return
            if state.get('fault'):
                with self.queue_lock:
                    self.offset_fault = True
                    self.action_queue.clear()
                    self.raw_action_queue.clear()
                    self.inference_generation += 1
                self.command_logger.error(
                    f"CONTROL ENGINE BLOCKED; policy publication and inference stopped: {state['fault']}. "
                    'Stop this executor and inspect/reset the engine before restarting.')
                return
        if self.engine_status_client.service_is_ready():
            self.engine_status_future = self.engine_status_client.call_async(Trigger.Request())

    def image(self, key, message):
        self.buffer.update_image(
            key,
            message.height,
            message.width,
            message.encoding,
            message.step,
            bytes(message.data),
            time.monotonic(),
        )

    def joints(self, message):
        ordered = reorder_joint_positions(
            message.name, message.position, UR5_JOINT_ORDER
        )
        self.buffer.update_joints(ordered, time.monotonic())
        self.measured_joints = list(ordered)
        self.measured_joints_at = time.monotonic()
        self.measured_sample = (self.measured_joints, self.measured_joints_at)

    def gripper_state(self, message):
        self.gripper = 1.0 if float(message.data) >= 0.5 else 0.0

    def gripper_confirmation(self, message):
        if float(message.data) in (0.0, 1.0):
            self.confirmed_gripper = float(message.data)
            self.confirmed_gripper_at = time.monotonic()
            self.gripper = self.confirmed_gripper
            self.buffer.update_gripper(self.confirmed_gripper, self.confirmed_gripper_at)
            if getattr(self, 'planned_transport', None) is not None:
                self.planned_transport.observe(self.confirmed_gripper, self.confirmed_gripper_at)
            if (getattr(self, 'carry_task_schedule', None) is not None and
                    self.carry_task_schedule.observe(self.confirmed_gripper, self.confirmed_gripper_at)):
                self.carry_task_pending = True
            if self.carry_lift is not None:
                if self.carry_lift.observe(self.confirmed_gripper, self.confirmed_gripper_at):
                    self.get_logger().info(f'CARRY LIFT: stable closed IO; ramping base Z up by {self.carry_lift.height_m*1000:.1f} mm through checked targets')
            if self.approach_schedule is not None:
                if self.approach_schedule.observe(self.confirmed_gripper, self.confirmed_gripper_at):
                    self.get_logger().info("APPROACH OFFSET FADING: stable closed IO acknowledged; smoothly returning carry targets to the unshifted policy trajectory")

    def apply_carry_task(self):
        """Invalidate pickup predictions before requesting a placement chunk."""
        with self.queue_lock:
            if not self.carry_task_pending:
                return
            self.carry_task_pending = False
            self.task = self.carry_task
            self.action_queue.clear()
            self.raw_action_queue.clear()
            self.inference_generation += 1
            # Hold only the last checked target while a fresh chunk is made.
            # Do not mark an old worker finished: it must discard its generation.
            self.inference_wait_started = time.monotonic()
        self.get_logger().info(f'POLICY TASK SWITCH: {self.task}; pickup queue discarded; route remains policy-generated')

    def schedule_inference(self):
        if self.offset_fault:
            return
        if getattr(self, 'planned_transport', None) is not None and self.planned_transport.active:
            return
        if getattr(self, 'pickup_shift', None) is not None and self.pickup_shift.active:
            return
        if getattr(self, 'carry_task_pending', False):
            return  # The action timer switches at a publication boundary.
        if self.grasp_assist is not None and self.grasp_assist.active:
            return
        with self.queue_lock:
            if self.inference_running or len(self.action_queue) > self.prefetch_actions:
                return
            self.inference_running = True
            generation = self.inference_generation
        now = time.monotonic()
        if (self.confirmed_gripper is not None
                and now - self.confirmed_gripper_at <= 0.2):
            self.gripper = self.confirmed_gripper
        if (self.grasp_assist is not None and
                self.grasp_assist.confirmed_gripper is not None and
                now - self.grasp_assist.confirmed_gripper_at <= 0.2):
            self.gripper = self.grasp_assist.confirmed_gripper
            self.buffer.update_gripper(self.gripper, self.grasp_assist.confirmed_gripper_at)
        # Only an actual IO confirmation refreshes gripper observation age.
        # Prediction timers and predicted actions are not feedback heartbeats.
        try:
            state, images, _ = self.buffer.snapshot(now, 0.5, 1.0, 0.5)
        except Exception as error:
            self.get_logger().warning(f"Waiting for observations: {error}")
            with self.queue_lock:
                self.inference_running = False
            return
        raw = None
        if self.carry_lift is not None:
            with self.queue_lock:
                raw = list(self.last_raw_action) if self.last_raw_action is not None else None
        threading.Thread(
            target=self.infer, args=(state, images, generation, 0., raw), daemon=True
        ).start()

    def infer(self, state, images, generation, observed_lift=0., raw_snapshot=None):
        try:
            if raw_snapshot is not None and self.trajectory_offset.model is not None:
                model = self.trajectory_offset.model
                actual, _ = model.fk_jacobian(state[:6])
                baseline, _ = model.fk_jacobian(raw_snapshot[:6])
                observed_lift = min(self.carry_lift.height_m, max(0.,
                    actual[2,3]-baseline[2,3]-self.trajectory_offset.vector[2]))
            if observed_lift:
                model = self.trajectory_offset.model
                virtual = model.translate(state[:6], [0.,0.,-observed_lift], carry_height=-observed_lift)
                state = [*virtual, state[6]]
            chunk, _ = request_prediction(
                SERVER_HOST, SERVER_PORT, 5.0, state, self.task, images
            )
            actions = [list(action) for action in chunk]
            if len(actions) != 50 or any(len(action) != 7 for action in actions):
                raise ValueError(f"Expected action chunk [50, 7], got {len(actions)} actions")
            full_span = max(max(a[j] for a in actions)-min(a[j] for a in actions)
                            for j in range(6))
            actions = actions[:self.actions_per_prediction]
            used_span = max(max(a[j] for a in actions)-min(a[j] for a in actions)
                            for j in range(6))
            first_error = max(abs(a-b) for a, b in zip(actions[0][:6], state[:6]))
            predicted_closed = sum(float(a[6]) < 0.5 for a in actions)
            with self.queue_lock:
                if generation != self.inference_generation:
                    return
                if self.grasp_assist is not None and self.grasp_assist.active:
                    return
                if self.trajectory_offset is not None:
                    boundary = list(self.raw_action_queue[-1]) if self.raw_action_queue else self.last_raw_action
                else:
                    boundary = list(self.action_queue[-1]) if self.action_queue else self.last_action
            if boundary is not None:
                count = min(self.blend_actions, len(actions))
                for index in range(count):
                    phase = (index + 1) / count
                    weight = phase * phase * (3.0 - 2.0 * phase)
                    actions[index][:6] = [
                        old + weight * (new - old)
                        for old, new in zip(boundary[:6], actions[index][:6])
                    ]
            raw_actions = [list(action) for action in actions]
            if self.trajectory_offset is not None:
                measured, measured_at = getattr(self, 'measured_sample', (self.measured_joints, self.measured_joints_at))
                if measured is None or time.monotonic() - measured_at > 0.1:
                    raise ValueError("fresh measured joints required for continuous correction")
                # If actions are already queued, check the path from their endpoint.
                with self.queue_lock:
                    reference = list(self.action_queue[-1][:6]) if self.action_queue else list(measured)
                if self.approach_schedule is None:
                    if getattr(self, 'carry_lift', None) is None:
                        actions, report = self.trajectory_offset.plan_actions(actions, reference)
                    else:
                        period = ACTION_PERIOD / self.speed_scale
                        heights = self.carry_lift.next_heights(self.planned_carry_height, len(actions), period)
                        actions, report = self.trajectory_offset.plan_actions(actions, reference, lift_heights=heights)
                else:
                    with self.queue_lock:
                        queued = len(self.action_queue)
                    period = ACTION_PERIOD / self.speed_scale
                    scales = self.approach_schedule.scales(time.monotonic()+queued*period, len(actions), period)
                    actions, report = self.trajectory_offset.plan_actions(actions, reference, scales=scales)
                self.get_logger().info(f"CONTINUOUS OFFSET CHECKED: {report}")
            with self.queue_lock:
                if generation != self.inference_generation or self.offset_fault:
                    return
                if self.grasp_assist is not None and self.grasp_assist.active:
                    return
                self.action_queue.extend(actions)
                if self.trajectory_offset is not None:
                    self.raw_action_queue.extend(raw_actions)
                    if getattr(self, 'carry_lift', None) is not None:
                        self.planned_carry_height = heights[-1]
            self.get_logger().info(
                f"Queued {len(actions)} policy actions with joint-space boundary blending"
            )
            self.get_logger().info(
                f"POLICY DIAGNOSTICS: predicted_span_50={full_span:.5f} rad; "
                f"used_span_{len(actions)}={used_span:.5f} rad; "
                f"first_prediction_vs_observed={first_error:.5f} rad; "
                f"observed_gripper={state[6]:.0f}; "
                f"predicted_close_actions={predicted_closed}/{len(actions)}"
            )
        except Exception as error:
            with self.queue_lock:
                if generation != self.inference_generation:
                    return  # An instruction switch invalidated this worker.
            if self.trajectory_offset is not None:
                with self.queue_lock:
                    self.offset_fault = True
                    self.action_queue.clear()
                    self.raw_action_queue.clear()
                self.get_logger().error(f"CONTINUOUS OFFSET BLOCKED; fault latched: {error}")
            self.get_logger().error(f"Inference failed: {error}")
        finally:
            with self.queue_lock:
                self.inference_running = False

    def publish_planned_transport(self, measured, measured_at):
        try:
            now = time.monotonic()
            self.buffer.snapshot(now, 0.5, 1.0, 0.5)
            if self.planned_transport.stage == 'requested':
                with self.queue_lock:
                    self.action_queue.clear()
                    self.raw_action_queue.clear()
                    self.inference_generation += 1
                # No logging or helper work while holding the queue lock.
                self.planned_transport.begin(self.last_action, now)
            action = self.planned_transport.tick(measured, measured_at, now)
            self.last_action = list(action)
            message = Float64MultiArray();message.data = action
            self.publisher.publish(message)
        except Exception as error:
            with self.queue_lock:
                self.offset_fault = True
                self.inference_generation += 1
                self.action_queue.clear()
                self.raw_action_queue.clear()
            getattr(self, 'command_logger', self.get_logger()).error(f'ASSISTED TRANSPORT BLOCKED; fault latched: {error}')

    def publish_action(self):
        if self.offset_fault:
            return
        if getattr(self, 'carry_task_pending', False):
            self.apply_carry_task()
        measured, measured_at = getattr(self, 'measured_sample', (self.measured_joints, self.measured_joints_at))
        if not self.execution_started:
            # DDS subscriptions become ready independently. Missing initial
            # samples mean WAITING, not a runtime loss of joint feedback.
            now = time.monotonic()
            try:
                if measured is None or now - measured_at > 0.1:
                    raise ValueError("waiting for fresh initial joints")
                self.buffer.snapshot(now, 0.5, 1.0, 0.5)
            except Exception:
                with self.queue_lock:
                    # A queued prediction based on older inputs must be
                    # regenerated, not replayed when those inputs return.
                    if self.action_queue:
                        self.action_queue.clear()
                        self.raw_action_queue.clear()
                        self.inference_generation += 1
                return
        if (self.trajectory_offset is not None and
                (measured is None or time.monotonic() - measured_at > 0.1)):
            with self.queue_lock:
                self.offset_fault = True
                self.action_queue.clear()
                self.raw_action_queue.clear()
            getattr(self, 'command_logger', self.get_logger()).error(f"CONTINUOUS OFFSET BLOCKED: stale measured joints; age_s={time.monotonic()-measured_at:.6f}, limit_s=0.100")
            return
        if getattr(self, 'planned_transport', None) is not None and self.planned_transport.active:
            self.publish_planned_transport(measured, measured_at)
            return
        if getattr(self, 'pickup_shift', None) is not None and self.pickup_shift.active:
            self.publish_pickup_shift(measured, measured_at)
            return
        if self.grasp_assist is not None and self.grasp_assist.active:
            action = self.grasp_assist.tick(
                measured, measured_at,
                self.grasp_assist.open_confirmed,
            )
            if action is not None:
                self.last_action = list(action)
                message = Float64MultiArray()
                message.data = action
                self.publisher.publish(message)
                self.execution_started = True
            elif self.grasp_assist.stage == "complete":
                self.gripper = 0.0
                with self.queue_lock:
                    self.action_queue.clear()
            return
        with self.queue_lock:
            empty = not self.action_queue
        if empty:
            self.publish_inference_wait()
            return
        self.inference_wait_started = None
        with self.queue_lock:
            if getattr(self, 'pickup_shift', None) is not None and float(self.action_queue[0][6]) < .5:
                try:
                    self.pickup_shift.begin(self.last_action, self.confirmed_gripper,
                                            self.confirmed_gripper_at, time.monotonic())
                    self.action_queue.clear(); self.raw_action_queue.clear()
                    self.inference_generation += 1
                except Exception as error:
                    self.offset_fault = True
                    self.action_queue.clear();self.raw_action_queue.clear()
                    getattr(self, 'command_logger', self.get_logger()).error(f'PICKUP SHIFT BLOCKED: {error}')
                return
            action = self.action_queue.popleft()
            if self.trajectory_offset is not None:
                self.last_raw_action = self.raw_action_queue.popleft()
            self.last_action = list(action)
            remaining = len(self.action_queue)
        if (self.grasp_assist is not None and
                self.grasp_assist.stage == "idle" and float(action[6]) < 0.5):
            if not self.grasp_assist.open_confirmed:
                self.grasp_assist.stage = "fault"
                getattr(self, 'command_logger', self.get_logger()).error("OFFSET BLOCKED: fresh gripper-open IO required before correction")
                return
            if measured is None or time.monotonic() - measured_at > 0.1:
                self.grasp_assist.stage = "fault"
                getattr(self, 'command_logger', self.get_logger()).error("OFFSET BLOCKED: fresh measured joints required before correction")
                return
            with self.queue_lock:
                self.action_queue.clear()
                self.inference_generation += 1
                self.grasp_assist.start(action[:6], measured)
            return
        desired_gripper = 1.0 if float(action[6]) >= 0.5 else 0.0
        self.gripper = desired_gripper
        message = Float64MultiArray()
        # Feedback runs independently; do not reread its shared estimate here.
        message.data = [float(value) for value in action[:6]] + [desired_gripper]
        self.publisher.publish(message)
        self.execution_started = True
        if remaining == 0:
            getattr(self, 'command_logger', self.get_logger()).info("Action chunk completed; requesting next observation")

    def publish_pickup_shift(self, measured, measured_at):
        try:
            now = time.monotonic()
            self.buffer.snapshot(now, 0.5, 1.0, 0.5)
            action = self.pickup_shift.tick(measured, measured_at,
                self.confirmed_gripper, self.confirmed_gripper_at, now)
            self.last_action = list(action)
            message = Float64MultiArray(); message.data = action
            self.publisher.publish(message)
            self.execution_started = True
        except Exception as error:
            with self.queue_lock:
                self.offset_fault = True
                self.action_queue.clear(); self.raw_action_queue.clear()
                self.inference_generation += 1
            getattr(self, 'command_logger', self.get_logger()).error(f'PICKUP SHIFT BLOCKED; fault latched: {error}')

    def publish_inference_wait(self):
        """Keep a bounded, unchanged target during normal inference gaps.

        This is not proof of physical standstill: the engine may still be
        converging to that target. No future policy action is extrapolated.
        """
        if self.offset_fault or self.last_action is None:
            return
        now = time.monotonic()
        if self.inference_wait_started is None:
            self.inference_wait_started = now
        try:
            if now - self.inference_wait_started >= MAX_INFERENCE_WAIT:
                raise TimeoutError("next action chunk unavailable for 2 seconds")
            measured, measured_at = getattr(self, 'measured_sample', (self.measured_joints, self.measured_joints_at))
            if measured is None or now - measured_at > 0.1:
                raise ValueError("stale joint feedback during inference wait")
            # Do not keep the engine alive on stale cameras or other inputs.
            self.buffer.snapshot(now, 0.5, 1.0, 0.5)
        except Exception as error:
            with self.queue_lock:
                self.offset_fault = True
                self.inference_generation += 1
                self.action_queue.clear()
                self.raw_action_queue.clear()
            getattr(self, 'command_logger', self.get_logger()).error(f"INFERENCE WAIT BLOCKED; fault latched: {error}")
            return
        if not self.inference_running:
            return
        message = Float64MultiArray()
        message.data = list(self.last_action)
        self.publisher.publish(message)


def wait_for_server(process, timeout=120.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Inference server exited during startup")
        try:
            with socket.create_connection((SERVER_HOST, SERVER_PORT), timeout=0.2):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError("Timed out waiting for inference server")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', default=TASK,
                        help='language instruction sent to the policy; default: %(default)s')
    parser.add_argument('--carry-task',
                        help='optional placement instruction: switch once after 0.5 s confirmed closed IO; IO is not object-contact sensing')
    parser.add_argument('--planned-transport-file', type=Path,
                        help='opt-in assisted transport: policy pickup then checked lift/carry to this bowl calibration; requires matching bowl release enabled in engine')
    parser.add_argument('--pickup-shift-file', type=Path,
                        help='separate checked OPEN reposition before first policy close; restricted to recorded attempts 12/13')
    parser.add_argument('--experiment-attempt', type=int, help='trial identifier required with --pickup-shift-file')
    parser.add_argument('--unrecorded-repeat', action='store_true',
                        help='explicit operator demonstration of completed attempt 13; requires recorder waiting; never edits trial result/configuration')
    parser.add_argument(
        "--grasp-offset-file", type=Path,
        help="measured Cartesian translation; applied throughout the trajectory by default",
    )
    parser.add_argument(
        "--offset-mode", choices=("continuous", "approach", "grasp"), default="continuous",
        help="continuous: all targets; approach: fade after stable close; grasp: earlier one-close intervention",
    )
    parser.add_argument("--speed-scale", type=validated_speed, default=DEFAULT_SPEED_SCALE)
    parser.add_argument("--actions-per-prediction", type=validated_horizon,
                        default=DEFAULT_ACTIONS_PER_PREDICTION)
    parser.add_argument("--prepare-episode", action="store_true",
                        help="new pickup trial: open gripper and confirm IO before inference; scene must be reset with no object held")
    parser.add_argument('--carry-lift-mm', type=float, default=None,
                        help='0-30 mm base-Z lift after stable closed IO; default 25 with continuous grasp offset, otherwise 0; use 0 to disable')
    args = parser.parse_args()
    args.task = args.task.strip()
    if not args.task:
        parser.error('--task must not be empty')
    if args.carry_task is not None:
        args.carry_task = args.carry_task.strip()
        if not args.carry_task:
            parser.error('--carry-task must not be empty')
    try:
        if args.planned_transport_file is not None:
            from bowl_release import BowlRelease
            BowlRelease(args.planned_transport_file)
            if args.carry_task is not None:
                parser.error('--planned-transport-file and --carry-task are separate execution modes')
            if args.carry_lift_mm not in (None, 0.):
                parser.error('Planned transport supplies its own lift; omit --carry-lift-mm or set it to 0')
            args.carry_lift_mm = 0.
        args.carry_lift_mm = resolve_carry_lift_mm(args.carry_lift_mm, args.offset_mode,
                                               args.grasp_offset_file is not None)
    except ValueError as error:
        parser.error(str(error))
    if args.grasp_offset_file is not None:
        from grasp_offset import load_offset
        load_offset(args.grasp_offset_file)
    if args.pickup_shift_file is not None:
        from planned_pickup_shift import load_shift, load_unrecorded_repeat
        if args.planned_transport_file is None:
            parser.error('--pickup-shift-file requires --planned-transport-file')
        try:
            if args.unrecorded_repeat:
                load_unrecorded_repeat(args.pickup_shift_file, args.experiment_attempt)
            else:
                load_shift(args.pickup_shift_file, args.experiment_attempt)
        except ValueError as error:
            parser.error(str(error))
    elif args.experiment_attempt is not None:
        parser.error('--experiment-attempt requires --pickup-shift-file')
    elif args.unrecorded_repeat:
        parser.error('--unrecorded-repeat requires --pickup-shift-file and --experiment-attempt 13')
    if not MODEL_PATH.is_dir():
        raise SystemExit(f"MODEL_PATH does not exist: {MODEL_PATH}")
    from engine_readiness import require_ready_engine
    robot_description = require_ready_engine(
        prepare_episode=args.prepare_episode,
        fetch_robot_description=(args.planned_transport_file is not None or
                                 args.grasp_offset_file is not None and args.offset_mode in ("continuous", "approach")),
        require_bowl_release=args.planned_transport_file is not None,
    )
    server = subprocess.Popen(
        [
            "/home/grasplab/miniconda3/bin/conda",
            "run",
            "--no-capture-output",
            "-n",
            "pi0",
            "python",
            "-m",
            "ur5_vla_dataset_recorder.policy_inference_server",
            "--checkpoint",
            str(MODEL_PATH),
            "--host",
            SERVER_HOST,
            "--port",
            str(SERVER_PORT),
            "--device",
            "cuda",
        ],
        cwd=REPOSITORY,
        env=inference_environment(),
    )
    try:
        wait_for_server(server)
        rclpy.init()
        node = PolicyExecutor(args.grasp_offset_file, args.offset_mode,
                              args.speed_scale, args.actions_per_prediction, robot_description,
                              carry_lift_mm=args.carry_lift_mm, task=args.task,
                              carry_task=args.carry_task,
                              planned_transport_file=args.planned_transport_file,
                              pickup_shift_file=args.pickup_shift_file, experiment_attempt=args.experiment_attempt,
                              unrecorded_repeat=args.unrecorded_repeat)
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        feedback_executor = SingleThreadedExecutor()
        feedback_executor.add_node(node.feedback_node)
        feedback_thread = threading.Thread(target=feedback_executor.spin, daemon=True)
        feedback_thread.start()
        command_executor = SingleThreadedExecutor()
        command_executor.add_node(node.command_node)
        command_thread = threading.Thread(target=command_executor.spin, daemon=True)
        command_thread.start()
        try:
            executor.spin()
        finally:
            # Stop publication first; sensor and service callbacks remain
            # available while an in-progress checked action callback returns.
            command_executor.shutdown()
            command_thread.join(timeout=2.)
            executor.shutdown()
            feedback_executor.shutdown()
            feedback_thread.join(timeout=2.)
            node.command_node.destroy_node()
            node.feedback_node.destroy_node()
            node.planning_node.close()
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    main()
