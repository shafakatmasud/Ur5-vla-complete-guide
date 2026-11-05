#!/usr/bin/python3
"""Read-only Pi0 robot checks and live-camera predictions; no command publishers."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import threading
import time
from tf2_ros import Buffer, TransformListener

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'scripts'))
sys.path.insert(0, str(HERE/'ros2_ws/src/ur5-vla-dataset-recorder'))

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64, Int8
from std_srvs.srv import Trigger
from ur_dashboard_msgs.srv import GetSafetyMode, IsProgramRunning
from controller_manager_msgs.srv import ListControllers
from engine_readiness import validate_engine_status
from ur5_vla_dataset_recorder.core import UR5_JOINT_ORDER, reorder_joint_positions
from ur5_vla_dataset_recorder.policy_executor import ShadowObservationBuffer
from ur5_vla_dataset_recorder.shadow_protocol import request_prediction


class Inputs(Node):
    def __init__(self):
        super().__init__('isolated_pi0_readonly_check')
        self.buffer = ShadowObservationBuffer()
        self.speed = None
        self.speed_at = 0.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.subscriptions_kept = []
        for view in ('base', 'wrist'):
            self.subscriptions_kept.append(self.create_subscription(
                Image, f'/{view}/camera/color/image_raw',
                lambda m, v=view: self.buffer.update_image(
                    f'observation.images.{v}', m.height, m.width, m.encoding,
                    m.step, m.data, time.monotonic()), qos_profile_sensor_data))
        self.subscriptions_kept.append(self.create_subscription(
            JointState, '/joint_states', lambda m: self.buffer.update_joints(
                reorder_joint_positions(m.name, m.position, UR5_JOINT_ORDER),
                time.monotonic()), qos_profile_sensor_data))
        self.subscriptions_kept.append(self.create_subscription(
            Int8, '/servo_node/status', lambda m: self.buffer.update_servo(
                m.data, time.monotonic()), qos_profile_sensor_data))
        self.subscriptions_kept.append(self.create_subscription(
            Float64, '/speed_scaling_state_broadcaster/speed_scaling', self.speed_cb,
            qos_profile_sensor_data))
        self.subscriptions_kept.append(self.create_subscription(
            Float64, '/control_engine/gripper_state', lambda m: self.buffer.update_gripper(
                m.data, time.monotonic()), QoSProfile(depth=1,
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL)))

    def speed_cb(self, msg):
        self.speed, self.speed_at = float(msg.data), time.monotonic()

    def call(self, typ, name):
        client = self.create_client(typ, name)
        if not client.wait_for_service(timeout_sec=4):
            raise RuntimeError(f'{name} unavailable')
        for attempt in range(2):
            future = client.call_async(typ.Request())
            deadline = time.monotonic()+5
            while not future.done() and time.monotonic()<deadline:
                time.sleep(.02)
            if future.done() and future.result() is not None:
                return future.result()
            future.cancel()
        raise RuntimeError(f'{name} did not respond after two read-only attempts')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shadow', action='store_true', help='Request predictions from a separately started port-8766 server')
    parser.add_argument('--samples', type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.samples <= 100: parser.error('--samples must be between 1 and 100')
    rclpy.init()
    node = Inputs()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic()+10
        while True:
            try:
                state, images, ages = node.buffer.snapshot(time.monotonic(), .25, .5, .5)
                break
            except ValueError as error:
                if time.monotonic()>deadline:
                    hint = (' Initialize the empty-gripper episode with bash "$PI0_ROOT/prepare.sh"; '
                            'the Pi0 run wrapper does this automatically.'
                            if 'gripper_state' in str(error) else ' Check the named live input publishers.')
                    raise RuntimeError(
                        f'{error}.{hint}') from None
                time.sleep(.05)
        response = node.call(Trigger, '/control_engine/status')
        if not response.success: raise RuntimeError('Engine status failed')
        engine = json.loads(response.message)
        validate_engine_status(engine, preparing=True)
        if not engine['bowl_release_enabled'] or engine.get('motion_profile') != 'pi0_smooth_v1':
            raise RuntimeError('Restart the dedicated Pi0 engine to load its motion and grasp-hold guards')
        if engine.get('gripper_profile') != 'pi0_single_close_v2':
            raise RuntimeError('Restart the Pi0 engine to load the single-action CLOSE fix')
        safety = node.call(GetSafetyMode, '/dashboard_client/get_safety_mode')
        if not safety.success or safety.safety_mode.mode != 1:
            raise RuntimeError('Robot safety mode must be NORMAL')
        program = node.call(IsProgramRunning, '/dashboard_client/program_running')
        if not program.success or not program.program_running:
            raise RuntimeError('Play the saved External Control program on the pendant')
        controllers = node.call(ListControllers, '/controller_manager/list_controllers')
        controller_states = {c.name:c.state for c in controllers.controller}
        if controller_states.get('forward_position_controller') != 'active':
            raise RuntimeError('forward_position_controller is not active')
        if (node.speed is None or time.monotonic()-node.speed_at>.5
                or not 0 < node.speed <= 20.01):
            raise RuntimeError(f'Speed feedback must be fresh and in (0, 20]%; got {node.speed}')
        try:
            transform = node.tf_buffer.lookup_transform('base', 'tool0_controller', rclpy.time.Time())
            stamp = transform.header.stamp.sec + transform.header.stamp.nanosec/1e9
            age = node.get_clock().now().nanoseconds/1e9-stamp
            if not 0 <= age <= .2: raise ValueError(f'TF age {age:.3f}s')
        except Exception as error:
            raise RuntimeError(f'Fresh base-to-tool TF is required for guarded bowl release: {error}') from None
        state, images, ages = node.buffer.snapshot(time.monotonic(), .25, .5, .5)
        report = dict(utc=datetime.now(timezone.utc).isoformat(), robot_commands_published=0,
                      safety='NORMAL', external_control_running=True, speed_percent=node.speed,
                      controller_states=controller_states, engine=engine, measured_state=state,
                      input_ages_seconds=ages, gripper_source='Acknowledged engine IO command history; not finger/contact sensing',
                      model='Pi0 20,000-step checkpoint plus motion smoothing and grasp-hold/bowl-release guard; no grasp offset or planned transport')
        if args.shadow:
            predictions=[]
            for index in range(args.samples):
                state, images, ages = node.buffer.snapshot(time.monotonic(), .25, .5, .5)
                actions, latency = request_prediction('127.0.0.1', 8766, 5., state,
                    'Pick up the orange cube and place it in the bowl', images)
                finite = len(actions)==50 and all(len(a)==7 and all(math.isfinite(v) for v in a) for a in actions)
                if not finite: raise RuntimeError('Pi0 returned an invalid action chunk')
                first_error=max(abs(a-b) for a,b in zip(state[:6],actions[0][:6]))
                step=max(abs(actions[k][j]-actions[k-1][j]) for k in range(1,50) for j in range(6))
                prediction=dict(sample=index+1, latency_seconds=latency, first_action=actions[0],
                    first_joint_error_rad=first_error, maximum_joint_step_rad=step,
                    close_actions=sum(a[6]<.5 for a in actions), input_ages_seconds=ages,
                    within_engine_envelope=first_error<=.1 and step<=.15)
                predictions.append(prediction)
                print(json.dumps(prediction),flush=True)
            report['predictions']=predictions
            report['all_within_engine_envelope']=all(p['within_engine_envelope'] for p in predictions)
        folder=HERE/'data/models/evaluation/pi0_checks'
        folder.mkdir(parents=True,exist_ok=True)
        output=folder/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+('_shadow.json' if args.shadow else '_ready.json'))
        output.write_text(json.dumps(report,indent=2)+'\n')
        print('Report:',output)
        if args.shadow and not report['all_within_engine_envelope']:
            raise RuntimeError('Predictions exceeded the existing engine envelope; inspect the report before motion')
        print('READ-ONLY CHECK PASSED; no robot command publishers were created.')
    finally:
        executor.shutdown(); thread.join(timeout=2)
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()


if __name__=='__main__':
    try:
        main()
    except (ValueError, RuntimeError) as error:
        raise SystemExit(f'Pi0 check blocked: {error}') from None
