"""Read-only preflight so a policy cannot start against a blocked engine."""
import json
import time

RESET_COMMAND = 'ros2 service call /control_engine/reset std_srvs/srv/Trigger "{}"'


class EngineWarmingUp(ValueError):
    """Feedback or Servo acknowledgment may still arrive during startup."""


def validate_engine_status(status, preparing=False):
    if status.get('policy_publishers', 0) != 0:
        raise ValueError('Another policy publisher is running. Stop it before starting a new test.')
    resumable_end = preparing and status.get('fault_kind') == 'source_disconnected'
    if (status.get('fault') and not resumable_end) or status.get('active_target'):
        raise ValueError(f'Control engine is blocked or still armed: {status.get("fault")}. After inspecting the cause, stop the old executor and run: {RESET_COMMAND}')
    if status.get('gripper_control') != 'policy':
        raise ValueError('Restart the updated engine: automatic-only gripper control is required.')
    if status.get('teleop_mapper_running'):
        raise ValueError('Stop logitech_servo before policy execution; it can send arm and gripper commands.')
    if not status.get('fresh_joints') or not status.get('servo_stopped') or status.get('gripper_busy'):
        raise EngineWarmingUp('Control engine is not ready: wait for fresh joints, Servo stop acknowledgment and idle gripper IO.')
    if not preparing and status.get('confirmed_gripper') != 1.0:
        raise ValueError('A new pickup must start with confirmed OPEN gripper IO. Reset the scene with no object held, then add --prepare-episode to your executor command.')


def fetch_moveit_description(node, executor):
    """Fetch the live model before image subscriptions start; never reuse disk cache."""
    import rclpy
    from rcl_interfaces.srv import GetParameters
    client = node.create_client(GetParameters, '/move_group/get_parameters')
    if not client.wait_for_service(timeout_sec=4.):
        raise ValueError('MoveIt robot-description service unavailable')
    for attempt in range(2):
        request = GetParameters.Request()
        request.names = ['robot_description']
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, executor=executor, timeout_sec=5.)
        if not future.done():
            future.cancel()
            if attempt == 0:
                continue
            raise ValueError('MoveIt robot-description request timed out after two read-only attempts')
        result = future.result()
        if result is None or len(result.values) != 1 or not result.values[0].string_value:
            raise ValueError('MoveIt returned no robot_description')
        return result.values[0].string_value


def require_ready_engine(prepare_episode=False, fetch_robot_description=False,
                         require_bowl_release=False):
    import rclpy
    from rclpy.node import Node
    from std_srvs.srv import Trigger
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    context = Context()
    rclpy.init(context=context)
    node = Node('policy_engine_preflight', context=context)
    executor = SingleThreadedExecutor(context=context)
    try:
        client = node.create_client(Trigger, '/control_engine/status')
        if not client.wait_for_service(timeout_sec=4.):
            raise ValueError('Control-engine status service unavailable. Start one updated control_engine.py before the executor.')
        deadline = time.monotonic()+5.
        prepare_pending = prepare_episode
        while True:
            future = client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(node, future, executor=executor, timeout_sec=2.)
            if not future.done() or future.result() is None or not future.result().success:
                future.cancel()
                raise ValueError('No successful control-engine status response')
            try:
                status = json.loads(future.result().message)
                if require_bowl_release and not status.get('bowl_release_enabled'):
                    raise ValueError('Planned transport requires the control engine calibrated bowl-release mode')
                validate_engine_status(status, preparing=prepare_pending)
                if prepare_pending:
                    prepare_client = node.create_client(Trigger, '/control_engine/prepare_episode')
                    if not prepare_client.wait_for_service(timeout_sec=2.):
                        raise ValueError('Updated engine prepare_episode service unavailable')
                    prepare_future = prepare_client.call_async(Trigger.Request())
                    rclpy.spin_until_future_complete(node, prepare_future, executor=executor, timeout_sec=2.)
                    if not prepare_future.done() or prepare_future.result() is None or not prepare_future.result().success:
                        detail = prepare_future.result().message if prepare_future.done() and prepare_future.result() is not None else 'no response'
                        raise ValueError('Episode preparation failed: '+detail)
                    prepare_pending = False
                    continue
                return fetch_moveit_description(node, executor) if fetch_robot_description else None
            except EngineWarmingUp:
                if time.monotonic() >= deadline:
                    raise
                rclpy.spin_once(node, executor=executor, timeout_sec=.1)
    except (ValueError, RuntimeError) as error:
        raise SystemExit(f'POLICY START BLOCKED: {error}') from None
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown(context=context)
