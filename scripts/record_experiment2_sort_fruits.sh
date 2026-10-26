#!/usr/bin/env bash
# Cameras, robot driver, MoveIt/Servo and one joy_node must already be running.
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export UR5_VLA_ROOT="$project_root"
prepare_only=false
if [[ ${1:-} == --prepare-only && $# == 1 ]]; then
  prepare_only=true
elif [[ $# != 0 ]]; then
  echo 'Usage: record_experiment2_sort_fruits.sh [--prepare-only]' >&2
  exit 2
fi
export UR5_RECORD_PREPARE_ONLY="$prepare_only"
set +u
source /opt/ros/humble/setup.bash
if [[ -f "$project_root/hardware_ws/install/local_setup.bash" ]]; then
  source "$project_root/hardware_ws/install/local_setup.bash"
fi
source "$project_root/ros2_ws/install/local_setup.bash"
set -u
export ROS_LOG_DIR="$project_root/data/dataset/staging/experiment_2_sort_fruits/setup/ros_logs"
mkdir -p "$ROS_LOG_DIR" "$project_root/data/dataset/staging/experiment_2_sort_fruits/episodes"
/usr/bin/python3 - <<'PY'
import rclpy,time,os
from rclpy.node import Node
from ur_dashboard_msgs.srv import GetSafetyMode,IsProgramRunning
rclpy.init();node=Node('experiment2_launch_preflight')
end=time.monotonic()+2
while time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.05)
conflicts=set(node.get_node_names()) & {'control_engine','bypass_policy_executor','wrist_calibration_teleop','logitech_servo_mapper','dataset_recorder'}
joy=node.count_publishers('/joy')
ready=True
for typ,service in [(GetSafetyMode,'/dashboard_client/get_safety_mode'),(IsProgramRunning,'/dashboard_client/program_running')]:
    client=node.create_client(typ,service)
    if not client.wait_for_service(timeout_sec=2):ready=False;continue
    future=client.call_async(typ.Request());rclpy.spin_until_future_complete(node,future,timeout_sec=3)
    response=future.result() if future.done() else None
    ready=ready and response is not None and response.success and (response.safety_mode.mode==1 if typ is GetSafetyMode else response.program_running)
node.destroy_node();rclpy.shutdown()
if conflicts:raise SystemExit('Stop existing control/recording nodes first: '+', '.join(sorted(conflicts)))
if joy!=1:raise SystemExit(f'Expected exactly one joystick input publisher; found {joy}')
if not ready and os.environ.get('UR5_RECORD_PREPARE_ONLY')!='true':raise SystemExit('Restore NORMAL safety mode and the running robot program on the pendant first; no remote safety reset is performed.')
PY
# Keep actual Servo hold targets fresh while the deadman is released. This is
# Servo's stationary-output setting; its incoming-command timeout stays active.
ros2 param set /servo_node moveit_servo.num_outgoing_halt_msgs_to_publish 0
if $prepare_only; then
  echo 'PREPARE ONLY: Servo remains stopped; restore the robot on the pendant before enabling arm motion.'
else
  ros2 service call /servo_node/start_servo std_srvs/srv/Trigger '{}'
fi
echo 'EXPERIMENT 2: Sort the fruits. One episode = both apples in bowl; cubes untouched.'
echo 'Joystick 9 START | 10 ABORT/DISCARD | 11 SAVE SUCCESS. Trigger enables arm motion.'
exec ros2 launch "$project_root/ros2_ws/src/ur5-vla-dataset-recorder/launch/experiment2_sort_fruits.launch.py"
