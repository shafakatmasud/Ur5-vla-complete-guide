#!/usr/bin/env bash
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source /opt/ros/humble/setup.bash
set -u
export PYTHONPATH="$here/ros2_ws/src/ur5-vla-dataset-recorder${PYTHONPATH:+:$PYTHONPATH}"
# Stop only the isolated Pi0 engine, and only after all policy/jog publishers
# have gone away. The copied standard-pose helper checks robot state and path.
/usr/bin/python3 - "$here" <<'PY'
import sys,time,subprocess
from pathlib import Path
import rclpy
from rclpy.node import Node
sys.path.insert(0,str(Path(sys.argv[1])/'scripts'))
from control_engine_lock import acquire_engine_lock
rclpy.init();n=Node('pi0_standard_pose_ownership_check')
try:
 deadline=time.monotonic()+2
 while time.monotonic()<deadline:rclpy.spin_once(n,timeout_sec=.05)
 for topic in ['/control_abstraction_layer','/servo_node/delta_joint_cmds','/servo_node/delta_twist_cmds']:
  if n.count_publishers(topic):raise SystemExit('Stop policy/teleop publishers first: '+topic)
 result=subprocess.run(['systemctl','--user','show','ur5-pi0-control-engine.service','--property=MainPID','--value'],capture_output=True,text=True,check=True)
 if not int(result.stdout.strip() or 0):
  lock=acquire_engine_lock();lock.close()
finally:n.destroy_node();rclpy.shutdown()
PY
systemctl --user stop ur5-pi0-control-engine.service
echo 'PHYSICAL STANDARD-POSE RESET: no object held; keep the swept path clear.'
/usr/bin/python3 -c 'from ur5_vla_dataset_recorder.standard_pose_node import prepare_dataset_start; prepare_dataset_start()'
bash "$here/prepare.sh"
