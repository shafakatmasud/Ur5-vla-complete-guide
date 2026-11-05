#!/usr/bin/env bash
# Frozen SmolVLA pickup with the existing checked bowl transport.
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export UR5_VLA_ROOT="$project_root"
export ORANGE_TEST_SESSION="$project_root/data/models/evaluation/orange_fruit_tests/session_20261023_20_attempts"
if [[ ${1:-} == --help ]]; then
  echo 'Usage: orange_fruit_test.sh [--run | --practice | --check | --prepare-start | --attempt 1..20]'
  echo 'One orange per execution cycle; two cycles per official attempt. Model pickup, planned bowl carry/release.'
  echo '--prepare-start physically moves to the guarded standard pose; keep its path clear.'
  echo 'The run command is independent of recording. No announcement or trial marker is required.'
  echo 'Enter runs policy; Ctrl+C stops it. Only the manual monitor controls official telemetry.'
  exit 0
fi
if [[ $# == 0 || ( $# == 1 && ${1:-} == --run ) ]]; then
  mode=--run
  unset ORANGE_TEST_ATTEMPT
elif [[ $# == 1 && $1 == --practice ]]; then
  mode=--practice
  unset ORANGE_TEST_ATTEMPT
elif [[ $# == 1 && ( $1 == --check || $1 == --prepare-start ) ]]; then
  mode=$1
elif [[ $# == 2 && $1 == --attempt && $2 =~ ^([1-9]|1[0-9]|20)$ ]]; then
  mode=--attempt
  export ORANGE_TEST_ATTEMPT="$2"
else
  echo 'Usage: orange_fruit_test.sh [--run | --practice | --check | --prepare-start | --attempt 1..20]' >&2
  exit 2
fi
set +u
source /opt/ros/humble/setup.bash
source "$project_root/hardware_ws/install/local_setup.bash"
source "$project_root/ros2_ws/install/local_setup.bash"
set -u
cd "$project_root/scripts"
# The other three GPUs are occupied by the separate Pi0 training run.
export CUDA_VISIBLE_DEVICES=3
export ORANGE_EXECUTION_MODE="$mode"
export PYTHONPATH="$project_root/scripts${PYTHONPATH:+:$PYTHONPATH}"
/usr/bin/python3 - <<'PY'
import json,os,time
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from std_srvs.srv import Trigger
from ur_dashboard_msgs.srv import GetSafetyMode,IsProgramRunning
from engine_readiness import validate_engine_status
from bowl_release import BowlRelease
from grasp_offset import load_offset
root=Path(os.environ['UR5_VLA_ROOT']);preparing=os.environ['ORANGE_EXECUTION_MODE']=='--prepare-start'
if not preparing:
 assert (root/'data/models/checkpoints/ur5_cube_110ep_v3_smolvla_a100_b128_18000_steps_20261003_180950/pretrained_model').is_dir()
 BowlRelease(root/'data/models/evaluation/codex test analysis/bowl_release_calibration.json')
 load_offset(root/'data/models/evaluation/codex test analysis/offset_calibration.json')
rclpy.init();n=Node('orange_fruit_launch_check');seen={};speed=[];subscriptions=[]
for camera in ['base','wrist']:
 p=f'/{camera}/camera/color/image_raw'
 subscriptions.append(n.create_subscription(Image,p,lambda m,p=p:seen.update({p:time.monotonic()}),qos_profile_sensor_data))
subscriptions.append(n.create_subscription(Float64,'/speed_scaling_state_broadcaster/speed_scaling',lambda m:speed.append(m.data),qos_profile_sensor_data))
def call(typ,name):
 c=n.create_client(typ,name)
 assert c.wait_for_service(timeout_sec=4),name+' unavailable'
 f=c.call_async(typ.Request());rclpy.spin_until_future_complete(n,f,timeout_sec=5)
 assert f.done() and f.result() is not None,name+' timed out'
 return f.result()
try:
 engine=call(Trigger,'/control_engine/status');assert engine.success
 state=json.loads(engine.message);validate_engine_status(state,preparing=True)
 if not preparing:assert state['bowl_release_enabled'],'Start engine with the taught bowl-release file'
 safety=call(GetSafetyMode,'/dashboard_client/get_safety_mode');assert safety.success and safety.safety_mode.mode==1,'Robot safety mode is not NORMAL'
 program=call(IsProgramRunning,'/dashboard_client/program_running');assert program.success and program.program_running,'External Control is not running'
 end=time.monotonic()+1
 while time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.02)
 if not preparing:assert len(seen)==2 and max(time.monotonic()-t for t in seen.values())<.5,'Camera feed missing or stale'
 assert speed and 0<speed[-1]<=20.01,'Restore robot speed slider to 20% or below before testing'
 if not preparing:assert '/compute_cartesian_path' in dict(n.get_service_names_and_types()),'MoveIt Cartesian planning service unavailable'
 print('READY: robot NORMAL, fresh joints, Servo stopped and robot speed at most 20%. Recording state does not gate execution.')
finally:n.destroy_node();rclpy.shutdown()
PY
if [[ $mode == --check ]]; then exit 0; fi
if [[ $mode == --prepare-start ]]; then
  echo 'GUARDED PHYSICAL RESET: no object held; keep the swept path clear.'
  ros2 run ur5_vla_dataset_recorder move_to_standard_pose
  ros2 control switch_controllers --activate forward_position_controller --deactivate scaled_joint_trajectory_controller --strict
  echo 'Standard pose ready; Servo stays stopped for policy control.'
  exit 0
fi
command=(/usr/bin/python3 -u "$project_root/scripts/policy_executor.py"
  --task 'Pick up the orange and place it in the bowl'
  --planned-transport-file "$project_root/data/models/evaluation/codex test analysis/bowl_release_calibration.json"
  --grasp-offset-file "$project_root/data/models/evaluation/codex test analysis/offset_calibration.json"
  --offset-mode continuous --prepare-episode --speed-scale 0.5 --actions-per-prediction 50)
if [[ $mode == --attempt ]]; then
  echo "ORANGE FRUIT TEST: attempt $ORANGE_TEST_ATTEMPT of 20. Existing 110-episode SmolVLA checkpoint."
else
  echo 'ORANGE FRUIT RUN: no attempt number required; official recording is controlled separately.'
fi
echo 'One orange per execution cycle; run two cycles for the two-orange official task. Model pickup + assisted planned bowl placement.'
echo 'Orange pickup correction: restored 1.3 mm calibrated forward + 0.3 mm right; checked continuous offset.'
echo 'Keep bowl at its taught location; start at standard pose with no object held. Episode preparation opens the gripper.'
echo 'This launcher never starts or finishes official telemetry recording.'
read -r -p 'Press Enter to execute, or Ctrl+C to cancel: ' || exit 0
mkdir -p "$ORANGE_TEST_SESSION/run_logs"
log_file="$ORANGE_TEST_SESSION/run_logs/executor_$(date +%Y%m%d_%H%M%S_%N).log"
printf 'Command: '; printf '%q ' "${command[@]}"; printf '\nLog: %s\n' "$log_file"
set +e
"${command[@]}" 2>&1 | tee "$log_file"
status=${PIPESTATUS[0]}
set -e
echo "Policy stopped (status $status). This launcher did not change official recording or results."
exit "$status"
