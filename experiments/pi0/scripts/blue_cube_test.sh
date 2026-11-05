#!/usr/bin/env bash
# Open this in a terminal; no robot commands are sent until the operator starts.
set -o pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export UR5_VLA_ROOT="$project_root"

if [[ ${1:-} == --help ]]; then
  echo 'Blue-cube generalization test using the existing 110-episode checkpoint.'
  echo 'Run: bash scripts/blue_cube_test.sh'
  echo 'The terminal waits for Enter before running the physical policy.'
  exit 0
fi

source /opt/ros/humble/setup.bash || exit 1
source "$project_root/hardware_ws/install/local_setup.bash" || exit 1
source "$project_root/ros2_ws/install/local_setup.bash" || exit 1
cd "$project_root/scripts" || exit 1

command=(/usr/bin/python3 -u "$project_root/scripts/policy_executor.py"
  --task 'Pick up the blue cube and place it in the bowl'
  --planned-transport-file "$project_root/data/models/evaluation/codex test analysis/bowl_release_calibration.json"
  --grasp-offset-file "$project_root/data/models/evaluation/codex test analysis/offset_calibration.json"
  --offset-mode continuous --prepare-episode --speed-scale 0.5
  --actions-per-prediction 50)

unrecorded_repeat=false
if [[ ${1:-} == --repeat-13-unrecorded && $# == 1 ]]; then
  command+=(--pickup-shift-file "$project_root/data/models/evaluation/blue_cube_tests/session_20261019_20_attempts/temporary_pickup_shift_12_13.json"
    --experiment-attempt 13 --unrecorded-repeat)
  unrecorded_repeat=true
  echo 'UNRECORDED REPEAT: same temporary shift as attempt 13; no trial result, camera recording or terminal log.'
elif [[ ${1:-} == --attempt && ( ${2:-} == 12 || ${2:-} == 13 ) && $# == 2 ]]; then
  command+=(--pickup-shift-file "$project_root/data/models/evaluation/blue_cube_tests/session_20261019_20_attempts/temporary_pickup_shift_12_13.json"
    --experiment-attempt "$2")
  echo "TEMPORARY ATTEMPT $2: checked 20 mm table-forward + 20 mm table-left OPEN reposition before first close."
elif [[ $# != 0 ]]; then
  echo 'Usage: blue_cube_test.sh [--attempt 12|13 | --repeat-13-unrecorded]' >&2
  exit 2
fi

echo 'BLUE CUBE TEST — existing 110-episode SmolVLA checkpoint'
echo
echo 'Instruction: Pick up the blue cube and place it in the bowl'
echo 'After stable closed IO: checked lift-and-carry route to calibrated bowl.'
echo 'ASSISTED TRANSPORT condition: model pickup, planned carry, automatic bowl release.'
echo 'Same pickup alignment; planned route supplies lift instead of policy carry offset.'
echo 'This is an unseen-instruction test; colour selection may fail.'
echo
echo 'Place the blue cube at a tested cross and keep the bowl in its calibrated position.'
echo 'Start from the established starting pose with no object held.'
echo 'Keep one control engine running and teleop stopped.'
echo 'Episode preparation opens the gripper; Ctrl+C stops this policy.'
echo
printf 'Command: '; printf '%q ' "${command[@]}"; printf '\n\n'
read -r -p 'Press Enter to START the physical blue-cube test, or Ctrl+C to cancel: ' || exit 0

if $unrecorded_repeat; then
  "${command[@]}"
  status=$?
else
  log_dir="$project_root/data/models/evaluation/blue_cube_tests"
  mkdir -p "$log_dir" || exit 1
  log_file="$log_dir/executor_$(date +%Y%m%d_%H%M%S).log"
  echo "Terminal log: $log_file"
  "${command[@]}" 2>&1 | tee "$log_file"
  status=${PIPESTATUS[0]}
fi
echo
echo "Policy exited with status $status. Inspect the terminal log before another test."
read -r -p 'Press Enter to close this window: '
exit "$status"
