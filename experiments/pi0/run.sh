#!/usr/bin/env bash
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$here"
source /opt/ros/humble/setup.bash
set -u
if [[ ${1:-} == --help ]]; then
  exec /usr/bin/python3 "$here/scripts/policy_executor.py" --help
fi
bash "$here/pause_training.sh"
/usr/bin/python3 - "$here" <<'PY'
from pathlib import Path
import json,subprocess,sys
root=Path(sys.argv[1])
check=root/'offline_inference_check.json'
if not check.exists() or not json.loads(check.read_text()).get('finite'):
    raise SystemExit('Pi0 offline inference check has not passed')
result=subprocess.run(['nvidia-smi','--id=0','--query-gpu=memory.used','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
if int(result.stdout.strip())>1200:
    raise SystemExit('Pi0 GPU 0 is busy. Checkpoint/pause Pi0.5 with: /usr/bin/python3 "$UR5_VLA_ROOT/experiments/pi05/pause.py". Wait for the training workers to exit.')
print('ISOLATED Pi0: separate 20,000-step checkpoint, source, environment, calibrations, logs and port 8766.')
print('This command starts physical policy execution through the existing single control engine.')
PY
bash "$here/prepare.sh"
log_root="$here/data/models/evaluation/pi0_tests"
mkdir -p "$log_root"
log_file="$log_root/executor_$(date +%Y%m%d_%H%M%S).log"
echo "Pi0 terminal log: $log_file"
/usr/bin/python3 -u "$here/scripts/policy_executor.py" \
  --speed-scale 0.5 --actions-per-prediction 20 "$@" \
  2>&1 | tee -i "$log_file"
