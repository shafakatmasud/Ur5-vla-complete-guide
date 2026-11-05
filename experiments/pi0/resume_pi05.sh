#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
if systemctl --user is-active --quiet ur5-pi05-training.service; then
  echo 'Pi0.5 training is already active.'
  exit 0
fi
used=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits)
if (( used > 1200 )); then
  echo 'Stop the Pi0/SmolVLA model server before resuming GPU training.' >&2
  exit 1
fi
run=$(/usr/bin/python3 - "$root" <<'PY'
import json,sys
from pathlib import Path
print(json.loads((Path(sys.argv[1])/'data/models/checkpoints/experiment1_pi05_local/active_job.json').read_text())['output'])
PY
)
systemd-run --user --unit=ur5-pi05-training --collect \
  --description='Isolated experiment 1 Pi0.5 training with W&B' \
  --property="WorkingDirectory=$root" --property=KillSignal=SIGINT \
  --property=TimeoutStopSec=180 /bin/bash "$root/experiments/pi05/train.sh" --resume "$run"
