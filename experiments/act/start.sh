#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$here/../.." && pwd)
if systemctl --user is-active --quiet ur5-act-training.service; then
  echo 'The isolated ACT training service is already running.'
  exit 0
fi
/usr/bin/python3 - "$here" "$root" <<'PY'
import hashlib,json,sys
from pathlib import Path
here,root=map(Path,sys.argv[1:])
probe=root/'data/models/checkpoints/experiment1_act_local/probe_b16_a1_c50_v1'
state=json.loads((probe/'status.json').read_text())
if state.get('phase')!='probe_passed' or not state.get('resume_checkpoint_loaded'):
    raise SystemExit('Run ACT --probe twice: update/checkpoint test, then fresh-process resume.')
signature=json.loads((probe/'run_config.json').read_text())['signature']
for name,expected in signature['source_sha256'].items():
    if hashlib.sha256((here/name).read_bytes()).hexdigest()!=expected:
        raise SystemExit('ACT source changed since its probe: '+name)
PY
exec systemd-run --user --unit=ur5-act-training --collect \
  --description='Isolated ACT experiment1 training with online W&B' \
  --property=Nice=10 --property=KillMode=mixed --property=KillSignal=SIGUSR1 \
  --property=TimeoutStopSec=300 /bin/bash "$here/train.sh"
