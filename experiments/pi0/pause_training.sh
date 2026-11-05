#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
if ! systemctl --user is-active --quiet ur5-pi05-training.service; then
  exit 0
fi
echo 'Saving a Pi0.5 checkpoint and waiting for its GPUs to be released...'
/usr/bin/python3 "$root/experiments/pi05/pause.py"
deadline=$((SECONDS + 300))
while systemctl --user is-active --quiet ur5-pi05-training.service; do
  if (( SECONDS >= deadline )); then
    echo 'Checkpoint pause is still pending; training was not killed. Inspect its service/log before testing.' >&2
    exit 1
  fi
  sleep 2
done
echo 'Pi0.5 workers have exited; continuing Pi0 startup.'
