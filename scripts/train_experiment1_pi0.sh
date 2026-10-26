#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
export CUDA_VISIBLE_DEVICES=0,1,2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false NCCL_P2P_DISABLE=1
precision=${UR5_PI0_PRECISION:-fp16}
python_root=/home/grasplab/miniconda3/envs/pi0/bin
weights=/home/grasplab/.cache/huggingface/hub/models--lerobot--pi0_base/snapshots/26b99b9439acb1e352439e34ee9c67af0d76efa3/model.safetensors
run_root="$root/data/models/checkpoints/experiment1_pi0_local"
mkdir -p "$run_root"
exec 9>"$run_root/.training.lock"
flock -n 9 || { echo 'An experiment-1 Pi0 job is already running' >&2; exit 1; }
stamp=$(date +%Y%m%d_%H%M%S)
if [[ ${1:-} == --probe ]]; then
  output="$run_root/probe_$stamp"
  extra=(--steps 2 --accumulate 1 --probe)
elif [[ ${1:-} == --resume && $# == 2 ]]; then
  output=$(realpath -- "$2")
  readarray -t recovered < <(/usr/bin/python3 - "$output" <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0,'scripts')
from pi0_training_recovery import latest_checkpoint
run=Path(sys.argv[1]);checkpoint=latest_checkpoint(run)
if not checkpoint:raise SystemExit('No complete checkpoint available')
c=json.loads((run/'training_config.json').read_text())
print(checkpoint);print(c['steps']);print(c['gradient_accumulation']);print(c['mixed_precision'])
PY
)
  [[ ${#recovered[@]} == 4 ]] || { echo 'Cannot resolve resume state' >&2; exit 1; }
  precision=${recovered[3]}
  extra=(--resume "${recovered[0]}" --steps "${recovered[1]}" --accumulate "${recovered[2]}" --save-every 150)
elif [[ $# == 0 ]]; then
  output="$run_root/ur5_cube_110ep_pi0_expert_$stamp"
  extra=(--steps 20000 --accumulate 4)
else
  echo 'Usage: train_experiment1_pi0.sh [--probe | --resume RUN_DIRECTORY]' >&2
  exit 2
fi
mkdir -p "$output"
echo "Training output: $output"
exec taskset -c 12-19 nice -n 10 ionice -c 2 -n 7 \
  "$python_root/torchrun" --standalone --nproc_per_node=3 \
  "$root/scripts/train_experiment1_pi0.py" \
  --dataset "$root/data/dataset/lerobot/ur5_cube_110ep_v3" \
  --weights "$weights" --output "$output" --mixed-precision "$precision" "${extra[@]}" \
  >> "$output/terminal.log" 2>&1
