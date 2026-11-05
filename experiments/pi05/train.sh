#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$here/../.." && pwd)
cd "$root"
export CUDA_VISIBLE_DEVICES=0,1,2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false NCCL_P2P_DISABLE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$here/lerobot/src"
export HF_HOME="$root/data/models/training_environments/pi05/hf_cache"
export HF_LEROBOT_HOME="$root/data/models/training_environments/pi05/dataset_cache"
export HF_HUB_OFFLINE=1 WANDB_PROJECT=ur5-experiment1-pi05
export WANDB_DISABLE_GIT=true WANDB_DISABLE_CODE=true
python_root=/home/grasplab/miniconda3/envs/pi05_ur5/bin
run_root="$root/data/models/checkpoints/experiment1_pi05_local"
weights="$HF_HOME/hub/models--lerobot--pi05_base/snapshots/22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0/model.safetensors"
mkdir -p "$run_root"
exec 9>"$run_root/.training.lock"
flock -n 9 || { echo 'A Pi0.5 training process already owns these GPUs' >&2; exit 1; }
stamp=$(date +%Y%m%d_%H%M%S)
if [[ ${1:-} == --probe && $# == 1 ]]; then
  output="$run_root/probe_$stamp"
  extra=(--steps 2 --accumulate 1 --probe)
elif [[ ${1:-} == --resume && $# == 2 ]]; then
  output=$(realpath -- "$2")
  readarray -t recovered < <("$python_root/python" - "$output" "$here" <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[2])
from training_recovery import latest_checkpoint
run=Path(sys.argv[1]); checkpoint=latest_checkpoint(run)
if checkpoint is None: raise SystemExit('No complete Pi0.5 checkpoint available')
c=json.loads((run/'training_config.json').read_text())
print(checkpoint); print(c['steps']); print(c['gradient_accumulation'])
PY
)
  [[ ${#recovered[@]} == 3 ]] || { echo 'Cannot resolve Pi0.5 resume state' >&2; exit 1; }
  extra=(--resume "${recovered[0]}" --steps "${recovered[1]}" --accumulate "${recovered[2]}")
  rm -f -- "$output/paused.json"
elif [[ $# == 0 ]]; then
  output="$run_root/ur5_cube_110ep_pi05_expert_$stamp"
  extra=(--steps 20000 --accumulate 4)
else
  echo 'Usage: experiments/pi05/train.sh [--probe | --resume RUN_DIRECTORY]' >&2
  exit 2
fi
mkdir -p "$output"
export UR5_PI05_OUTPUT="$output"
"$python_root/python" - "$run_root/active_job.json" <<'PY'
import json,os,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'launcher_pid':os.getppid(),'output':os.environ['UR5_PI05_OUTPUT'],'gpus':[0,1,2]},indent=2)+'\n')
PY
echo "Pi0.5 output: $output"
exec taskset -c 12-19 nice -n 10 ionice -c 2 -n 7 \
  "$python_root/torchrun" --standalone --nproc_per_node=3 \
  "$here/train.py" --dataset "$root/data/dataset/lerobot/ur5_cube_110ep_v3" \
  --weights "$weights" --output "$output" --mixed-precision fp16 \
  --save-every 250 --keep-checkpoints 2 "${extra[@]}" \
  >> "$output/terminal.log" 2>&1
