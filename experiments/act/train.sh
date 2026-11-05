#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$here/../.." && pwd)
environment="$root/data/models/training_environments/act"
python="$environment/env/bin/python"
[[ -x "$python" ]] || { echo 'Run experiments/act/setup.sh first.' >&2; exit 1; }
export PYTHONNOUSERSITE=1 PYTHONPATH="$here/lerobot/src"
export HF_HOME="$environment/hf_cache" HF_LEROBOT_HOME="$environment/lerobot_cache"
export HF_DATASETS_CACHE="$environment/datasets_cache" TORCH_HOME="$environment/torch_cache"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_DISABLE_GIT=true WANDB_DISABLE_CODE=true
run_root="$root/data/models/checkpoints/experiment1_act_local"
mkdir -p "$run_root" "$environment"
if [[ ${1:-} == --probe && $# == 1 ]]; then
  output="$run_root/probe_b16_a1_c50_v1"
  extra=(--probe)
elif [[ ${1:-} == --resume && $# == 2 ]]; then
  output=$(realpath -- "$2")
  [[ "$output" == "$run_root/"* ]] || { echo 'ACT resume must use the ACT checkpoint root.' >&2; exit 1; }
  extra=()
elif [[ $# == 0 ]]; then
  output="$run_root/ur5_cube_110ep_act_b16_c50_20000_steps_v1"
  extra=()
else
  echo 'Usage: experiments/act/train.sh [--probe | --resume ACT_RUN_DIRECTORY]' >&2
  exit 2
fi
mkdir -p "$output"
cd "$root"
exec /usr/bin/python3 "$root/experiments/model_job.py" --gpus 2 -- \
  "$python" -u "$here/train.py" --dataset "$environment/dataset" --output "$output" \
  --batch-size 16 --accumulate 1 --workers 4 --chunk-size 50 --action-steps 20 \
  --steps 20000 --save-every 250 "${extra[@]}" >> "$output/terminal.log" 2>&1
