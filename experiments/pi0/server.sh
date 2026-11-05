#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 HF_HOME="$here/data/huggingface_cache"
export HF_LEROBOT_HOME="$here/data/hf_lerobot_cache"
export PYTHONPATH="$here/ros2_ws/src/ur5-vla-dataset-recorder:$here/lerobot/src"
exec /home/grasplab/miniconda3/envs/pi0_ur5_runtime/bin/python \
  -m ur5_vla_dataset_recorder.policy_inference_server \
  --checkpoint "$here/data/models/checkpoints/pi0_20000/pretrained_model" \
  --host 127.0.0.1 --port 8766 --device cuda "$@"
