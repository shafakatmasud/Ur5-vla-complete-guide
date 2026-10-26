#!/usr/bin/env bash
# Source in every ROS terminal: source /path/to/repository/scripts/env.sh
export UR5_VLA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$UR5_VLA_ROOT/.env" ]]; then
  set -a
  source "$UR5_VLA_ROOT/.env"
  set +a
fi
export VLA_PYTHON="${VLA_PYTHON:-$UR5_VLA_ROOT/.venv-vla/bin/python}"
export HF_HOME="${HF_HOME:-$UR5_VLA_ROOT/data/huggingface_cache}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$UR5_VLA_ROOT/data/hf_lerobot_cache}"
export UR5_PACKAGE="$UR5_VLA_ROOT/ros2_ws/src/ur5-vla-dataset-recorder"
export PYTHONPATH="$UR5_PACKAGE${PYTHONPATH:+:$PYTHONPATH}"
if [[ -f /opt/ros/humble/setup.bash ]]; then
  source /opt/ros/humble/setup.bash
fi
for workspace in hardware_ws ros2_ws; do
  if [[ -f "$UR5_VLA_ROOT/$workspace/install/local_setup.bash" ]]; then
    source "$UR5_VLA_ROOT/$workspace/install/local_setup.bash"
  fi
done
