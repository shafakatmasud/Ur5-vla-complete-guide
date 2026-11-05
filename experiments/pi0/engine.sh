#!/usr/bin/env bash
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source /opt/ros/humble/setup.bash
set -u
export PYTHONNOUSERSITE=1
cd "$here/scripts"
exec /usr/bin/python3 -u control_engine.py --speed-scale 0.5 \
  --bowl-release-file "$here/data/models/evaluation/calibration/bowl_release_calibration.json"
