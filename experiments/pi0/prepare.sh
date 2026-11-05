#!/usr/bin/env bash
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source /opt/ros/humble/setup.bash
set -u
systemctl --user start ur5-pi0-control-engine.service
/usr/bin/python3 "$here/prepare.py"
/usr/bin/python3 "$here/check.py"
