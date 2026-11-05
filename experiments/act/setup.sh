#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$here/../.." && pwd)
environment="$root/data/models/training_environments/act"
conda=/home/grasplab/miniconda3/bin/conda
mkdir -p "$environment"
if [[ ! -x "$environment/env/bin/python" ]]; then
  "$conda" create -y --prefix "$environment/env" --clone /home/grasplab/miniconda3/envs/pi0
fi
if [[ ! -d "$here/lerobot/src/lerobot" ]]; then
  /usr/bin/python3 - "$here" <<'PY'
import hashlib,json,sys,tarfile
from pathlib import Path
here=Path(sys.argv[1]);record=json.loads((here/'source_archive.json').read_text())
archive=here/record['archive']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=record['sha256']:
    raise RuntimeError('ACT source archive checksum mismatch')
with tarfile.open(archive,'r:gz') as tar:
    for entry in tar.getmembers():
        target=(here/entry.name).resolve()
        if not target.is_relative_to(here/'lerobot') or not entry.isfile():
            raise RuntimeError('Unexpected source archive entry')
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(tar.extractfile(entry).read())
PY
fi
"$environment/env/bin/python" -m pip install --no-deps --no-build-isolation -e "$here/lerobot"
if [[ ! -d "$environment/dataset" ]]; then
  cp -a --reflink=auto "$root/data/dataset/lerobot/ur5_cube_110ep_v3" "$environment/dataset"
fi
PYTHONNOUSERSITE=1 PYTHONPATH="$here/lerobot/src" "$environment/env/bin/python" - "$here" <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from data import sha256,validate_dataset
here=Path(sys.argv[1]); root=here.parents[1]
manifest=json.loads((here/'lerobot_source_snapshot.json').read_text())
for relative,expected in manifest['files'].items():
    if sha256(here/'lerobot'/relative)!=expected:raise RuntimeError('ACT private source mismatch: '+relative)
print(validate_dataset(root/'data/models/training_environments/act/dataset'))
import lerobot
if not Path(lerobot.__file__).resolve().is_relative_to(here):raise RuntimeError('ACT imported another model source')
print('ACT ISOLATION CHECK PASSED:',sys.prefix,lerobot.__file__)
PY
