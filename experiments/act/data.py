"""Read-only validation and deterministic sampling for the isolated ACT experiment."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def validate_dataset(root):
    root = Path(root).resolve()
    reviewed = json.loads((Path(__file__).parent / 'reviewed_source.json').read_text())
    for relative, expected in reviewed['sha256'].items():
        if sha256(root / relative) != expected:
            raise ValueError(f'Reviewed dataset mismatch: {relative}')
    info = json.loads((root / 'meta/info.json').read_text())
    if (info['total_episodes'], info['total_frames'], info['fps']) != (110, 285204, 20):
        raise ValueError('Expected the reviewed 110 episodes, 285204 frames at 20 Hz')
    for key in ('action', 'observation.state'):
        if info['features'][key]['shape'] != [7]:
            raise ValueError(f'Expected seven dimensions for {key}')
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    tables = [pq.read_table(p, columns=['index', 'episode_index', 'frame_index', 'action', 'observation.state'])
              for p in sorted((root / 'data').rglob('*.parquet'))]
    table = pa.concat_tables(tables).sort_by([('index', 'ascending')])
    if not np.array_equal(table['index'].to_numpy(), np.arange(285204)):
        raise ValueError('Dataset indices are not contiguous and unique')
    for key in ('action', 'observation.state'):
        values = np.asarray(table[key].to_pylist(), dtype=np.float32)
        if values.shape != (285204, 7) or not np.isfinite(values).all():
            raise ValueError(f'Invalid {key} values')
        if key == 'action' and not np.isin(values[:, 6], [0., 1.]).all():
            raise ValueError('Action gripper must be binary: 0 closed / 1 open')
    episodes = table['episode_index'].to_numpy()
    frames = table['frame_index'].to_numpy()
    if not np.array_equal(np.unique(episodes), np.arange(110)):
        raise ValueError('Episode IDs differ from 0..109')
    for episode in range(110):
        selected = frames[episodes == episode]
        if not np.array_equal(selected, np.arange(len(selected))):
            raise ValueError(f'Invalid frame order in episode {episode}')
    return {'episodes': 110, 'frames': 285204, 'fps': 20,
            'files_verified': len(reviewed['sha256']),
            'source_manifest_sha256': sha256(Path(__file__).parent / 'reviewed_source.json'),
            'action': 'six absolute joint positions in radians; gripper 0=CLOSED/1=OPEN'}


def validate_private_source():
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / 'lerobot_source_snapshot.json').read_text())
    for relative, expected in manifest['files'].items():
        if sha256(here / 'lerobot' / relative) != expected:
            raise ValueError(f'ACT private LeRobot source changed: {relative}')
    import lerobot
    if not Path(lerobot.__file__).resolve().is_relative_to(here / 'lerobot'):
        raise ValueError('ACT imported a different experiment source')


class ResumableSampler:
    def __init__(self, length, consumed=0, seed=42):
        self.length, self.consumed, self.seed = length, consumed, seed

    def __iter__(self):
        import torch
        epoch, offset = divmod(self.consumed, self.length)
        while True:
            order = torch.randperm(self.length, generator=torch.Generator().manual_seed(self.seed + epoch))
            yield from order[offset:].tolist()
            epoch, offset = epoch + 1, 0


def verify_checkpoint(path):
    path = Path(path)
    marker = json.loads((path / 'COMPLETE.json').read_text())
    if marker['step'] != int(path.name):
        raise ValueError('Checkpoint step/name mismatch')
    for relative, expected in marker['sha256'].items():
        if sha256(path / relative) != expected:
            raise ValueError(f'Checkpoint checksum mismatch: {relative}')
    return marker


def latest_checkpoint(output):
    candidates = sorted(p for p in (Path(output) / 'checkpoints').glob('*')
                        if p.is_dir() and p.name.isdigit() and (p / 'COMPLETE.json').exists())
    if not candidates:
        return None
    verify_checkpoint(candidates[-1])
    return candidates[-1]
