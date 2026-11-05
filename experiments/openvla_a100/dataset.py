"""Standalone LeRobot-v3 -> original OpenVLA adapter. No ROS or LeRobot imports.

Action token discretization and prompt formatting follow the MIT-licensed
openvla/openvla ActionTokenizer and PurePromptBuilder implementation:
https://github.com/openvla/openvla/tree/c8f03f48af692657d3060c19588038c7220e9af9
"""
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3

import numpy as np

DATASET_KEY = 'ur5_cube_110ep_v3'
CAMERA = 'observation.images.base'
EXPECTED_HASHES = {
    'meta/info.json': 'de71d93083204c98babf100c8331061badae2df7aa1a3bf23de5e8deecfa4fdf',
    'meta/ur5_vla_conversion.json': 'eda8704603a39667c9b540b966cb8efdfee54e8b7d2ab4f34765ca24de592d21',
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read_source(root):
    """Check reviewed metadata and align scalar rows with per-episode video offsets."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = Path(root)
    for relative, expected in EXPECTED_HASHES.items():
        if sha256(root / relative) != expected:
            raise ValueError(f'Reviewed dataset metadata mismatch: {relative}')
    info = json.loads((root / 'meta/info.json').read_text())
    conversion = json.loads((root / 'meta/ur5_vla_conversion.json').read_text())
    columns = ['index', 'episode_index', 'frame_index', 'timestamp', 'action']
    files = sorted((root / 'data').rglob('*.parquet'))
    table = pa.concat_tables([pq.read_table(p, columns=columns) for p in files])
    table = table.sort_by([('index', 'ascending')])
    ids = table['index'].to_numpy()
    actions = np.asarray(table['action'].to_pylist(), dtype=np.float32)
    episodes = table['episode_index'].to_numpy()
    frame_ids = table['frame_index'].to_numpy()
    times = table['timestamp'].to_numpy().astype(np.float64)
    if not np.array_equal(ids, np.arange(285204)) or actions.shape != (285204, 7):
        raise ValueError('Expected exactly 285204 unique, contiguous seven-dimensional action rows')
    if not np.isfinite(actions).all() or not np.isin(actions[:, 6], [0., 1.]).all():
        raise ValueError('Invalid actions or nonbinary gripper commands')
    metadata_files = sorted((root / 'meta/episodes').rglob('*.parquet'))
    meta_columns = ['episode_index', 'tasks', 'length', 'dataset_from_index', 'dataset_to_index']
    meta_columns += [f'videos/{CAMERA}/{x}' for x in ('chunk_index', 'file_index', 'from_timestamp')]
    metadata = pa.concat_tables([pq.read_table(p, columns=meta_columns) for p in metadata_files]).to_pylist()
    metadata.sort(key=lambda x: x['episode_index'])
    if [e['episode_index'] for e in metadata] != list(range(110)):
        raise ValueError('Expected episodes 0..109 exactly once')
    video_requests = {}
    for e in metadata:
        lo, hi = e['dataset_from_index'], e['dataset_to_index']
        if hi - lo != e['length'] or not np.all(episodes[lo:hi] == e['episode_index']):
            raise ValueError('Episode/global-row alignment mismatch')
        if not np.array_equal(frame_ids[lo:hi], np.arange(hi - lo)):
            raise ValueError('Episode frame order mismatch')
        if len(e['tasks']) != 1 or e['tasks'][0] != 'Pick up the orange cube and place it in the bowl':
            raise ValueError('Unexpected task instruction')
        relative = info['video_path'].format(
            video_key=CAMERA, chunk_index=e[f'videos/{CAMERA}/chunk_index'],
            file_index=e[f'videos/{CAMERA}/file_index'])
        absolute_times = times[lo:hi] + e[f'videos/{CAMERA}/from_timestamp']
        frame_numbers = np.rint(absolute_times * info['fps']).astype(np.int64)
        if np.max(np.abs(frame_numbers / info['fps'] - absolute_times)) > .002:
            raise ValueError('Video timestamp is not on the expected 20 Hz grid')
        pairs = zip(frame_numbers.tolist(), range(lo, hi))
        requests = video_requests.setdefault(relative, {})
        for frame_number, row in pairs:
            if frame_number in requests:
                raise ValueError('Two dataset rows unexpectedly reference the same video frame')
            requests[frame_number] = row
    return info, conversion, actions, episodes, video_requests, files + metadata_files


def action_statistics(actions):
    low, high = np.quantile(actions, [.01, .99], axis=0)
    if not np.isfinite(actions).all() or np.any(high - low <= 0):
        raise ValueError('Action dimensions must have finite, nonzero normalization ranges')
    return {'q01': low.tolist(), 'q99': high.tolist(), 'min': actions.min(0).tolist(),
            'max': actions.max(0).tolist(), 'mean': actions.mean(0).tolist(),
            'std': actions.std(0).tolist(), 'mask': [True] * 7}


def normalize(actions, stats):
    lo, hi = np.asarray(stats['q01']), np.asarray(stats['q99'])
    return np.clip(2 * (np.asarray(actions) - lo) / (hi - lo) - 1, -1, 1)


def encode_actions(actions, vocab_size):
    bins = np.linspace(-1, 1, 256)
    return vocab_size - np.digitize(np.clip(actions, -1, 1), bins)


def decode_actions(ids, vocab_size):
    bins = np.linspace(-1, 1, 256)
    centers = (bins[:-1] + bins[1:]) / 2
    return centers[np.clip(vocab_size - np.asarray(ids) - 1, 0, 254)]


def unnormalize(actions, stats):
    low, high = np.asarray(stats['q01']), np.asarray(stats['q99'])
    return .5 * (np.asarray(actions) + 1) * (high - low) + low


class UR5Dataset:
    """Map-style RGB/action dataset; SQLite connections are private to each worker."""
    def __init__(self, cache, processor):
        self.cache, self.processor = Path(cache), processor
        self.manifest = json.loads((self.cache / 'manifest.json').read_text())
        if not self.manifest['complete']:
            raise ValueError('Incomplete prepared dataset')
        arrays = np.load(self.cache / 'actions.npz')
        self.actions = arrays['actions']
        self.episodes = arrays['episodes']
        self.stats = self.manifest['dataset_statistics'][DATASET_KEY]['action']
        self.normalized = normalize(self.actions, self.stats)
        self.connection = None
        self.connection_pid = None

    def __len__(self):
        return len(self.actions)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['connection'] = None
        return state

    def __getitem__(self, index):
        import torch
        from PIL import Image
        if self.connection is None or self.connection_pid != os.getpid():
            uri = (self.cache / 'images.sqlite').resolve().as_uri() + '?mode=ro&immutable=1'
            self.connection = sqlite3.connect(uri, uri=True)
            self.connection_pid = os.getpid()
        record = self.connection.execute('SELECT jpeg FROM images WHERE id=?', (int(index),)).fetchone()
        if record is None:
            raise ValueError(f'Missing cached camera frame {index}')
        with Image.open(io.BytesIO(record[0])) as image:
            pixels = self.processor.image_processor(image.convert('RGB'), return_tensors='pt')['pixel_values'][0]
        tokenizer = self.processor.tokenizer
        action_ids = encode_actions(self.normalized[index], tokenizer.vocab_size).tolist()
        # Match upstream PurePromptBuilder including the space preceding action tokens.
        answer = tokenizer.decode(action_ids).strip()
        task = self.manifest['task'].lower()
        text = f'In: What action should the robot take to {task}?\nOut: {answer}</s>'
        ids = tokenizer(text, add_special_tokens=True).input_ids
        if ids[-1] != tokenizer.eos_token_id or ids[-8:-1] != action_ids:
            raise ValueError('Action text did not round-trip to exactly seven OpenVLA action tokens')
        if len(ids) > tokenizer.model_max_length:
            raise ValueError('Prompt exceeds model token limit')
        labels = [-100] * (len(ids) - 8) + ids[-8:]
        return {'pixel_values': pixels, 'input_ids': torch.tensor(ids), 'labels': torch.tensor(labels)}


def collate(examples, pad_token_id):
    import torch
    from torch.nn.utils.rnn import pad_sequence
    ids = pad_sequence([x['input_ids'] for x in examples], batch_first=True, padding_value=pad_token_id)
    labels = pad_sequence([x['labels'] for x in examples], batch_first=True, padding_value=-100)
    return {'input_ids': ids, 'labels': labels, 'attention_mask': ids.ne(pad_token_id),
            'pixel_values': torch.stack([x['pixel_values'] for x in examples])}


class ResumableSampler:
    """Deterministic per-epoch permutations; resume at the exact consumed sample."""
    def __init__(self, length, consumed=0, seed=42):
        self.length, self.consumed, self.seed = length, consumed, seed

    def __iter__(self):
        import torch
        epoch, offset = divmod(self.consumed, self.length)
        while True:
            order = torch.randperm(self.length, generator=torch.Generator().manual_seed(self.seed + epoch))
            yield from order[offset:].tolist()
            epoch, offset = epoch + 1, 0
