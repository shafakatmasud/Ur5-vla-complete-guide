#!/usr/bin/env python3
"""Validate the reviewed 110 episodes and cache synchronized base-camera RGB frames."""
import argparse
import io
import json
from pathlib import Path
import sqlite3
import time

from dataset import CAMERA, DATASET_KEY, action_statistics, read_source, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    import av
    import numpy as np
    info, conversion, actions, episodes, requests, parquet_files = read_source(args.dataset)
    reviewed = json.loads((Path(__file__).parent / 'reviewed_source.json').read_text())
    for relative, expected in reviewed['sha256'].items():
        if sha256(args.dataset / relative) != expected:
            raise ValueError(f'Source file differs from the reviewed local artifact: {relative}')
    print(f'Validated {len(actions)} scalar rows in {len(np.unique(episodes))} episodes; {len(requests)} base videos', flush=True)
    print('Action: six ABSOLUTE joint targets in radians, then gripper 0=CLOSED/1=OPEN.', flush=True)
    if args.validate_only:
        return
    args.cache.mkdir(parents=True, exist_ok=True)
    manifest_path = args.cache / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for relative, expected in manifest['source_sha256'].items():
            if sha256(args.dataset / relative) != expected:
                raise ValueError(f'Prepared cache belongs to different source data: {relative}')
        if manifest.get('complete') and manifest['num_frames'] == len(actions):
            for relative, expected in manifest['cache_sha256'].items():
                if sha256(args.cache / relative) != expected:
                    raise ValueError(f'Prepared cache corrupted: {relative}')
            print('Reusing complete, source-verified cache', flush=True)
            return
        raise ValueError('Incomplete cache; use a new cache directory')
    db_path = args.cache / 'images.sqlite'
    if db_path.exists():
        raise ValueError('Partial cache exists; choose a new cache directory')
    source_paths = set(parquet_files) | {args.dataset / p for p in requests}
    source_paths |= {args.dataset / 'meta/info.json', args.dataset / 'meta/ur5_vla_conversion.json'}
    source_hashes = {str(p.relative_to(args.dataset)): sha256(p) for p in sorted(source_paths)}
    connection = sqlite3.connect(db_path)
    connection.execute('CREATE TABLE images (id INTEGER PRIMARY KEY, jpeg BLOB NOT NULL)')
    count, started = 0, time.monotonic()
    try:
        for relative, targets in sorted(requests.items()):
            found = set()
            with av.open(str(args.dataset / relative)) as container:
                stream = container.streams.video[0]
                stream.thread_type = 'AUTO'
                stream.codec_context.thread_count = 4
                for frame in container.decode(stream):
                    if frame.pts is None:
                        raise ValueError('Video frame has no timestamp')
                    timestamp = float(frame.pts * stream.time_base)
                    frame_number = round(timestamp * info['fps'])
                    if frame_number not in targets:
                        continue
                    if frame_number in found or abs(timestamp - frame_number / info['fps']) > .002:
                        raise ValueError('Duplicated or misaligned decoded video timestamp')
                    image = frame.to_image().convert('RGB')
                    if image.size != (640, 480):
                        raise ValueError('Unexpected base-camera resolution')
                    output = io.BytesIO()
                    # Original geometry retained; JPEG compression is recorded in manifest.
                    image.save(output, format='JPEG', quality=92)
                    connection.execute('INSERT INTO images VALUES (?,?)', (targets[frame_number], output.getvalue()))
                    found.add(frame_number)
                    count += 1
                    if count % 1000 == 0:
                        connection.commit()
                        print(f'Cached {count}/{len(actions)} frames, elapsed {time.monotonic()-started:.0f}s', flush=True)
            if found != set(targets):
                raise ValueError(f'Missing {len(set(targets)-found)} requested frames in {relative}')
        connection.commit()
        if connection.execute('SELECT COUNT(*) FROM images').fetchone()[0] != len(actions):
            raise ValueError('Prepared camera/action count mismatch')
    finally:
        connection.close()
    np.savez(args.cache / 'actions.npz', actions=actions, episodes=episodes)
    statistics = {DATASET_KEY: {'action': action_statistics(actions)}}
    manifest = {'complete': True, 'num_frames': len(actions), 'num_episodes': 110,
                'task': 'Pick up the orange cube and place it in the bowl', 'fps': info['fps'],
                'camera': CAMERA, 'input_cameras': 1, 'proprioception_input': False,
                'image_cache_encoding': 'JPEG quality=92, original 640x480 geometry',
                'split': 'all 110 episodes used for training; no held-out validation split',
                'dataset_statistics': statistics, 'action_contract': conversion,
                'source_sha256': source_hashes,
                'cache_sha256': {p: sha256(args.cache / p) for p in ('images.sqlite', 'actions.npz')}}
    temp = manifest_path.with_suffix('.json.partial')
    temp.write_text(json.dumps(manifest, indent=2))
    temp.replace(manifest_path)
    print(f'PREPARATION COMPLETE: {count} synchronized RGB/action pairs; {args.cache}', flush=True)


if __name__ == '__main__':
    main()
