#!/usr/bin/env python3
"""Restore absent private dependency source copies from the frozen source archive.

No environment installation, overwriting, training or robot connection occurs.
The archived LeRobot Python source matches all three local model snapshots.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model',choices=['pi0','pi05','act'])
    args=parser.parse_args()
    experiments=Path(__file__).resolve().parent
    destination=experiments/args.model/'lerobot'
    if destination.exists():
        print('Existing private source retained without changes:',destination)
        return
    record=json.loads((experiments/'act/source_archive.json').read_text())
    archive=experiments/'act'/record['archive']
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=record['sha256']:
        raise RuntimeError('Frozen source archive checksum mismatch')
    with tarfile.open(archive,'r:gz') as tar:
        for entry in tar.getmembers():
            path=Path(entry.name)
            if path.parts[0]!='lerobot' or not entry.isfile():
                raise RuntimeError('Unexpected source archive member')
            target=(destination/Path(*path.parts[1:])).resolve()
            if not target.is_relative_to(destination):raise RuntimeError('Unsafe archive path')
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(tar.extractfile(entry).read())
    print('Restored independent private source:',destination)
    print('Use only this model\'s own Python environment and launcher.')


if __name__=='__main__':main()
