#!/usr/bin/env python3
"""Run one model job while holding cooperative GPU locks. Never stops another job."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', required=True, help='Physical nvidia-smi indices, e.g. 2 or 0,1,2')
    parser.add_argument('--max-used-mib', type=int, default=1200)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:parser.error('A command is required after --')
    ids = sorted(set(int(x) for x in args.gpus.split(',')))
    if not ids or min(ids) < 0:parser.error('Invalid GPU list')
    root = Path(__file__).resolve().parents[1]
    locks_root = root/'data/models/resource_locks'; locks_root.mkdir(parents=True, exist_ok=True)
    handles = []
    for index in ids:
        handle = open(locks_root/f'gpu_{index}.lock', 'a+')
        try:fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit(f'GPU {index} is reserved by another guarded model job; nothing was stopped.')
        os.set_inheritable(handle.fileno(), True)
        handles.append(handle)
        result = subprocess.run(['nvidia-smi', f'--id={index}', '--query-gpu=memory.used',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
        used = int(result.stdout.strip())
        if used > args.max_used_mib:
            raise SystemExit(f'GPU {index} is already using {used} MiB; inspect the active process first. Nothing was stopped.')
    env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, ids))
    # Exec preserves GPU lock descriptors for this process and all normal descendants.
    # Locks are advisory: use this wrapper for all training commands sharing GPUs.
    print(f'MODEL GPU RESERVATION: {ids}; existing jobs are never killed.', flush=True)
    os.execvpe(command[0], command, env)


if __name__ == '__main__':main()
