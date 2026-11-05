#!/usr/bin/env python3
"""Save a complete Pi0.5 checkpoint and release GPUs before robot testing."""
import json
from pathlib import Path
import os
import signal

root=Path(__file__).resolve().parents[2]
job=json.loads((root/'data/models/checkpoints/experiment1_pi05_local/active_job.json').read_text())
launcher=int(job['launcher_pid'])
proc=Path('/proc')/str(launcher)
if not proc.exists() or b'torchrun' not in (proc/'cmdline').read_bytes():
    raise SystemExit('No active Pi0.5 torchrun launcher matches the saved process')
children=[]
for candidate in Path('/proc').iterdir():
    if not candidate.name.isdigit(): continue
    try:
        stat=(candidate/'stat').read_text().rsplit(')',1)[1].split()
        if int(stat[1])==launcher and b'experiments/pi05/train.py' in (candidate/'cmdline').read_bytes():
            children.append(int(candidate.name))
    except (OSError,ValueError): pass
if len(children)!=3: raise SystemExit(f'Expected 3 Pi0.5 workers; found {children}; no signals sent')
for pid in children: os.kill(pid,signal.SIGUSR1)
print('Checkpoint pause requested. Wait for all workers to exit and paused.json before using their GPUs.')
print('Resume: bash experiments/pi05/train.sh --resume "'+job['output']+'"')
