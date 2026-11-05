#!/usr/bin/env python3
"""Request a checkpoint from only this experiment's ACT trainer."""
from pathlib import Path
import os
import signal

target = Path(__file__).resolve().parent/'train.py'
found = []
for process in Path('/proc').iterdir():
    if not process.name.isdigit():continue
    try:args=(process/'cmdline').read_bytes().decode().split('\0')
    except (OSError,UnicodeError):continue
    if str(target) not in args:continue
    if '--output' not in args:continue
    output=Path(args[args.index('--output')+1])
    if '/experiment1_act_local/' not in str(output):continue
    os.kill(int(process.name),signal.SIGUSR1);found.append(int(process.name))
print('ACT checkpoint/pause requested for:',found or 'no active ACT trainer')
