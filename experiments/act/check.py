#!/usr/bin/env python3
"""Inspect only ACT status/checkpoint files; never creates a robot publisher."""
import argparse
import json
from pathlib import Path
from data import latest_checkpoint, validate_private_source, verify_checkpoint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    run=args.run or root/'data/models/checkpoints/experiment1_act_local/ur5_cube_110ep_act_b16_c50_20000_steps_v1'
    checkpoint=args.checkpoint
    if checkpoint is None:
        status=run/'status.json'
        print(status.read_text() if status.exists() else 'ACT has not written status yet.')
        checkpoint=latest_checkpoint(run)
    if checkpoint:
        marker=verify_checkpoint(checkpoint)
        print('ACT COMPLETE CHECKPOINT VERIFIED:',checkpoint,'step',marker['step'])
    else:print('No complete ACT checkpoint saved yet; initial checkpoint interval is250 updates.')
    print('Read-only file inspection; no CUDA model or ROS publisher created.')


if __name__=='__main__':main()
