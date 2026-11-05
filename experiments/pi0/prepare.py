#!/usr/bin/python3
"""Explicitly prepare an empty-gripper Pi0 episode after engine startup."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent/'scripts'))
from engine_readiness import require_ready_engine


def main():
    print('Waiting for Servo-stop acknowledgment and fresh joints before opening the empty gripper...', flush=True)
    require_ready_engine(prepare_episode=True)
    print('Pi0 engine ready: gripper OPEN IO acknowledged; no arm target sent.', flush=True)


if __name__ == '__main__':
    main()
