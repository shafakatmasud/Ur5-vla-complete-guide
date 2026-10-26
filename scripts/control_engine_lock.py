"""Exclusive process ownership shared by both standalone control engines."""
import fcntl
import os
from pathlib import Path
import tempfile


def acquire_engine_lock(path=None):
    path = Path(path) if path is not None else Path(tempfile.gettempdir()) / f'ur5-vla-engine-{os.getuid()}.lock'
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    handle = os.fdopen(fd, 'r+')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        owner = handle.read().strip() or 'unknown'
        handle.close()
        raise RuntimeError(f'Another control engine is already running (PID {owner}). Keep one engine only.') from None
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()) + '\n')
    handle.flush()
    return handle  # Keep open for the entire engine lifetime; never unlink it.
