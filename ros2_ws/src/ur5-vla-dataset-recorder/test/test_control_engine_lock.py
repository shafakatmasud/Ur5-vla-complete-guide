"""The base and phased entrypoints must share one exclusive engine owner."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'scripts'))
from control_engine_lock import acquire_engine_lock


def test_duplicate_owner_is_rejected_and_release_allows_restart(tmp_path):
    path = tmp_path/'engine.lock'
    first = acquire_engine_lock(path)
    try:
        with pytest.raises(RuntimeError, match='already running'):
            acquire_engine_lock(path)
        assert path.read_text().strip().isdigit()
    finally:
        first.close()
    restarted = acquire_engine_lock(path)
    restarted.close()


def test_engine_entrypoints_acquire_lock_before_ros_initialization():
    import ast
    for name in ('control_engine.py', 'control_engine_phased.py'):
        tree = ast.parse((ROOT/'scripts'/name).read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        lock = next(n for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'acquire_engine_lock')
        ros = next(n for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'init')
        assert lock.lineno < ros.lineno
