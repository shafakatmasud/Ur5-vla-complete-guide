#!/usr/bin/env python3
"""Verify frozen pre-ACT model sources without importing policies or ROS."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
record=json.loads((root/'experiments/protected_model_sources.json').read_text())
changed=[]
for relative,expected in record['sha256'].items():
    path=root/relative
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        changed.append(relative)
if changed:raise SystemExit('Protected existing model files changed:\n'+'\n'.join(changed))
print(f"PASS: {len(record['sha256'])} pre-existing model/helper source files unchanged.")
print('No model loading, environment edits, GPU allocation or robot commands performed.')
