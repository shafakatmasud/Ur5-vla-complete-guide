"""Training contract validation without ML downloads, ROS, or robot commands."""
import argparse
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_policy import build_command, validate_dataset


def dataset(tmp_path):
    root = tmp_path / 'dataset'
    (root / 'meta').mkdir(parents=True)
    features = {key: {'shape': shape} for key, shape in {
        'observation.state': [7], 'action': [7],
        'observation.images.base': [480, 640, 3],
        'observation.images.wrist': [480, 640, 3],
    }.items()}
    (root / 'meta/info.json').write_text(json.dumps(dict(features=features, fps=20, total_episodes=2)))
    (root / 'meta/ur5_vla_conversion.json').write_text(json.dumps(dict(action_mode='absolute')))
    return root


@pytest.mark.parametrize('model', ['smolvla', 'pi0', 'act', 'diffusion'])
def test_all_recipes_use_dataset_contract_without_robot_specific_pretrained_config(tmp_path, model):
    args = argparse.Namespace(dataset=dataset(tmp_path), output=tmp_path/'new_run', model=model,
                              repo_id='local/test', batch_size=1, steps=100, num_workers=0,
                              save_freq=50, no_wandb=False, wandb_project='test', wandb_entity=None,
                              pretrained_path=None, dtype='float32')
    command = build_command(args)
    assert '--policy.path' not in ' '.join(command)
    assert '--policy.push_to_hub=false' in command
    if model in ('smolvla', 'pi0'):
        assert '--policy.empty_cameras=0' in command
        assert any(s.startswith('--policy.pretrained_path=') for s in command)
    assert not args.output.exists()


@pytest.mark.parametrize('field,value', [('action', [6]), ('observation.state', [6]),
                                         ('observation.images.base', [3,480,640])])
def test_wrong_schema_is_rejected(tmp_path, field, value):
    root = dataset(tmp_path)
    path = root/'meta/info.json'
    info = json.loads(path.read_text())
    info['features'][field]['shape'] = value
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match=field):
        validate_dataset(root)


def test_delta_actions_are_not_silently_used_as_absolute(tmp_path):
    root = dataset(tmp_path)
    (root/'meta/ur5_vla_conversion.json').write_text('{"action_mode":"delta"}')
    with pytest.raises(ValueError, match='absolute'):
        validate_dataset(root)
