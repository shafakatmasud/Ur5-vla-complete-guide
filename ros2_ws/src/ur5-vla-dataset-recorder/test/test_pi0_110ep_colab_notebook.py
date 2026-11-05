"""Check Pi0 semantics and verified persistent checkpoint handling offline."""
import ast
import importlib.util
import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1]/'notebooks/train_ur5_cube_110ep_v3_pi0_a100_colab.ipynb'


def source(index):
    return ''.join(json.loads(NOTEBOOK.read_text())['cells'][index]['source'])


def test_clean_notebook_and_pi0_arguments():
    notebook=json.loads(NOTEBOOK.read_text())
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            compile('\n'.join(l for l in ''.join(cell['source']).splitlines() if not l.startswith('%pip ')), 'cell', 'exec')
            assert cell['outputs']==[] and cell['execution_count'] is None
    namespace=dict(json=json,BASE_MODEL_PATH='/cached/pi0',TRAIN_EXPERT_ONLY=True,
                   NORMALIZATION={'STATE':'MEAN_STD','ACTION':'MEAN_STD','VISUAL':'IDENTITY'})
    fn=next(n for n in ast.parse(source(9)).body if isinstance(n,ast.FunctionDef) and n.name=='policy_arguments')
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'policy_arguments','exec'),namespace)
    args=namespace['policy_arguments']()
    assert '--policy.type=pi0' in args and '--policy.pretrained_path=/cached/pi0' in args
    assert '--policy.use_relative_actions=false' in args and '--policy.train_expert_only=true' in args
    assert '--policy.empty_cameras=1' in args
    assert json.loads(next(a.split('=',1)[1] for a in args if a.startswith('--policy.normalization_mapping=')))==namespace['NORMALIZATION']


def shared_checks():
    path=Path(__file__).with_name('test_pi05_110ep_colab_notebook.py')
    spec=importlib.util.spec_from_file_location('shared_checkpoint_validation',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.NOTEBOOK=NOTEBOOK
    return module


def test_checkpoint_copy_and_retention(tmp_path):
    shared_checks().test_prunes_only_verified_local_copies_after_new_checkpoint_sync(tmp_path)


def test_corrupt_persistent_copy_cannot_delete_local_checkpoint(tmp_path):
    shared_checks().test_corrupted_drive_copy_prevents_local_pruning(tmp_path)
