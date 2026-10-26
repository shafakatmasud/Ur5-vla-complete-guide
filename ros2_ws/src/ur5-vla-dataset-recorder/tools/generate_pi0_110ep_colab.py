#!/usr/bin/env python3
"""Create experiment-1 Pi0 Colab using the verified dataset/copy workflow."""
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[2]
SOURCE = PACKAGE/'notebooks/train_ur5_cube_110ep_v3_pi05_a100_colab.ipynb'
OUTPUT = PACKAGE/'notebooks/train_ur5_cube_110ep_v3_pi0_a100_colab.ipynb'


def main():
    notebook = json.loads(SOURCE.read_text())
    def get(index):
        return ''.join(notebook['cells'][index]['source'])
    def put(index, text):
        notebook['cells'][index]['source'] = (text.strip()+'\n').splitlines(keepends=True)
    for i, cell in enumerate(notebook['cells']):
        put(i, get(i).replace('pi05','pi0').replace('Pi0.5','Pi0').replace('π0.5','π0'))
        cell.setdefault('metadata', {})['id'] = f'ur5-pi0-110-{i:02d}'
        cell['id'] = f'ur5-pi0-110-{i:02d}'
        if cell['cell_type']=='code':
            cell['outputs']=[]
            cell['execution_count']=None
    put(0, '''# Experiment 1: UR5 Pi0 training — Colab A100

Train **lerobot/pi0_base** on the reviewed **110 orange-cube episodes / 285,204 frames at 20 Hz**.
This is **Pi0**, not Pi0.5 or SmolVLA. The default trains the action expert and projections,
freezing the vision-language backbone. Inputs: base/wrist RGB, 7D measured joint/gripper state,
and the original orange-cube instruction. Targets: six absolute joint targets in radians plus
binary gripper (0 closed, 1 open). Pi0 uses **MEAN_STD** state/action normalization.

1. Select **Runtime → Change runtime type → A100 GPU**, preferably high RAM.
2. Add **HF_TOKEN** and **WANDB_API_KEY** in Colab Secrets and enable notebook access.
   Hidden login prompts are also available. Accept the PaliGemma tokenizer license with the same HF account.
3. Choose **Runtime → Run all**, authorize Drive, and complete any authentication prompts.

The existing Drive dataset is used directly as the source, then checksum-copied to fast local storage.
The notebook validates all state/action rows and 330 paired camera samples. A real optimizer-step
probe chooses a batch with measured memory reserve. Training uses bfloat16, gradient checkpointing,
20,000 optimizer steps, online W&B graphs, and verified Drive checkpoints every 2,500 steps.
The live dashboard URL is printed by the trainer and saved with the run. ETA comes from actual
probe timings; an A100 allocation or a fixed completion time is not guaranteed.

This starts a **new cloud run from the Pi0 base weights**. It does not resume the local FSDP run.
The same dataset is used, but the automatically selected cloud batch may differ; configuration records this.
All 110 episodes are training data; training loss is not physical task success or held-out accuracy.
Keep Drive capacity for eight checkpoints plus final exports. Latest two local copies are retained
only after verifying their persistent Drive copies. Completed checkpoints survive a runtime disconnect;
fresh Run all starts a new run. Resume instructions are at the end.

References: [Pi0](https://huggingface.co/docs/lerobot/main/pi0),
[base weights](https://huggingface.co/lerobot/pi0_base),
[LeRobot 0.6.1](https://pypi.org/project/lerobot/0.6.1/),
[Colab runtime limits](https://research.google.com/colaboratory/faq.html).
''')
    put(2, get(2).replace("USE_WANDB = True", "USE_WANDB = True  # Required for this experiment; online graphs"))
    put(3, '''## Install the pinned Pi0 stack

Use a fresh A100 runtime with Python 3.12/3.13. Installation is isolated to Colab; it does not
modify the recording workstation. The software versions, dataset hashes and model revision
are saved with each cloud run. If Colab asks for a runtime restart after installation, restart
and rerun the cells, inspecting the installed-version checks before proceeding.
''')
    put(4, get(4).replace('"av==15.1.0" matplotlib psutil',
                         '"av==15.1.0" "wandb==0.24.2" "accelerate==1.14.0" matplotlib psutil'))
    validation = get(6).split("# Pi0's tokenizer expects quantile-normalized state.")[0]
    put(6, validation+'''
# Pi0 normalizes continuous state/action with the dataset mean and standard deviation.
for key in ('observation.state', 'action'):
    for statistic in ('mean', 'std'):
        value = np.asarray(dataset.meta.stats[key][statistic]).reshape(-1)
        assert value.shape == (7,) and np.isfinite(value).all()
        if statistic == 'std': assert np.all(value >= 0)
validation_report['normalization'] = 'MEAN_STD for STATE/ACTION; IDENTITY for VISUAL'
Path('/content/dataset_validation.json').write_text(json.dumps(validation_report, indent=2))
''')
    put(7, get(7).replace("WANDB_PROJECT = 'pi0_ur5_cube_110ep_v3_a100'", "WANDB_PROJECT = 'ur5-experiment1-pi0'")
        .replace('    wandb.login()', '''    try:
        wandb_key = userdata.get('WANDB_API_KEY')
    except Exception:
        wandb_key = None
    wandb.login(key=wandb_key) if wandb_key else wandb.login()'''))
    put(8, '''## Benchmark actual Pi0 optimizer steps

Both benchmark and training use bfloat16, checkpointing, mean/std normalization and absolute
actions. Base and wrist camera keys stay unchanged; one masked empty camera supplies the third
pretraining slot. The model pads 7D vectors internally. Compilation stays disabled.
Only probes that load all pretrained keys, complete optimizer steps and retain at least 12%
measured VRAM reserve are eligible. Non-memory errors stop with their traceback.
''')
    probe = get(9).replace("'QUANTILES'", "'MEAN_STD'")
    probe = probe.replace('"success": code == 0,',
                          '"success": code == 0 and "All keys loaded successfully!" in text,')
    probe = probe.replace("if code != 0:\n        result[", "if not result['success']:\n        result[")
    # The upstream Pi0 loader may return a random model after a load exception.
    # Never accept a probe on that basis, even if its process exit code is zero.
    probe = probe.replace("if not result['success']:\n        result[", "if not result['success']:\n        result[")
    put(9, probe)
    put(10, get(10).replace("f'--wandb.enable={str(USE_WANDB).lower()}',", "'--wandb.enable=true', '--wandb.mode=online', '--wandb.disable_artifact=true',")
        .replace("'--log_freq=10'", "'--log_freq=1'"))
    train = get(12).replace("print(line, end=''); log.write(line); log.flush()", '''print(line, end=''); log.write(line); log.flush()
            if any(message in line for message in ('Returning model without loading pretrained weights', 'Warning: Could not load state dict:')):
                raise RuntimeError('Pi0 pretrained loading failed; refusing random initialization')
            if 'wandb.ai/' in line:
                for url in re.findall(r'https://wandb\\.ai/[^\\s\\x1b]+', line):
                    if '/runs/' in url:
                        (DRIVE_RUN_DIR / 'wandb_url.txt').write_text(url+'\\n')''')
    put(12, train)
    put(15, '''## Completed model and resuming an interrupted cloud run

The final Pi0 model is in `checkpoints/ur5_pi0_trained/<run-name>/pretrained_model/`.
A portable tar and checksum are in `deployment/downloaded_checkpoints/`. The run records versions,
source dataset hashes, normalization, model revision, selected batch and actual metrics.
Use the exported Pi0 processors for offline evaluation before any physical deployment.

After a disconnect, restore the same environment, dataset staging and authentication cells.
Instead of launching a fresh training run, point `--config_path` at a completed Drive checkpoint:

```python
checkpoint = Path('/content/drive/MyDrive/VLA_Thesis/Sorted VLA Backup Final/pHYSICAL UR5/checkpoints/ur5_pi0_trained/YOUR_RUN/checkpoints/002500')
command = ['lerobot-train', '--config_path='+str(checkpoint/'pretrained_model/train_config.json'), '--resume=true']
subprocess.run(command, check=True)
```

The saved dataset path expects `/content/ur5_cube_110ep_v3`, so restaging is required.
This direct resume command restores training state and W&B identity but **does not run this notebook's
Drive-sync/export loop**. For a long resumed session, arrange checkpoint syncing before relying on
disconnect protection. Do not run a fresh training subprocess and a resume subprocess together.
''')
    notebook['metadata'].update({'accelerator':'GPU','colab':{'name':OUTPUT.name,'provenance':[]}})
    text = json.dumps(notebook,indent=2,ensure_ascii=False)+'\n'
    OUTPUT.write_text(text)
    (REPO/'notebooks'/OUTPUT.name).write_text(text)
    print(OUTPUT)


if __name__=='__main__':
    main()
