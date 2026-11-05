#!/usr/bin/env python3
"""Build a self-contained cloud notebook; embedding scripts avoids repo uploads."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parent
TARGET = ROOT.parent.parent / 'notebooks/train_ur5_cube_110ep_openvla_a100_80gb.ipynb'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-memory-gb', type=int, choices=(40, 80), default=80)
    args = parser.parse_args()
    target = TARGET.with_name(TARGET.name.replace('80gb', f'{args.gpu_memory_gb}gb'))
    files = ('train.py', 'dataset.py', 'prepare_data.py', 'requirements.txt', 'reviewed_source.json', 'load_checkpoint.py')
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in files:
            # Fixed archive timestamps make regenerated notebooks reproducible.
            entry = zipfile.ZipInfo(name, date_time=(2026, 11, 4, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(entry, (ROOT / name).read_bytes())
    payload = base64.b64encode(buffer.getvalue()).decode()
    digest = hashlib.sha256(buffer.getvalue()).hexdigest()
    cells = []

    def add(kind, source):
        cell = {'cell_type': kind, 'metadata': {}, 'source': textwrap.dedent(source).strip().splitlines(True)}
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add('markdown', '''
        # UR5 orange-cube dataset: OpenVLA on one A100 80 GB

        Run the cells in order. This notebook **has not been GPU-tested or used to start training yet**.
        It first checks the hardware and performs two optimizer updates, saves a checkpoint, and
        reloads that checkpoint in a new process. The full run starts only after those checks pass.

        **Recipe:** original `openvla/openvla-7b`, LoRA rank 32, BF16, microbatch 4 × accumulation 4
        (effective batch 16), constant learning rate 0.0005, 20,000 optimizer steps, save every 250
        steps, retain the latest two complete checkpoints. The base weights are pinned to a commit.

        **Data:** the reviewed 110 episodes / 285,204 frames at 20 Hz. The input is the **base RGB
        camera only**; original OpenVLA does not use our wrist camera or joint-state observation.
        Labels preserve six **absolute UR5 joint targets in radians**, followed by gripper
        `0 = closed`, `1 = open`. All seven dimensions use stored training-set 1st/99th-percentile
        normalization and the original 256-bin action tokenizer. Do not interpret these outputs
        as Cartesian displacements. Cached frames retain 640×480 geometry and use JPEG quality 92.

        All 110 episodes are training data; there is **no held-out validation split**. Loss, action
        accuracy, and joint error shown here are teacher-forced training diagnostics, not robot
        success rates. This recipe is the original OpenVLA baseline, **not OpenVLA-OFT**.

        The notebook creates a separate Python 3.10 environment and OpenVLA output folder. It does
        not modify SmolVLA/Pi0/Pi0.5 or connect to the robot. Training is not deployment.

        In Colab's **Secrets** panel, add `WANDB_API_KEY` and optionally `HF_TOKEN`, and enable
        **Notebook access** for each. Cell 6 reads them with `google.colab.userdata.get` without
        displaying their values. A hidden-input prompt remains available if the W&B secret is absent.

        Use a provider/runtime that actually supplies **A100 80 GB** and persistent storage.
        A runtime labelled only “A100” may have 40 GB; this notebook will stop on that hardware.
        Budget at least 100 GiB of free local disk. Persistent checkpoints are necessary to recover
        after a session ends; Colab cannot guarantee uninterrupted long training sessions.

        References: [OpenVLA](https://github.com/openvla/openvla),
        [official LoRA recipe](https://github.com/openvla/openvla/blob/main/vla-scripts/finetune.py),
        [OpenVLA model](https://huggingface.co/openvla/openvla-7b).
    ''')
    add('code', '''
        # 1. Verify the GPU before downloading packages or model weights.
        import os, sys, json, subprocess, shutil, hashlib, time, signal
        from pathlib import Path

        if sys.platform != 'linux':
            raise RuntimeError('Use a Linux NVIDIA GPU notebook runtime.')
        gpu_lines = subprocess.check_output([
            'nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'
        ], text=True).strip().splitlines()
        name, memory_mib = [x.strip() for x in gpu_lines[0].rsplit(',', 1)]
        print('Selected GPU 0:', name, memory_mib, 'MiB')
        if 'A100' not in name or int(memory_mib) < 76800:
            raise RuntimeError('Select an A100 80 GB runtime, then reconnect. No training was started.')
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        print('GPU capacity check passed.')
    ''')
    add('code', '''
        # 2. Configure source data and a PERSISTENT output location.
        IN_COLAB = 'google.colab' in sys.modules
        try:
            from google.colab import drive
        except ImportError:
            IN_COLAB = False
        else:
            IN_COLAB = True
            drive.mount('/content/drive')

        WORK_ROOT = Path('/content/openvla_ur5') if IN_COLAB else Path.cwd() / 'openvla_ur5_work'
        WORK_ROOT.mkdir(parents=True, exist_ok=True)
        # This is the same Drive dataset location used by the previous training notebooks.
        DRIVE_ROOT = Path('/content/drive/MyDrive/VLA_Thesis/Sorted VLA Backup Final/pHYSICAL UR5')
        DATASET_SOURCE_DIR = DRIVE_ROOT / 'datasets/lerobot/ur5_cube_110ep_v3' if IN_COLAB else Path('/workspace/datasets/ur5_cube_110ep_v3')
        DATASET_ARCHIVE = DRIVE_ROOT / 'datasets/lerobot/ur5_cube_110ep_v3.tar' if IN_COLAB else Path('/workspace/datasets/ur5_cube_110ep_v3.tar')
        # For a provider other than Colab, EDIT SAVE_ROOT to its mounted persistent volume.
        SAVE_ROOT = DRIVE_ROOT / 'checkpoints/ur5_openvla_trained' if IN_COLAB else Path('/workspace/persistent/openvla_ur5')
        RUN_NAME = 'ur5_cube_110ep_openvla_lora32_b16_20000_steps_v1'
        OUTPUT = SAVE_ROOT / RUN_NAME
        LOCAL_DATASET = WORK_ROOT / 'ur5_cube_110ep_v3'
        CACHE = WORK_ROOT / 'prepared_base_rgb'
        CODE = WORK_ROOT / 'code'
        VENV = WORK_ROOT / 'env'
        TRAIN_STEPS, BATCH_SIZE, ACCUMULATE, SAVE_EVERY = 20000, 4, 4, 250
        MIN_GPU_MEMORY_GIB = 75
        WANDB_ENTITY = 'mdshafakat-masud-ontario-tech-university'
        WANDB_PROJECT = 'ur5-experiment1-openvla'

        required_gib = 20 if (CACHE / 'manifest.json').exists() and VENV.exists() else 100
        if shutil.disk_usage(WORK_ROOT).free < required_gib * 1024**3:
            raise RuntimeError(f'Need at least {required_gib} GiB free local disk for this setup/resume.')
        SAVE_ROOT.mkdir(parents=True, exist_ok=True)
        print('Dataset folder:', DATASET_SOURCE_DIR)
        print('Alternative dataset archive:', DATASET_ARCHIVE)
        print('Persistent run folder:', OUTPUT)
        print('Verify that SAVE_ROOT survives runtime termination before proceeding.')
    ''')
    add('code', f'''
        #@title 3. Unpack the embedded training source (click to expand code)
        import base64, io, zipfile
        payload = base64.b64decode({payload!r})
        if hashlib.sha256(payload).hexdigest() != {digest!r}:
            raise RuntimeError('Embedded training bundle checksum mismatch')
        CODE.mkdir(exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for member in archive.infolist():
                if Path(member.filename).name != member.filename:
                    raise RuntimeError('Unexpected embedded filename')
                (CODE / member.filename).write_bytes(archive.read(member))
        print('Embedded source verified:', ', '.join(sorted(p.name for p in CODE.iterdir())))
    ''')
    cells[-1]['metadata']['cellView'] = 'form'
    add('code', '''
        # 4. Create an isolated Python 3.10 environment, including on newer Colab kernels.
        # Kernel packages and existing robot/ML environments are not modified.
        def run(command, env=None, log=None):
            process = subprocess.Popen([str(x) for x in command], stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
            log_file = Path(log).open('a') if log else None
            try:
                try:
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        if log_file:
                            log_file.write(line)
                            log_file.flush()
                except KeyboardInterrupt:
                    if process.poll() is None:
                        print('Requesting graceful stop; wait for CHECKPOINT SAVED before closing.')
                        process.send_signal(signal.SIGUSR1)
                        for line in process.stdout:
                            print(line, end='', flush=True)
                            if log_file:
                                log_file.write(line)
                    process.wait()
                    raise
                status = process.wait()
                if status:
                    raise RuntimeError(f'Process exited with status {status}; inspect the error above.')
            finally:
                if log_file:
                    log_file.close()

        BOOTSTRAP = WORK_ROOT / 'bootstrap'
        uv_binary = BOOTSTRAP / 'bin/uv'
        if not uv_binary.exists():
            run([sys.executable, '-m', 'pip', 'install', '--target', BOOTSTRAP, 'uv==0.9.8'])
        CHILD_ENV = os.environ.copy()
        CHILD_ENV.update({'UV_PYTHON_INSTALL_DIR': str(WORK_ROOT / 'python'),
                          'UV_CACHE_DIR': str(WORK_ROOT / 'uv_cache'),
                          'HF_HOME': str(WORK_ROOT / 'hf_cache'),
                          'TOKENIZERS_PARALLELISM': 'false', 'CUDA_VISIBLE_DEVICES': '0'})
        PYTHON = VENV / 'bin/python'
        if not PYTHON.exists():
            run([uv_binary, 'python', 'install', '3.10'], env=CHILD_ENV)
            run([uv_binary, 'venv', '--seed', '--python', '3.10', VENV], env=CHILD_ENV)
        run([uv_binary, 'pip', 'install', '--python', PYTHON,
             'torch==2.2.2', 'torchvision==0.17.2', '--index-url',
             'https://download.pytorch.org/whl/cu121'], env=CHILD_ENV)
        run([uv_binary, 'pip', 'install', '--python', PYTHON, '-r', CODE / 'requirements.txt'], env=CHILD_ENV)
        run([PYTHON, '-c',
             'import torch,transformers,peft,timm,av; '
             'print("Torch:",torch.__version__,"Transformers:",transformers.__version__); '
             'assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(); '
             'print("GPU:",torch.cuda.get_device_name(0))'], env=CHILD_ENV)
    ''')
    add('code', '''
        # 5. Stage the reviewed dataset on local SSD; verify its actual file hashes.
        import tarfile
        from pathlib import PurePosixPath

        def sha256_file(path):
            h = hashlib.sha256()
            with Path(path).open('rb') as f:
                for block in iter(lambda: f.read(8*1024**2), b''):
                    h.update(block)
            return h.hexdigest()

        reviewed = json.loads((CODE / 'reviewed_source.json').read_text())
        if not LOCAL_DATASET.exists():
            staging = WORK_ROOT / ('.staging-' + str(time.time_ns()))
            staging.mkdir()
            if DATASET_SOURCE_DIR.is_dir():
                shutil.copytree(DATASET_SOURCE_DIR, staging / LOCAL_DATASET.name)
            elif DATASET_ARCHIVE.is_file():
                # Reject links and paths outside the one expected dataset root.
                with tarfile.open(DATASET_ARCHIVE, 'r:*') as archive:
                    members = archive.getmembers()
                    for member in members:
                        p = PurePosixPath(member.name)
                        if (not p.parts or p.parts[0] != LOCAL_DATASET.name or
                            p.is_absolute() or '..' in p.parts or
                            not (member.isfile() or member.isdir())):
                            raise ValueError(f'Unsafe or unexpected dataset archive member: {member.name}')
                    archive.extractall(staging, members=members)
            else:
                raise FileNotFoundError('Set DATASET_SOURCE_DIR or DATASET_ARCHIVE to the reviewed 110-episode dataset.')
            (staging / LOCAL_DATASET.name).rename(LOCAL_DATASET)
            staging.rmdir()
        for i, (relative, expected) in enumerate(reviewed['sha256'].items(), 1):
            if sha256_file(LOCAL_DATASET / relative) != expected:
                raise ValueError(f'Source artifact checksum mismatch: {relative}')
            if i % 10 == 0:
                print('Verified dataset files:', i, '/', len(reviewed['sha256']))
        print('All reviewed dataset file checksums match.')
        run([PYTHON, CODE / 'prepare_data.py', '--dataset', LOCAL_DATASET, '--cache', CACHE], env=CHILD_ENV)
    ''')
    add('code', '''
        # 6. Read Colab Secrets. Enable Notebook access in Colab's Secrets panel.
        # Secret names: WANDB_API_KEY (required for online graphs), HF_TOKEN (optional).
        # Values stay in child-process environment variables and are never printed.
        from getpass import getpass

        def runtime_secret(name, required=False):
            existing = CHILD_ENV.get(name, '').strip()
            if existing:
                return existing
            if IN_COLAB:
                from google.colab import userdata
                try:
                    value = userdata.get(name).strip()
                except userdata.SecretNotFoundError:
                    if not required:
                        return ''
                except userdata.NotebookAccessError:
                    raise RuntimeError(f'Enable Notebook access for {name} in Colab Secrets, then rerun this cell.') from None
                except userdata.TimeoutException:
                    raise RuntimeError('Run this cell from the Colab notebook UI to access Secrets.') from None
                else:
                    if value or not required:
                        return value
            return getpass(f'{name} (hidden; Enter uses existing credentials or skips optional token): ').strip()

        key = runtime_secret('WANDB_API_KEY', required=True)
        if key:
            CHILD_ENV['WANDB_API_KEY'] = key
        del key
        token = runtime_secret('HF_TOKEN')
        if token:
            CHILD_ENV['HF_TOKEN'] = token
        del token
        run([PYTHON, '-c', 'import wandb; wandb.login(verify=True)'], env=CHILD_ENV)
        print('W&B project: https://wandb.ai/' + WANDB_ENTITY + '/' + WANDB_PROJECT)
        print('Full training will create/resume its own run and print the live run URL.')
        print('Charts include train/loss, train/learning_rate, action-token accuracy, joint MAE and GPU memory.')
    ''')
    add('code', '''
        # 7. Real memory/update/save test, followed by a fresh-process resume test.
        def training_command(output, probe=False):
            command = [PYTHON, '-u', CODE / 'train.py', '--cache', CACHE, '--output', output,
                       '--steps', str(TRAIN_STEPS), '--batch-size', str(BATCH_SIZE),
                       '--accumulate', str(ACCUMULATE), '--save-every', str(SAVE_EVERY),
                       '--min-gpu-memory-gib', str(MIN_GPU_MEMORY_GIB),
                       '--wandb-entity', WANDB_ENTITY, '--wandb-project', WANDB_PROJECT]
            return command + (['--probe'] if probe else [])

        PROBE = WORK_ROOT / f'probe_b{BATCH_SIZE}_a{ACCUMULATE}'
        run(training_command(PROBE, probe=True), env=CHILD_ENV)
        first = json.loads((PROBE / 'status.json').read_text())
        if first['phase'] != 'probe_passed' or first['step'] != 2:
            raise RuntimeError('The two-update probe did not finish.')
        run(training_command(PROBE, probe=True), env=CHILD_ENV)
        second = json.loads((PROBE / 'status.json').read_text())
        if second['phase'] != 'probe_passed' or not second['resume_checkpoint_loaded']:
            raise RuntimeError('Fresh-process checkpoint resume was not verified.')
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / 'probe_report.json').write_text(json.dumps(second, indent=2))
        PROBE_PASSED = True
        print('MEMORY, OPTIMIZER UPDATE, CHECKPOINT AND COLD-RESUME CHECKS PASSED.')
        print('If the probe runs out of memory, lower BATCH_SIZE and raise ACCUMULATE to retain batch 16, then repeat.')
    ''')
    add('code', '''
        # 8. START / RESUME FULL TRAINING. This cell stays active while training runs.
        # A same-name, same-settings run resumes its newest COMPLETE checkpoint automatically.
        # To pause, interrupt this cell ONCE and wait for the graceful checkpoint-save message.
        # Runtime crashes can lose up to 249 updates; existing completed checkpoints remain usable.
        assert PROBE_PASSED, 'Complete the memory and cold-resume probe first.'
        print('Starting original OpenVLA LoRA training on all 110 episodes.')
        print('Use the LIVE W&B URL printed below to view the graphs during this cell.')
        print('Persistent output:', OUTPUT)
        run(training_command(OUTPUT), env=CHILD_ENV, log=OUTPUT / 'terminal.log')
    ''')
    add('code', '''
        # 9. Inspect completion or pause state and the latest checkpoint.
        state = json.loads((OUTPUT / 'status.json').read_text())
        print(json.dumps(state, indent=2))
        pointer = OUTPUT / 'latest_checkpoint.json'
        if pointer.exists():
            print('Latest complete checkpoint:', pointer.read_text())
        if state.get('wandb_url'):
            from IPython.display import display, HTML
            display(HTML('<a target="_blank" href="' + state['wandb_url'] + '">Open training graphs in W&B</a>'))
        if state['phase'] == 'finished':
            print('Training is complete. The checkpoint contains a LoRA adapter plus optimizer/RNG state.')
            print('It is not a standalone merged model; load it with the pinned base model and saved normalization.')
            print('No robot execution setup was changed. A separate read-only inference check is needed before deployment.')
    ''')
    add('markdown', '''
        ## Resume and retrieve

        Keep the same `RUN_NAME`, batch/accumulation, source, and training settings. In a fresh
        runtime, rerun the setup, data, authentication and probe cells, then rerun the training cell.
        It restores the adapter, optimizer, scheduler, Python/NumPy/Torch/CUDA random states and
        exact deterministic sample position from the newest checksum-verified complete checkpoint.

        The run directory contains `terminal.log`, `metrics.jsonl`, `run_config.json`, `status.json`,
        source snapshots, package versions, the W&B files and `checkpoints/XXXXXX/`. Each checkpoint
        includes `adapter_model.safetensors`, `adapter_config.json`, the processor/tokenizer,
        `dataset_statistics.json`, `action_contract.json`, `openvla_config.json`, `training_state.pt`
        and `COMPLETE.json` checksums. Only the newest two complete checkpoints are retained.

        Use `load_checkpoint.py` from the accompanying local experiment folder for **read-only**
        inference loading later. Preserve the base-model commit and seven-dimensional action
        contract. This notebook deliberately does not install or alter robot execution scripts.
    ''')
    if args.gpu_memory_gb == 40:
        for cell in cells:
            source = ''.join(cell['source'])
            source = source.replace('This notebook **has not been GPU-tested or used to start training yet**.',
                                    'This A100 40 GB profile has passed real GPU memory/update tests; each setup still requires the full-data probe.')
            source = source.replace(
                'Use a provider/runtime that actually supplies **A100 80 GB** and persistent storage.\n'
                'A runtime labelled only “A100” may have 40 GB; this notebook will stop on that hardware.',
                'This separate **A100 40 GB** profile uses microbatch 16 and accumulation 1.\n'
                'The effective batch is still 16. A real memory/update/resume probe is mandatory.')
            source = source.replace('microbatch 4 × accumulation 4', 'microbatch 16 × accumulation 1')
            source = source.replace('80 GB', '40 GB')
            source = source.replace('int(memory_mib) < 76800', 'int(memory_mib) < 35840')
            source = source.replace('= 20000, 4, 4, 250', '= 20000, 16, 1, 250')
            source = source.replace('MIN_GPU_MEMORY_GIB = 75', 'MIN_GPU_MEMORY_GIB = 35')
            source = source.replace('20000_steps_v1', '20000_steps_a10040_b16a1_v2')
            cell['source'] = source.splitlines(True)
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3',
                'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'},
                'accelerator': 'GPU', 'colab': {'gpuType': 'A100'}, 'openvla_bundle_sha256': digest,
                'gpu_memory_profile_gb': args.gpu_memory_gb,
                'training_status_at_creation': 'not_started; A100 validation pending'},
                'nbformat': 4, 'nbformat_minor': 4}
    target.write_text(json.dumps(notebook, indent=1) + '\n')
    print(target)
    print('Self-contained notebook:', len(cells), 'cells;', target.stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
