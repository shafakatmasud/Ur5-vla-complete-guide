#!/usr/bin/env python3
"""Single-A100 original OpenVLA LoRA training with exact sample-position resume.

Training-only process. No ROS, robot connections, or command publishers.
All metrics are teacher-forced training diagnostics, not physical task success.
"""
import argparse
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import time

MODEL = 'openvla/openvla-7b'
MODEL_REVISION = '47a0ec7fc4ec123775a391911046cf33cf9ed83f'


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def verify_checkpoint(path):
    from dataset import sha256
    marker = json.loads((Path(path) / 'COMPLETE.json').read_text())
    if marker['step'] != int(Path(path).name):
        raise ValueError('Checkpoint step/name mismatch')
    for relative, expected in marker['sha256'].items():
        if sha256(Path(path) / relative) != expected:
            raise ValueError(f'Corrupt checkpoint: {path}/{relative}')
    return marker


def latest_checkpoint(output):
    candidates = sorted(p for p in (Path(output) / 'checkpoints').glob('*')
                        if p.is_dir() and p.name.isdigit() and (p / 'COMPLETE.json').exists())
    if not candidates:
        return None
    verify_checkpoint(candidates[-1])
    return candidates[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--accumulate', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=5e-4)
    parser.add_argument('--save-every', type=int, default=250)
    # HF's dynamically loaded processor is not reliably pickleable with spawn.
    # Cached JPEG decoding is cheap; keep preprocessing in the training process.
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--min-gpu-memory-gib', type=int, default=75)
    parser.add_argument('--wandb-project', default='ur5-experiment1-openvla')
    parser.add_argument('--wandb-entity', default='mdshafakat-masud-ontario-tech-university')
    parser.add_argument('--probe', action='store_true', help='Two optimizer updates plus checkpoint; no W&B run')
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.accumulate, args.save_every) < 1 or args.workers < 0:
        parser.error('Step counts, batch size, and accumulation must be positive')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        parser.error('This recipe is for exactly one A100 GPU')
    if args.min_gpu_memory_gib < 35:
        parser.error('Use an A100 40 GB or larger for this BF16 LoRA recipe')
    import fcntl
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from peft import LoraConfig, PeftModel, get_peft_model
    from dataset import (DATASET_KEY, UR5Dataset, ResumableSampler, collate,
                         decode_actions, sha256, unnormalize)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    properties = torch.cuda.get_device_properties(0)
    if 'A100' not in properties.name or properties.total_memory / 1024**3 < args.min_gpu_memory_gib:
        raise RuntimeError(f'Requires A100 with at least {args.min_gpu_memory_gib} GiB; '
                           f'received {properties.name}, {properties.total_memory/1024**3:.1f} GiB')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 support required')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    # Prevent duplicate trainers targeting this run on the same host.
    lock_name = hashlib.sha256(str(args.output).encode()).hexdigest()[:24]
    lock = open(f'/tmp/ur5_openvla_{lock_name}.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    manifest = json.loads((args.cache / 'manifest.json').read_text())
    if not manifest.get('complete'):
        raise ValueError('Data preparation is incomplete')
    for relative, expected in manifest['cache_sha256'].items():
        if sha256(args.cache / relative) != expected:
            raise ValueError(f'Prepared cache corrupted: {relative}')
    dataset_digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    signature = {'dataset_manifest_sha256': dataset_digest, 'model': MODEL, 'model_revision': MODEL_REVISION,
                 'variant': 'original OpenVLA; single base RGB; discrete action tokens; no proprio input',
                 'batch_size': args.batch_size, 'accumulate': args.accumulate,
                 'effective_batch_size': args.batch_size * args.accumulate,
                 'learning_rate': args.learning_rate, 'learning_rate_schedule': 'constant',
                 'steps': args.steps, 'lora_rank': 32, 'lora_alpha': 16, 'lora_dropout': 0.,
                 'lora_target_modules': 'all-linear', 'precision': 'bfloat16',
                 'attention': 'eager; no FlashAttention installation needed', 'seed': 42,
                 'minimum_gpu_memory_gib': args.min_gpu_memory_gib,
                 'dataloader_workers': args.workers,
                 'probe': args.probe, 'gradient_checkpointing': 'language model, non-reentrant',
                 'augmentation': 'none', 'training_split': manifest['split'],
                 'implementation_sha256': {name: sha256(Path(__file__).parent / name)
                     for name in ('train.py', 'dataset.py', 'prepare_data.py', 'requirements.txt')}}
    config_path = args.output / 'run_config.json'
    resume = latest_checkpoint(args.output)
    if config_path.exists():
        old = json.loads(config_path.read_text())
        if old['signature'] != signature:
            raise ValueError('Resume settings differ from this run. Use its original settings or a new output folder.')
        if resume is None:
            previous_metrics = args.output / 'metrics.jsonl'
            if previous_metrics.exists() and previous_metrics.stat().st_size:
                raise ValueError('Run contains metrics but no complete checkpoint. Use a new output folder.')
        wandb_id = old['wandb_id']
    else:
        wandb_id = hashlib.sha256((str(args.output) + str(time.time_ns())).encode()).hexdigest()[:8]
        write_json(config_path, {'signature': signature, 'wandb_id': wandb_id,
                                'created_utc': datetime.now(timezone.utc).isoformat(),
                                'hardware': {'gpu': properties.name, 'gpu_memory_bytes': properties.total_memory},
                                'source_dataset_manifest': manifest})
        source_dir = args.output / 'source'
        source_dir.mkdir(exist_ok=True)
        for name in ('train.py', 'dataset.py', 'prepare_data.py', 'requirements.txt', 'reviewed_source.json'):
            shutil.copy2(Path(__file__).parent / name, source_dir / name)
        (args.output / 'pip_freeze.txt').write_text(subprocess.check_output(
            [os.sys.executable, '-m', 'pip', 'freeze'], text=True))
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True
        print('STOP REQUESTED: finishing this optimizer update, then saving a resumable checkpoint.', flush=True)

    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)
    write_json(args.output / 'status.json', {'phase': 'loading_model', 'training_started': False})
    processor = AutoProcessor.from_pretrained(MODEL, revision=MODEL_REVISION, trust_remote_code=True)
    base, loading = AutoModelForVision2Seq.from_pretrained(
        MODEL, revision=MODEL_REVISION, trust_remote_code=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, attn_implementation='eager', output_loading_info=True)
    if any(loading.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise RuntimeError(f'Pretrained weights did not load exactly: {loading}')
    base = base.to('cuda:0')
    base.config.use_cache = False
    base.language_model.config.use_cache = False
    base.norm_stats = manifest['dataset_statistics']
    base.config.norm_stats = manifest['dataset_statistics']
    base.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    if resume:
        model = PeftModel.from_pretrained(base, str(resume), is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(r=32, lora_alpha=16, lora_dropout=0.,
                                              target_modules='all-linear', init_lora_weights='gaussian'))
    model.print_trainable_parameters()
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.)
    step = 0
    if resume:
        state = torch.load(resume / 'training_state.pt', map_location='cpu')
        if state['signature'] != signature:
            raise ValueError('Checkpoint run signature differs')
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        step = state['step']
        if state['consumed_samples'] != step * args.batch_size * args.accumulate:
            raise ValueError('Checkpoint sample-position mismatch')
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        del state
        print(f'RESUMED model, optimizer, scheduler, random states and sample position at step {step}.', flush=True)
    dataset = UR5Dataset(args.cache, processor)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        sampler=ResumableSampler(len(dataset), step * args.batch_size * args.accumulate),
                        num_workers=args.workers, pin_memory=True,
                        collate_fn=partial(collate, pad_token_id=processor.tokenizer.pad_token_id),
                        generator=torch.Generator().manual_seed(42),
                        multiprocessing_context='spawn' if args.workers else None)
    # Validate token alignment and real preprocessor before entering a long run.
    sample = dataset[0]
    if int((sample['labels'] != -100).sum()) != 8:
        raise ValueError('Expected seven supervised action tokens plus EOS')
    run = None
    if not args.probe:
        import wandb
        run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                         id=wandb_id, resume='allow', name=args.output.name,
                         config=signature, mode='online', dir=str(args.output))
        if run is None or run.offline:
            raise RuntimeError('Online W&B logging is required for the full training run')
        wandb.define_metric('train/step')
        wandb.define_metric('train/*', step_metric='train/step')
        print(f'LIVE W&B: {run.url}', flush=True)
    iterator = iter(loader)
    torch.cuda.reset_peak_memory_stats()
    started, start_step = time.monotonic(), step
    goal = min(2, args.steps) if args.probe else args.steps

    def save_checkpoint():
        path = args.output / 'checkpoints' / f'{step:06d}'
        path.parent.mkdir(exist_ok=True)
        if path.exists():
            verify_checkpoint(path)
            return path
        temporary = path.with_name(path.name + f'.incomplete-{time.time_ns()}')
        temporary.mkdir()
        model.save_pretrained(temporary, safe_serialization=True)
        processor.save_pretrained(temporary)
        base.config.to_json_file(temporary / 'openvla_config.json')
        write_json(temporary / 'dataset_statistics.json', manifest['dataset_statistics'])
        write_json(temporary / 'action_contract.json', manifest['action_contract'])
        shutil.copy2(config_path, temporary / 'run_config.json')
        torch.save({'step': step, 'consumed_samples': step * args.batch_size * args.accumulate,
                    'signature': signature, 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'python_rng': random.getstate(),
                    'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all()}, temporary / 'training_state.pt')
        hashes = {str(p.relative_to(temporary)): sha256(p) for p in sorted(temporary.rglob('*')) if p.is_file()}
        write_json(temporary / 'COMPLETE.json', {'step': step, 'sha256': hashes})
        temporary.rename(path)
        verify_checkpoint(path)
        write_json(args.output / 'latest_checkpoint.json', {'step': step, 'path': str(path)})
        complete = sorted(p for p in path.parent.iterdir()
                          if p.is_dir() and p.name.isdigit() and (p / 'COMPLETE.json').exists())
        for older in complete[:-2]:
            shutil.rmtree(older)
        print(f'CHECKPOINT SAVED AND VERIFIED: {path}', flush=True)
        return path

    try:
        while step < goal:
            optimizer.zero_grad(set_to_none=True)
            loss_sum, correct, action_count, valid_count, joint_error_sum, grip_error_sum = 0., 0, 0, 0, 0., 0.
            for _ in range(args.accumulate):
                batch = {k: v.to('cuda:0', non_blocking=True) for k, v in next(iterator).items()}
                batch['pixel_values'] = batch['pixel_values'].to(torch.bfloat16)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    output = model(**batch, use_cache=False)
                    loss = output.loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite loss before optimizer step {step+1}')
                (loss / args.accumulate).backward()
                loss_sum += loss.detach().float().item()
                with torch.no_grad():
                    prefix = output.logits.shape[1] - batch['labels'].shape[1]
                    predicted = output.logits[:, prefix:-1].argmax(-1)
                    truth = batch['labels'][:, 1:]
                    vocab = processor.tokenizer.vocab_size
                    mask = (truth >= vocab - 256) & (truth < vocab)
                    if int(mask.sum()) != args.batch_size * 7:
                        raise ValueError('Multimodal logits/action-label alignment failed')
                    pred_ids, truth_ids = predicted[mask], truth[mask]
                    correct += int((pred_ids == truth_ids).sum())
                    action_count += int(mask.sum())
                    valid_count += int(((pred_ids >= vocab-256) & (pred_ids < vocab)).sum())
                    pred_norm = decode_actions(pred_ids.cpu().numpy().reshape(-1, 7), vocab)
                    gt_norm = decode_actions(truth_ids.cpu().numpy().reshape(-1, 7), vocab)
                    errors = np.abs(unnormalize(pred_norm, dataset.stats) - unnormalize(gt_norm, dataset.stats))
                    joint_error_sum += float(errors[:, :6].sum())
                    grip_error_sum += float(errors[:, 6].sum())
                del output, loss, batch
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            elapsed = time.monotonic() - started
            updates = step - start_step
            metrics = {'train/step': step, 'train/loss': loss_sum / args.accumulate,
                       'train/learning_rate': optimizer.param_groups[0]['lr'],
                       'train/teacher_forced_action_token_accuracy': correct / action_count,
                       'train/predicted_token_in_action_vocabulary_fraction': valid_count / action_count,
                       'train/teacher_forced_joint_mae_rad': joint_error_sum / (args.batch_size * args.accumulate * 6),
                       'train/teacher_forced_gripper_mae': grip_error_sum / (args.batch_size * args.accumulate),
                       'train/gradient_norm_before_clip': float(norm), 'train/consumed_samples': step*args.batch_size*args.accumulate,
                       'train/seconds_per_optimizer_step': elapsed / updates,
                       'train/estimated_remaining_hours': (args.steps-step) * elapsed / updates / 3600,
                       'train/gpu_peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
                       'train/gpu_peak_reserved_gib': torch.cuda.max_memory_reserved() / 1024**3}
            if step == 1 or step % 10 == 0 or step == goal or args.probe:
                print(json.dumps(metrics), flush=True)
                with (args.output / 'metrics.jsonl').open('a') as f:
                    f.write(json.dumps(metrics) + '\n')
                if run:
                    # W&B internal step stays monotonic after a rollback to the last
                    # durable checkpoint. Charts explicitly use train/step as x-axis.
                    run.log(metrics)
            if step % args.save_every == 0 or step == goal or stopping:
                save_checkpoint()
            write_json(args.output / 'status.json', {'phase': 'running', 'step': step, 'training_started': True,
                                                     'updated_utc': datetime.now(timezone.utc).isoformat()})
            if stopping:
                break
        phase = 'paused' if stopping else ('probe_passed' if args.probe else 'finished')
        write_json(args.output / 'status.json', {'phase': phase, 'step': step,
                                                 'resume_checkpoint_loaded': str(resume) if resume else None,
                                                 'training_started': step > 0,
                                                 'wandb_url': run.url if run else None})
        print(f'{phase.upper()}: optimizer step {step}/{goal}', flush=True)
        if run:
            run.summary['final_step'] = step
            run.summary['training_complete'] = step == args.steps
    except Exception as e:
        write_json(args.output / 'status.json', {'phase': 'failed', 'last_completed_optimizer_step': step,
                                                 'error': str(e), 'training_started': step > 0})
        if run:
            run.finish(exit_code=1)
        raise
    else:
        if run:
            run.finish()


if __name__ == '__main__':
    main()
