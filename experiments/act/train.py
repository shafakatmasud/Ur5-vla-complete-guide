#!/usr/bin/env python3
"""Isolated ACT training: two RGB views, state, absolute joint actions; no ROS."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import time

from data import (ResumableSampler, latest_checkpoint, sha256, validate_dataset,
                  validate_private_source, verify_checkpoint)


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(value, indent=2, default=lambda x: x.tolist()) + '\n')
    temp.replace(path)


def worker_init(_):
    import torch
    torch.set_num_threads(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--accumulate', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--chunk-size', type=int, default=50)
    parser.add_argument('--action-steps', type=int, default=20)
    parser.add_argument('--save-every', type=int, default=250)
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.accumulate, args.chunk_size, args.action_steps, args.save_every) < 1:
        parser.error('Training sizes and intervals must be positive')
    if args.workers < 0 or args.action_steps > args.chunk_size:
        parser.error('Invalid worker count or action steps')
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from torch.utils.data import DataLoader
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.processor_act import make_act_pre_post_processors
    from lerobot.utils.feature_utils import dataset_to_policy_features
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one free GPU through the ACT launcher')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = open(args.output / '.training.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    validate_private_source()
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    report = validate_dataset(args.dataset)
    print('DATASET VERIFIED:', json.dumps(report), flush=True)
    cfg = ACTConfig(device='cuda', push_to_hub=False, chunk_size=args.chunk_size,
                    n_action_steps=args.action_steps)
    dataset = make_dataset(TrainPipelineConfig(
        dataset=DatasetConfig(repo_id='local/ur5_cube_110ep_v3', root=str(args.dataset),
                              video_backend='pyav', use_imagenet_stats=True), policy=cfg))
    features = dataset_to_policy_features(dataset.meta.features)
    inputs = ['observation.state', 'observation.images.base', 'observation.images.wrist']
    cfg.input_features = {k: features[k] for k in inputs}
    cfg.output_features = {'action': features['action']}
    cfg.validate_features()
    from importlib.metadata import version
    signature = {'model': 'ACT', 'steps': args.steps, 'microbatch': args.batch_size,
                 'accumulate': args.accumulate, 'effective_batch': args.batch_size * args.accumulate,
                 'chunk_size': args.chunk_size, 'n_action_steps': args.action_steps,
                 'workers': args.workers, 'seed': 42, 'precision': 'float32',
                 'learning_rate': cfg.optimizer_lr, 'backbone_learning_rate': cfg.optimizer_lr_backbone,
                 'weight_decay': cfg.optimizer_weight_decay, 'kl_weight': cfg.kl_weight,
                 'input_features': inputs, 'image_shape': [3, 480, 640],
                 'normalization': 'LeRobot ACT mean/std; ImageNet RGB statistics',
                 'dataset': report, 'probe': args.probe,
                 'split': 'all 110 episodes; no held-out validation',
                 'language_conditioned': False,
                 'source_sha256': {n: sha256(Path(__file__).parent / n)
                                    for n in ('train.py', 'data.py', 'reviewed_source.json',
                                              'lerobot_source_snapshot.json')},
                 'versions': {n: version(n) for n in ('torch', 'torchvision', 'lerobot', 'wandb', 'av')}}
    config_file = args.output / 'run_config.json'
    resume = latest_checkpoint(args.output)
    if config_file.exists():
        stored = json.loads(config_file.read_text())
        if stored['signature'] != signature:
            raise ValueError('ACT resume settings/source differ: use original settings or a new run')
        if resume is None and (args.output / 'metrics.jsonl').exists():
            raise ValueError('Metrics exist without a complete checkpoint; choose a new output')
        wandb_id = stored['wandb_id']
    else:
        wandb_id = hashlib.sha256((str(args.output) + str(time.time_ns())).encode()).hexdigest()[:8]
        write_json(config_file, {'signature': signature, 'wandb_id': wandb_id,
                                'created_utc': datetime.now(timezone.utc).isoformat(),
                                'gpu': torch.cuda.get_device_name(0)})
        source = args.output / 'source'; source.mkdir(exist_ok=True)
        for name in ('train.py', 'data.py', 'reviewed_source.json', 'lerobot_source_snapshot.json'):
            shutil.copy2(Path(__file__).parent / name, source / name)
        (args.output / 'pip_freeze.txt').write_text(subprocess.check_output(
            [os.sys.executable, '-m', 'pip', 'freeze'], text=True))
    write_json(args.output / 'status.json', {'phase': 'loading_model', 'step': 0})
    policy = ACTPolicy(cfg).to('cuda')
    pre, post = make_act_pre_post_processors(cfg, dataset_stats=dataset.meta.stats)
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=cfg.optimizer_lr,
                                   weight_decay=cfg.optimizer_weight_decay)
    step = 0
    if resume:
        policy.load_state_dict(load_file(str(resume / 'pretrained_model/model.safetensors')), strict=True)
        state = torch.load(resume / 'training_state.pt', map_location='cpu', weights_only=False)
        if state['signature'] != signature:
            raise ValueError('ACT checkpoint signature mismatch')
        optimizer.load_state_dict(state['optimizer']); step = state['step']
        if state['consumed_samples'] != step * args.batch_size * args.accumulate:
            raise ValueError('ACT checkpoint sample position mismatch')
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        del state
        print(f'RESUMED ACT model/optimizer/RNG/sample position at step {step}', flush=True)
    loader_options = dict(num_workers=args.workers, pin_memory=True,
                          worker_init_fn=worker_init, generator=torch.Generator().manual_seed(42))
    if args.workers:
        loader_options.update(multiprocessing_context='spawn', persistent_workers=True, prefetch_factor=1)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        sampler=ResumableSampler(len(dataset), step * args.batch_size * args.accumulate),
                        **loader_options)
    iterator = iter(loader)
    run = None
    if not args.probe:
        import wandb
        run = wandb.init(entity='mdshafakat-masud-ontario-tech-university',
                         project='ur5-experiment1-act', id=wandb_id, resume='allow',
                         name=args.output.name, config=signature, dir=str(args.output), mode='online')
        if run is None or run.offline:
            raise RuntimeError('ACT requires online W&B logging')
        wandb.define_metric('train/step'); wandb.define_metric('train/*', step_metric='train/step')
        write_json(args.output / 'wandb_run.json', {'id': run.id, 'url': run.url})
        print('LIVE W&B:', run.url, flush=True)
    stopping = False
    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True
        print('ACT pause requested; finish current optimizer update and save.', flush=True)
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)
    def save_checkpoint():
        path = args.output / 'checkpoints' / f'{step:06d}'
        path.parent.mkdir(exist_ok=True)
        if path.exists():
            verify_checkpoint(path); return path
        temp = path.with_name(path.name + f'.incomplete-{time.time_ns()}'); temp.mkdir()
        exported = temp / 'pretrained_model'
        policy.save_pretrained(exported); pre.save_pretrained(exported); post.save_pretrained(exported)
        write_json(exported / 'dataset_statistics.json', dataset.meta.stats)
        write_json(exported / 'action_contract.json', report)
        shutil.copy2(config_file, temp / 'run_config.json')
        torch.save({'step': step, 'consumed_samples': step * args.batch_size * args.accumulate,
                    'signature': signature, 'optimizer': optimizer.state_dict(),
                    'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                    'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()},
                   temp / 'training_state.pt')
        hashes = {str(p.relative_to(temp)): sha256(p) for p in sorted(temp.rglob('*')) if p.is_file()}
        write_json(temp / 'COMPLETE.json', {'step': step, 'sha256': hashes})
        temp.rename(path); verify_checkpoint(path)
        write_json(args.output / 'latest_checkpoint.json', {'step': step, 'path': str(path)})
        complete = sorted(p for p in path.parent.iterdir()
                          if p.name.isdigit() and (p / 'COMPLETE.json').exists())
        for older in complete[:-2]:shutil.rmtree(older)
        print('CHECKPOINT SAVED AND VERIFIED:', path, flush=True)
        return path
    policy.train(); torch.cuda.reset_peak_memory_stats()
    started, start_step = time.monotonic(), step
    goal = min(2, args.steps) if args.probe else args.steps
    try:
        while step < goal:
            optimizer.zero_grad(set_to_none=True)
            losses = {'loss': 0., 'l1_loss': 0., 'kld_loss': 0.}
            for _ in range(args.accumulate):
                raw = next(iterator)
                batch = pre(raw)
                loss, components = policy.forward(batch)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite ACT loss')
                (loss / args.accumulate).backward()
                losses['loss'] += loss.detach().item() / args.accumulate
                for key in ('l1_loss', 'kld_loss'):losses[key] += components[key] / args.accumulate
            grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10., error_if_nonfinite=True)
            optimizer.step(); step += 1
            elapsed = time.monotonic() - started
            metrics = {f'train/{k}': v for k, v in losses.items()}
            metrics.update({'train/step': step, 'train/learning_rate': optimizer.param_groups[0]['lr'],
                            'train/backbone_learning_rate': optimizer.param_groups[1]['lr'],
                            'train/gradient_norm_before_clip': float(grad),
                            'train/seconds_per_optimizer_step': elapsed / (step-start_step),
                            'train/estimated_remaining_hours': (args.steps-step)*elapsed/(step-start_step)/3600,
                            'train/consumed_samples': step*args.batch_size*args.accumulate,
                            'train/gpu_peak_allocated_gib': torch.cuda.max_memory_allocated()/1024**3})
            if step == 1 or step % 250 == 0 or args.probe:
                with torch.inference_mode():
                    policy.eval()
                    prediction = post(policy.predict_action_chunk(batch))
                    if prediction.shape != (args.batch_size, args.chunk_size, 7) or not torch.isfinite(prediction).all():
                        raise RuntimeError('Invalid free-running ACT action chunk')
                    truth = raw['action'].cpu()
                    valid = (~raw['action_is_pad']).unsqueeze(-1).cpu()
                    errors = (prediction.cpu()-truth).abs()*valid
                    metrics['train/free_running_joint_mae_rad'] = errors[..., :6].sum().item()/(valid.sum().item()*6)
                    metrics['train/free_running_gripper_mae'] = errors[..., 6].sum().item()/valid.sum().item()
                    policy.train()
            if step == 1 or step % 10 == 0 or step == goal or args.probe:
                print(json.dumps(metrics), flush=True)
                with (args.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
                if run:run.log(metrics)
            if step % args.save_every == 0 or step == goal or stopping:save_checkpoint()
            write_json(args.output/'status.json', {'phase': 'running', 'step': step,
                       'updated_utc': datetime.now(timezone.utc).isoformat(),
                       'wandb_url': run.url if run else None})
            if stopping:break
        phase = 'paused' if stopping else ('probe_passed' if args.probe else 'finished')
        write_json(args.output/'status.json', {'phase': phase, 'step': step,
                   'resume_checkpoint_loaded': str(resume) if resume else None,
                   'wandb_url': run.url if run else None})
        print(f'{phase.upper()}: ACT step {step}/{goal}', flush=True)
    except Exception as e:
        write_json(args.output/'status.json', {'phase': 'failed', 'last_completed_step': step, 'error': str(e)})
        if run:run.finish(exit_code=1)
        raise
    else:
        if run:
            run.summary['final_step'] = step; run.summary['training_complete'] = step == args.steps
            run.finish()


if __name__ == '__main__':main()
