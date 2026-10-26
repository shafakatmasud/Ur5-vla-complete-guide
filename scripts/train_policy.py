#!/usr/bin/env python3
"""Build reproducible UR5 training commands against the bundled LeRobot API.

--dry-run uses only the standard library and never starts a training process.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


def validate_dataset(root):
    info = json.loads((root / 'meta/info.json').read_text())
    features = info['features']
    expected = {'observation.state': [7], 'action': [7],
                'observation.images.base': [480, 640, 3],
                'observation.images.wrist': [480, 640, 3]}
    for key, shape in expected.items():
        if features.get(key, {}).get('shape') != shape:
            raise ValueError(f'{key} must have shape {shape}')
    if info.get('total_episodes', 0) < 1 or info.get('fps') != 20:
        raise ValueError('Expected a nonempty 20 Hz dataset')
    conversion = json.loads((root / 'meta/ur5_vla_conversion.json').read_text())
    if conversion.get('action_mode') != 'absolute':
        raise ValueError('This recipe requires the UR5 converter absolute action mode')
    return info


def build_command(args):
    validate_dataset(args.dataset)
    if args.output.exists():
        raise ValueError('Use a new output directory; use the documented native resume command for recovery')
    command = [sys.executable, '-m', 'lerobot.scripts.lerobot_train',
               f'--dataset.root={args.dataset.resolve()}', f'--dataset.repo_id={args.repo_id}',
               '--dataset.video_backend=pyav', '--dataset.use_imagenet_stats=false',
               f'--policy.type={args.model}', '--policy.device=cuda', '--policy.push_to_hub=false',
               f'--policy.repo_id=local/{args.output.name}',
               f'--output_dir={args.output.resolve()}', f'--job_name={args.output.name}',
               f'--batch_size={args.batch_size}', f'--steps={args.steps}',
               f'--num_workers={args.num_workers}', '--persistent_workers=false',
               f'--save_freq={args.save_freq}', '--log_freq=20', '--eval_freq=0',
               '--eval.n_episodes=0', '--eval.batch_size=1',
               f'--wandb.enable={str(not args.no_wandb).lower()}',
               f'--wandb.project={args.wandb_project}']
    if args.wandb_entity:
        command.append(f'--wandb.entity={args.wandb_entity}')
    if args.model in ('smolvla', 'pi0'):
        base = args.pretrained_path or {'smolvla': 'lerobot/smolvla_base', 'pi0': 'lerobot/pi0_base'}[args.model]
        command += [f'--policy.pretrained_path={base}', '--policy.empty_cameras=0',
                    '--policy.train_expert_only=true', '--policy.freeze_vision_encoder=true']
        if args.model == 'smolvla':
            command += ['--policy.adapt_to_pi_aloha=false', '--policy.use_delta_joint_actions_aloha=false']
        else:
            command += ['--policy.use_relative_actions=false', '--policy.gradient_checkpointing=true',
                        '--policy.compile_model=false', f'--policy.dtype={args.dtype}']
    elif args.pretrained_path:
        raise ValueError('This launcher trains ACT/Diffusion on the UR5 dataset without policy pretrained_path')
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('smolvla', 'pi0', 'act', 'diffusion'), required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--save-freq', type=int, default=1000)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--wandb-project', default='ur5-vla')
    parser.add_argument('--wandb-entity')
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--pretrained-path')
    parser.add_argument('--dtype', choices=('float32', 'bfloat16'), default='float32')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.save_freq) < 1 or args.num_workers < 0:
        parser.error('Steps, batch size and save interval must be positive; workers must be nonnegative')
    try:
        command = build_command(args)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raise SystemExit(subprocess.call(command))


if __name__ == '__main__':
    main()
