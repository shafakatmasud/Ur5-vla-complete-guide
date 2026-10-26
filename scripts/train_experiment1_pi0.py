#!/usr/bin/env python3
"""Local float32 Pi0 expert finetuning, sharded across recording-host GPUs.

Launch with torchrun. No ROS imports or robot command publishers.
Leaf projection wrapping is intentional: Pi0's combined attention accesses the
decoder layers' children directly instead of calling decoder-layer forward.
"""
import argparse
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--accumulate', type=int, default=4)
    parser.add_argument('--keep-checkpoints', type=int, default=2)
    parser.add_argument('--mixed-precision', choices=['float32','fp16'], default='float32')
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--resume', type=Path, help='Complete checkpoint from this output run')
    args = parser.parse_args()
    if min(args.steps,args.save_every,args.accumulate,args.keep_checkpoints)<1:
        parser.error('steps, save interval, accumulation and checkpoint retention must be positive')
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig
    from torch.distributed.fsdp import MixedPrecision
    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
    from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
    from torch.utils.data import DataLoader, DistributedSampler
    from safetensors.torch import load_file, save_file
    from lerobot.policies.pi0.configuration_pi0 import PI0Config
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.datasets.factory import make_dataset
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.utils.feature_utils import dataset_to_policy_features
    from lerobot.configs.types import FeatureType
    from pi0_training_recovery import OffsetSampler, sampler_position, restore_optimizer

    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local)
    torch.set_num_threads(1)
    dist.init_process_group('nccl')
    torch.manual_seed(42+rank)
    args.output.mkdir(parents=True, exist_ok=True)
    info_path = args.dataset/'meta/info.json'
    info = json.loads(info_path.read_text())
    resume_state=None
    resume_meta=None
    if args.resume:
        args.resume=args.resume.resolve()
        if args.resume.parent != (args.output/'checkpoints').resolve():
            raise RuntimeError('Resume checkpoint must belong to this output run')
        if rank==0:
            resume_state=torch.load(args.resume/'training_state.pt',map_location='cpu',weights_only=True,mmap=True)
            resume_meta={k:v for k,v in resume_state.items() if k!='optimizer'}
        packet=[resume_meta]
        dist.broadcast_object_list(packet,src=0)
        resume_meta=packet[0]
        previous=resume_meta['config']
        expected={'world_size':world,'steps':args.steps,'gradient_accumulation':args.accumulate,
                  'mixed_precision':args.mixed_precision,
                  'dataset_info_sha256':hashlib.sha256(info_path.read_bytes()).hexdigest()}
        for key,value in expected.items():
            if previous[key]!=value:raise RuntimeError(f'Resume configuration mismatch: {key}')
    start_step=resume_meta['step'] if resume_meta else 0
    assert info['total_episodes'] == 110 and info['features']['action']['shape'] == [7]
    assert info['features']['observation.state']['shape'] == [7]
    cfg = PI0Config(device='cpu', dtype='float32', train_expert_only=True,
                    freeze_vision_encoder=True, gradient_checkpointing=True,
                    compile_model=False, use_relative_actions=False,
                    empty_cameras=1, push_to_hub=False)
    pipeline = TrainPipelineConfig(dataset=DatasetConfig(
        repo_id='local/ur5_cube_110ep_v3', root=str(args.dataset),
        video_backend='pyav', use_imagenet_stats=False), policy=cfg)
    dataset = make_dataset(pipeline)
    features = dataset_to_policy_features(dataset.meta.features)
    cfg.input_features = {k:v for k,v in features.items() if v.type != FeatureType.ACTION}
    cfg.output_features = {k:v for k,v in features.items() if v.type == FeatureType.ACTION}
    print(f'Rank {rank}: constructing Pi0 on CPU', flush=True)
    from transformers.initialization import no_init_weights
    # Every parameter is replaced by a strictly checked pretrained tensor below.
    with no_init_weights():
        policy = PI0Policy(cfg)
    original = load_file(str(args.resume/'pretrained_model/model.safetensors' if args.resume else args.weights), device='cpu')
    fixed = original if args.resume else policy._fix_pytorch_state_dict_keys(original, cfg)
    state = {k if k.startswith('model.') else f'model.{k}':v for k,v in fixed.items()}
    # Unlike the upstream permissive loader, failure must not train random weights.
    policy.load_state_dict(state, strict=True)
    del original, fixed, state
    print(f'Rank {rank}: all pretrained weights loaded strictly', flush=True)
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy.parameters())
    # Shared embeddings/LM heads remain in the root wrapper to preserve ties.
    owners = {}
    for module in policy.modules():
        for parameter in module.parameters(recurse=False):
            owners.setdefault(id(parameter), []).append(module)
    tied = {id(m) for modules in owners.values() if len(modules)>1 for m in modules}
    wrap = partial(lambda_auto_wrap_policy, lambda_fn=lambda m:
                   isinstance(m, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d))
                   and id(m) not in tied)
    model = FSDP(policy, auto_wrap_policy=wrap, device_id=local,
                 use_orig_params=True, limit_all_gathers=True,
                 sync_module_states=True,
                 mixed_precision=MixedPrecision(param_dtype=torch.float16,
                    reduce_dtype=torch.float32,buffer_dtype=torch.float32)
                    if args.mixed_precision=='fp16' else None)
    scaler=ShardedGradScaler(enabled=args.mixed_precision=='fp16',init_scale=128.)
    cfg.device = f'cuda:{local}'
    pre, post = make_pre_post_processors(cfg, dataset_stats=dataset.meta.stats)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=2.5e-5, betas=(.9,.95), eps=1e-8, weight_decay=.01)
    if resume_meta:
        restore_optimizer(model,optimizer,resume_state,scaler)
        # GradScaler state is small and identical across the synchronized ranks.
        scaler.load_state_dict(resume_meta['scaler'])
        del resume_state
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=42)
    epoch,offset=sampler_position(start_step,args.accumulate,sampler.num_samples)
    sampler.set_epoch(epoch)
    offset_sampler=OffsetSampler(sampler,offset)
    loader = DataLoader(dataset, batch_size=1, sampler=offset_sampler, num_workers=0,
                        generator=torch.Generator().manual_seed(42+rank))
    iterator = iter(loader)
    if resume_meta and resume_meta.get('rng_per_rank'):
        rng=resume_meta['rng_per_rank'][rank]
        torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state(rng['cuda'],local)
    elif resume_meta:
        print('Legacy checkpoint: model/optimizer/scaler/data position restored; original RNG state was not saved',flush=True)
    configuration = dict(task='Pick up the orange cube and place it in the bowl',
        dataset=str(args.dataset), dataset_info_sha256=hashlib.sha256(info_path.read_bytes()).hexdigest(),
        episodes=info['total_episodes'], frames=info['total_frames'], model='lerobot/pi0_base',
        weights=str(args.weights), dtype='float32', expert_only=True,
        trainable_parameters=trainable, total_parameters=total, world_size=world,
        batch_per_gpu=1, gradient_accumulation=args.accumulate,
        effective_batch=world*args.accumulate, steps=args.steps,
        seed=42, learning_rate=2.5e-5, action_semantics='absolute six joints plus binary gripper',
        camera_features=list(dataset.meta.camera_keys), missing_camera_slots=1,
        probe=args.probe, started_utc=datetime.now(timezone.utc).isoformat())
    configuration['mixed_precision']=args.mixed_precision
    from importlib.metadata import version
    configuration['versions']={name:version(name) for name in ['torch','lerobot','transformers','wandb']}
    configuration['training_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    configuration['resumed_from']=str(args.resume) if args.resume else None
    if rank==0:
        config_file=args.output/('resume_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json' if args.resume else 'training_config.json')
        config_file.write_text(json.dumps(configuration,indent=2)+'\n')
        print(json.dumps(configuration),flush=True)
    run = None
    if rank==0 and not args.probe:
        import wandb
        existing=json.loads((args.output/'wandb_run.json').read_text()) if args.resume else None
        resume_kwargs={}
        if existing:
            from urllib.parse import urlsplit
            entity,project=urlsplit(existing['url']).path.strip('/').split('/')[:2]
            resume_kwargs=dict(id=existing['id'],resume='must',entity=entity)
        run = wandb.init(project='ur5-experiment1-pi0', name=args.output.name,
                         dir=str(args.output), config=configuration, mode='online',allow_val_change=True,
                         **resume_kwargs,
                         tags=['experiment1','pi0','110episodes','expert-only','fsdp','absolute-actions'])
        (args.output/'wandb_run.json').write_text(json.dumps({'url':run.url,'id':run.id},indent=2)+'\n')
        print('WANDB_URL='+run.url,flush=True)
        if args.resume:
            run.log({'recovery/checkpoint_step':start_step,'recovery/utc':datetime.now(timezone.utc).isoformat()},commit=True)
    def checkpoint(step):
        enough=torch.tensor(int(shutil.disk_usage(args.output).free > 50*1024**3),device=local)
        dist.all_reduce(enough,op=dist.ReduceOp.MIN)
        if not enough.item():raise RuntimeError('Checkpoint blocked: retain at least 50 GiB of disk reserve')
        # All ranks participate in gathers; only rank 0 writes portable files.
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                                 FullStateDictConfig(offload_to_cpu=True,rank0_only=True)):
            weights = model.state_dict()
        optim_state = FSDP.full_optim_state_dict(model, optimizer, rank0_only=True)
        rng={'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(local)}
        rngs=[None]*world if rank==0 else None
        dist.gather_object(rng,rngs,dst=0)
        if rank==0:
            folder=args.output/'checkpoints'/f'{step:06d}'
            temp=folder.with_name('.'+folder.name+'.incomplete');temp.mkdir(parents=True)
            exported=temp/'pretrained_model';exported.mkdir()
            save_file({k:v.contiguous().clone() for k,v in weights.items()},str(exported/'model.safetensors'))
            cfg.device='cuda';cfg.save_pretrained(exported)
            pre.save_pretrained(exported);post.save_pretrained(exported)
            torch.save({'step':step,'optimizer':optim_state,'scaler':scaler.state_dict(),
                        'config':configuration,'schema_version':2,'rng_per_rank':rngs},temp/'training_state.pt')
            temp.rename(folder)
            print(f'CHECKPOINT_SAVED={folder}',flush=True)
            complete=sorted(p for p in folder.parent.iterdir() if p.name.isdigit() and p.is_dir())
            for obsolete in complete[:-args.keep_checkpoints]:
                shutil.rmtree(obsolete)
        del weights, optim_state
        dist.barrier()
    model.train()
    started=time.monotonic()
    for step in range(start_step+1,args.steps+1):
        tick=time.monotonic();optimizer.zero_grad(set_to_none=True)
        loss_sum=torch.zeros((),device=local)
        per_dim=torch.zeros(7,device=local)
        # Synchronize each microbatch: no_sync accumulates full gradients and
        # defeats sharding's memory savings on these 11 GB cards.
        for _ in range(args.accumulate):
            try: batch=next(iterator)
            except StopIteration:
                epoch+=1;sampler.set_epoch(epoch);offset_sampler.offset=0;iterator=iter(loader);batch=next(iterator)
            for k in dataset.meta.camera_keys:
                batch[k]=batch[k].float().div_(255.)
            batch=pre(batch)
            with torch.autocast('cuda',dtype=torch.float16,enabled=args.mixed_precision=='fp16'):
                loss,details=model(batch)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite Pi0 loss')
            scaler.scale(loss/args.accumulate).backward()
            loss_sum+=loss.detach()/args.accumulate
            per_dim+=torch.tensor(details['loss_per_dim'],device=local)/args.accumulate
        scaler.unscale_(optimizer)
        grad=model.clip_grad_norm_(1.)
        if not torch.isfinite(grad):raise RuntimeError('Non-finite Pi0 gradient norm')
        warmup=min(1000,max(1,args.steps//10))
        fraction=min(1.,step/warmup) if step<=warmup else .1+.9*.5*(1+math.cos(math.pi*(step-warmup)/max(1,args.steps-warmup)))
        for group in optimizer.param_groups:group['lr']=2.5e-5*fraction
        scaler.step(optimizer)
        scaler.update()
        dist.all_reduce(loss_sum);loss_sum/=world
        dist.all_reduce(per_dim);per_dim/=world
        memory=torch.tensor(torch.cuda.max_memory_allocated()/2**20,device=local)
        dist.all_reduce(memory,op=dist.ReduceOp.MAX)
        if rank==0:
            metrics={'train/loss':loss_sum.item(),'train/grad_norm':float(grad),
                     'train/learning_rate':optimizer.param_groups[0]['lr'],
                     'train/step_seconds':time.monotonic()-tick,
                     'train/elapsed_seconds':time.monotonic()-started,
                     'system/max_gpu_memory_mib':memory.item(), 'step':step}
            metrics.update({f'train/loss_dim_{i}':value for i,value in enumerate(per_dim.tolist())})
            metrics['train/samples_seen']=step*world*args.accumulate
            with (args.output/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(metrics)+'\n')
            # W&B's history index continues across rollbacks; the explicit step
            # metric records the actual restored optimizer step without dropping logs.
            if run:run.log(metrics,commit=True)
            print(json.dumps(metrics),flush=True)
        if not args.probe and (step==1 or step%args.save_every==0 or step==args.steps):checkpoint(step)
    if run:run.finish()
    dist.destroy_process_group()


if __name__=='__main__':
    main()
