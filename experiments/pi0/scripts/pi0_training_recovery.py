"""Checkpoint selection and deterministic sampler-position helpers (no ROS)."""
import json
from itertools import islice
from pathlib import Path


def latest_checkpoint(run):
    checkpoints = Path(run)/'checkpoints'
    valid=[]
    for folder in checkpoints.glob('*'):
        if not folder.name.isdigit() or not folder.is_dir():
            continue
        files=[folder/'training_state.pt',folder/'pretrained_model/model.safetensors',
               folder/'pretrained_model/config.json']
        try:
            if not all(p.is_file() and p.stat().st_size>0 for p in files):
                continue
            if json.loads(files[-1].read_text())['type']!='pi0':
                continue
        except (ValueError,KeyError,OSError):
            continue
        valid.append(folder)
    return max(valid,key=lambda p:int(p.name)) if valid else None


def sampler_position(step, accumulation, samples_per_rank):
    if min(step,accumulation,samples_per_rank)<0 or not accumulation or not samples_per_rank:
        raise ValueError('Invalid sampler position')
    return divmod(step*accumulation,samples_per_rank)


class OffsetSampler:
    def __init__(self,base,offset=0):
        self.base,self.offset=base,offset

    def __iter__(self):
        return islice(iter(self.base),self.offset,None)

    def __len__(self):
        return len(self.base)-self.offset


def restore_optimizer(model,optimizer,state,scaler):
    """All ranks call; rank zero supplies the trusted complete checkpoint state."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    sharded=FSDP.scatter_full_optim_state_dict(state['optimizer'] if state else None,
                                            model,optim=optimizer)
    optimizer.load_state_dict(sharded)
    if state and state.get('scaler'):
        scaler.load_state_dict(state['scaler'])
