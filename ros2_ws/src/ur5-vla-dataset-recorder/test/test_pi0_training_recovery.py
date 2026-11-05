import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'scripts'))
from pi0_training_recovery import latest_checkpoint,OffsetSampler,sampler_position
from pi0_training_watchdog import assess


def test_only_complete_numeric_checkpoint_is_selected(tmp_path):
    def create(name,complete=True):
        p=tmp_path/'checkpoints'/name;(p/'pretrained_model').mkdir(parents=True)
        (p/'pretrained_model/config.json').write_text(json.dumps({'type':'pi0'}))
        (p/'pretrained_model/model.safetensors').write_bytes(b'weights')
        if complete:(p/'training_state.pt').write_bytes(b'state')
        return p
    older=create('000001');create('.000500.incomplete');create('001000',False)
    assert latest_checkpoint(tmp_path)==older
    newer=create('000150');assert latest_checkpoint(tmp_path)==newer
    (newer/'pretrained_model/config.json').write_text('broken')
    assert latest_checkpoint(tmp_path)==older


def test_sampler_resumes_without_decoding_previous_samples():
    epoch,offset=sampler_position(7,4,10)
    assert (epoch,offset)==(2,8)
    sampler=OffsetSampler(list(range(10)),offset)
    assert list(sampler)==[8,9] and len(sampler)==2
    sampler.offset=0;assert list(sampler)==list(range(10))


@pytest.mark.parametrize('alive,age,step,checkpoint,expected',[
    (True,27,150,'checkpoint','healthy'),
    (True,1900,150,'checkpoint','stalled'),
    (False,27,150,'checkpoint','stopped'),
    (False,27,150,None,'blocked_no_checkpoint'),
    (False,0,20000,'checkpoint','complete'),
])
def test_health_decision(alive,age,step,checkpoint,expected):
    assert assess(alive,age,step,20000,checkpoint)==expected


def test_stopped_job_launches_resume_not_fresh_training(tmp_path,monkeypatch):
    import pi0_training_watchdog as watch
    from types import SimpleNamespace
    run=tmp_path/'owned_run';run.mkdir()
    (tmp_path/'active_job.json').write_text(json.dumps({'run_directory':str(run),'pid':123}))
    (run/'training_config.json').write_text(json.dumps({'steps':20000}))
    (run/'metrics.jsonl').write_text(json.dumps({'step':40})+'\n')
    checkpoint=run/'checkpoints/000020';checkpoint.mkdir(parents=True)
    calls=[]
    monkeypatch.setattr(watch,'BASE',tmp_path)
    monkeypatch.setattr(watch,'owned_processes',lambda _:[])
    monkeypatch.setattr(watch,'latest_checkpoint',lambda _:checkpoint)
    monkeypatch.setattr(watch.os,'statvfs',lambda _:SimpleNamespace(f_bavail=100*1024**3,f_frsize=1))
    monkeypatch.setattr(watch,'start_recovery',lambda run:(calls.append(run) or 456))
    watch.main()
    assert calls==[run]
    result=json.loads((tmp_path/'watchdog_status.json').read_text())
    assert result['event']=='recovery_launched' and result['restored_step']==20
    assert result['last_observed_step']==40


def test_recovery_has_its_own_service_cgroup(tmp_path,monkeypatch):
    import pi0_training_watchdog as watch
    commands=[]
    monkeypatch.setattr(watch,'BASE',tmp_path)
    monkeypatch.setattr(watch.subprocess,'run',lambda cmd,**kw:commands.append(cmd))
    monkeypatch.setattr(watch.subprocess,'check_output',lambda *a,**kw:'456\n')
    assert watch.start_recovery(tmp_path/'run')==456
    assert commands[0][:3]==['systemd-run','--user','--collect']
    assert commands[0][-2:]==['--resume',str(tmp_path/'run')]


@pytest.mark.skipif(os.environ.get('UR5_TEST_FSDP_RECOVERY')!='1',reason='Explicit spare-GPU validation only')
def test_real_fsdp_optimizer_and_rng_resume(tmp_path):
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
    from pi0_training_recovery import restore_optimizer
    torch.cuda.set_device(0)
    dist.init_process_group('nccl',init_method='file://'+str(tmp_path/'rendezvous'),rank=0,world_size=1)
    try:
        torch.manual_seed(42)
        original=torch.nn.Linear(4,2).cuda()
        model=FSDP(original,device_id=0,use_orig_params=True)
        opt=torch.optim.AdamW(model.parameters(),lr=.001)
        scaler=ShardedGradScaler(enabled=False)
        def update(m,o):
            o.zero_grad();x=torch.randn(3,4,device='cuda');loss=m(x).square().sum();loss.backward();o.step()
        update(model,opt)
        weights={k:v.clone() for k,v in original.state_dict().items()}
        state={'optimizer':FSDP.full_optim_state_dict(model,opt),'scaler':scaler.state_dict()}
        # Real checkpoints serialize before the next optimizer update; freeze
        # these tensors too rather than retaining aliases to live Adam moments.
        import io
        stream=io.BytesIO();torch.save(state,stream);stream.seek(0)
        state=torch.load(stream,weights_only=True)
        rng=torch.cuda.get_rng_state()
        update(model,opt)
        expected={k:v.clone() for k,v in original.state_dict().items()}
        restored_base=torch.nn.Linear(4,2).cuda();restored_base.load_state_dict(weights)
        restored=FSDP(restored_base,device_id=0,use_orig_params=True)
        restored_opt=torch.optim.AdamW(restored.parameters(),lr=.001)
        restore_optimizer(restored,restored_opt,state,scaler)
        torch.cuda.set_rng_state(rng)
        update(restored,restored_opt)
        for k,value in restored_base.state_dict().items():torch.testing.assert_close(value,expected[k],rtol=0,atol=0)
    finally:
        dist.destroy_process_group()
