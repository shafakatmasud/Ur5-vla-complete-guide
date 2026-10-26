#!/usr/bin/python3
"""Check only the designated Pi0 job; recover it from a complete checkpoint."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from pi0_training_recovery import latest_checkpoint

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/models/checkpoints/experiment1_pi0_local'
STALE_SECONDS=1800


def owned_processes(run):
    result=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            args=[a.decode() for a in (proc/'cmdline').read_bytes().split(b'\0') if a]
            # Exact arguments, not substring matches or unverified PID-file contents.
            if str(ROOT/'scripts/train_experiment1_pi0.py') not in args:continue
            index=args.index('--output')
            if Path(args[index+1]).resolve()==run:result.append(int(proc.name))
        except (OSError,ValueError,IndexError,UnicodeError):continue
    return result


def last_metric(path):
    if not path.exists():return None
    with path.open('rb') as stream:
        stream.seek(max(0,path.stat().st_size-65536))
        lines=stream.read().splitlines()
    for line in reversed(lines):
        try:return json.loads(line)
        except ValueError:continue
    return None


def assess(alive,age,step,total,checkpoint):
    if step>=total:return 'complete'
    if alive and age<STALE_SECONDS:return 'healthy'
    if not checkpoint:return 'blocked_no_checkpoint'
    return 'stalled' if alive else 'stopped'


def record(event,**details):
    entry={'time':datetime.now(ZoneInfo('America/Toronto')).isoformat(),'event':event,**details}
    with (BASE/'watchdog_checks.jsonl').open('a') as stream:stream.write(json.dumps(entry)+'\n')
    temp=BASE/'watchdog_status.json.tmp';temp.write_text(json.dumps(entry,indent=2)+'\n');temp.replace(BASE/'watchdog_status.json')
    print(json.dumps(entry),flush=True)


def start_recovery(run):
    # A separate user unit owns the long-running trainer. Children left in the
    # watchdog's oneshot cgroup would otherwise be killed when the check exits.
    unit='ur5-pi0-auto-recovery'
    command=['systemd-run','--user','--collect','--unit='+unit,
             '--property=Type=exec','--property=WorkingDirectory='+str(ROOT),
             '/bin/bash',str(ROOT/'scripts/train_experiment1_pi0.sh'),'--resume',str(run)]
    with (BASE/'watchdog_launcher.log').open('a') as log:
        subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=20)
    pid=subprocess.check_output(['systemctl','--user','show',unit+'.service','--property=MainPID','--value'],text=True,timeout=10)
    return int(pid.strip())


def main():
    lock=(BASE/'.watchdog.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    job=json.loads((BASE/'active_job.json').read_text())
    run=Path(job['run_directory']).resolve()
    if run.parent!=BASE.resolve():raise RuntimeError('Job is outside the designated training root')
    config=json.loads((run/'training_config.json').read_text())
    metrics=run/'metrics.jsonl';last=last_metric(metrics) or {'step':0}
    age=time.time()-metrics.stat().st_mtime if metrics.exists() else float('inf')
    processes=owned_processes(run)
    checkpoint=latest_checkpoint(run)
    state=assess(bool(processes),age,last['step'],config['steps'],checkpoint)
    if state=='complete':
        # A final logged update precedes its checkpoint. Allow a live process
        # to finish saving; a dead job without that final checkpoint must recover.
        if processes and age<STALE_SECONDS:
            record('finalizing',step=last['step'],pids=processes);return
        if checkpoint and int(checkpoint.name)>=config['steps']:
            record('complete',step=last['step']);return
        state='stopped' if checkpoint else 'blocked_no_checkpoint'
    if state=='healthy':record('healthy',step=last['step'],metrics_age_s=age,pids=processes);return
    if state.startswith('blocked'):
        record(state,step=last['step']);return
    if os.statvfs(BASE).f_bavail*os.statvfs(BASE).f_frsize<60*1024**3:
        record('blocked_low_disk',step=last['step']);return
    recovery_path=BASE/'watchdog_restarts.json'
    restarts=json.loads(recovery_path.read_text()) if recovery_path.exists() else []
    recent=[x for x in restarts if time.time()-x['epoch_s']<86400]
    if len(recent)>=3:
        record('blocked_restart_limit',step=last['step'],reason='Three recovery attempts within 24 hours; inspect logs');return
    if processes:
        record('stalled_stopping_owned_training_only',step=last['step'],pids=processes,metrics_age_s=age)
        # torchrun forwards TERM to its workers. Never signal unrelated GPU/ROS jobs.
        for pid in processes:
            if pid in owned_processes(run):
                try:os.kill(pid,signal.SIGTERM)
                except ProcessLookupError:pass
        deadline=time.monotonic()+30
        while owned_processes(run) and time.monotonic()<deadline:time.sleep(.5)
        for pid in set(processes)&set(owned_processes(run)):
            try:os.kill(pid,signal.SIGKILL)
            except ProcessLookupError:pass
        if owned_processes(run):record('blocked_existing_processes_remain');return
    pid=start_recovery(run)
    entry={'epoch_s':time.time(),'pid':pid,'checkpoint':str(checkpoint),
           'last_observed_step':last['step'],'restored_step':int(checkpoint.name)}
    restarts.append(entry)
    recovery_path.write_text(json.dumps(restarts,indent=2)+'\n')
    job.update(pid=pid,last_recovery=entry)
    temp=BASE/'active_job.json.tmp';temp.write_text(json.dumps(job,indent=2)+'\n');temp.replace(BASE/'active_job.json')
    record('recovery_launched',**entry)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        if BASE.is_dir():record('watchdog_error',error=str(error))
        raise
