#!/usr/bin/python3
"""Install the two-hour per-user timer for the designated Pi0 training run."""
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[1]
units=Path.home()/'.config/systemd/user'
units.mkdir(parents=True,exist_ok=True)
(units/'ur5-pi0-training-watchdog.service').write_text(f'''[Unit]
Description=Check and recover the experiment-1 Pi0 training job
ConditionPathExists={root}/data/models/checkpoints/experiment1_pi0_local/active_job.json

[Service]
Type=oneshot
WorkingDirectory={root}
ExecStart=/usr/bin/python3 "{root}/scripts/pi0_training_watchdog.py"
TimeoutStartSec=90
''')
(units/'ur5-pi0-training-watchdog.timer').write_text('''[Unit]
Description=Check experiment-1 Pi0 training every two hours (Toronto)

[Timer]
OnCalendar=*-*-* 00/2:00:00 America/Toronto
Persistent=true
AccuracySec=1min
Unit=ur5-pi0-training-watchdog.service

[Install]
WantedBy=timers.target
''')
subprocess.run(['systemctl','--user','daemon-reload'],check=True)
subprocess.run(['systemctl','--user','enable','--now','ur5-pi0-training-watchdog.timer'],check=True)
subprocess.run(['systemctl','--user','start','ur5-pi0-training-watchdog.service'],check=True)
subprocess.run(['systemctl','--user','list-timers','ur5-pi0-training-watchdog.timer','--no-pager'],check=True)
