#!/usr/bin/python3
"""Send experiment markers to the read-only monitor; never command the robot."""
import argparse
import json
from pathlib import Path
import time
from cross_grid_results import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True)
    sub=p.add_subparsers(dest='action',required=True)
    start=sub.add_parser('start');start.add_argument('cross',type=int);start.add_argument('attempt',type=int)
    finish=sub.add_parser('finish');finish.add_argument('outcome',choices=('pass','partial','fail','aborted','unknown'));finish.add_argument('--notes',default='')
    for stage in ('pickup','carry','placement'):finish.add_argument('--'+stage,choices=('yes','no','unknown'),default='unknown')
    sub.add_parser('status')
    args=p.parse_args();root=args.session.resolve()
    status=json.loads((root/'status.json').read_text())
    if args.action=='status':
        mapping_path=root/'pending_trial_renumber.json'
        if mapping_path.exists():
            mapping=json.loads(mapping_path.read_text())
            if (mapping.get('state')=='active_recording_renumbered' and
                    status.get('state')=='recording' and
                    status.get('attempt')==mapping['internal_monitor_attempt']):
                status['recorder_internal_attempt']=status['attempt']
                status['attempt']=mapping['canonical_attempt']
        print(json.dumps(status,indent=2));return
    if status['state']=='stopped':p.error('Monitor is stopped')
    command={'action':args.action}
    if args.action=='start':
        if status['state']!='waiting':p.error('An attempt is already recording')
        command.update(cross=args.cross,attempt=args.attempt)
    else:
        if status['state']!='recording':p.error('No attempt is recording')
        command.update(outcome=args.outcome,notes=args.notes)
        for stage in ('pickup','carry','placement'):
            command[stage]={'yes':True,'no':False,'unknown':None}[getattr(args,stage)]
    path=root/'commands'/f'{time.time_ns()}_{args.action}.json'
    write_json(path,command)
    print('Marker queued:',path.name)


if __name__=='__main__':main()
