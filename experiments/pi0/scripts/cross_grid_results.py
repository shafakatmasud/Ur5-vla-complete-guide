"""Durable, non-overwriting records for a numbered-location evaluation."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


class TrialStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.config = json.loads((self.root/'session.json').read_text())
        self.active = None

    def start(self, cross, attempt):
        if self.active is not None:
            raise ValueError('Finish the active attempt before starting another')
        if not 1 <= cross <= self.config['cross_count'] or not 1 <= attempt <= self.config['attempts_per_cross']:
            raise ValueError('Cross or attempt outside the planned protocol')
        folder = self.root/f'cross_{cross:02d}'/f'attempt_{attempt:02d}'
        folder.mkdir(parents=True, exist_ok=False)
        for name in ('camera_snapshots','video','depth','configuration'):
            (folder/name).mkdir()
        trial = dict(cross=cross, attempt=attempt, status='recording', started_utc=utc(),
                     start_epoch_s=time.time(), timeout_s=self.config.get('timeout_s'),
                     operator_outcome=None, pickup=None, carry=None, placement=None,
                     result_authority='Operator report; IO confirmation is not object-contact sensing')
        write_json(folder/'trial.json', trial)
        self.active = (folder, trial)
        self.update_results(trial)
        return folder, trial

    def update_results(self, trial):
        path = self.root/'results.csv'
        with path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            rows = list(reader)
        for row in rows:
            if int(row['cross']) == trial['cross'] and int(row['attempt']) == trial['attempt']:
                for key in fields:
                    if key in trial:
                        row[key] = '' if trial[key] is None else str(trial[key])
                break
        else:
            raise ValueError('Attempt not present in results table')
        temp = path.with_suffix('.csv.tmp')
        with temp.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        temp.replace(path)

    def aggregate(self):
        with (self.root/'results.csv').open(newline='') as stream:
            rows=list(csv.DictReader(stream))
        result={'updated_utc':utc(),'planned_attempts':len(rows),'crosses':[]}
        for cross in range(1,self.config['cross_count']+1):
            completed=[r for r in rows if int(r['cross'])==cross and r['status']=='finished']
            passed=sum(r.get('operator_outcome')=='pass' for r in completed)
            partial=sum(r.get('operator_outcome')=='partial' for r in completed)
            within=sum(r.get('reported_pass_within_limit')=='True' for r in completed)
            result['crosses'].append({'cross':cross,'completed':len(completed),'operator_passes':passed,'partial_successes':partial,
                'passes_within_limit':within,'pass_rate_within_limit':within/len(completed) if completed else None,
                'unknown_or_aborted':sum(r.get('operator_outcome') in ('unknown','aborted') for r in completed)})
        result['completed']=sum(r['completed'] for r in result['crosses'])
        result['operator_passes']=sum(r['operator_passes'] for r in result['crosses'])
        result['partial_successes']=sum(r['partial_successes'] for r in result['crosses'])
        result['passes_within_limit']=sum(r['passes_within_limit'] for r in result['crosses'])
        write_json(self.root/'aggregate.json',result)

    def finish(self, outcome, summary, notes='', pickup=None, carry=None, placement=None):
        if self.active is None:
            raise ValueError('No active attempt')
        if outcome not in ('pass','partial','fail','aborted','unknown'):
            raise ValueError('Outcome must be pass, partial, fail, aborted, or unknown')
        folder, trial = self.active
        trial.update(status='finished', operator_outcome=outcome, notes=notes,
                     pickup=pickup, carry=carry, placement=placement, ended_utc=utc(),
                     duration_s=summary['duration_s'])
        limit=trial['timeout_s']
        trial['timeout_reached']=limit is not None and trial['duration_s'] >= limit
        trial['reported_pass_within_limit'] = None if outcome == 'unknown' else outcome == 'pass' and not trial['timeout_reached']
        write_json(folder/'summary.json', {**summary, **trial})
        write_json(folder/'trial.json', trial)
        self.update_results(trial)
        self.aggregate()
        text=[f"Cross {trial['cross']:02d}, attempt {trial['attempt']:02d}",
              f"Operator outcome: {outcome}",f"Duration: {trial['duration_s']:.2f} seconds",
              f"{limit}-second cutoff reached: {trial['timeout_reached']}",
              f"Pickup / carry / placement: {pickup} / {carry} / {placement}",
              f"Operator notes: {notes}", '',
              'See summary.json for measured rates, tracking errors, gripper IO and runtime faults.',
              'See observations.jsonl, camera snapshots, video and depth for source evidence.',
              'Cube/base-frame position and physical contact are not established by gripper IO.']
        (folder/'analysis.txt').write_text('\n'.join(text)+'\n')
        self.active = None
        return trial
