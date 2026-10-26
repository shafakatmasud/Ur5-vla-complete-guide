#!/usr/bin/env python3
"""Initialize an empty, numbered evaluation session without ROS or robot control."""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--crosses', type=int, default=1)
    parser.add_argument('--attempts-per-cross', type=int, default=20)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--task', required=True)
    parser.add_argument('--execution-condition', choices=('policy-only', 'policy-with-offset', 'planned-transport'), required=True)
    args = parser.parse_args()
    if min(args.crosses, args.attempts_per_cross, args.timeout) <= 0:
        parser.error('Counts and timeout must be positive')
    args.session.mkdir(parents=True, exist_ok=False)
    (args.session / 'commands').mkdir()
    config = dict(cross_count=args.crosses, attempts_per_cross=args.attempts_per_cross,
                  timeout_s=args.timeout, task=args.task, execution_condition=args.execution_condition,
                  created_utc=datetime.now(timezone.utc).isoformat())
    (args.session / 'session.json').write_text(json.dumps(config, indent=2) + '\n')
    fields = ['cross', 'attempt', 'status', 'started_utc', 'ended_utc', 'duration_s',
              'operator_outcome', 'pickup', 'carry', 'placement', 'timeout_reached',
              'reported_pass_within_limit', 'notes']
    with (args.session / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for cross in range(1, args.crosses + 1):
            for attempt in range(1, args.attempts_per_cross + 1):
                writer.writerow(dict(cross=cross, attempt=attempt, status='planned'))
    print(args.session.resolve())


if __name__ == '__main__':
    main()
