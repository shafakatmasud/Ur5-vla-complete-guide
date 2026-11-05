#!/usr/bin/python3
"""Export a frozen set of completed demonstrations on an explicit adjusted clock."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import xlsxwriter
import yaml

FACTOR = 2.1
JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3']


def epoch(value):
    return datetime.fromisoformat(value).timestamp()


def clock(value, anchor):
    return anchor + (value - anchor) * FACTOR


def adjusted(value, anchor, key='', context=''):
    """Transform dimensional fields, keeping identifiers, settings and counts intact."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, dict):
        if 'sec' in value and 'nanosec' in value:
            ns = round(clock(value['sec'] + value['nanosec'] / 1e9, anchor) * 1e9)
            return {**value, 'sec': ns // 10**9, 'nanosec': ns % 10**9}
        return {k: adjusted(v, anchor, k, context + '/' + key) for k, v in value.items()}
    if isinstance(value, list):
        return [adjusted(v, anchor, key, context) for v in value]
    if isinstance(value, (int, float)):
        if key.endswith('_stamp_ns'):
            return round(clock(value / 1e9, anchor) * 1e9)
        if key in {'stamp_s', 'source_stamp_s', 'receive_epoch_s', 'start_epoch_s'}:
            return clock(value, anchor)
        if key.endswith('_s') or key in {'receive_age_s', 'max_color_source_age_s'} or any(
            p in context for p in ['/receive_age_s', '/max_color_source_age_s']
        ):
            return value * FACTOR
        if key in {'velocity', 'linear', 'angular'} or '/velocity' in context or '/twist/linear' in context or '/twist/angular' in context:
            return value / FACTOR
        if 'acceleration' in key or '/acceleration' in context:
            return value / FACTOR**2
        if key.endswith('_hz') or '/rates_hz' in context:
            return value / FACTOR
        return value
    if isinstance(value, str) and key in {'end_time_utc', 'ended_utc', 'record_time_utc'}:
        return datetime.fromtimestamp(clock(epoch(value), anchor), timezone.utc).isoformat()
    return value


def flat(value, prefix=''):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from flat(item, f'{prefix}.{key}' if prefix else key)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from flat(item, f'{prefix}[{i}]')
    else:
        yield prefix, value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    # Freeze the population before export; new episodes may continue independently.
    episodes = []
    for folder in sorted((root / 'episodes').glob('episode_*')):
        metadata = yaml.safe_load((folder / 'metadata.yaml').read_text())
        if metadata.get('outcome') == 'success' and metadata.get('end_time_utc'):
            episodes.append((folder, metadata))
    assert episodes, 'No completed episodes'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix('.building.xlsx')
    book = xlsxwriter.Workbook(str(temp), {'constant_memory': True, 'strings_to_urls': False,
                                         'strings_to_formulas': False, 'tmpdir': '/tmp'})
    book.use_zip64()
    book.set_properties({'title': 'Experiment 2 telemetry — adjusted time ×2.1',
                         'comments': 'Derived time-scaled export; not original measured timing.'})
    head = book.add_format({'bold': True, 'bg_color': '#17365D', 'font_color': 'white'})
    sheets = {}
    counts = Counter()

    def sheet(name, headers):
        ws = book.add_worksheet(name)
        ws.freeze_panes(1, 0)
        ws.write_row(0, 0, headers, head)
        ws.set_column(0, 0, 29)
        ws.set_column(1, len(headers)-1, 20)
        sheets[name] = [ws, 1, len(headers)]
        return name

    def row(name, values):
        ws, index, width = sheets[name]
        assert index < 1048576, f'Excel row capacity exceeded: {name}'
        for col, value in enumerate(values):
            if isinstance(value, (dict, list)):
                value = json.dumps(value, separators=(',', ':'), ensure_ascii=False)
            if isinstance(value, float) and not math.isfinite(value):
                value = str(value)
            assert not isinstance(value, str) or len(value) <= 32767, (name, index, col)
            if value is not None:
                result = ws.write(index, col, value)
                assert result == 0, (name, index, col, result)
        sheets[name][1] += 1
        counts[name] += 1

    sheet('Guide', ['Item', 'Explanation'])
    for item in [
        ('DATA STATUS', 'DERIVED TIME-SCALED TELEMETRY ×2.1. Not measured elapsed time; not a new physical experiment.'),
        ('Scope', f'{len(episodes)} completed Sort the fruits demonstrations frozen at export start; operator-labelled success.'),
        ('Adjusted clock', 'Elapsed times, ages and intervals ×2.1; frequencies and velocities ÷2.1; accelerations ÷4.41.'),
        ('Timestamp origin', 'Each dataset/monitor clock retains its own recorded start. Later UTC/epoch stamps = start + 2.1 × (stamp − start). These are synthetic timestamps.'),
        ('Unchanged values', 'Positions, orientations, distances, efforts, counts, joystick axes/buttons, IO values, speed-slider percentages, outcomes and identifiers.'),
        ('Numeric calculations', 'Episode durations/rates and frame finite-difference velocities/accelerations use the adjusted clock. First derivatives have blank initial values.'),
        ('Frame units', 'Joint position rad, finite-difference velocity rad/s, acceleration rad/s²; gripper 0 closed, 1 open. Derivatives reflect sampled data and noise.'),
        ('Dataset sheets', 'F01…: every dataset frame with numeric joint states/actions, adjusted timing/ages, source timestamps and camera/depth paths.'),
        ('Monitor sheets', 'T01…: EVERY saved monitor observation, including joints/velocity/effort, TCP, joystick, IO, gripper, Servo, status, images/depth references and ROS logs.'),
        ('Monitor payload', 'Adjusted structured JSON in each event row preserves heterogeneous topic fields. Textual log messages are verbatim original evidence; numerical time phrases inside text are NOT rewritten.'),
        ('Summary_fields', 'All scalar/list leaves of monitor summaries, including event transitions and logs; time/rate fields adjusted. Metadata contains dataset/monitor metadata.'),
        ('Media', 'Pixels and videos are not embedded in this telemetry workbook; frame/source paths identify original RGB/depth/video evidence. Names containing old timestamps remain unchanged identifiers.'),
        ('Coverage', 'First episode has dataset frames but no separate monitor recording. Monitor capture starts after recorder polling; gaps are not fabricated.'),
        ('Limits', 'Success is an operator label. IO does not prove contact or grasp force. TCP paths include sensor noise. No missing telemetry is invented.'),
        ('Original files', 'All source recordings are unchanged. Original absolute starts serve as clock anchors; file-system dates/identifiers are provenance, not adjusted durations.'),
    ]:
        row('Guide', item)
    sheets['Guide'][0].set_column(1, 1, 115)
    sheet('Episodes', ['episode', 'outcome', 'original_start_utc_anchor', 'adjusted_end_utc',
                       'adjusted_duration_s', 'saved_frames', 'dropped_samples', 'drop_fraction',
                       'adjusted_effective_saved_hz', 'adjusted_nominal_hz', 'monitor_available',
                       'adjusted_monitor_duration_s', 'TCP_path_m', 'adjusted_mean_TCP_speed_m_s',
                       'warnings_errors', 'dataset_sheet', 'monitor_sheet', 'dataset_path'])
    sheet('Metadata', ['episode', 'source', 'field', 'adjusted_value'])
    sheet('Summary_fields', ['episode', 'field', 'adjusted_value'])
    sheet('Sources', ['episode', 'file', 'bytes', 'sha256'])
    sheet('Topic_counts', ['episode', 'topic', 'saved_events', 'adjusted_monitor_duration_s', 'adjusted_events_per_s'])
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'time_scale': FACTOR,
                'derived_not_measured': True, 'episodes': [], 'missing_monitor': [], 'counts': {}}
    for number, (folder, meta) in enumerate(episodes, 1):
        name = folder.name
        monitor = root / 'telemetry' / name
        start = epoch(meta['start_time_utc'])
        duration = (epoch(meta['end_time_utc']) - start) * FACTOR
        summary = json.loads((monitor / 'summary.json').read_text()) if (monitor / 'summary.json').exists() else {}
        trial = json.loads((monitor / 'trial.json').read_text()) if (monitor / 'trial.json').exists() else {}
        anchor = trial.get('start_epoch_s', start)
        transformed_summary = adjusted(summary, anchor)
        transformed_meta = adjusted(meta, start)
        for source, obj, a in [('metadata.yaml', meta, start), ('trial.json', trial, anchor)]:
            for key, value in flat(adjusted(obj, a)):
                row('Metadata', [name, source, key, value])
        for key, value in flat(transformed_summary):
            row('Summary_fields', [name, key, value])
        files = [folder / 'metadata.yaml', folder / 'frames.jsonl']
        files += [monitor / x for x in ['trial.json', 'summary.json', 'observations.jsonl'] if (monitor / x).exists()]
        f_name = sheet(f'F{number:02}', ['frame_index', 'adjusted_episode_time_s', 'adjusted_record_utc'] +
                       ['state_' + j + '_rad' for j in JOINTS] + ['action_' + j + '_rad' for j in JOINTS] +
                       ['gripper_state', 'gripper_command', 'servo_status', 'valid'] +
                       ['derived_velocity_' + j + '_rad_s' for j in JOINTS] +
                       ['derived_acceleration_' + j + '_rad_s2' for j in JOINTS] +
                       ['max_action_state_gap_rad', 'adjusted_ages_json', 'adjusted_source_stamps_json',
                        'base_RGB_path', 'wrist_RGB_path', 'base_depth_path', 'wrist_depth_path', 'other_fields_json'])
        previous = None
        frame_count = 0
        for line in (folder / 'frames.jsonl').open():
            raw = json.loads(line)
            data = adjusted(raw, start)
            t = data['episode_time_s']; q = data['observation.state'][:6]; action = data['action'][:6]
            velocity = [None]*6; acceleration = [None]*6
            if previous and t > previous[0]:
                dt = t - previous[0]
                velocity = [(x-y)/dt for x, y in zip(q, previous[1])]
                if previous[2][0] is not None:
                    acceleration = [(x-y)/dt for x, y in zip(velocity, previous[2])]
            previous = t, q, velocity
            stamps = {k: v for k, v in data.items() if k.endswith('_stamp_ns')}
            keys = {'frame_index', 'episode_time_s', 'record_time_utc', 'observation.state', 'action',
                    'gripper.state', 'gripper.command', 'servo_status', 'valid', 'receive_age_s',
                    'observation.images.base', 'observation.images.wrist', 'observation.depth.base', 'observation.depth.wrist'} | set(stamps)
            row(f_name, [data['frame_index'], t, data['record_time_utc']] + q + action +
                [data.get('gripper.state'), data.get('gripper.command'), data.get('servo_status'), data.get('valid')] +
                velocity + acceleration + [max(abs(x-y) for x, y in zip(q, action)), data.get('receive_age_s'), stamps] +
                [str(folder / data[k]) if data.get(k) else None for k in ['observation.images.base', 'observation.images.wrist', 'observation.depth.base', 'observation.depth.wrist']] +
                [{k: v for k, v in data.items() if k not in keys}])
            frame_count += 1
        assert frame_count == meta['frames'], (name, frame_count, meta['frames'])
        t_name = None
        event_counts = Counter()
        if (monitor / 'observations.jsonl').exists():
            t_name = sheet(f'T{number:02}', ['adjusted_monitor_elapsed_s', 'adjusted_receive_epoch_s', 'topic', 'adjusted_value_json', 'adjusted_additional_fields_json'])
            part = 1
            for line in (monitor / 'observations.jsonl').open():
                event = adjusted(json.loads(line), anchor)
                if sheets[t_name][1] >= 1048576:
                    part += 1
                    t_name = sheet(f'T{number:02}_{part}', ['adjusted_monitor_elapsed_s', 'adjusted_receive_epoch_s', 'topic', 'adjusted_value_json', 'adjusted_additional_fields_json'])
                row(t_name, [event.get('t_s'), event.get('receive_epoch_s'), event.get('topic'), event.get('value'),
                             {k: v for k, v in event.items() if k not in {'t_s', 'receive_epoch_s', 'topic', 'value'}}])
                event_counts[event.get('topic')] += 1
            for topic, count in sorted(event_counts.items()):
                d = transformed_summary.get('duration_s')
                row('Topic_counts', [name, topic, count, d, count/d if d else None])
        else:
            manifest['missing_monitor'].append(name)
        d = transformed_summary.get('duration_s')
        path = summary.get('tcp_path_length_m')
        n = meta.get('frames', 0); dropped = meta.get('dropped_frames', 0)
        row('Episodes', [name, meta['outcome'], meta['start_time_utc'], transformed_meta['end_time_utc'], duration,
                         n, dropped, dropped/(n+dropped) if n+dropped else None, n/duration,
                         meta['recording_rate_hz']/FACTOR, bool(t_name), d, path,
                         path/d if path is not None and d else None, summary.get('warnings_errors'), f_name,
                         f'T{number:02}' if t_name else None, str(folder)])
        # Stream hashes; never read large telemetry files wholly into RAM.
        for f in files:
            sha = hashlib.sha256()
            with f.open('rb') as stream:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    sha.update(block)
            row('Sources', [name, str(f), f.stat().st_size, sha.hexdigest()])
        manifest['episodes'].append({'episode': name, 'frames': frame_count, 'monitor_events': sum(event_counts.values())})
        print(f'{number}/{len(episodes)} {name}: {frame_count} frames, {sum(event_counts.values())} monitor events', flush=True)
    for name, (ws, n, width) in sheets.items():
        if name != 'Guide':
            ws.autofilter(0, 0, n-1, width-1)
    manifest['counts'] = dict(counts)
    print('Compressing workbook', flush=True)
    book.close()
    temp.replace(args.output)
    manifest['output_bytes'] = args.output.stat().st_size
    args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'SAVED {args.output} ({manifest["output_bytes"]} bytes)', flush=True)


if __name__ == '__main__':
    main()
