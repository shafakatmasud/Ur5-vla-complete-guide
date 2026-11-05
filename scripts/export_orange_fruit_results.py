#!/usr/bin/python3
"""Build a results workbook from retained orange-test records and their evidence."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import xlsxwriter


OUTCOMES = {'pass': 'Full success', 'partial': 'Partial success', 'fail': 'Failure'}
JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3']


def read(path):
    return json.loads(path.read_text())


def leaves(value, prefix=''):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from leaves(child, f'{prefix}.{key}' if prefix else key)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from leaves(child, f'{prefix}[{i}]')
        if not value:
            yield prefix, '[]', 'empty list'
    else:
        yield prefix, value, type(value).__name__


def local(stamp):
    return datetime.fromisoformat(stamp).astimezone(ZoneInfo('America/Toronto')).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.session.resolve(), args.output.resolve()
    records = []
    with (root/'results.csv').open(newline='') as stream:
        table = list(csv.DictReader(stream))
    for row in table:
        if row['status'] != 'finished' or row['operator_outcome'] not in OUTCOMES:
            continue
        n = int(row['attempt'])
        folder = root/f'cross_01/attempt_{n:02}'
        trial, summary = read(folder/'trial.json'), read(folder/'summary.json')
        assert trial['attempt'] == summary['attempt'] == n
        assert trial['operator_outcome'] == summary['operator_outcome'] == row['operator_outcome']
        assert not summary.get('excluded_from_scored_results', False)
        assert math.isclose(float(row['duration_s']), summary['duration_s'], abs_tol=1e-6)
        manifest = read(folder/'report_frames/source_manifest.json')
        assert len(manifest['panels']) == 20
        records.append(dict(n=n, folder=folder, trial=trial, summary=summary, manifest=manifest))
    records.sort(key=lambda record: record['n'])
    assert [r['n'] for r in records] == list(range(1, 21))
    assert [r['trial']['start_epoch_s'] for r in records] == sorted(r['trial']['start_epoch_s'] for r in records)
    counts = {label: sum(r['summary']['operator_outcome'] == code for r in records)
              for code, label in OUTCOMES.items()}
    assert list(counts.values()) == [6, 3, 11]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.building.xlsx')
    workbook = xlsxwriter.Workbook(str(temporary), {
        'strings_to_formulas': False, 'strings_to_urls': False,
        'default_date_format': 'yyyy-mm-dd hh:mm:ss'})
    workbook.set_properties({'title': 'Sort the oranges — 20-attempt results',
                             'subject': 'Observed UR5 system results and recorded telemetry',
                             'comments': 'Operator-reported outcomes; original retained chronology and measured times.'})
    header = workbook.add_format({'bold': True, 'font_color': 'white', 'bg_color': '#17344A', 'text_wrap': True, 'valign': 'vcenter'})
    body = workbook.add_format({'valign': 'top', 'text_wrap': True})
    number = workbook.add_format({'num_format': '0.000', 'valign': 'top'})
    percent = workbook.add_format({'num_format': '0.0%', 'valign': 'top'})
    colors = {label: workbook.add_format({'bg_color': color, 'font_color': 'white', 'bold': True})
              for label, color in [('Full success', '#23845B'), ('Partial success', '#AA7912'), ('Failure', '#AD4141')]}
    sheets, dimensions = {}, {}

    def sheet(name, headers, rows, widths=None):
        ws = workbook.add_worksheet(name)
        sheets[name] = ws
        ws.freeze_panes(1, 1)
        ws.set_row(0, 34)
        ws.set_default_row(30)
        ws.set_column(0, len(headers)-1, 22, body)
        for column, width in (widths or {}).items():
            ws.set_column(column, column, width, body)
        ws.write_row(0, 0, headers, header)
        for i, row in enumerate(rows, 1):
            assert len(row) == len(headers), (name, i, len(row), len(headers))
            for j, value in enumerate(row):
                if value is None:
                    ws.write_blank(i, j, None)
                elif isinstance(value, (dict, list)):
                    ws.write_string(i, j, json.dumps(value, ensure_ascii=False))
                elif isinstance(value, float):
                    ws.write_number(i, j, value, number)
                else:
                    ws.write(i, j, value)
        if rows:
            ws.add_table(0, 0, len(rows), len(headers)-1, {
                'name': name.replace('_', '')+'Table', 'style': 'Table Style Medium 2',
                'columns': [{'header': h, 'header_format': header} for h in headers]})
        ws.set_landscape()
        ws.fit_to_pages(1, 0)
        ws.repeat_rows(0)
        ws.set_footer('&LSort the oranges&RCaptured results | &P / &N')
        dimensions[name] = {'data_rows': len(rows), 'columns': len(headers)}
        return ws

    summary_rows = [
        ['Task', 'Sort the oranges', 'Place both oranges into the fixed bowl.'],
        ['Total retained attempts', 20, 'Consecutive retained attempt IDs; no presentation shuffle.'],
        ['Full success', 6, 'Both oranges placed, according to the operator.'],
        ['Partial success', 3, 'One orange placed; second orange not completed.'],
        ['Failure', 11, 'Operator reported failure; unspecified individual stages remain unknown.'],
        ['Full success rate', .30, '6 / 20; operator-reported system completion.'],
        ['Partial success rate', .15, '3 / 20.'],
        ['Failure rate', .55, '11 / 20.'],
        ['At least one orange completed', .45, '(6 full + 3 partial) / 20.'],
        ['Known orange placements', sum(r['summary'].get('placed_count') or 0 for r in records), 'Sum of reported counts only; unknown counts are not zero.'],
        ['Attempts with unknown placed counts', sum(r['summary'].get('placed_count') is None for r in records), 'No overall per-object placement rate is calculated from missing counts.'],
        ['Camera images indexed', 400, '10 base + 10 wrist images per retained attempt.'],
        ['Time basis', 'Original measured seconds', 'Recording duration includes repositioning, pauses, setup and reporting delay; it is not verified task completion time.'],
        ['Execution', 'Policy pickup + planned bowl placement', 'Two pickup/place cycles, with operator standard-pose reset and relaunch between fruits; this is an assisted system evaluation.'],
        ['Model', 'Frozen 110-episode, 18,000-step SmolVLA', 'Orange-cube checkpoint, tested on orange fruit; not the separate Pi0 training run.'],
        ['Telemetry scope', 'All stored summary fields + events + source links', 'Full per-message traces and videos remain in the linked original files; they are not embedded in this workbook.'],
    ]
    ws = sheet('Summary', ['Metric', 'Value', 'Meaning'], summary_rows, {0: 34, 1: 51, 2: 94})
    for row in (6, 7, 8, 9):
        ws.write_number(row, 1, summary_rows[row-1][1], percent)
    chart = workbook.add_chart({'type': 'column'})
    chart.add_series({'name': 'Retained attempts', 'categories': ['Summary', 3, 0, 5, 0], 'values': ['Summary', 3, 1, 5, 1],
                      'points': [{'fill': {'color': c}} for c in ['#23845B', '#AA7912', '#AD4141']], 'data_labels': {'value': True}})
    chart.set_title({'name': 'Observed outcomes (20 attempts)'})
    chart.set_legend({'none': True})
    chart.set_y_axis({'name': 'Attempt count', 'major_unit': 2})
    ws.insert_chart('E2', chart, {'x_scale': 1.15, 'y_scale': 1.1})

    attempt_rows = []
    object_rows, telemetry_rows, topic_rows, event_rows, condition_rows = [], [], [], [], []
    all_fields, image_rows, evidence_rows = [], [], []
    for r in records:
        n, folder, s, t = r['n'], r['folder'], r['summary'], r['trial']
        attempt_rows.append([n, OUTCOMES[s['operator_outcome']], 2, s.get('picked_count'), s.get('placed_count'),
                             s.get('full_task_complete'), s.get('pickup'), s.get('carry'), s.get('placement'),
                             local(s['started_utc']), local(s['ended_utc']), s['duration_s'], s['notes'], str(folder)])
        for item in s.get('item_outcomes', []):
            object_rows.append([n, item['item'], item.get('pickup'), item.get('carry'), item.get('placement'), item.get('status'), item.get('authority')])
        telemetry_rows.append([n, s['duration_s'], s.get('first_policy_command_s'), s.get('first_arm_command_s'),
                               s.get('tcp_path_length_m'), s.get('tcp_samples'), s.get('invalid_action_count'),
                               s.get('warnings_errors'), s.get('max_consecutive_policy_jump_rad'),
                               *s.get('max_policy_measured_gap_rad', [None]*6), *s.get('max_command_measured_gap_rad', [None]*6),
                               *[s.get('video_frames', {}).get(c) for c in ['base', 'wrist']],
                               *[s.get('max_color_source_age_s', {}).get(c) for c in ['base', 'wrist']],
                               len(s.get('gripper_prediction_transitions', [])), len(s.get('digital_output_transitions', []))])
        topics = sorted(set(s.get('native_message_counts', {})) | set(s.get('rates_hz', {})) | set(s.get('message_counts', {})))
        for topic in topics:
            topic_rows.append([n, topic, s.get('native_message_counts', {}).get(topic), s.get('message_counts', {}).get(topic), s.get('rates_hz', {}).get(topic)])
        for family in ['notable_logs', 'gripper_prediction_transitions', 'digital_output_transitions']:
            for event in s.get(family, []):
                event_rows.append([n, family, event.get('t_s'), event.get('node'), event.get('level'), event.get('message'), event])
        snapshot = read(folder/'configuration/execution_start.json')
        offset = snapshot.get('offset_configuration', {})
        code = (folder/'configuration/policy_executor.py').read_text()
        condition_rows.append([n, offset.get('requested_forward_mm'), offset.get('requested_wrist_image_right_mm'),
                               snapshot.get('code_sha256', {}).get('policy_executor.py'),
                               snapshot.get('code_sha256', {}).get('control_engine.py'),
                               'check_engine_status' in code, snapshot.get('processes'), str(folder/'configuration')])
        for key, value, kind in leaves(s):
            all_fields.append([n, key, value, kind, str(folder/'summary.json')])
        for panel in r['manifest']['panels']:
            path = folder/'report_frames'/panel['file']
            assert path.is_file()
            image_rows.append([n, panel['camera'], panel['step'], panel['capture_time_s'], panel.get('source_stamp_s'),
                               panel.get('source_age_s'), panel['video_frame_index_zero_based'], panel['width'], panel['height'], str(path)])
        for name in ['trial.json', 'summary.json', 'operator_outcome.json', 'observations.jsonl',
                     'report_frames/source_manifest.json', 'video/base.avi', 'video/wrist.avi']:
            path = folder/name
            assert path.is_file()
            evidence_rows.append([n, name, str(path), path.stat().st_size, digest(path) if path.suffix == '.json' else 'Original large stream; see source file'])
    ws = sheet('Attempt_results', ['Attempt', 'Outcome', 'Oranges requested', 'Oranges picked (reported)', 'Oranges placed (reported)',
               'Full task complete', 'Any pickup', 'Any carry', 'Any placement', 'Capture start (Toronto)', 'Capture end (Toronto)',
               'Recording duration (s)', 'Recorded notes', 'Evidence folder'], attempt_rows,
               {0: 11, 1: 22, 9: 29, 10: 29, 12: 100, 13: 90})
    for i, row in enumerate(attempt_rows, 1):
        ws.write_string(i, 1, row[1], colors[row[1]])
        ws.write_url(i, 13, Path(row[13]).as_uri(), string='Open attempt evidence')
    sheet('Object_outcomes', ['Attempt', 'Orange', 'Pickup', 'Carry', 'Placement', 'Status / unknown details', 'Authority'], object_rows, {5: 72, 6: 100})
    metric_headers = ['Attempt', 'Recording duration (s)', 'First policy command (s from marker)', 'First arm command (s from marker)',
                      'TCP path (m; includes noise)', 'TCP samples', 'Invalid action count', 'Warning/error message count', 'Max consecutive abstract target jump (rad)']
    metric_headers += [f'Max policy-measured {j} (rad)' for j in JOINTS] + [f'Max command-measured {j} (rad)' for j in JOINTS]
    metric_headers += ['Base video frames', 'Wrist video frames', 'Max base image source age (s)', 'Max wrist image source age (s)', 'Gripper request transitions', 'Digital output transitions']
    sheet('Telemetry', metric_headers, telemetry_rows)
    sheet('Topic_rates', ['Attempt', 'Topic', 'Native callback count', 'Summary message count', 'Received mean rate (Hz)'], topic_rows, {1: 72})
    sheet('Events', ['Attempt', 'Family', 'Time from marker (s)', 'Node', 'Log level', 'Message', 'Stored event JSON'], event_rows, {1: 35, 3: 30, 5: 110, 6: 110})
    sheet('Conditions', ['Attempt', 'Captured forward offset (mm)', 'Captured right offset (mm)', 'Policy code SHA-256', 'Engine code SHA-256',
                        'Engine-fault guard in captured code', 'Captured processes / arguments', 'Configuration folder'], condition_rows, {3: 72, 4: 72, 6: 110, 7: 100})
    ws = sheet('Camera_images', ['Attempt', 'Camera', 'Image number', 'Capture-relative time (s)', 'Source timestamp (epoch s)', 'Source age (s)',
                                'Video frame index (zero based)', 'Width (px)', 'Height (px)', 'Image path'], image_rows, {9: 110})
    for i, row in enumerate(image_rows, 1):
        ws.write_url(i, 9, Path(row[9]).as_uri(), string=row[9])
    ws = sheet('Evidence', ['Attempt', 'File', 'Absolute source path', 'Size (bytes)', 'SHA-256 for JSON / stream note'], evidence_rows, {1: 40, 2: 110, 4: 75})
    for i, row in enumerate(evidence_rows, 1):
        ws.write_url(i, 2, Path(row[2]).as_uri(), string=row[2])
    sheet('All_summary_fields', ['Attempt', 'JSON field path', 'Stored value', 'Value type', 'Summary path'], all_fields, {1: 75, 2: 115, 4: 110})
    methods = [
        ['Task', 'Sort the oranges: place both orange fruits into the fixed bowl.'],
        ['Execution unit', 'Each retained attempt includes two sequential pickup/place cycles. The operator returns the arm to standard pose and launches the same executor for the second fruit.'],
        ['Assistance', 'Frozen SmolVLA generates pickup actions. Checked planned transport and automatic bowl release assist carry/placement after closed IO; IO does not prove a held fruit.'],
        ['Full success', 'Operator reported both oranges placed into the bowl.'],
        ['Partial success', 'Operator reported one orange placed, with the second incomplete.'],
        ['Failure', 'Operator reported failure; individual stage/count fields remain blank when not provided.'],
        ['Outcome authority', 'Operator reports. Image candidates, command values and IO transitions do not independently prove pickup or placement.'],
        ['Any-stage flags', 'Pickup/carry/placement flags mean at least one orange completed that stage. Use item rows/counts to assess both fruits.'],
        ['Unknown values', 'Blank cells and null entries indicate unreported or unavailable measurements. They are not zeros or false.'],
        ['Timing', 'Original recorded seconds, unscaled. Capture duration includes pauses, repositioning, relaunches and reporting delay; it is not verified active task completion time.'],
        ['300-second reference', 'The stored 300 s field is a reference, not an enforced robot stop or validated completion deadline. Reported success rates use operator outcomes, not legacy within-window flags.'],
        ['Attempt order', 'Retained chronological sequence. Abandoned captures were removed at operator request and retained IDs made consecutive; performance order was not randomized.'],
        ['Changing conditions', 'Diagnostic/runtime handling changed during testing. Inspect captured code hashes and configurations. This is not a fixed-condition autonomous-policy benchmark.'],
        ['TCP metrics', 'Path length includes noise and all recorded movement, including repositioning. Joint target/measured error is not fruit localization or grasp error.'],
        ['Abstract target jumps', 'The peak metric spans the entire recording and can cross executor restarts; it is not automatically an executed joint discontinuity.'],
        ['Warnings and errors', 'Counts include startup waits and publisher disconnection after operator stop. Severity alone does not establish a task-causing fault.'],
        ['ROS rates', 'Reported rates are summary receive rates. Native callback counts, recorded rows and throttled unchanged IO events have different meanings.'],
        ['Camera images', '400 original recorded candidate frames (200 per camera); linked files remain alongside source evidence. Stage captions require visual review. AVI nominal playback is 10 FPS; JSONL timestamps are authoritative.'],
        ['Localization interpretation', 'Attempt 20: operator suspected failure to locate the second orange. This cause remains unconfirmed; no measured object localization result is provided.'],
        ['Telemetry scope', 'All scalar/list leaves of saved summaries, recorded summary events, topic rates, camera index and source links are included. Full raw per-message JSONL and videos are external evidence, not embedded.'],
        ['Result reproducibility', 'Generated exclusively from retained trial/summary files and camera manifests. No outcomes, times, trajectories or imagery were synthesized.'],
    ]
    sheet('Methods_definitions', ['Field', 'Definition / limitation'], methods, {0: 35, 1: 145})
    sequence = workbook.add_chart({'type': 'column', 'subtype': 'stacked'})
    ws = sheets['Summary']
    sequence_rows = []
    for r in records:
        code = r['summary']['operator_outcome']
        sequence_rows.append([r['n'], int(code == 'pass'), int(code == 'partial'), int(code == 'fail')])
    sheet('Outcome_sequence', ['Attempt', 'Full success', 'Partial success', 'Failure'], sequence_rows)
    for col, label, color in [(1, 'Full success', '#23845B'), (2, 'Partial success', '#AA7912'), (3, 'Failure', '#AD4141')]:
        sequence.add_series({'name': label, 'categories': ['Outcome_sequence', 1, 0, 20, 0], 'values': ['Outcome_sequence', 1, col, 20, col],
                             'fill': {'color': color}, 'border': {'none': True}})
    sequence.set_title({'name': 'Retained attempt sequence'})
    sequence.set_x_axis({'name': 'Attempt'})
    sequence.set_y_axis({'visible': False, 'min': 0, 'max': 1})
    sequence.set_legend({'position': 'bottom'})
    ws.insert_chart('E18', sequence, {'x_scale': 1.6, 'y_scale': .75})
    workbook.close()
    temporary.replace(output)
    manifest = {'workbook': str(output), 'sha256': digest(output), 'task': 'Sort the oranges', 'attempts': 20,
                'outcomes': counts, 'full_success_rate': .30, 'partial_success_rate': .15, 'failure_rate': .55,
                'retained_sequence_preserved': True, 'synthetic_telemetry': False, 'time_scale': 1,
                'images_indexed': len(image_rows), 'sheets': dimensions, 'source_session': str(root),
                'source_summary_sha256': {str(r['n']): digest(r['folder']/'summary.json') for r in records}}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
