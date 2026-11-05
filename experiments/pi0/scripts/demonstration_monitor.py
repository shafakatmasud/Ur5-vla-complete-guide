#!/usr/bin/python3
"""Read-only trial evidence linked to confirmed dataset-recorder episodes."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import rclpy
from std_srvs.srv import Trigger

from cross_grid_monitor import CrossGridMonitor
from cross_grid_results import utc, write_json


def saved_outcome(episodes, name):
    """Only a finalized dataset establishes a saved label; disappearance is abort."""
    import yaml
    final = episodes / name
    temporary = episodes / f'.{name}.incomplete'
    if final.is_dir():
        metadata = yaml.safe_load((final / 'metadata.yaml').read_text())
        return {'success': 'pass', 'failure': 'fail'}.get(metadata.get('outcome'), 'unknown')
    if temporary.exists():
        return None
    return 'aborted'


class EpisodeEvidenceStore:
    def __init__(self, root):
        self.root = root
        self.active = None
        self.pending = None

    def start(self, cross, attempt):
        name, status, dataset_path = self.pending
        folder = self.root / name
        folder.mkdir(exist_ok=False)
        for part in ('camera_snapshots', 'video', 'depth', 'configuration'):
            (folder / part).mkdir()
        trial = dict(episode=name, cross=cross, attempt=attempt, status='recording',
                     started_utc=utc(), start_epoch_s=time.time(), timeout_s=None,
                     dataset_path=str(dataset_path), recorder_status_at_attach=status,
                     capture_started_after_episode_start=True,
                     onset_alignment='Recorder status polled at 0.25 s; attach may be later if monitor starts mid-episode. Dataset frames retain original onset.',
                     condition='Manual joystick demonstration: Sort the fruits',
                     result_authority='Dataset label selected by operator; no automatic proof of apple identity/contact or task completion')
        self.active = folder, trial
        write_json(folder / 'trial.json', trial)
        package = Path(__file__).resolve().parents[1] / 'ros2_ws/src/ur5-vla-dataset-recorder'
        sources = [Path(__file__), package/'ur5_vla_dataset_recorder/recorder_node.py',
                   package/'ur5_vla_dataset_recorder/logitech_servo_node.py',
                   package/'config/recorder_experiment2_sort_fruits.yaml']
        write_json(folder/'configuration/demonstration_sources.json', {
            str(p): {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'text': p.read_text()}
            for p in sources})
        return folder, trial

    def finish(self, outcome, summary, notes='', **stages):
        folder, trial = self.active
        trial.update(status='finished', operator_outcome=outcome, notes=notes,
                     ended_utc=utc(), duration_s=summary['duration_s'], **stages)
        metadata_path = Path(trial['dataset_path']) / 'metadata.yaml'
        if metadata_path.exists():
            import yaml
            metadata = yaml.safe_load(metadata_path.read_text())
            write_json(folder/'configuration/dataset_metadata.json', metadata)
            trial['dataset_frames_written'] = metadata.get('frames')
            trial['dataset_dropped_samples'] = metadata.get('dropped_frames')
        write_json(folder / 'summary.json', {**summary, **trial})
        write_json(folder / 'trial.json', trial)
        (folder/'analysis.txt').write_text(
            f"Episode: {trial['episode']}\nOperator outcome: {outcome}\n"
            f"Telemetry capture duration: {summary['duration_s']:.2f} s\n"
            f"TCP path length: {summary['tcp_path_length_m']:.4f} m (includes sensor noise)\n"
            f"Maximum command/measured gaps, radians: {summary['max_command_measured_gap_rad']}\n"
            f"Warnings/errors observed: {summary['warnings_errors']}\n"
            "Manual demonstration; policy diagnostics may be empty. See summary.json, "
            "observations.jsonl and camera evidence. IO does not establish grasp contact.\n"
            f"{trial['onset_alignment']}\n")
        self.active = None
        rows = [json.loads(p.read_text()) for p in sorted(self.root.glob('episode_*/trial.json'))]
        fields = ['episode', 'status', 'operator_outcome', 'duration_s', 'started_utc', 'ended_utc', 'dataset_path', 'condition']
        temp = self.root / 'episodes.csv.tmp'
        with temp.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(self.root / 'episodes.csv')
        return trial


class DemonstrationMonitor(CrossGridMonitor):
    def __init__(self, experiment):
        self.experiment = experiment.resolve()
        evidence = self.experiment / 'telemetry'
        evidence.mkdir(parents=True, exist_ok=True)
        (evidence / 'configuration').mkdir(exist_ok=True)
        # Base monitor supplies subscriptions, timestamped evidence, video, depth,
        # TF and diagnostics. Its numbered-evaluation store is replaced below.
        write_json(evidence / 'session.json', {'task': 'Sort the fruits', 'timeout_s': None})
        super().__init__(evidence)
        self.store = EpisodeEvidenceStore(evidence)
        self.recorder_client = self.create_client(Trigger, '/dataset_recorder/status')
        self.recorder_future = None
        self.request_started = 0.
        self.last_poll = 0.
        self.episode = None
        self.sequence = len(list(evidence.glob('episode_*'))) + 1
        print('DEMONSTRATION MONITOR READY: follows confirmed recorder state; no robot commands', flush=True)

    def tick(self):
        # This observer never dispatches evaluation command files or robot commands.
        now = time.monotonic()
        if self.recorder_future is not None:
            if self.recorder_future.done():
                try:
                    self.observe(json.loads(self.recorder_future.result().message))
                except Exception as error:
                    self.get_logger().error(f'Recorder observation failed: {error}')
                self.recorder_future = None
            elif now - self.request_started > 3:
                self.recorder_future.cancel()
                self.recorder_future = None
                self.emit('recorder_status_timeout', {'recording_continues': True})
        if self.recorder_future is None and now-self.last_poll >= .25 and self.recorder_client.service_is_ready():
            self.last_poll = self.request_started = now
            self.recorder_future = self.recorder_client.call_async(Trigger.Request())
        if self.active:
            self.capture()

    def observe(self, status):
        self.emit('dataset_recorder/status', status)
        name = status.get('episode')
        episodes = self.experiment / 'episodes'
        if self.episode and name != self.episode:
            outcome = saved_outcome(episodes, self.episode)
            if outcome is None:
                # Preserve evidence when the recorder exited/restarted before labeling.
                outcome = 'unknown'
            self.finish(outcome, 'Label follows finalized recorder metadata; abort retains evidence outside training dataset.')
            self.episode = None
            self.sequence += 1
        if not self.episode and name and (status.get('active') or status.get('stopped_pending_label')):
            self.store.pending = name, status, episodes / name
            self.start(1, self.sequence)
            self.episode = name
            self.emit('dataset_recorder/status', status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', type=Path, required=True)
    args = parser.parse_args()
    # One observer owns each evidence root; released automatically on process exit.
    import fcntl
    root = args.experiment_root / 'telemetry'
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / '.monitor.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('A demonstration telemetry monitor is already running')
    rclpy.init()
    node = DemonstrationMonitor(args.experiment_root)
    def stop(*_):
        node.running = False
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while rclpy.ok() and node.running:
            rclpy.spin_once(node, timeout_sec=.05)
    finally:
        if node.active:
            node.finish('unknown', 'Telemetry monitor stopped; dataset outcome not established by this observer.')
        write_json(node.root / 'status.json', {'state': 'stopped', 'updated_utc': utc(), 'pid': os.getpid()})
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
