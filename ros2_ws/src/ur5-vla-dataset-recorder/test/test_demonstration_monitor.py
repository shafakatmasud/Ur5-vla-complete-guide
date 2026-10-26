import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'scripts'))
from demonstration_monitor import DemonstrationMonitor, saved_outcome


def test_saved_labels_require_finalized_folder(tmp_path):
    name = 'episode_example'
    incomplete = tmp_path / f'.{name}.incomplete'
    incomplete.mkdir()
    (incomplete/'metadata.yaml').write_text('outcome: success\n')
    assert saved_outcome(tmp_path, name) is None
    incomplete.rename(tmp_path/name)
    assert saved_outcome(tmp_path, name) == 'pass'
    assert saved_outcome(tmp_path, 'discarded') == 'aborted'


def test_monitor_follows_recorder_and_preserves_abort(tmp_path):
    episodes = tmp_path/'episodes'
    episodes.mkdir()
    events = []
    node = SimpleNamespace(experiment=tmp_path, episode=None, sequence=1,
                           store=SimpleNamespace(pending=None),
                           emit=lambda *a: events.append(a))
    node.start = lambda *a: events.append(('start', a))
    node.finish = lambda *a: events.append(('finish', a))
    status = {'episode': 'episode_one', 'active': True}
    DemonstrationMonitor.observe(node, status)
    DemonstrationMonitor.observe(node, status)
    assert sum(e[0]=='start' for e in events) == 1
    DemonstrationMonitor.observe(node, {'episode': None, 'active': False})
    assert [e[1][0] for e in events if e[0]=='finish'] == ['aborted']
    assert node.episode is None


def test_pending_label_keeps_evidence_running(tmp_path):
    events = []
    node = SimpleNamespace(experiment=tmp_path, episode='episode_one', sequence=1,
                           emit=lambda *a: None, finish=lambda *a: events.append(a))
    DemonstrationMonitor.observe(node, {'episode': 'episode_one', 'active': False,
                                      'stopped_pending_label': True})
    assert not events
