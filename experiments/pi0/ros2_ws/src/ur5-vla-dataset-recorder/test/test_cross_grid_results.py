"""Trial recording must preserve operator outcomes and never overwrite evidence."""
import csv
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'scripts'))
from cross_grid_results import TrialStore


def session(tmp_path):
    (tmp_path/'session.json').write_text(json.dumps({'cross_count':2,'attempts_per_cross':5,'timeout_s':300}))
    with (tmp_path/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['cross','attempt','status','operator_outcome','duration_s','timeout_reached','reported_pass_within_limit']);w.writeheader()
        for cross in (1,2):
            for attempt in range(1,6):w.writerow(dict(cross=cross,attempt=attempt,status='pending'))
    return TrialStore(tmp_path)


def test_trial_boundaries_duplicate_protection_and_aggregate(tmp_path):
    store=session(tmp_path)
    folder,_=store.start(1,1)
    with pytest.raises(ValueError,match='active'):store.start(2,1)
    result=store.finish('pass',{'duration_s':20},'cube released into bowl',pickup=True,carry=True,placement=True)
    assert result['reported_pass_within_limit'] and (folder/'analysis.txt').exists()
    with pytest.raises(FileExistsError):store.start(1,1)
    aggregate=json.loads((tmp_path/'aggregate.json').read_text())
    assert aggregate['completed']==1 and aggregate['passes_within_limit']==1


def test_operator_pass_after_timeout_is_preserved_separately(tmp_path):
    store=session(tmp_path);folder,_=store.start(2,5)
    store.finish('pass',{'duration_s':301})
    summary=json.loads((folder/'summary.json').read_text())
    assert summary['operator_outcome']=='pass' and summary['timeout_reached']
    assert not summary['reported_pass_within_limit']


def test_unknown_outcome_never_becomes_a_measured_success(tmp_path):
    store=session(tmp_path);folder,_=store.start(1,1)
    store.finish('unknown',{'duration_s':5})
    assert json.loads((folder/'summary.json').read_text())['reported_pass_within_limit'] is None


def test_partial_completion_is_preserved_without_counting_as_full_pass(tmp_path):
    store=session(tmp_path);folder,_=store.start(1,1)
    store.finish('partial',{'duration_s':200},'One of two oranges placed')
    summary=json.loads((folder/'summary.json').read_text())
    aggregate=json.loads((tmp_path/'aggregate.json').read_text())
    assert summary['operator_outcome']=='partial'
    assert not summary['reported_pass_within_limit']
    assert aggregate['partial_successes']==1 and aggregate['operator_passes']==0


@pytest.mark.parametrize('cross,attempt',[(0,1),(3,1),(1,0),(1,6)])
def test_out_of_protocol_ids_are_rejected(tmp_path,cross,attempt):
    store=session(tmp_path)
    with pytest.raises(ValueError):store.start(cross,attempt)
    assert store.active is None


def test_invalid_outcome_keeps_attempt_active(tmp_path):
    store=session(tmp_path);store.start(1,1)
    with pytest.raises(ValueError):store.finish('maybe',{'duration_s':10})
    assert store.active is not None
