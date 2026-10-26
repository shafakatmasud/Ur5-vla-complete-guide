"""Test pose-triggered release without ROS or physical outputs."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'scripts'))
from bowl_release import BowlRelease


@pytest.fixture
def guard(tmp_path):
    path = tmp_path/'pose.json'
    path.write_text(json.dumps(dict(frame='base',tip='tool0_controller',
                                    position_m=[0.,0.,0.],quaternion_xyzw=[0.,0.,0.,1.])))
    return BowlRelease(path)


def sample(g,t,xyz=(0.,0.,0.),q=(0.,0.,0.,1.),closed=0.,age=.01,busy=False):
    return g.update(t,t,age,xyz,q,closed,busy)


def arm(g):
    for i in range(11):
        assert not sample(g,i*.1,xyz=(.1,0.,0.))
    assert g.armed


def settle(g,**kw):
    return [sample(g,2.+i*.1,**kw) for i in range(10)]


def test_arrival_without_closed_away_from_bowl_cannot_release(guard):
    assert not any(settle(guard))
    assert not guard.armed


def test_release_requires_arrival_dwell_and_is_one_shot(guard):
    arm(guard)
    hits=settle(guard)
    assert hits.count(True)==1 and not any(hits[:4])
    assert guard.released
    assert not sample(guard,4.)
    guard.reset()
    assert not guard.released and not guard.armed


@pytest.mark.parametrize('kw',[
    {'xyz':(.011,0.,0.)}, {'xyz':(0.,0.,.011)},
    {'q':(0.,0.,1.,0.)}, {'closed':1.}, {'age':.21},
    {'age':-.1}, {'busy':True},
])
def test_wrong_pose_open_io_stale_feedback_or_busy_blocks_release(guard,kw):
    arm(guard)
    assert not any(settle(guard,**kw))
    assert not guard.released


def test_duplicate_tf_does_not_accumulate_dwell(guard):
    arm(guard)
    assert not sample(guard,2.)
    for i in range(10):
        assert not guard.update(2.+i*.1,2.,.01,[0.,0.,0.],[0.,0.,0.,1.],0.,False)


def test_orientation_quaternion_sign_is_equivalent(guard):
    arm(guard)
    assert any(settle(guard,q=(0.,0.,0.,-1.)))


def test_stale_transform_breaks_confirmation(guard):
    arm(guard)
    sample(guard,2.)
    sample(guard,2.1)
    sample(guard,2.2,age=.3)
    assert not any(settle(guard))


def test_fast_pass_through_does_not_release(guard):
    arm(guard)
    assert not sample(guard,2.,xyz=(.009,0.,0.))
    assert not sample(guard,2.1,xyz=(-.009,0.,0.))
    assert not sample(guard,2.2,xyz=(-.03,0.,0.))


def test_first_closed_away_from_bowl_latches_transport_until_release_or_reset(guard):
    sample(guard,0.,xyz=(.1,0.,0.))
    assert guard.transport_hold
    sample(guard,.1,age=.3)
    assert guard.transport_hold and not guard.released
    guard.reset()
    assert not guard.transport_hold


@pytest.mark.parametrize('xyz,expected', [((.02,0.,.07),True), ((.026,0.,.04),False),
    ((0.,0.,-.011),False), ((0.,0.,.091),False)])
def test_bowl_region_allows_release_height_variation_but_not_outside_bowl(guard,xyz,expected):
    guard.zone = dict(horizontal_radius_m=.025, below_reference_m=.01,
                      above_reference_m=.09,orientation_tolerance_deg=30.)
    arm(guard)
    assert any(settle(guard,xyz=xyz)) == expected


def test_operator_requested_30mm_symmetric_region_loads_and_checks_boundaries(tmp_path):
    path=tmp_path/'region.json'
    path.write_text(json.dumps(dict(frame='base',tip='tool0_controller',
        position_m=[0.,0.,0.],quaternion_xyzw=[0.,0.,0.,1.],
        release_zone=dict(horizontal_radius_m=.03,below_reference_m=.03,
                          above_reference_m=.03,orientation_tolerance_deg=30.))))
    for xyz,expected in [((.029,0.,-.029),True),((0.,.029,.029),True),
                         ((.031,0.,0.),False),((0.,0.,-.031),False),((0.,0.,.031),False)]:
        g=BowlRelease(path)
        arm(g)
        assert any(settle(g,xyz=xyz)) == expected
