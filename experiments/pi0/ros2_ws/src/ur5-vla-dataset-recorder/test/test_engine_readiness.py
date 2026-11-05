"""Competing control paths and stale engine faults must not masquerade as policy runs."""
import json
from types import SimpleNamespace

import pytest

from test_control_engine_startup import engine, measured
from engine_readiness import validate_engine_status


def ready():
    return dict(fault=None, fault_kind=None, confirmed_gripper=1., active_target=False, policy_publishers=0,
                fresh_joints=True, servo_stopped=True, gripper_busy=False,
                gripper_control="policy", teleop_mapper_running=False)


def test_clean_idle_engine_is_ready():
    validate_engine_status(ready())


@pytest.mark.parametrize('key,value,reason', [
    ('fault','policy publisher disconnected','blocked'),
    ('gripper_control','manual','automatic-only'),
    ('teleop_mapper_running',True,'Stop logitech_servo'),
    ('policy_publishers',1,'Another policy'),
    ('active_target',True,'still armed'),
    ('fresh_joints',False,'not ready'),
    ('servo_stopped',False,'not ready'),
    ('gripper_busy',True,'not ready'),
])
def test_preflight_rejects_unready_engine(key, value, reason):
    status = ready(); status[key] = value
    with pytest.raises(ValueError,match=reason):validate_engine_status(status)


def test_explicit_reset_clears_latches_without_actuator_commands():
    h, _, sent, gripper = engine()
    measured(h, .3)
    h.count_publishers = lambda topic: 0
    h.fault = 'policy publisher disconnected'
    h.confirmed_gripper = 1.
    response = h.reset_callback(None, SimpleNamespace())
    assert response.success and not sent and not gripper
    assert h.fault is None
    assert h.command == h.filtered_target == [.3]*6
    assert h.target is None and h.last_raw_target is None
    status = h.status_callback(None, SimpleNamespace())
    validate_engine_status(json.loads(status.message))


@pytest.mark.parametrize('reason', ['publisher', 'stale_joints', 'servo', 'busy'])
def test_reset_cannot_override_active_execution_or_unready_hardware(reason):
    h, clock, sent, gripper = engine()
    measured(h, 0.)
    h.count_publishers = lambda topic: 0
    h.fault = 'still blocked'
    if reason == 'publisher':h.count_publishers = lambda topic: 1
    elif reason == 'stale_joints':clock[0] += .21
    elif reason == 'servo':h.servo_stopped = False
    else:h.gripper_busy = True
    assert not h.reset_callback(None, SimpleNamespace()).success
    assert h.fault == 'still blocked'
    assert not sent and not gripper


@pytest.mark.parametrize('blocked', [False, True])
def test_preflight_waits_for_startup_feedback_but_never_clears_fault(monkeypatch, blocked):
    import sys
    import engine_readiness
    statuses = [ready(), ready()]
    statuses[0]['servo_stopped'] = False
    if blocked: statuses[0]['fault'] = 'tracking fault'
    calls, spins = [], []
    class Client:
        def wait_for_service(self, timeout_sec): return True
        def call_async(self, request):
            status = statuses.pop(0)
            calls.append(status)
            return SimpleNamespace(done=lambda: True, result=lambda: SimpleNamespace(success=True, message=json.dumps(status)), cancel=lambda: None)
    class Node:
        def __init__(self, *args, **kwargs): pass
        def create_client(self, *args): return Client()
        def destroy_node(self): pass
    monkeypatch.setitem(sys.modules, 'rclpy', SimpleNamespace(init=lambda **kw: None, shutdown=lambda **kw: None,
        spin_until_future_complete=lambda *a, **kw: None, spin_once=lambda *a, **kw: spins.append(True)))
    monkeypatch.setitem(sys.modules, 'rclpy.node', SimpleNamespace(Node=Node))
    monkeypatch.setitem(sys.modules, 'rclpy.context', SimpleNamespace(Context=object))
    monkeypatch.setitem(sys.modules, 'rclpy.executors', SimpleNamespace(SingleThreadedExecutor=lambda **kw: SimpleNamespace(shutdown=lambda: None)))
    monkeypatch.setitem(sys.modules, 'std_srvs', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'std_srvs.srv', SimpleNamespace(Trigger=SimpleNamespace(Request=object)))
    if blocked:
        with pytest.raises(SystemExit, match='tracking fault'): engine_readiness.require_ready_engine()
        assert len(calls) == 1 and not spins
    else:
        engine_readiness.require_ready_engine()
        assert len(calls) == 2 and len(spins) == 1


@pytest.mark.parametrize('gripper', [0., None])
def test_closed_or_unknown_initial_gripper_requires_explicit_episode_preparation(gripper):
    status = ready(); status['confirmed_gripper'] = gripper
    with pytest.raises(ValueError, match='prepare-episode'):validate_engine_status(status)
    validate_engine_status(status, preparing=True)


def test_new_episode_opens_only_after_guards_and_requires_successful_confirmation():
    h, _, sent, gripper = engine()
    measured(h, .3)
    h.count_publishers = lambda topic: 0
    h.fault, h.fault_kind = 'old executor disconnected', 'source_disconnected'
    h.confirmed_gripper = 0.
    def start(value):
        gripper.append(value)
        h.gripper_busy = True
    h.start_gripper = start
    response = h.prepare_episode_callback(None, SimpleNamespace())
    assert response.success and gripper == [1.] and not sent
    assert h.fault is None and h.confirmed_gripper == 0.
    status = json.loads(h.status_callback(None, SimpleNamespace()).message)
    with pytest.raises(ValueError, match='not ready'):validate_engine_status(status)
    h.gripper_state_publisher = SimpleNamespace(publish=lambda msg: None)
    h.io_finished(SimpleNamespace(result=lambda: SimpleNamespace(success=True)), [(0, 0.), (1, 1.)], 1, 1.)
    validate_engine_status(json.loads(h.status_callback(None, SimpleNamespace()).message))
    assert h.confirmed_gripper == 1. and not sent


@pytest.mark.parametrize('fault_kind', ['execution', 'source_conflict'])
def test_new_episode_cannot_automatically_clear_tracking_or_conflicting_source_fault(fault_kind):
    h, _, sent, gripper = engine()
    measured(h, 0.); h.count_publishers = lambda _: 0
    h.fault, h.fault_kind = 'inspect this fault', fault_kind
    assert not h.prepare_episode_callback(None, SimpleNamespace()).success
    assert not sent and not gripper and h.fault == 'inspect this fault'
    status = ready(); status.update(fault=h.fault, fault_kind=fault_kind)
    with pytest.raises(ValueError, match='blocked'):validate_engine_status(status, preparing=True)


@pytest.mark.parametrize('reason', ['policy', 'stale', 'servo', 'busy', 'mapper'])
def test_episode_preparation_guards_prevent_io(reason):
    h, clock, sent, gripper = engine()
    measured(h, 0.); h.count_publishers = lambda _: 0
    if reason == 'policy':h.count_publishers = lambda _: 1
    elif reason == 'stale':clock[0] += .21
    elif reason == 'servo':h.servo_stopped = False
    elif reason == 'busy':h.gripper_busy = True
    else:h.get_node_names = lambda: ['control_engine', 'logitech_servo_mapper']
    assert not h.prepare_episode_callback(None, SimpleNamespace()).success
    assert not sent and not gripper


def test_prepare_preflight_waits_for_open_ack_and_only_requests_one_open(monkeypatch):
    import sys
    import engine_readiness
    closed = ready(); closed.update(fault='executor ended', fault_kind='source_disconnected', confirmed_gripper=0.)
    busy = ready(); busy.update(confirmed_gripper=0., gripper_busy=True)
    statuses = [closed, busy, ready()]
    requests = []
    class Client:
        def __init__(self, topic): self.topic = topic
        def wait_for_service(self, timeout_sec): return True
        def call_async(self, request):
            requests.append(self.topic)
            msg = json.dumps(statuses.pop(0)) if self.topic.endswith('/status') else 'opening requested'
            return SimpleNamespace(done=lambda: True, result=lambda: SimpleNamespace(success=True, message=msg), cancel=lambda: None)
    class Node:
        def __init__(self, *a, **kw): pass
        def create_client(self, kind, topic): return Client(topic)
        def destroy_node(self): pass
    monkeypatch.setitem(sys.modules, 'rclpy', SimpleNamespace(init=lambda **kw: None, shutdown=lambda **kw: None,
        spin_until_future_complete=lambda *a, **kw: None, spin_once=lambda *a, **kw: None))
    monkeypatch.setitem(sys.modules, 'rclpy.node', SimpleNamespace(Node=Node))
    monkeypatch.setitem(sys.modules, 'rclpy.context', SimpleNamespace(Context=object))
    monkeypatch.setitem(sys.modules, 'rclpy.executors', SimpleNamespace(SingleThreadedExecutor=lambda **kw: SimpleNamespace(shutdown=lambda: None)))
    monkeypatch.setitem(sys.modules, 'std_srvs', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'std_srvs.srv', SimpleNamespace(Trigger=SimpleNamespace(Request=object)))
    engine_readiness.require_ready_engine(prepare_episode=True)
    assert requests.count('/control_engine/prepare_episode') == 1
    assert requests.count('/control_engine/status') == 3


@pytest.mark.parametrize('both_lost', [False, True])
def test_moveit_description_preload_has_one_bounded_read_only_retry(monkeypatch, both_lost):
    import sys
    from engine_readiness import fetch_moveit_description
    replies = [False, not both_lost]
    requests, cancelled = [], []
    class Client:
        def wait_for_service(self, timeout_sec):return True
        def call_async(self, request):
            requests.append(request.names)
            delivered = replies.pop(0)
            return SimpleNamespace(done=lambda: delivered,
                result=lambda: SimpleNamespace(values=[SimpleNamespace(string_value='<robot/>')]),
                cancel=lambda: cancelled.append(True))
    node = SimpleNamespace(create_client=lambda *args: Client())
    monkeypatch.setitem(sys.modules, 'rclpy', SimpleNamespace(spin_until_future_complete=lambda *args, **kwargs: None))
    monkeypatch.setitem(sys.modules, 'rcl_interfaces', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'rcl_interfaces.srv', SimpleNamespace(GetParameters=SimpleNamespace(Request=SimpleNamespace)))
    if both_lost:
        with pytest.raises(ValueError,match='two read-only attempts'):fetch_moveit_description(node, object())
    else:
        assert fetch_moveit_description(node, object()) == '<robot/>'
    assert requests == [['robot_description']]*2
    assert len(cancelled) == (2 if both_lost else 1)
