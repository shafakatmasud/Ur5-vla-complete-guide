"""Geometry and queue regressions for the continuous Cartesian modifier."""
import ast
from collections import deque
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts'))
from trajectory_offset import CartesianOffsetModel, TrajectoryOffsetPlanner, rotation_error


def test_path_checks_every_state_sequentially_with_bounded_read_only_retry():
    calls = []
    planner = object.__new__(TrajectoryOffsetPlanner)
    planner.validity = SimpleNamespace(srv_name='/check_state_validity')
    planner.require_service = lambda client: None
    def call(client, request, timeout, attempts):
        calls.append((request, timeout, attempts))
        return SimpleNamespace(valid=True)
    planner.call = call
    report = planner.check_path(list(range(51)))
    assert [r for r, _, _ in calls] == list(range(51))
    assert all(0 < timeout <= .25 and attempts == 2 for _, timeout, attempts in calls)
    assert report['validity_request_mode'] == 'sequential_bounded_retry'


def test_path_rejection_stops_before_later_states():
    planner = object.__new__(TrajectoryOffsetPlanner)
    planner.validity = SimpleNamespace(srv_name='/check_state_validity')
    planner.require_service = lambda client: None
    calls = []
    def call(client, request, **kwargs):
        calls.append(request)
        return SimpleNamespace(valid=request != 1)
    planner.call = call
    with pytest.raises(ValueError, match='invalid state 2/3'):
        planner.check_path([0, 1, 2])
    assert calls == [0, 1]


def test_path_timeout_never_accepts_unchecked_states():
    planner = object.__new__(TrajectoryOffsetPlanner)
    planner.validity = SimpleNamespace(srv_name='/check_state_validity')
    planner.require_service = lambda client: None
    calls = []
    def call(client, request, **kwargs):
        calls.append(request)
        raise TimeoutError('missing response')
    planner.call = call
    with pytest.raises(TimeoutError, match='state 1/2 after read-only retry'):
        planner.check_path([0, 1])
    assert calls == [0]


def model():
    # A bounded six-joint chain exercises XML origins, frame handling and IK
    # without relying on a running ROS installation or robot.
    xml = ['<robot name="test"><link name="base"/>']
    previous = 'base'
    for i in range(6):
        link = f'link_{i}'
        xml.append(f'<link name="{link}"/><joint name="q{i}" type="revolute">'
                   f'<parent link="{previous}"/><child link="{link}"/>'
                   '<origin xyz="0.2 0 0"/><axis xyz="0 0 1"/>'
                   '<limit lower="-3" upper="3" effort="1" velocity="1"/></joint>')
        previous = link
    xml.append(f'<link name="tool0"/><joint name="tip" type="fixed">'
               f'<parent link="{previous}"/><child link="tool0"/>'
               '<origin xyz="0.1 0 0"/></joint></robot>')
    return CartesianOffsetModel(''.join(xml), [f'q{i}' for i in range(6)])


def test_every_target_has_one_translation_and_preserves_orientation():
    m = model()
    for i in range(50):
        q = [.5 + .0001*i, -.8, .6, .3, -.7, .2]
        original, _ = m.fk_jacobian(q)
        corrected, _ = m.fk_jacobian(m.translate(q, [.005, 0, 0]))
        assert np.linalg.norm(corrected[:3, 3]-original[:3, 3]-[.005, 0, 0]) < .00005
        assert np.linalg.norm(rotation_error(original[:3, :3], corrected[:3, :3])) < .001


def test_analytic_jacobian_matches_finite_difference():
    m = model()
    q = [.5, -.8, .6, .3, -.7, .2]
    original, jacobian = m.fk_jacobian(q)
    for i in range(6):
        perturbed = list(q)
        perturbed[i] += 1e-6
        changed, _ = m.fk_jacobian(perturbed)
        assert np.linalg.norm((changed[:3, 3]-original[:3, 3])/1e-6-jacobian[:3, i]) < 1e-5


@pytest.mark.parametrize('vector', [[.008, 0, 0], [float('nan'), 0, 0], [0, 0]])
def test_invalid_offsets_are_rejected(vector):
    with pytest.raises(ValueError):
        model().translate([.5, -.8, .6, .3, -.7, .2], vector)


def test_blending_uses_raw_boundary_and_keeps_gripper_sequence():
    # Exercise the actual executor methods with fake model/ROS endpoints.
    source = ast.parse((ROOT / 'scripts/policy_executor.py').read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'PolicyExecutor')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('infer', 'publish_action')]
    module = ast.Module(body=[ast.ClassDef(name='Harness', bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
    ast.fix_missing_locations(module)
    class Message:
        pass
    calls = []
    def transform(actions, reference):
        calls.append([list(a) for a in actions])
        return [[*[v+.01 for v in a[:6]], a[6]] for a in actions], {}
    chunk = [[.1]*6+[float(i<25)] for i in range(50)]
    namespace = dict(time=time, Float64MultiArray=Message,
                     request_prediction=lambda *args: (chunk, 0),
                     SERVER_HOST='localhost', SERVER_PORT=8765, TASK='test', BLEND_ACTIONS=10)
    exec(compile(module, 'executor_methods', 'exec'), namespace)
    h = namespace['Harness']()
    h.queue_lock = threading.Lock()
    h.action_queue = deque([[.06]*6+[1.]])
    h.raw_action_queue = deque([[.05]*6+[1.]])
    h.last_action = h.last_raw_action = None
    h.trajectory_offset = SimpleNamespace(plan_actions=transform)
    h.grasp_assist = None
    h.approach_schedule = None
    h.carry_lift = None
    h.task = 'test'
    h.offset_fault = False
    h.execution_started = True
    h.inference_generation = 0
    h.actions_per_prediction = 50
    h.blend_actions = 10
    h.inference_running = True
    h.measured_joints = [.06]*6
    h.measured_joints_at = time.monotonic()
    h.get_logger = lambda: SimpleNamespace(info=lambda msg: None, error=lambda msg: None)
    sent = []
    h.publisher = SimpleNamespace(publish=lambda msg: sent.append(msg.data))
    h.infer([0]*7, [], 0)
    assert not h.offset_fault and len(h.action_queue) == len(h.raw_action_queue) == 51
    weight = .1*.1*(3-2*.1)
    assert calls[0][0][0] == pytest.approx(.05+weight*(.1-.05))
    for _ in range(51):
        h.measured_joints_at = time.monotonic()
        h.publish_action()
    assert [a[6] for a in sent[1:]] == [a[6] for a in chunk]
    assert sent[-1][0] == pytest.approx(.11)
    assert not h.raw_action_queue
    h.inference_running = True
    h.infer([0]*7, [], 0)
    # The next blend begins at 0.1 raw, never the already corrected 0.11.
    assert calls[1][0][0] == pytest.approx(.1)


def test_moveit_client_waits_for_discovery_before_request():
    from grasp_offset import OffsetPlanner
    calls = []
    future = SimpleNamespace(done=lambda: True, cancelled=lambda: False, result=lambda: 'response')
    client = SimpleNamespace(
        srv_name='/move_group/get_parameters',
        service_is_ready=lambda: False,
        wait_for_service=lambda timeout_sec: calls.append(('wait', timeout_sec)) or True,
        call_async=lambda request: calls.append(('request', request)) or future,
    )
    assert OffsetPlanner.call(client, 'request', timeout=5.) == 'response'
    assert calls == [('wait', 5.), ('request', 'request')]


def test_missing_moveit_service_names_fault_and_sends_no_request():
    from grasp_offset import OffsetPlanner
    calls = []
    client = SimpleNamespace(
        srv_name='/compute_fk',
        service_is_ready=lambda: False,
        wait_for_service=lambda timeout_sec: False,
        call_async=lambda request: calls.append(request),
    )
    with pytest.raises(RuntimeError, match='/compute_fk'):
        OffsetPlanner.call(client, 'request')
    assert not calls


def test_planning_read_only_retry_cancels_lost_response(monkeypatch):
    import grasp_offset
    calls, cancelled = [], []
    def request(_):
        calls.append(True)
        finished = len(calls)==2
        return SimpleNamespace(done=lambda:finished, cancelled=lambda:False,
                               cancel=lambda:cancelled.append(True), result=lambda:'ok')
    client=SimpleNamespace(service_is_ready=lambda:True,call_async=request,srv_name='/move_group/get_parameters')
    assert grasp_offset.OffsetPlanner.call(client,object(),timeout=.01,attempts=2)=='ok'
    assert len(calls)==2 and len(cancelled)==1


def test_ready_moveit_result_does_not_depend_on_scheduled_done_callback():
    from grasp_offset import OffsetPlanner
    def forbidden_callback(_):
        raise AssertionError('Do not wait on a second executor callback')
    future=SimpleNamespace(done=lambda:True,cancelled=lambda:False,
                           add_done_callback=forbidden_callback,result=lambda:'valid response')
    client=SimpleNamespace(service_is_ready=lambda:True,call_async=lambda _:future)
    assert OffsetPlanner.call(client,object())=='valid response'


def test_lift_observation_inverse_prevents_repeated_height_addition():
    axes=['0 0 1','0 1 0','0 1 0','1 0 0','0 1 0','1 0 0']
    xml=['<robot name="spatial"><link name="base"/>'];previous='base'
    for i,axis in enumerate(axes):
        link=f'link_{i}'
        xml.append(f'<link name="{link}"/><joint name="q{i}" type="revolute">'
                   f'<parent link="{previous}"/><child link="{link}"/>'
                   f'<origin xyz="0.15 0 0.03"/><axis xyz="{axis}"/>'
                   '<limit lower="-3" upper="3" effort="1" velocity="1"/></joint>')
        previous=link
    xml.append(f'<link name="tool0"/><joint name="tip" type="fixed"><parent link="{previous}"/>'
               '<child link="tool0"/><origin xyz="0.1 0 0"/></joint></robot>')
    m=CartesianOffsetModel(''.join(xml),[f'q{i}' for i in range(6)])
    raw=[.5,-1.2,1.5,.8,-.9,.2];base,_=m.fk_jacobian(raw)
    lifted=m.translate(raw,[0.,0.,.025],carry_height=.025)
    for _ in range(8):
        virtual=m.translate(lifted,[0.,0.,-.025],carry_height=-.025)
        lifted=m.translate(virtual,[0.,0.,.025],carry_height=.025)
        actual,_=m.fk_jacobian(lifted)
        assert np.linalg.norm(actual[:3,3]-base[:3,3]-[0.,0.,.025]) < .0001
