"""Standalone standard-pose shutdown regressions; never command a robot."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


ROOT=Path(__file__).resolve().parents[4]


def test_final_command_cancels_timer_without_shutting_down_inside_callback():
    tree=ast.parse((ROOT/'scripts/standard_pose.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='update')
    module=ast.Module(body=[method],type_ignores=[]);ast.fix_missing_locations(module)
    shutdown=Mock(side_effect=AssertionError('shutdown called inside callback'))
    namespace=dict(time=SimpleNamespace(monotonic=lambda:20.),STANDARD_POSE=(1.,)*6,
                   Float64MultiArray=SimpleNamespace,rclpy=SimpleNamespace(shutdown=shutdown))
    exec(compile(module,'standard_pose_update','exec'),namespace)
    node=SimpleNamespace(start_joints=[0.]*6,started_at=0.,duration=20.,completed=False,
                         publisher=Mock(),timer=Mock(),get_logger=lambda:Mock())
    namespace['update'](node)
    assert node.completed
    assert node.publisher.publish.call_args.args[0].data==[1.]*6
    node.timer.cancel.assert_called_once()
    shutdown.assert_not_called()
    namespace['update'](node)
    node.publisher.publish.assert_called_once()


def test_main_exits_and_shuts_down_after_spin_once_returns(monkeypatch):
    import argparse
    import sys
    monkeypatch.setattr(sys,'argv',['standard_pose.py'])
    tree=ast.parse((ROOT/'scripts/standard_pose.py').read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    module=ast.Module(body=[main],type_ignores=[]);ast.fix_missing_locations(module)
    events=[]
    node=SimpleNamespace(completed=False,destroy_node=lambda:events.append('destroy'))
    def spin_once(n,timeout_sec):
        events.append('spin')
        n.completed=True
    namespace=dict(argparse=argparse,StandardPose=lambda duration:node,
        rclpy=SimpleNamespace(init=lambda:None,ok=lambda:True,spin_once=spin_once,
                             shutdown=lambda:events.append('shutdown')))
    exec(compile(module,'standard_pose_main','exec'),namespace)
    namespace['main']()
    assert events==['spin','destroy','shutdown']
