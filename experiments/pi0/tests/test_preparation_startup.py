"""Cold startup must wait for hardware feedback and never invent gripper ACKs."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('isolated_pi0_readiness',
    Path(__file__).resolve().parents[1]/'scripts/engine_readiness.py')
readiness=importlib.util.module_from_spec(spec)
spec.loader.exec_module(readiness)


def status(**changes):
    value=dict(fault=None, fault_kind=None, active_target=False, policy_publishers=0,
        gripper_control='policy', teleop_mapper_running=False, fresh_joints=True,
        servo_stopped=True, gripper_busy=False, confirmed_gripper=None)
    value.update(changes)
    return value


class PreparationStartupTests(unittest.TestCase):
    def scenario(self, states, *, accepted=True, expect_error=None):
        import json
        clock=[0.]
        events=[]
        current=[None]
        class Client:
            def __init__(self,name): self.name=name
            def wait_for_service(self,timeout_sec):return True
            def call_async(self,request):
                if self.name.endswith('/status'):
                    if states:current[0]=states.pop(0)
                    response=SimpleNamespace(success=True,message=json.dumps(current[0]))
                else:
                    events.append(dict(current[0]))
                    response=SimpleNamespace(success=accepted,message='opening requested' if accepted else 'Reset requires Servo stop acknowledgment and fresh measured joints')
                return SimpleNamespace(done=lambda:True,result=lambda:response,cancel=lambda:None)
        class Node:
            def __init__(self,*args,**kwargs):pass
            def create_client(self,typ,name):return Client(name)
            def destroy_node(self):pass
        def tick(*args,**kwargs):clock[0]+=.1
        modules={
            'rclpy':SimpleNamespace(init=lambda **kw:None,shutdown=lambda **kw:None,
                spin_until_future_complete=lambda *a,**kw:None,spin_once=tick),
            'rclpy.node':SimpleNamespace(Node=Node),
            'rclpy.context':SimpleNamespace(Context=object),
            'rclpy.executors':SimpleNamespace(SingleThreadedExecutor=lambda **kw:SimpleNamespace(shutdown=lambda:None)),
            'std_srvs':SimpleNamespace(),
            'std_srvs.srv':SimpleNamespace(Trigger=SimpleNamespace(Request=object)),
        }
        with patch.dict(sys.modules,modules),patch.object(readiness,'time',SimpleNamespace(monotonic=lambda:clock[0])):
            if expect_error:
                with self.assertRaisesRegex(SystemExit,expect_error):
                    readiness.require_ready_engine(prepare_episode=True)
            else:
                readiness.require_ready_engine(prepare_episode=True)
        return events

    def test_cold_start_waits_then_opens_once_then_waits_for_ack(self):
        events=self.scenario([
            status(servo_stopped=False,fresh_joints=False),
            status(servo_stopped=True,fresh_joints=False),
            status(),status(gripper_busy=True),status(confirmed_gripper=1.),
        ])
        self.assertEqual(len(events),1)
        self.assertTrue(events[0]['servo_stopped'] and events[0]['fresh_joints'])

    def test_failed_prepare_response_is_not_treated_as_success(self):
        events=self.scenario([status()],accepted=False,
            expect_error='Episode preparation failed: Reset requires')
        self.assertEqual(len(events),1)

    def test_missing_io_ack_times_out_without_repeating_open(self):
        events=self.scenario([status(),status(gripper_busy=True)],expect_error='not ready')
        self.assertEqual(len(events),1)


if __name__=='__main__':unittest.main()
