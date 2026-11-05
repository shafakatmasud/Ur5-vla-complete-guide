"""Exercise Pi0 filtering and real control callbacks without robot outputs."""
import ast
import json
import math
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from motion_smoothing import smooth_motion_step, smooth_action_chunk, InvalidPolicyTargets
from execution_profile import StableGripperCommand
from bowl_release import BowlRelease


def harness():
    tree=ast.parse((ROOT/'scripts/control_engine.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ControlEngine')
    cls.bases=[]
    cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name!='__init__']
    module=ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[]))
    clock=[10.];writes=[];arm=[]
    names=dict(json=json,math=math,time=SimpleNamespace(monotonic=lambda:clock[0]),
        smooth_motion_step=smooth_motion_step,JOINT_TIMEOUT=.20,COMMAND_TIMEOUT=.50,
        MAX_INITIAL_ERROR=.10,MAX_TARGET_STEP=.15,MAX_TRACKING_ERROR=.05,
        Float64MultiArray=SimpleNamespace,Float64=SimpleNamespace,
        SetIO=SimpleNamespace(Request=SimpleNamespace),rclpy=SimpleNamespace(time=SimpleNamespace(Time=lambda:None)))
    exec(compile(module,'pi0_control_callbacks','exec'),names)
    h=names['ControlEngine']()
    h.fault=h.fault_kind=None;h.episode_completed=False;h.servo_stopped=True
    h.target=h.last_raw_target=None;h.command=h.filtered_target=[0.]*6;h.velocity=[0.]*6
    h.measured=[0.]*6;h.measured_at=clock[0];h.last_command_at=0.;h.last_tick=clock[0]
    h.startup_hold=False;h.gripper_busy=h.dry_run=False
    h.last_gripper=h.confirmed_gripper=1.;h.gripper_filter=StableGripperCommand(stable_s=.3)
    h.bowl_release=BowlRelease(ROOT/'data/models/evaluation/calibration/bowl_release_calibration.json')
    h.last_bowl_diagnostics=h.last_motion_diagnostics=0.;h.output_count=0
    h.max_velocity=.04;h.max_acceleration=.10;h.filter_time_constant=.8
    h.count_publishers=lambda topic:1;h.get_node_names=lambda:['control_engine']
    h.get_logger=lambda:SimpleNamespace(info=lambda msg:None,error=lambda msg:None,warning=lambda msg:None)
    h.publish_gripper_confirmation=lambda:None
    h.arm_publisher=SimpleNamespace(publish=lambda msg:arm.append(msg.data))
    class Future:
        def result(self):return SimpleNamespace(success=True)
        def add_done_callback(self,fn):fn(self)
    def io(request):writes.append((request.pin,request.state));return Future()
    h.io_client=SimpleNamespace(service_is_ready=lambda:True,call_async=io)
    return h,clock,writes,arm


def tick(h,clock,gripper):
    h.measured_at=clock[0]
    h.command_callback(SimpleNamespace(data=[.01]*6+[gripper]))
    clock[0]+=.1


class MotionTests(unittest.TestCase):
    def test_arm_average_preserves_gripper_and_rejects_raw_jumps(self):
        raw=[[.005*(-1)**i]*6+[i%2] for i in range(50)]
        smooth=smooth_action_chunk(raw,[0.]*6)
        self.assertEqual([a[6] for a in raw],[a[6] for a in smooth])
        self.assertLess(statistics.pstdev(a[0] for a in smooth),statistics.pstdev(a[0] for a in raw)/2)
        for bad in ([[.11]*6+[1.]]*50, [[0.]*6+[1.]]+[[.16]*6+[1.]]*49,
                    [[float('nan')]*6+[1.]]*50):
            with self.assertRaises(InvalidPolicyTargets):smooth_action_chunk(bad,[0.]*6)

    def test_speed_acceleration_and_sustained_reversal(self):
        q=v=f=0.;dt=.008
        for k in range(5000):
            target=.20 if k<2500 else -.20
            fs,qs,vs=smooth_motion_step([q],[v],[f],[target],dt,.04,.10,.8)
            self.assertLessEqual(abs(vs[0]),.04+1e-12)
            self.assertLessEqual(abs(vs[0]-v)/dt,.10+1e-12)
            f,q,v=fs[0],qs[0],vs[0]
            if k==2499:self.assertGreater(q,.19)
        self.assertLess(q,-.19)

    def test_stationary_pose_jitter_is_suppressed(self):
        q=v=f=0.;positions=[];dt=.008
        for k in range(2500):
            fs,qs,vs=smooth_motion_step([q],[v],[f],[.04+.008*math.sin(2*math.pi*2*k*dt)],dt,.04,.10,.8)
            f,q,v=fs[0],qs[0],vs[0];positions.append(q)
        self.assertLess(statistics.pstdev(positions[1250:]),.0001)
        self.assertLess(abs(q-.04),.002)


class GripperTests(unittest.TestCase):
    def test_single_action_close_is_sent_and_following_open_requests_are_held(self):
        h,clock,writes,_=harness()
        # The reported run had only 1 CLOSE action in its selected 20-action chunk.
        tick(h,clock,1.)
        tick(h,clock,0.)
        self.assertEqual(writes,[(1,0.),(0,1.)])
        self.assertEqual(h.confirmed_gripper,0.)
        for _ in range(30):tick(h,clock,1.)
        self.assertEqual(writes,[(1,0.),(0,1.)])

    def test_repeated_policy_open_cannot_drop_after_closed_ack_even_without_tf(self):
        h,clock,writes,_=harness()
        for _ in range(5):tick(h,clock,0.)
        self.assertEqual(writes,[(1,0.),(0,1.)])
        self.assertTrue(h.bowl_release.transport_hold)
        h.release_tf=SimpleNamespace(lookup_transform=lambda *a:(_ for _ in ()).throw(RuntimeError('TF missing')))
        h.update_bowl_release(clock[0])
        for value in [1.]*15+[0.]*5+[1.]*20+[0.]*5:
            tick(h,clock,value)
        self.assertEqual(writes,[(1,0.),(0,1.)])

    def test_bowl_arrival_releases_once_then_ignores_arm_and_gripper_retries(self):
        h,clock,writes,arm=harness()
        for _ in range(5):tick(h,clock,0.)
        xyz=[h.bowl_release.position[0]-.1,*h.bowl_release.position[1:]]
        def transform(*args):
            stamp=clock[0];sec=int(stamp)
            return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=sec,nanosec=int((stamp-sec)*1e9))),
                transform=SimpleNamespace(translation=SimpleNamespace(x=xyz[0],y=xyz[1],z=xyz[2]),
                    rotation=SimpleNamespace(**dict(zip(['x','y','z','w'],h.bowl_release.orientation)))))
        h.release_tf=SimpleNamespace(lookup_transform=transform)
        h.get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=int(clock[0]*1e9)))
        for _ in range(10):h.update_bowl_release(clock[0]);clock[0]+=.1
        xyz[:]=h.bowl_release.position
        for _ in range(10):h.update_bowl_release(clock[0]);clock[0]+=.1
        self.assertEqual(writes,[(1,0.),(0,1.),(0,0.),(1,1.)])
        self.assertTrue(h.episode_completed)
        self.assertIsNone(h.target)
        for _ in range(20):tick(h,clock,0.);h.publish_smoothed_arm()
        self.assertEqual(len(writes),4);self.assertEqual(arm,[])
        h.count_publishers=lambda topic:0;h.measured_at=clock[0]
        result=h.reset_callback(None,SimpleNamespace())
        self.assertTrue(result.success)
        self.assertFalse(h.episode_completed or h.bowl_release.transport_hold or h.bowl_release.released)
        self.assertEqual(len(writes),4)

    def test_raw_arm_guard_still_blocks_gripper_io(self):
        h,clock,writes,arm=harness()
        h.command_callback(SimpleNamespace(data=[.11]*6+[0.]))
        h.publish_smoothed_arm()
        self.assertTrue(h.fault);self.assertEqual(writes,[]);self.assertEqual(arm,[])

    def test_tracking_and_feedback_watchdogs_remain_active(self):
        for fault in ['stale','tracking','timeout']:
            h,clock,writes,arm=harness()
            tick(h,clock,1.)
            if fault=='stale':clock[0]+=.21
            elif fault=='tracking':h.measured=[.06]*6;h.measured_at=clock[0]
            else:clock[0]+=.51;h.measured_at=clock[0]
            h.publish_smoothed_arm()
            self.assertTrue(h.fault,fault);self.assertEqual(arm,[])


if __name__=='__main__':unittest.main()
