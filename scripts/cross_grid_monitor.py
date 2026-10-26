#!/usr/bin/python3
"""Read-only evaluation capture, controlled by JSON files rather than robot topics."""
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rosidl_runtime_py.utilities import get_message
from rosidl_runtime_py.convert import message_to_ordereddict
from sensor_msgs.msg import Image, JointState, CameraInfo
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger
from rcl_interfaces.msg import Log
from rcl_interfaces.srv import GetParameters
from controller_manager_msgs.srv import ListControllers
from tf2_ros import Buffer, TransformListener

from cross_grid_results import TrialStore, utc, write_json
from project_paths import ROOT

ORDER=['shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint']
NUMERIC_TOPICS=['/gripper/state','/gripper/command','/control_engine/gripper_state',
 '/servo_node/status','/servo_node/delta_twist_cmds','/joy',
 '/speed_scaling_state_broadcaster/speed_scaling','/io_and_status_controller/robot_program_running',
 '/io_and_status_controller/robot_mode','/io_and_status_controller/safety_mode',
 '/io_and_status_controller/io_states']


def plain(value):
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(x) for x in value]
    if isinstance(value,float) and not math.isfinite(value):return str(value)
    if isinstance(value,np.generic):return plain(value.item())
    return value


def execution_snapshot():
    result={'captured_utc':utc(),'processes':[],'code_sha256':{}}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            args=proc.joinpath('cmdline').read_bytes().split(b'\0')
            if not any(Path(a.decode(errors='replace')).name in ('policy_executor.py','control_engine.py','control_engine_phased.py','policy_executor_phased.py') for a in args if a):continue
            result['processes'].append({'pid':int(proc.name),'arguments':[a.decode(errors='replace') for a in args if a]})
        except (OSError,ValueError):pass
    for name in ('control_engine.py','policy_executor.py','trajectory_offset.py','grasp_offset.py','execution_profile.py','engine_readiness.py','approach_offset.py','planned_transport.py','planning_service_node.py'):
        path=ROOT/'scripts'/name
        result['code_sha256'][name]=hashlib.sha256(path.read_bytes()).hexdigest()
    path=ROOT/'data/models/evaluation/codex test analysis/offset_calibration.json'
    if path.exists():result['offset_configuration']=json.loads(path.read_text())
    return result


class CrossGridMonitor(Node):
    def __init__(self,root):
        super().__init__('codex_readonly_cross_grid_monitor')
        self.store=TrialStore(root);self.root=self.store.root
        self.bridge=CvBridge();self.latest={};self.images={};self.depth={};self.intrinsics={}
        self.active=None;self.running=True;self.processed=set();self.subscriptions_kept=[]
        self.source_counts=collections.Counter();self.last_io=None
        self.buffer=Buffer();self.listener=TransformListener(self.buffer,self)
        self.model=None;self.parameter_future=None;self.parameter_started=0.
        self.parameter_client=self.create_client(GetParameters,'/move_group/get_parameters')
        self.engine_client=self.create_client(Trigger,'/control_engine/status');self.engine_future=None;self.engine_started=0.
        self.controllers_client=self.create_client(ListControllers,"/controller_manager/list_controllers");self.controllers_future=None
        self.subscribe('/joint_states',JointState)
        self.subscribe('/control_abstraction_layer',Float64MultiArray)
        self.subscribe('/forward_position_controller/commands',Float64MultiArray)
        self.subscribe('/rosout',Log)
        for camera in ('base','wrist'):
            color=f'/{camera}/camera/color/image_raw'
            self.subscriptions_kept.append(self.create_subscription(Image,color,lambda m,c=camera:self.color(c,m),qos_profile_sensor_data))
            depth=f'/{camera}/camera/aligned_depth_to_color/image_raw'
            self.subscriptions_kept.append(self.create_subscription(Image,depth,lambda m,c=camera:self.depth_image(c,m),qos_profile_sensor_data))
            self.subscriptions_kept.append(self.create_subscription(CameraInfo,f'/{camera}/camera/color/camera_info',lambda m,c=camera:self.camera_info(c,m),qos_profile_sensor_data))
        self.optional=set();self.discover()
        self.create_timer(.1,self.tick)
        self.create_timer(2.,self.health)
        self.health()
        print('MONITOR READY: read-only; waiting for cross/attempt start',flush=True)

    def subscribe(self,topic,kind,transient=False):
        qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL) if transient else qos_profile_sensor_data
        self.subscriptions_kept.append(self.create_subscription(kind,topic,lambda msg,t=topic:self.numeric(t,msg),qos))

    def discover(self):
        topics=dict(self.get_topic_names_and_types())
        for topic in NUMERIC_TOPICS:
            if topic not in topics or topic in self.optional:continue
            try:
                kind=get_message(topics[topic][0]);self.subscribe(topic,kind,topic.endswith('/robot_program_running'));self.optional.add(topic)
            except Exception as error:self.latest[f'unsupported:{topic}']=str(error)

    def emit(self,topic,value,**extra):
        if not self.active:return
        now=time.monotonic();a=self.active
        event={'t_s':now-a['start'],'receive_epoch_s':time.time(),'topic':topic,'value':plain(value),**extra}
        a['file'].write(json.dumps(plain(event),separators=(',',':'),allow_nan=False)+'\n')

    def numeric(self,topic,msg):
        now=time.monotonic();self.source_counts[topic]+=1
        if isinstance(msg,Log):
            if msg.level<30 and msg.name not in ('control_engine','bypass_policy_executor','servo_node','move_group'):return
            value={'node':msg.name,'level':msg.level,'message':msg.msg,'stamp_s':msg.stamp.sec+msg.stamp.nanosec/1e9}
        elif isinstance(msg,JointState):
            try:q=[float(msg.position[msg.name.index(n)]) for n in ORDER]
            except (ValueError,IndexError):return
            value={'name':list(msg.name),'position':list(msg.position),'velocity':list(msg.velocity),'effort':list(msg.effort),'stamp_s':msg.header.stamp.sec+msg.header.stamp.nanosec/1e9}
            self.latest['ordered_joints']={'q':q,'received_monotonic':now}
        elif isinstance(msg,Float64MultiArray):value=list(msg.data)
        else:value=message_to_ordereddict(msg)
        self.latest[topic]={'value':plain(value),'received_monotonic':now}
        if not self.active:return
        a=self.active;a['counts'][topic]+=1;a['first'].setdefault(topic,now);a['last'][topic]=now
        if topic=='/io_and_status_controller/io_states':
            pins={str(x.pin):bool(x.state) for x in msg.digital_out_states}
            if pins==a['last_io'] and now-a['last_io_event']<1:return
            if pins!=a['last_io']:a['io_transitions'].append({'t_s':now-a['start'],'pins':pins})
            a['last_io']=pins;a['last_io_event']=now
        if topic=='/control_abstraction_layer' and (len(value)!=7 or not all(math.isfinite(float(x)) for x in value)):
            a['invalid_actions']+=1
        if topic=='/control_abstraction_layer' and len(value)==7:
            if a['first_policy_s'] is None:a['first_policy_s']=now-a['start']
            if a['last_policy'] is not None:a['max_policy_jump']=max(a['max_policy_jump'],max(abs(x-y) for x,y in zip(value[:6],a['last_policy'][:6])))
            if a['last_policy'] is not None and (value[6]>=.5)!=(a['last_policy'][6]>=.5):a['gripper_transitions'].append({'t_s':now-a['start'],'from':a['last_policy'][6],'to':value[6]})
            a['last_policy']=value
        if topic=='/forward_position_controller/commands' and a['first_arm_s'] is None:a['first_arm_s']=now-a['start']
        if topic in ('/control_abstraction_layer','/forward_position_controller/commands') and len(value)>=6:
            joint=self.latest.get('ordered_joints')
            if joint and now-joint['received_monotonic']<.1:
                delta=[float(x)-y for x,y in zip(value[:6],joint['q'])]
                key='policy_gap' if topic=='/control_abstraction_layer' else 'tracking_gap'
                a[key]=[max(old,abs(new)) for old,new in zip(a[key],delta)]
                self.emit(topic,value,joint_feedback_age_s=now-joint['received_monotonic'],target_minus_measured_rad=delta)
                return
        if isinstance(msg,Log) and msg.level>=30:
            a['fault_count']+=1
            if len(a['notable_logs'])<200:a['notable_logs'].append({'t_s':now-a['start'],**value})
        self.emit(topic,value)

    def color(self,camera,msg):
        self.source_counts[f'{camera}_color']+=1;self.images[camera]=msg

    def depth_image(self,camera,msg):
        self.source_counts[f'{camera}_depth']+=1;self.depth[camera]=msg

    def camera_info(self,camera,msg):
        self.intrinsics[camera]=plain(message_to_ordereddict(msg))

    def graph(self):
        return {topic:[{'node':x.node_name,'namespace':x.node_namespace} for x in self.get_publishers_info_by_topic(topic)] for topic in ('/control_abstraction_layer','/forward_position_controller/commands','/servo_node/delta_twist_cmds','/control_engine/gripper_state')}

    def start(self,cross,attempt):
        folder,trial=self.store.start(cross,attempt)
        self.active={'folder':folder,'source_counts_start':dict(self.source_counts),'start':time.monotonic(),'file':(folder/'observations.jsonl').open('w'),
            'counts':collections.Counter(),'first':{},'last':{},'policy_gap':[0.]*6,'tracking_gap':[0.]*6,
            'max_policy_jump':0.,'invalid_actions':0,'last_policy':None,'last_io':None,'last_io_event':0.,'gripper_transitions':[],
            'io_transitions':[],'notable_logs':[],'fault_count':0,'videos':{},'video_frames':collections.Counter(),
            'last_snapshot':{},'last_depth':{},'camera_age_max':{},'timeout_noted':False,
            'first_policy_s':None,'first_arm_s':None,'last_tcp':None,'tcp_path_m':0.,'tcp_samples':0}
        write_json(folder/'configuration/execution_start.json',execution_snapshot())
        revision = self.root/'active_revision.json'
        if revision.exists():
            shutil.copy2(revision, folder/'configuration/evaluation_revision.json')
        for name in execution_snapshot()['code_sha256']:
            shutil.copy2(ROOT/'scripts'/name,folder/'configuration'/name)
        write_json(folder/'configuration/camera_intrinsics.json',self.intrinsics)
        write_json(folder/'configuration/graph_start.json',self.graph())
        write_json(folder/'configuration/tf_frames_start.json',{'frames_yaml':self.buffer.all_frames_as_yaml()})
        if self.model is not None:write_json(folder/'configuration/kinematic_model.json',{'frame':'base','source':'live MoveIt robot_description','purpose':'FK target comparison, not cube position'})
        print(f'CROSS {cross:02d} ATTEMPT {attempt:02d} RECORDING',flush=True)

    def capture(self):
        if not self.active:return
        a=self.active;now=time.monotonic();elapsed=now-a['start']
        for camera,msg in self.images.items():
            try:
                bgr=self.bridge.imgmsg_to_cv2(msg,'bgr8')
                if camera not in a['videos']:
                    path=a['folder']/'video'/f'{camera}.avi'
                    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10.,(msg.width,msg.height))
                    if not writer.isOpened():raise RuntimeError('MJPG video writer unavailable')
                    a['videos'][camera]=writer
                a['videos'][camera].write(bgr);a['video_frames'][camera]+=1
                stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
                age=self.get_clock().now().nanoseconds/1e9-stamp
                a['camera_age_max'][camera]=max(age,a['camera_age_max'].get(camera,age))
                self.emit(f'camera/{camera}/video_frame',{'index':a['video_frames'][camera]-1,'source_stamp_s':stamp,'source_age_s':age})
                if now-a['last_snapshot'].get(camera,0)>=1:
                    filename=f'{camera}_{elapsed:09.3f}.jpg';cv2.imwrite(str(a['folder']/'camera_snapshots'/filename),bgr)
                    self.emit(f'camera/{camera}/snapshot',{'file':filename,'source_stamp_s':stamp});a['last_snapshot'][camera]=now
            except Exception as error:self.emit(f'camera/{camera}/error',str(error))
        for camera,msg in self.depth.items():
            if now-a['last_depth'].get(camera,0)<1:continue
            try:
                depth=self.bridge.imgmsg_to_cv2(msg,'passthrough')
                filename=f'{camera}_{elapsed:09.3f}.npz'
                np.savez_compressed(a['folder']/'depth'/filename,depth=depth)
                self.emit(f'depth/{camera}',{'file':filename,'encoding':msg.encoding,'frame':msg.header.frame_id,'source_stamp_s':msg.header.stamp.sec+msg.header.stamp.nanosec/1e9})
                a['last_depth'][camera]=now
            except Exception as error:self.emit(f'depth/{camera}/error',str(error));a['last_depth'][camera]=now
        try:
            tf=self.buffer.lookup_transform('base','tool0_controller',rclpy.time.Time())
            t,q=tf.transform.translation,tf.transform.rotation;xyz=[t.x,t.y,t.z]
            stamp=tf.header.stamp.sec+tf.header.stamp.nanosec/1e9
            event={'frame':'base','source':'tool0_controller','xyz_m':xyz,'quaternion_xyzw':[q.x,q.y,q.z,q.w],'stamp_s':stamp,'age_s':self.get_clock().now().nanoseconds/1e9-stamp}
            if event['age_s']<.2:
                if a['last_tcp'] is not None:a['tcp_path_m']+=math.sqrt(sum((x-y)**2 for x,y in zip(xyz,a['last_tcp'])))
                a['last_tcp']=xyz;a['tcp_samples']+=1
            if self.model is not None:
                for topic,key in (('/control_abstraction_layer','policy_fk'),('/forward_position_controller/commands','command_fk')):
                    latest=self.latest.get(topic)
                    if latest and len(latest['value'])>=6:
                        pose,_=self.model.fk_jacobian(latest['value'][:6]);predicted=pose[:3,3].tolist()
                        event[key+'_xyz_m']=predicted
                        event[key+'_vs_tcp_m']=math.sqrt(sum((x-y)**2 for x,y in zip(predicted,xyz)))
                        event[key+'_age_s']=now-latest['received_monotonic']
            self.emit('tcp_pose',event)
        except Exception as error:self.emit('tcp_pose_unavailable',str(error))

    def finish(self,outcome='unknown',notes='',**stages):
        if not self.active:raise ValueError('No active attempt')
        if outcome not in ('pass','partial','fail','aborted','unknown'):raise ValueError('Invalid outcome; recording continues')
        a=self.active;duration=time.monotonic()-a['start']
        for writer in a['videos'].values():writer.release()
        a['file'].flush();a['file'].close()
        rates={k:(a['counts'][k]-1)/(a['last'][k]-a['first'][k]) if a['counts'][k]>1 and a['last'][k]>a['first'][k] else 0. for k in a['counts']}
        summary={'native_message_counts':{k:n-a['source_counts_start'].get(k,0) for k,n in self.source_counts.items()},'duration_s':duration,'first_policy_command_s':a['first_policy_s'],'first_arm_command_s':a['first_arm_s'],
            'message_counts':dict(a['counts']),'rates_hz':rates,'max_policy_measured_gap_rad':a['policy_gap'],
            'max_command_measured_gap_rad':a['tracking_gap'],'max_consecutive_policy_jump_rad':a['max_policy_jump'],
            'invalid_action_count':a['invalid_actions'],'gripper_prediction_transitions':a['gripper_transitions'],'digital_output_transitions':a['io_transitions'],
            'warnings_errors':a['fault_count'],'notable_logs':a['notable_logs'],'video_frames':dict(a['video_frames']),
            'max_color_source_age_s':a['camera_age_max'],'tcp_path_length_m':a['tcp_path_m'],'tcp_samples':a['tcp_samples'],
            'measurement_limits':['IO states do not prove object contact','TCP error is not grasp/cube error','Video timestamps in observations.jsonl are authoritative; AVI playback uses nominal 10 FPS','TCP path length includes sensor noise','Cross positions are photo coordinates until separately calibrated']}
        write_json(a['folder']/'configuration/execution_end.json',execution_snapshot())
        write_json(a['folder']/'configuration/graph_end.json',self.graph())
        trial=self.store.finish(outcome,summary,notes,**stages);self.active=None
        print(f"SAVED cross {trial['cross']:02d} attempt {trial['attempt']:02d}: {outcome}",flush=True)

    def tick(self):
        for path in sorted((self.root/'commands').glob('*.json')):
            if path.name in self.processed:continue
            self.processed.add(path.name)
            try:
                cmd=json.loads(path.read_text())
                if cmd['action']=='start':self.start(int(cmd['cross']),int(cmd['attempt']))
                elif cmd['action']=='finish':self.finish(cmd.get('outcome','unknown'),cmd.get('notes',''),**{k:cmd.get(k) for k in ('pickup','carry','placement')})
                elif cmd['action']=='shutdown':self.running=False
                else:raise ValueError('Unknown monitor command')
            except Exception as error:
                write_json(self.root/f'command_error_{path.stem}.json',{'error':str(error),'command':path.name,'utc':utc()});print('COMMAND ERROR:',error,flush=True)
        if self.active:
            limit=self.store.active[1]['timeout_s']
            if limit is not None and time.monotonic()-self.active['start']>=limit and not self.active['timeout_noted']:
                self.active['timeout_noted']=True;self.emit('time_limit',{'seconds':limit,'action':'recording continues; no robot command sent'})
                print('TIME LIMIT REACHED: recording continues until operator finishes',flush=True)
            self.capture()

    def health(self):
        self.discover()
        if self.parameter_future is None and self.model is None and self.parameter_client.service_is_ready():
            request=GetParameters.Request();request.names=['robot_description'];self.parameter_started=time.monotonic();self.parameter_future=self.parameter_client.call_async(request)
        if self.parameter_future is not None:
            if self.parameter_future.done():
                try:
                    from trajectory_offset import CartesianOffsetModel
                    result=self.parameter_future.result();xml=result.values[0].string_value
                    self.model=CartesianOffsetModel(xml,ORDER,'base')
                    write_json(self.root/'configuration/moveit_model.json',{'bytes':len(xml),'sha256':hashlib.sha256(xml.encode()).hexdigest()})
                except Exception as error:self.latest['kinematic_model_error']=str(error)
                self.parameter_future=None
            elif time.monotonic()-self.parameter_started>5:
                self.parameter_future.cancel();self.parameter_future=None
        if self.engine_future is None and self.engine_client.service_is_ready():
            self.engine_future=self.engine_client.call_async(Trigger.Request());self.engine_started=time.monotonic()
        elif self.engine_future is not None and self.engine_future.done():
            try:
                self.latest['engine_status']=json.loads(self.engine_future.result().message);self.emit('engine_status',self.latest['engine_status'])
            except Exception as error:self.latest['engine_status_error']=str(error)
            self.engine_future=None
        elif self.engine_future is not None and time.monotonic()-self.engine_started>5:
            self.engine_future.cancel();self.engine_future=None;self.latest['engine_status_error']='status request timeout'
        if self.controllers_future is None and self.controllers_client.service_is_ready():
            self.controllers_future=self.controllers_client.call_async(ListControllers.Request())
        elif self.controllers_future is not None and self.controllers_future.done():
            try:
                info={'captured_utc':utc(),'controllers':[{'name':c.name,'type':c.type,'state':c.state} for c in self.controllers_future.result().controller], 'topic_types':self.get_topic_names_and_types()}
                write_json(self.root/'configuration/hardware_inventory.json',info)
            except Exception as error:self.latest['controller_inventory_error']=str(error)
            self.controllers_future=None
        if self.active:self.active['file'].flush()
        folder,trial=self.store.active if self.store.active else (None,None)
        status={'state':'recording' if self.active else 'waiting','pid':os.getpid(),'updated_utc':utc(),
            'cross':trial['cross'] if trial else None,'attempt':trial['attempt'] if trial else None,
            'elapsed_s':time.monotonic()-self.active['start'] if self.active else None,
            'topics_seen':dict(self.source_counts),'color_cameras_seen':list(self.images),'depth_cameras_seen':list(self.depth),
            'engine_status':self.latest.get('engine_status'),'kinematic_model_ready':self.model is not None,
            'read_only':True,'command_publishers_created':0}
        write_json(self.root/'status.json',status)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--session',type=Path,required=True);args=parser.parse_args()
    rclpy.init();node=CrossGridMonitor(args.session)
    def stop(signum,frame):node.running=False
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        while rclpy.ok() and node.running:rclpy.spin_once(node,timeout_sec=.05)
    finally:
        if node.active:node.finish('aborted','Monitor stopped before operator outcome')
        write_json(node.root/'status.json',{'state':'stopped','updated_utc':utc(),'pid':os.getpid()})
        node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
