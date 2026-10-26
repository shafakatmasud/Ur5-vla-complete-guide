"""Opt-in planned bowl transport after policy closure; never publishes commands.

Gripper IO is not object-contact sensing. MoveIt checks its registered scene,
which may not contain the real table, bowl or a held cube.
"""
import json
import math
from pathlib import Path
import threading
import time

import numpy as np

from approach_offset import ApproachOffsetSchedule
from bowl_release import BowlRelease
from grasp_offset import OffsetPlanner, checked_joints
from trajectory_offset import CartesianOffsetModel


def route_waypoints(start_xyz, start_quaternion, bowl):
    start = BowlRelease.vector(start_xyz, 3)
    orientation = BowlRelease.quaternion(start_quaternion)
    if not .20 <= bowl.position[2] <= .50:
        raise ValueError('Bowl reference outside the permitted height envelope')
    # Lift vertically before translating across the table; no policy prediction
    # supplies the carry destination. All dimensions are in base-frame metres.
    height = max(start[2] + .05, bowl.position[2] + .08)
    if height > .60 or height-start[2] > .30:
        raise ValueError('Planned carry clearance is outside the permitted height envelope')
    if math.hypot(*bowl.position[:2]) > .85:
        raise ValueError('Bowl reference outside the permitted radial envelope')
    final_z = bowl.position[2] + .025
    return [(start[0], start[1], height, *orientation),
            (bowl.position[0], bowl.position[1], height, *orientation),
            (bowl.position[0], bowl.position[1], final_z, *bowl.orientation)]


class PathKnotGap(ValueError):
    def __init__(self, index, jump, previous, target):
        self.index = index
        super().__init__(f'Planned path discontinuity: index={index}, jump_rad={jump:.6f}, previous={previous}, target={target}')


def interpolate_pose(start, goal):
    """Cartesian midpoint with shortest-arc quaternion interpolation."""
    a = np.array(BowlRelease.quaternion(start[3:]));b = np.array(BowlRelease.quaternion(goal[3:]))
    if np.dot(a,b) < 0:b = -b
    orientation = a+b
    orientation /= np.linalg.norm(orientation)
    return [float(x) for x in [*(.5*(np.array(start[:3])+np.array(goal[:3]))), *orientation]]


def refined_joint_path(start, goal, solve, pose_of, order, limits, depth=0):
    """Split Cartesian legs when time-parameterization returns sparse knots.

    No knot is accepted by relaxing its limit. Initial discontinuities, invalid
    states and incomplete Cartesian plans still fail immediately.
    """
    points = solve(start,goal)
    try:
        return dense_joint_path(points,start,order,limits), 0
    except PathKnotGap as error:
        if error.index == 0 or depth >= 6:
            raise
        origin = pose_of(start)
        dot = abs(sum(a*b for a,b in zip(origin[3:],goal[3:])))
        angle = 2*math.acos(min(1.,dot))
        if math.dist(origin[:3],goal[:3]) < .001 and angle < .005:
            raise
        midpoint = interpolate_pose(origin,goal)
        first,n1 = refined_joint_path(start,midpoint,solve,pose_of,order,limits,depth+1)
        second,n2 = refined_joint_path(first[-1],goal,solve,pose_of,order,limits,depth+1)
        return first+second, 1+n1+n2


def dense_joint_path(points, start, order, limits, step=.002):
    if not points:
        raise ValueError('MoveIt returned an empty path')
    start = checked_joints(start)
    previous = start
    dense = []
    for index, point in enumerate(points):
        q = checked_joints(point)
        for name, value in zip(order, q):
            lo, hi = limits[name]
            if not lo <= value <= hi:
                raise ValueError('Planned path violates a joint limit')
        jump = max(abs(a-b) for a,b in zip(previous,q))
        # MoveIt time-parameterizes and sparsifies Cartesian path knots. They
        # are not playback targets: validate the branch, then resample/check
        # every actual command below at <= 0.002 rad.
        if jump > (.005 if index == 0 else .15):
            raise PathKnotGap(index,jump,previous,q)
        count = max(1, math.ceil(jump/step))
        dense.extend([[a+(b-a)*k/count for a,b in zip(previous,q)] for k in range(1,count+1)])
        previous = q
    return dense


class BowlTransportPlanner(OffsetPlanner):
    def __init__(self, node, order, calibration, robot_description):
        from moveit_msgs.srv import GetCartesianPath, GetPlanningScene
        super().__init__(node, order, 'base', [0.,0.,0.])
        self.bowl = BowlRelease(calibration)
        if self.bowl.zone is None:
            raise ValueError('Planned transport requires the calibrated bowl release region')
        self.model = CartesianOffsetModel(robot_description, order)
        self.cartesian = node.create_client(GetCartesianPath, '/compute_cartesian_path')
        self.scene = node.create_client(GetPlanningScene, '/get_planning_scene')

    def plan(self, start, waypoints=None):
        from geometry_msgs.msg import Pose
        from moveit_msgs.srv import GetCartesianPath, GetStateValidity, GetPlanningScene
        from moveit_msgs.msg import PlanningSceneComponents
        start = checked_joints(start)
        pose = self.pose(start).pose
        local,_ = self.model.fk_jacobian(start)
        if np.linalg.norm(local[:3,3]-[pose.position.x,pose.position.y,pose.position.z]) > .002:
            raise ValueError('Local FK disagrees with live MoveIt')
        if waypoints is None:
            waypoints = route_waypoints([pose.position.x,pose.position.y,pose.position.z],
                                        [pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w],self.bowl)
        request = GetCartesianPath.Request()
        request.header.frame_id = 'base'
        request.group_name = 'ur_manipulator';request.link_name = 'tool0'
        request.start_state.is_diff = True
        request.start_state.joint_state.name = self.order
        request.start_state.joint_state.position = start
        request.max_step = .005
        request.jump_threshold = 0.0
        request.revolute_jump_threshold = .05
        request.avoid_collisions = True
        dense = [];fractions = [];segment_start = start;refinements = 0
        def solve(q, values):
            p = Pose()
            p.position.x,p.position.y,p.position.z = values[:3]
            p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w = values[3:]
            request.waypoints = [p]
            request.start_state.joint_state.position = list(q)
            response = self.call(self.cartesian, request, timeout=8.)
            if response.error_code.val != 1 or response.fraction < .999999:
                raise ValueError(f'Complete Cartesian carry route unavailable: fraction={response.fraction:.4f}, code={response.error_code.val}')
            fractions.append(response.fraction)
            trajectory = response.solution.joint_trajectory
            if len(trajectory.joint_names) != len(self.order) or set(trajectory.joint_names) != set(self.order):
                raise ValueError('MoveIt trajectory joint names differ from the UR5 order')
            indices = [trajectory.joint_names.index(k) for k in self.order]
            points = [[point.positions[i] for i in indices] for point in trajectory.points]
            return points
        def pose_of(q):
            p = self.pose(q).pose
            return [p.position.x,p.position.y,p.position.z,p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w]
        for values in waypoints:
            segment,count = refined_joint_path(segment_start,values,solve,pose_of,self.order,self.model.limits)
            refinements += count
            origin = self.model.fk_jacobian(segment_start)[0][:3,3]
            destination = np.array(values[:3]);direction = destination-origin
            for q in segment:
                position = self.model.fk_jacobian(q)[0][:3,3]
                phase = float(np.clip(np.dot(position-origin,direction)/max(np.dot(direction,direction),1e-12),0,1))
                if np.linalg.norm(position-(origin+phase*direction)) > .003:
                    raise ValueError('Resampled carry path cuts outside the checked Cartesian segment')
            if np.linalg.norm(self.model.fk_jacobian(segment[-1])[0][:3,3]-destination) > .002:
                raise ValueError('Cartesian segment did not reach its waypoint')
            dense.extend(segment);segment_start = segment[-1]
        if len(dense) > 1600:
            raise ValueError('Carry route exceeds bounded execution duration')
        # Check every target in the exact playback sequence, not just waypoints.
        self.require_service(self.validity)
        for q in dense:
            req = GetStateValidity.Request();req.group_name = 'ur_manipulator'
            req.robot_state.is_diff = True;req.robot_state.joint_state.name = self.order
            req.robot_state.joint_state.position = q
            result = self.call(self.validity, req, timeout=.25, attempts=2)
            if not result.valid:
                raise ValueError('Planned transport state rejected: collision/invalid state')
        scene_request = GetPlanningScene.Request()
        scene_request.components.components = PlanningSceneComponents.WORLD_OBJECT_NAMES | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
        scene = self.call(self.scene,scene_request)
        report = {'checked_request_fractions':fractions,'cartesian_refinements':refinements,
                  'start_joints':start,'waypoints_base_xyz_xyzw':waypoints,
                  'playback_targets':len(dense),'joint_target_step_limit_rad':.002,
                  'registered_world_objects':[x.id for x in scene.scene.world.collision_objects],
                  'registered_attached_objects':[x.object.id for x in scene.scene.robot_state.attached_collision_objects],
                  'limitation':'Checks cover registered MoveIt geometry only; unregistered physical fixtures and cube contact are not established.'}
        return dense, report

    def inside_release_region(self, joints):
        pose,_ = self.model.fk_jacobian(joints)
        delta = pose[:3,3]-self.bowl.position;zone = self.bowl.zone
        # Compare rotation matrices so quaternion sign does not matter.
        x,y,z,w = self.bowl.orientation
        desired = np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                            [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                            [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
        angle = math.acos(float(np.clip((np.trace(desired@pose[:3,:3].T)-1)/2,-1,1)))
        return (math.hypot(*delta[:2]) <= zone['horizontal_radius_m'] and
                -zone['below_reference_m'] <= delta[2] <= zone['above_reference_m'] and
                angle <= math.radians(zone['orientation_tolerance_deg']))


class PlannedBowlTransport:
    """Bounded handoff; caller owns the one command publisher and watchdogs."""
    def __init__(self, planner, logger, log_dir):
        self.planner,self.logger,self.log_dir = planner,logger,Path(log_dir)
        self.close = ApproachOffsetSchedule(close_hold=.5)
        self.stage = 'policy';self.started = self.stage_started = None
        self.hold = None;self.path = None;self.report = None;self.error = None
        self.index = 0;self.settled_since = None;self.worker = None
        self.previous_measured = None;self.previous_time = None
        self.closed = None;self.closed_at = 0.
        self.release_seen = False

    @property
    def active(self):
        return self.stage != 'policy'

    def observe(self, value, now):
        self.closed,self.closed_at = value,now
        if self.stage == 'policy' and self.close.observe(value,now):
            self.stage = 'requested'

    def begin(self, last_action, now):
        if self.stage != 'requested' or last_action is None:
            raise ValueError('No checked policy target for planned handoff')
        self.hold = checked_joints(last_action[:6]);self.started = self.stage_started = now
        self.stage = 'settling'
        self.logger.info('ASSISTED TRANSPORT: pickup policy paused; settling at last checked target; CLOSED IO does not prove contact')

    def _plan(self, start):
        try:
            path,report = self.planner.plan(start)
            self.log_dir.mkdir(parents=True,exist_ok=True)
            filename = self.log_dir/f'plan_{time.time_ns()}.json'
            filename.write_text(json.dumps({'report':report,'joint_targets':path},indent=2)+'\n')
            self.report = {**report,'plan_file':str(filename)}
            self.path = path  # Publish readiness only after the full plan is checked/saved.
        except Exception as error:self.error = str(error)

    def tick(self, measured, measured_at, now):
        if measured is None or now-measured_at > .1:
            raise ValueError('Stale joint feedback during planned transport')
        if self.closed is None or now-self.closed_at > .2:
            raise ValueError('Stale gripper confirmation during planned transport')
        if self.error:raise ValueError(self.error)
        if self.stage != 'complete' and now-self.started > 210:
            raise TimeoutError('Planned transport duration exceeded 210 seconds')
        q = checked_joints(measured)
        if self.closed == 1. and not self.release_seen:
            if self.stage not in ('executing','arrival') or not self.planner.inside_release_region(q):
                raise ValueError('Unexpected gripper OPEN outside checked bowl arrival')
            self.release_seen = True;self.stage = 'complete'
            self.logger.info('ASSISTED TRANSPORT COMPLETE: OPEN IO confirmed in calibrated bowl region; physical placement still requires operator/video verification')
        if self.stage == 'complete':return [*self.hold,1.]
        if self.stage == 'settling':
            speed = float('inf')
            if self.previous_measured is not None and now>self.previous_time:
                speed = max(abs(a-b)/(now-self.previous_time) for a,b in zip(q,self.previous_measured))
            if max(abs(a-b) for a,b in zip(q,self.hold)) <= .003 and speed <= .005:
                if self.settled_since is None:self.settled_since = now
                if now-self.settled_since >= .3:
                    self.stage = 'planning';self.stage_started = now
                    self.worker = threading.Thread(target=self._plan,args=(list(q),),daemon=True);self.worker.start()
            else:self.settled_since = None
            if now-self.started > 10 and self.stage == 'settling':raise TimeoutError('Arm did not settle for planned transport')
            self.previous_measured,self.previous_time = q,now
        elif self.stage == 'planning':
            if now-self.stage_started > 30:raise TimeoutError('Checked carry planning exceeded 30 seconds')
            if self.path is not None:
                if max(abs(a-b) for a,b in zip(q,self.path[0]))>.005:
                    raise ValueError('Arm moved while planned transport was being checked')
                self.stage = 'executing';self.stage_started = now
                self.logger.info('ASSISTED TRANSPORT PLAN: '+json.dumps(self.report))
        elif self.stage == 'executing':
            candidate = self.path[self.index]
            # Slow playback (~0.02 rad/s at 10 Hz) plus feedback gating prevents
            # outrunning the existing smoothed engine. No policy overshoot.
            if max(abs(a-b) for a,b in zip(q,candidate)) <= .025:
                self.hold = list(candidate);self.index += 1;self.stage_started = now
                if self.index == len(self.path):self.stage = 'arrival'
            elif now-self.stage_started > 10:raise TimeoutError('Measured joints stopped following planned carry')
        elif self.stage == 'arrival':
            if now-self.stage_started > 10:
                raise TimeoutError('Reached checked bowl target but automatic OPEN was not confirmed')
        return [*self.hold,0.]
