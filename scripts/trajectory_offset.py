"""Constant Cartesian translation of every policy target, using the MoveIt model."""
from __future__ import annotations

import math
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np

from grasp_offset import OffsetPlanner, checked_joints


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + math.sin(angle)*skew + (1-math.cos(angle))*(skew@skew)


def origin_transform(element):
    result = np.eye(4)
    if element is None:
        return result
    result[:3, 3] = [float(x) for x in element.get('xyz', '0 0 0').split()]
    roll, pitch, yaw = [float(x) for x in element.get('rpy', '0 0 0').split()]
    result[:3, :3] = rotation([0, 0, 1], yaw) @ rotation([0, 1, 0], pitch) @ rotation([1, 0, 0], roll)
    return result


def rotation_error(desired, actual):
    r = desired @ actual.T
    angle = math.acos(float(np.clip((np.trace(r)-1)/2, -1, 1)))
    vector = np.array([r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1]])
    return .5*vector if angle < 1e-7 else angle/(2*math.sin(angle))*vector


class CartesianOffsetModel:
    def __init__(self, urdf, order, frame='base', tip='tool0'):
        xml = ET.fromstring(urdf)
        links = {link.get('name') for link in xml.findall('link')}
        if frame not in links or tip not in links:
            raise ValueError('offset frame or tool link missing from robot description')
        self.order = list(order)
        self.parents = {}
        self.limits = {}
        for joint in xml.findall('joint'):
            name, kind = joint.get('name'), joint.get('type')
            axis = joint.find('axis')
            limit = joint.find('limit')
            data = {'name': name, 'type': kind, 'parent': joint.find('parent').get('link'),
                    'origin': origin_transform(joint.find('origin')),
                    'axis': np.array([float(x) for x in (axis.get('xyz') if axis is not None else '1 0 0').split()])}
            self.parents[joint.find('child').get('link')] = data
            if name in self.order:
                if kind != 'revolute' or limit is None:
                    raise ValueError('offset model requires bounded UR5 revolute joints')
                self.limits[name] = (float(limit.get('lower')), float(limit.get('upper')))
        self.chain = self.path(tip)
        if {j['name'] for j in self.chain if j['type'] != 'fixed'} != set(self.order):
            raise ValueError('robot model joint chain differs from the policy joint order')
        base_path = self.path(frame)
        if any(j['type'] != 'fixed' for j in base_path):
            raise ValueError('offset reference frame must be fixed')
        transform = np.eye(4)
        for joint in base_path:
            transform = transform @ joint['origin']
        self.root_to_base = np.linalg.inv(transform)

    def path(self, link):
        path, visited = [], set()
        while link in self.parents:
            if link in visited:
                raise ValueError('cyclic robot description')
            visited.add(link)
            joint = self.parents[link]
            path.append(joint)
            link = joint['parent']
        return list(reversed(path))

    def fk_jacobian(self, values):
        q = dict(zip(self.order, checked_joints(values)))
        transform = self.root_to_base.copy()
        axes = {}
        for joint in self.chain:
            transform = transform @ joint['origin']
            if joint['type'] != 'fixed':
                axes[joint['name']] = (transform[:3, 3].copy(), transform[:3, :3] @ joint['axis'])
                turn = np.eye(4)
                turn[:3, :3] = rotation(joint['axis'], q[joint['name']])
                transform = transform @ turn
        jacobian = np.zeros((6, 6))
        for i, name in enumerate(self.order):
            point, axis = axes[name]
            jacobian[:3, i] = np.cross(axis, transform[:3, 3]-point)
            jacobian[3:, i] = axis
        return transform, jacobian

    def translate(self, target, vector, carry_height=0.):
        original = np.asarray(checked_joints(target))
        vector = np.asarray(vector, dtype=float)
        if not math.isfinite(carry_height) or not abs(carry_height) <= .03:
            raise ValueError('carry transform magnitude must be at most 30 mm')
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError('invalid Cartesian translation')
        residual = vector - np.array([0.,0.,carry_height])
        if np.linalg.norm(residual) > .00701:
            raise ValueError('invalid or excessive Cartesian translation')
        desired, _ = self.fk_jacobian(original)
        desired[:3, 3] += np.asarray(vector)
        solution = original.copy()
        for _ in range(12):
            actual, jacobian = self.fk_jacobian(solution)
            position = desired[:3, 3]-actual[:3, 3]
            orientation = rotation_error(desired[:3, :3], actual[:3, :3])
            if np.linalg.norm(position) <= .00001 and np.linalg.norm(orientation) <= .0001:
                break
            error = np.concatenate([position, orientation])
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-8*np.eye(6), error)
            largest = np.max(np.abs(delta))
            if largest > .01:
                delta *= .01/largest
            solution += delta
        actual, _ = self.fk_jacobian(solution)
        if (np.linalg.norm(desired[:3, 3]-actual[:3, 3]) > .00005 or
                np.linalg.norm(rotation_error(desired[:3, :3], actual[:3, :3])) > .001):
            raise ValueError('continuous offset IK did not converge')
        if np.max(np.abs(solution-original)) > .05+4.*abs(carry_height):
            raise ValueError('offset correction exceeds the configured joint correction envelope')
        for name, value in zip(self.order, solution):
            lo, hi = self.limits[name]
            if not lo <= value <= hi:
                raise ValueError(f'corrected {name} exceeds joint limits')
        return solution.tolist()


class TrajectoryOffsetPlanner(OffsetPlanner):
    def __init__(self, node, order, frame, vector, robot_description=None):
        super().__init__(node, order, frame, vector)
        from rcl_interfaces.srv import GetParameters
        self.description = node.create_client(GetParameters, '/move_group/get_parameters')
        self.robot_description = robot_description
        self.model = None

    def initialize(self, reference):
        from rcl_interfaces.srv import GetParameters
        description = self.robot_description
        if description is None:
            request = GetParameters.Request()
            request.names = ['robot_description']
            response = self.call(self.description, request, timeout=5.0, attempts=2)
            if len(response.values) != 1 or not response.values[0].string_value:
                raise ValueError('MoveIt robot_description is unavailable')
            description = response.values[0].string_value
        model = CartesianOffsetModel(description, self.order, self.frame)
        actual, _ = model.fk_jacobian(reference)
        pose = self.pose(reference).pose
        if np.linalg.norm(actual[:3, 3]-np.array([pose.position.x, pose.position.y, pose.position.z])) > .0001:
            raise ValueError('local kinematic model does not match MoveIt FK')
        # Compare the local and MoveIt solutions for the initial correction.
        local = model.translate(reference, self.vector)
        reached, _ = model.fk_jacobian(local)
        # Orientation comparison uses the MoveIt quaternion directly.
        q = pose.orientation
        x, y, z, w = q.x, q.y, q.z, q.w
        norm = math.sqrt(x*x+y*y+z*z+w*w)
        if norm == 0:
            raise ValueError('MoveIt returned invalid orientation')
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
        orientation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                                [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                                [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        if np.linalg.norm(rotation_error(orientation, reached[:3, :3])) > .001:
            raise ValueError('local kinematic orientation differs from MoveIt FK')
        self.model = model

    def plan_actions(self, actions, measured, scales=None, lift_heights=None):
        from moveit_msgs.srv import GetStateValidity
        if not actions:
            raise ValueError('empty trajectory')
        measured = checked_joints(measured)
        if self.model is None:
            self.initialize(measured)
        if scales is None:
            scales = [1.]*len(actions)
        if len(scales) != len(actions) or any(not math.isfinite(x) or not 0 <= x <= 1 for x in scales):
            raise ValueError("offset scales must match actions and be in [0, 1]")
        if lift_heights is None:
            lift_heights = [0.]*len(actions)
        if (len(lift_heights) != len(actions) or
                any(not math.isfinite(h) or not 0 <= h <= .03 for h in lift_heights)):
            raise ValueError('carry heights must match actions and be between 0 and 30 mm')
        if any(lift_heights) and self.frame != 'base':
            raise ValueError('carry lift requires the robot base frame')
        corrected = []
        for action, scale, height in zip(actions, scales, lift_heights):
            if len(action) != 7 or not all(math.isfinite(float(x)) for x in action):
                raise ValueError('invalid policy action')
            vector = [scale*v for v in self.vector]
            vector[2] += height
            corrected.append([*self.model.translate(action[:6], vector, carry_height=height), action[6]])
            # Yield between CPU-heavy IK rows so live feedback callbacks run.
            time.sleep(0)
        previous = measured
        requests = []
        for index, action in enumerate(corrected):
            jump = max(abs(a-b) for a,b in zip(previous, action[:6]))
            if jump > (.05 if index == 0 else .15):
                raise ValueError(f'corrected trajectory exceeds the existing target-step envelope: action={index}, jump={jump:.6f} rad, limit={.05 if index==0 else .15:.3f} rad')
            steps = max(1, math.ceil(jump/.005))
            for step in range(1, steps+1):
                request = GetStateValidity.Request()
                request.group_name = 'ur_manipulator'
                request.robot_state.is_diff = True
                request.robot_state.joint_state.name = self.order
                request.robot_state.joint_state.position = [a+(b-a)*step/steps for a,b in zip(previous, action[:6])]
                requests.append(request)
            previous = action[:6]
        checks = self.check_path(requests)
        return corrected, {'frame':self.frame, 'offset_m':self.vector, 'actions':len(corrected), 'path_checks':len(requests), 'offset_scale_start':scales[0], 'offset_scale_end':scales[-1], 'carry_lift_start_m':lift_heights[0], 'carry_lift_end_m':lift_heights[-1], **checks}

    def check_path(self, requests):
        # A missing reply in a burst used to stall all checking for two seconds.
        # Keep only one request outstanding and retry that read-only query once.
        # Every interpolated state must still pass; never retry a rejection.
        self.require_service(self.validity)
        started = time.monotonic()
        deadline = started + 1.25
        for index, request in enumerate(requests):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f'continuous offset path checking timed out before state {index+1}/{len(requests)}')
            try:
                result = self.call(self.validity, request,
                                   timeout=min(.25, remaining/2), attempts=2)
            except TimeoutError as error:
                raise TimeoutError(f'continuous offset path checking timed out at state {index+1}/{len(requests)} after read-only retry: {self.validity.srv_name}') from error
            if not result.valid:
                raise ValueError(f'MoveIt rejected the continuously offset path: collision/invalid state {index+1}/{len(requests)}')
        return {'validity_elapsed_s':time.monotonic()-started,
                'validity_request_mode':'sequential_bounded_retry'}
