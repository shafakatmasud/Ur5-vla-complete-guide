"""One-shot gripper release at an operator-taught, fixed bowl pose."""
import json
import math
from pathlib import Path


class BowlRelease:
    radius_m = 0.010
    orientation_rad = math.radians(15.)
    dwell_s = 0.3
    max_speed_m_s = 0.020

    def __init__(self, calibration):
        data = json.loads(Path(calibration).read_text())
        if data['frame'] != 'base' or data['tip'] != 'tool0_controller':
            raise ValueError('Bowl release requires base -> tool0_controller calibration')
        self.position = self.vector(data['position_m'], 3)
        self.orientation = self.quaternion(data['quaternion_xyzw'])
        self.zone = data.get('release_zone')
        if self.zone is not None:
            limits = {'horizontal_radius_m': (.001,.05), 'below_reference_m': (0.,.05),
                      'above_reference_m': (0.,.12), 'orientation_tolerance_deg': (1.,45.)}
            self.zone = {k:float(self.zone[k]) for k in limits}
            if any(not math.isfinite(self.zone[k]) or not lo <= self.zone[k] <= hi
                   for k,(lo,hi) in limits.items()):
                raise ValueError('Invalid bowl release region')
        self.reset()

    @staticmethod
    def vector(values, length):
        result = tuple(float(v) for v in values)
        if len(result) != length or not all(math.isfinite(v) for v in result):
            raise ValueError('Invalid bowl release pose')
        return result

    @classmethod
    def quaternion(cls, values):
        q = cls.vector(values, 4)
        norm = math.sqrt(sum(v*v for v in q))
        if not .99 <= norm <= 1.01:
            raise ValueError('Invalid bowl release quaternion')
        return tuple(v/norm for v in q)

    def reset(self):
        self.closed_since = self.inside_since = None
        self.armed = self.released = False
        self.transport_hold = False
        self.previous = None
        self.diagnostics = {'reason': 'waiting for measured pose'}

    def update(self, now, stamp, age, xyz, quaternion, confirmed, busy):
        if self.released:
            return False
        if not 0 <= age <= .2:
            self.previous = self.inside_since = self.closed_since = None
            self.armed = False
            self.diagnostics = {'reason': 'TF stale or future stamped', 'tf_age_s': age}
            return False
        xyz = self.vector(xyz, 3)
        q = self.quaternion(quaternion)
        distance = math.dist(xyz, self.position)
        angle = 2*math.acos(min(1., abs(sum(a*b for a,b in zip(q,self.orientation)))))
        delta = [a-b for a,b in zip(xyz,self.position)]
        horizontal = math.hypot(delta[0],delta[1])
        inside = distance <= self.radius_m and angle <= self.orientation_rad
        if self.zone is not None:
            inside = (horizontal <= self.zone['horizontal_radius_m']
                      and -self.zone['below_reference_m'] <= delta[2] <= self.zone['above_reference_m']
                      and angle <= math.radians(self.zone['orientation_tolerance_deg']))
        self.diagnostics = {'distance_mm': 1000*distance,
                            'delta_mm': [1000*(a-b) for a,b in zip(xyz,self.position)],
                            'horizontal_distance_mm': horizontal*1000, 'inside_release_region': inside,
                            'orientation_error_deg': math.degrees(angle), 'tf_age_s': age,
                            'reason': 'waiting for closed IO away from bowl'}
        if confirmed == 0. and distance >= .025:
            self.transport_hold = True
        if confirmed != 0.:
            self.closed_since = self.inside_since = None
            self.armed = False
        elif self.closed_since is None:
            self.closed_since = now
        if confirmed == 0. and distance >= .025 and now-self.closed_since >= .5:
            self.armed = True
        if self.armed:
            self.diagnostics['reason'] = 'outside release position/orientation or waiting for slow dwell'
        if self.previous is None:
            self.previous = (stamp, xyz)
            return False
        old_stamp, old_xyz = self.previous
        if stamp <= old_stamp:
            return False
        self.previous = (stamp, xyz)
        dt = stamp-old_stamp
        speed = math.dist(xyz, old_xyz)/dt
        self.diagnostics['speed_mm_s'] = 1000*speed
        if (not self.armed or busy or dt > .2 or not inside or speed > self.max_speed_m_s):
            self.inside_since = None
            return False
        if self.inside_since is None:
            self.inside_since = now
        self.diagnostics['reason'] = 'confirming arrival dwell'
        self.diagnostics['dwell_s'] = now-self.inside_since
        if now-self.inside_since + 1e-9 < self.dwell_s:
            return False
        self.released = True
        self.transport_hold = False
        self.diagnostics['reason'] = 'release requested'
        return True
