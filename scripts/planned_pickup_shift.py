"""Explicit two-trial planned pickup intervention; no command publishers.

This is a larger Cartesian plan, separate from the <=7 mm offset experiment.
The last checked OPEN target is settled, shifted, then CLOSED; neural pickup
does not resume. Existing planned bowl transport owns the post-close handoff.
"""
import json
import math
import threading
from pathlib import Path

from planned_transport import BowlTransportPlanner


def load_shift(path, attempt, require_recording=True):
    data = json.loads(Path(path).read_text())
    if attempt not in (12, 13) or data.get('allowed_attempts') != [12, 13]:
        raise ValueError('Temporary pickup intervention is restricted to attempts 12 and 13')
    vector = [float(v) for v in data.get('additional_base_vector_m', [])]
    if data.get('frame') != 'base' or len(vector) != 3 or not all(math.isfinite(v) for v in vector):
        raise ValueError('Need an explicit finite base-frame translation')
    if not 0 < math.sqrt(sum(v*v for v in vector)) <= .03 or abs(vector[2]) > .001:
        raise ValueError('Pickup shift must be at most 30 mm total and 1 mm vertically')
    if require_recording:
        session = Path(data['session_root']).resolve()
        status = json.loads((session/'status.json').read_text())
        if status.get('state') != 'recording' or status.get('attempt') != attempt or status.get('cross') != 1:
            raise ValueError(f'Start recording attempt {attempt} before launching its temporary pickup intervention')
        trial = json.loads((session/f'cross_01/attempt_{attempt:02d}/trial.json').read_text())
        if trial.get('status') != 'recording':
            raise ValueError('Recorded trial is not active')
    return vector


def record_intervention(path, attempt):
    import hashlib
    data = json.loads(Path(path).read_text())
    directory = Path(data['session_root'])/f'cross_01/attempt_{attempt:02d}/configuration'
    directory.mkdir(parents=True, exist_ok=True)
    record = {'revision':'temporary_planned_pickup_shift_12_13', 'attempt':attempt,
              'status':'configured; execution must be verified from telemetry and saved plan',
              'configuration':data, 'code_sha256':{}}
    for name in ('policy_executor.py','planned_pickup_shift.py','planned_transport.py'):
        record['code_sha256'][name] = hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
    (directory/'temporary_pickup_shift_runtime.json').write_text(json.dumps(record,indent=2)+'\n')


def load_unrecorded_repeat(path, attempt):
    if attempt != 13:
        raise ValueError('Unrecorded repeat is restricted to the completed attempt-13 condition')
    vector = load_shift(path, attempt, require_recording=False)
    session = Path(json.loads(Path(path).read_text())['session_root'])
    if json.loads((session/'status.json').read_text()).get('state') != 'waiting':
        raise ValueError('Finish the active trial before an unrecorded repeat')
    if json.loads((session/'cross_01/attempt_13/trial.json').read_text()).get('status') != 'finished':
        raise ValueError('Recorded attempt 13 must be finished before an unrecorded repeat')
    return vector


class PickupShiftPlanner(BowlTransportPlanner):
    def __init__(self, node, order, calibration, description, vector):
        super().__init__(node, order, calibration, description)
        self.vector = list(vector)

    def plan(self, start):
        pose = self.pose(start).pose
        p, q = pose.position, pose.orientation
        target = [p.x+self.vector[0], p.y+self.vector[1], p.z+self.vector[2], q.x, q.y, q.z, q.w]
        path, report = super().plan(start, [target])
        report.update(intervention='temporary planned pre-close pickup shift', additional_base_vector_m=self.vector)
        return path, report


class PlannedPickupShift:
    def __init__(self, planner, log_dir):
        self.planner, self.log_dir = planner, Path(log_dir)
        self.stage = 'policy'
        self.hold = self.path = self.error = None
        self.index = 0
        self.started = self.changed = self.settled = None
        self.previous = self.previous_at = None

    @property
    def active(self):
        return self.stage != 'policy'

    def begin(self, last_open, confirmed, confirmed_at, now):
        if last_open is None or float(last_open[6]) < .5 or confirmed != 1. or now-confirmed_at > .2:
            raise ValueError('Fresh OPEN IO and a previously published OPEN arm target are required')
        self.hold = list(last_open[:6])
        self.started = self.changed = now
        self.stage = 'settling'

    def _plan(self, q):
        try:
            path, report = self.planner.plan(q)
            if not path:
                raise ValueError('Empty pickup shift path')
            self.log_dir.mkdir(parents=True, exist_ok=True)
            import time
            (self.log_dir/f'pickup_shift_{time.time_ns()}.json').write_text(json.dumps({'report':report, 'joint_targets':path},indent=2)+'\n')
            self.path = path
        except Exception as error:
            self.error = str(error)

    def tick(self, measured, measured_at, confirmed, confirmed_at, now):
        if measured is None or now-measured_at > .1 or confirmed is None or now-confirmed_at > .2:
            raise ValueError('Stale feedback during planned pickup shift')
        if self.error:
            raise ValueError(self.error)
        if now-self.started > 60:
            raise TimeoutError('Pickup shift exceeded 60 seconds')
        if self.stage == 'closed':
            return [*self.hold, 0.]
        if self.stage == 'closing':
            if confirmed == 0.:
                self.stage = 'closed'
            elif now-self.changed > 8:
                raise TimeoutError('Pickup CLOSE was not acknowledged')
            return [*self.hold, 0.]
        if confirmed != 1.:
            raise ValueError('Gripper closed before pickup reposition completed')
        if self.stage == 'settling':
            speed = float('inf') if self.previous is None else max(abs(a-b) for a,b in zip(measured,self.previous))/max(now-self.previous_at,1e-6)
            if max(abs(a-b) for a,b in zip(measured,self.hold)) <= .003 and speed <= .005:
                if self.settled is None:
                    self.settled = now
                if now-self.settled >= .3:
                    self.stage = 'planning'; self.changed = now
                    threading.Thread(target=self._plan,args=(list(measured),),daemon=True).start()
            else:
                self.settled = None
            self.previous, self.previous_at = list(measured), now
            if self.stage == 'settling' and now-self.started > 10:
                raise TimeoutError('Arm did not settle before pickup reposition')
        elif self.stage == 'planning':
            if now-self.changed > 30:
                raise TimeoutError('Pickup path planning timed out')
            if self.path is not None:
                if max(abs(a-b) for a,b in zip(measured,self.path[0])) > .005:
                    raise ValueError('Arm moved during pickup path checking')
                self.stage = 'executing';self.changed = now
        elif self.stage == 'executing':
            candidate = self.path[self.index]
            if max(abs(a-b) for a,b in zip(measured,candidate)) <= .025:
                self.hold = list(candidate); self.index += 1;self.changed = now
                if self.index == len(self.path):
                    self.stage = 'arrival';self.settled = None
            elif now-self.changed > 10:
                raise TimeoutError('Pickup path tracking stalled')
        elif self.stage == 'arrival':
            if max(abs(a-b) for a,b in zip(measured,self.hold)) <= .0015:
                if self.settled is None:
                    self.settled = now
                if now-self.settled >= .3:
                    self.stage = 'closing';self.changed = now
                    return [*self.hold, 0.]
            else:
                self.settled = None
            if now-self.changed > 10:
                raise TimeoutError('Pickup target did not settle for closing')
        return [*self.hold, 1.]
