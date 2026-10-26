"""Shared speed settings for the standalone executor/control-engine pair."""
import math

DEFAULT_SPEED_SCALE = 0.5
DEFAULT_ACTIONS_PER_PREDICTION = 10


def resolve_carry_lift_mm(value, offset_mode, has_offset_file):
    if value is None:
        value = 25. if has_offset_file and offset_mode == 'continuous' else 0.
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 30:
        raise ValueError('--carry-lift-mm must be between 0 and 30')
    if value and (not has_offset_file or offset_mode != 'continuous'):
        raise ValueError('--carry-lift-mm requires a grasp offset file and continuous mode')
    return value


class StableGripperCommand:
    """Require sustained policy intent before changing the gripper IO.

    No grasp/contact or bowl location is inferred. Both directions use the
    same short confirmation window; sustained retry and release remain legal.
    """

    def __init__(self, stable_s=0.2, max_gap_s=0.25):
        self.stable_s = stable_s
        self.max_gap_s = max_gap_s
        self.reset()

    def reset(self):
        self.candidate = None
        self.since = None
        self.last_at = None

    def update(self, requested, now, confirmed):
        if requested == confirmed:
            self.reset()
            return confirmed
        if (requested != self.candidate or self.last_at is None
                or now - self.last_at > self.max_gap_s or now < self.last_at):
            self.candidate = requested
            self.since = now
        self.last_at = now
        if now - self.since + 1e-9 >= self.stable_s:
            return requested
        return confirmed


def validated_speed(value):
    value = float(value)
    if not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError('speed scale must be finite and in (0, 1]')
    return value


def validated_horizon(value):
    value = int(value)
    if not 1 <= value <= 50:
        raise ValueError('actions per prediction must be in [1, 50]')
    return value
