"""Fade a measured grasp correction after a stable IO-confirmed close."""
import math


class ApproachOffsetSchedule:
    def __init__(self, close_hold=0.5, fade_seconds=2.0):
        if not math.isfinite(close_hold) or close_hold <= 0:
            raise ValueError('close hold must be finite and positive')
        if not math.isfinite(fade_seconds) or fade_seconds <= 0:
            raise ValueError('fade duration must be finite and positive')
        self.close_hold = close_hold
        self.fade_seconds = fade_seconds
        self.seen_open = False
        self.closed_since = None
        self.release_started = None
        self.last_confirmed_at = None

    def observe(self, value, now):
        """Return True once, when a stable close commits the fade.

        IO confirmation establishes command history, not object contact.
        Starting with a closed gripper cannot trigger this transition.
        """
        if value not in (0., 1.) or not math.isfinite(now):
            raise ValueError('invalid gripper confirmation')
        if self.last_confirmed_at is not None and now-self.last_confirmed_at > .2:
            self.closed_since = None
        self.last_confirmed_at = now
        if self.release_started is not None:
            return False
        if value == 1.:
            self.seen_open = True
            self.closed_since = None
        elif self.seen_open:
            if self.closed_since is None:
                self.closed_since = now
            elif now-self.closed_since >= self.close_hold:
                self.release_started = now
                return True
        return False

    def scales(self, start, count, period):
        if count < 1 or not math.isfinite(start) or not math.isfinite(period) or period <= 0:
            raise ValueError('invalid action timing')
        # Snapshot once: the ROS confirmation callback may update this object
        # while the inference worker plans a chunk.
        released = self.release_started
        if released is None:
            return [1.]*count
        result = []
        for index in range(count):
            phase = min(1., max(0., (start+index*period-released)/self.fade_seconds))
            result.append(1.-phase*phase*(3.-2.*phase))
        return result


class CarryLiftSchedule(ApproachOffsetSchedule):
    """Ramp upward after stable closed IO, retaining height for this episode."""

    def __init__(self, height_m):
        if not math.isfinite(height_m) or not 0 <= height_m <= .03:
            raise ValueError('carry lift must be between 0 and 30 mm')
        super().__init__(close_hold=.5, fade_seconds=2.)
        self.height_m = height_m

    def heights(self, start, count, period):
        return [self.height_m*(1.-scale) for scale in self.scales(start,count,period)]

    def next_heights(self, previous_height, count, period):
        """Ramp from the committed queue endpoint, never elapsed idle time."""
        if not 0 <= previous_height <= self.height_m or count < 1 or period <= 0:
            raise ValueError('invalid committed carry height or action timing')
        if self.release_started is None:
            return [0.]*count
        result=[]
        for i in range(count):
            phase=min(1., i*period/self.fade_seconds)
            weight=phase*phase*(3.-2.*phase)
            result.append(previous_height+(self.height_m-previous_height)*weight)
        return result
