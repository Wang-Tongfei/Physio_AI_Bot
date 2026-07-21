import enum

import numpy as np


def calculate_angle(a, b, c):
    """Angle in degrees at point b, formed by the points a-b-c.

    Each point is an (x, y) pair. Used for the elbow angle
    (shoulder -> elbow -> wrist).
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    if angle > 180.0:
        angle = 360.0 - angle
    return float(angle)


class RepEvent(enum.Enum):
    """What happened on the latest RepCounter.update() call."""
    NONE = "none"
    GOOD_REP = "good_rep"
    PARTIAL_REP = "partial_rep"


class RepCounter:
    def __init__(self, target_reps=10, extended_angle=160,
                 flexed_angle=45, partial_up_angle=80):
        self.target_reps = target_reps
        self.extended_angle = extended_angle
        self.flexed_angle = flexed_angle
        self.partial_up_angle = partial_up_angle
        self.reset()

    def reset(self):
        self.reps = 0
        self._stage = "down"           # "down" = extended, "up" = curled
        self._reached_bottom = False   # fully straightened this rep?
        self._reached_top = False      # fully curled this rep?

    def update(self, angle):
        """Advance the state machine by one frame; return a RepEvent."""
        event = RepEvent.NONE
        if angle > self.extended_angle:
            # Arm fully straight = bottom of the curl.
            if self._stage == "up":
                # Came back down after going up -> evaluate the rep.
                if self._reached_bottom and self._reached_top:
                    self.reps += 1
                    event = RepEvent.GOOD_REP
                else:
                    event = RepEvent.PARTIAL_REP
            self._stage = "down"
            self._reached_bottom = True
            self._reached_top = False
        elif angle < self.flexed_angle:
            # Arm fully curled = top of a good rep.
            self._stage = "up"
            self._reached_top = True
        elif angle < self.partial_up_angle and self._stage == "down":
            # Started curling up from the bottom (but not yet fully).
            self._stage = "up"
            self._reached_top = False
        return event

    @property
    def is_complete(self):
        return self.reps >= self.target_reps
