"""Exercise-specific movement and form analysis.

Every analyzer exposes the same small interface so the camera/UI loop does not
need exercise-specific branches.  Thresholds are deliberately conservative:
they are useful defaults, but should be calibrated with representative patient
videos before clinical use.
"""
from dataclasses import dataclass
import math
import time

from form_analysis import calculate_angle
from pose_estimator import PoseLandmark


READY_STABLE_FRAMES = 5
TARGET_STABLE_FRAMES = 3
RETURN_STABLE_FRAMES = 4


@dataclass
class AnalysisResult:
    status: str
    form_ok: bool
    metric_label: str
    metric_value: float | None
    fault: str | None = None


def _mean_joint_angle(landmarks, left_indices, right_indices):
    left = calculate_angle(*(landmarks[i] for i in left_indices))
    right = calculate_angle(*(landmarks[i] for i in right_indices))
    return (left + right) / 2.0, left, right


def _body_alignment(landmarks):
    left = calculate_angle(
        landmarks[PoseLandmark.LEFT_SHOULDER],
        landmarks[PoseLandmark.LEFT_HIP],
        landmarks[PoseLandmark.LEFT_ANKLE],
    )
    right = calculate_angle(
        landmarks[PoseLandmark.RIGHT_SHOULDER],
        landmarks[PoseLandmark.RIGHT_HIP],
        landmarks[PoseLandmark.RIGHT_ANKLE],
    )
    return (left + right) / 2.0


def _body_angle_from_horizontal(landmarks):
    shoulder_x = (
        landmarks[PoseLandmark.LEFT_SHOULDER][0]
        + landmarks[PoseLandmark.RIGHT_SHOULDER][0]
    ) / 2.0
    shoulder_y = (
        landmarks[PoseLandmark.LEFT_SHOULDER][1]
        + landmarks[PoseLandmark.RIGHT_SHOULDER][1]
    ) / 2.0
    ankle_x = (
        landmarks[PoseLandmark.LEFT_ANKLE][0]
        + landmarks[PoseLandmark.RIGHT_ANKLE][0]
    ) / 2.0
    ankle_y = (
        landmarks[PoseLandmark.LEFT_ANKLE][1]
        + landmarks[PoseLandmark.RIGHT_ANKLE][1]
    ) / 2.0
    return abs(math.degrees(math.atan2(ankle_y - shoulder_y,
                                       ankle_x - shoulder_x)))


def _is_horizontal(landmarks, limit=35):
    angle = _body_angle_from_horizontal(landmarks)
    return min(angle, abs(180.0 - angle)) < limit


def _has_upper_body_support(landmarks):
    """Approximate a high/forearm support position from a side view."""
    shoulder_y = (
        landmarks[PoseLandmark.LEFT_SHOULDER][1]
        + landmarks[PoseLandmark.RIGHT_SHOULDER][1]
    ) / 2.0
    elbow_y = (
        landmarks[PoseLandmark.LEFT_ELBOW][1]
        + landmarks[PoseLandmark.RIGHT_ELBOW][1]
    ) / 2.0
    wrist_y = (
        landmarks[PoseLandmark.LEFT_WRIST][1]
        + landmarks[PoseLandmark.RIGHT_WRIST][1]
    ) / 2.0
    return elbow_y > shoulder_y + 0.035 or wrist_y > shoulder_y + 0.055


def _landmarks_reliable(
    landmarks, indices, min_visibility=0.55, min_presence=0.50
):
    """Use confidence metadata when supplied by PoseEstimator.

    Plain coordinate lists remain supported for tests and callers that do not
    yet provide confidence metadata.
    """
    reliable = getattr(landmarks, "reliable", None)
    if reliable is None:
        return True
    return reliable(indices, min_visibility, min_presence)


class RepetitionAnalyzer:
    unit = "reps"

    def __init__(self, name, target):
        self.name = name
        self.target = target
        self.progress = 0
        self._stage = "up"
        self._movement_started = False
        self._invalid_pose_frames = 0
        self._movement_started_at = None

    @property
    def is_complete(self):
        return self.progress >= self.target

    def reset(self):
        self.progress = 0
        self._stage = "up"
        self._movement_started = False
        self._invalid_pose_frames = 0
        self._movement_started_at = None

    def _guard_pose(self, landmarks, indices, guidance):
        if _landmarks_reliable(landmarks, indices):
            self._invalid_pose_frames = 0
            return None
        self._invalid_pose_frames += 1
        if self._invalid_pose_frames >= 2:
            self._reset_incomplete_rep()
        return AnalysisResult(guidance, False, "", None)

    def _reset_incomplete_rep(self):
        """Cancel only the current partial repetition; keep completed reps."""
        self._stage = "up"
        self._movement_started = False
        self._movement_started_at = None


class BicepCurlAnalyzer(RepetitionAnalyzer):
    def __init__(self, target, arm, extended_angle, flexed_angle,
                 partial_up_angle):
        super().__init__("Bicep curl", target)
        self.arm = arm
        self.extended_angle = extended_angle
        self.flexed_angle = flexed_angle
        self.partial_up_angle = partial_up_angle
        self._ready = False
        self._rep_valid = True
        self._bad_arm_position = False
        self._extended_frames = 0
        self._flexed_frames = 0

    def reset(self):
        super().reset()
        self._ready = False
        self._rep_valid = True
        self._bad_arm_position = False
        self._extended_frames = 0
        self._flexed_frames = 0

    def _reset_incomplete_rep(self):
        super()._reset_incomplete_rep()
        self._ready = False
        self._rep_valid = True
        self._bad_arm_position = False
        self._extended_frames = 0
        self._flexed_frames = 0

    def update(self, landmarks, now=None):
        shoulder, elbow, wrist = PoseLandmark.arm_joints(self.arm)
        hip = (PoseLandmark.LEFT_HIP if self.arm == "left"
               else PoseLandmark.RIGHT_HIP)
        guard = self._guard_pose(
            landmarks,
            (shoulder, elbow, wrist, hip),
            "Keep the tracked shoulder, arm and hip fully in view",
        )
        if guard:
            return guard
        now = time.monotonic() if now is None else now
        angle = calculate_angle(landmarks[shoulder], landmarks[elbow],
                                landmarks[wrist])
        upper_arm_angle = calculate_angle(
            landmarks[hip], landmarks[shoulder], landmarks[elbow])
        arm_position_ok = upper_arm_angle <= 35
        fault = None

        self._extended_frames = (
            self._extended_frames + 1 if angle >= self.extended_angle else 0
        )
        self._flexed_frames = (
            self._flexed_frames + 1 if angle <= self.flexed_angle else 0
        )

        if (not self._movement_started
                and self._extended_frames >= READY_STABLE_FRAMES):
            self._ready = True
        if (not self._movement_started and self._ready
                and angle < self.extended_angle - 10):
            self._movement_started = True
            self._movement_started_at = now
            self._stage = "curling"
            self._rep_valid = arm_position_ok
        if self._movement_started and not arm_position_ok:
            self._rep_valid = False
        if (self._movement_started
                and self._flexed_frames >= TARGET_STABLE_FRAMES):
            self._stage = "flexed"

        if (self._movement_started
                and self._extended_frames >= RETURN_STABLE_FRAMES):
            duration = now - (self._movement_started_at or now)
            if (self._stage == "flexed" and self._rep_valid
                    and duration >= 0.6):
                self.progress += 1
                completed_good_rep = True
            else:
                completed_good_rep = False
                if duration < 0.6:
                    fault = "movement_too_fast"
                else:
                    fault = ("upper_arm_drift" if not self._rep_valid
                             else "partial_curl")
            self._stage = "up"
            self._movement_started = False
            self._movement_started_at = None
            self._ready = True
            self._rep_valid = True
            if completed_good_rep:
                return AnalysisResult(
                    f"Good rep! ({self.progress}/{self.target})", True,
                    "Elbow", angle)

        if (self._movement_started and not arm_position_ok
                and not self._bad_arm_position):
            fault = fault or "upper_arm_drift"
        self._bad_arm_position = self._movement_started and not arm_position_ok
        if not arm_position_ok:
            return AnalysisResult(
                "Keep the upper arm close to the torso", False,
                "Elbow", angle, fault)
        if fault:
            if fault == "upper_arm_drift":
                message = "Keep the upper arm close to the torso"
            elif fault == "movement_too_fast":
                message = "Move slowly and with control"
            else:
                message = "Partial rep - curl all the way up"
            return AnalysisResult(
                message, False, "Elbow", angle, fault)
        if not self._ready:
            status = "Fully extend the elbow before starting"
        elif self._stage == "flexed":
            status = "Good top position - lower with control"
        else:
            status = "Curl through the full range"
        return AnalysisResult(status, True, "Elbow", angle)


class SquatAnalyzer(RepetitionAnalyzer):
    """Count squats from the mean knee angle of both legs."""

    def __init__(self, target):
        super().__init__("Squat", target)
        self._bad_asymmetry = False
        self._ready = False
        self._rep_valid = True
        self._standing_frames = 0
        self._depth_frames = 0

    def reset(self):
        super().reset()
        self._bad_asymmetry = False
        self._ready = False
        self._rep_valid = True
        self._standing_frames = 0
        self._depth_frames = 0

    def _reset_incomplete_rep(self):
        super()._reset_incomplete_rep()
        self._ready = False
        self._rep_valid = True
        self._bad_asymmetry = False
        self._standing_frames = 0
        self._depth_frames = 0

    def update(self, landmarks, now=None):
        required = (
            PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP,
            PoseLandmark.LEFT_KNEE, PoseLandmark.RIGHT_KNEE,
            PoseLandmark.LEFT_ANKLE, PoseLandmark.RIGHT_ANKLE,
        )
        guard = self._guard_pose(
            landmarks, required, "Keep both hips, knees and ankles in view"
        )
        if guard:
            return guard
        now = time.monotonic() if now is None else now
        knee_angle, left, right = _mean_joint_angle(
            landmarks,
            PoseLandmark.leg_joints("left"),
            PoseLandmark.leg_joints("right"),
        )
        asymmetry = abs(left - right)
        asymmetry_bad = asymmetry > 20
        standing = left > 160 and right > 160
        fault = None
        self._standing_frames = (
            self._standing_frames + 1 if standing else 0
        )
        self._depth_frames = (
            self._depth_frames + 1
            if left <= 100 and right <= 100 else 0
        )

        if (not self._movement_started
                and self._standing_frames >= READY_STABLE_FRAMES):
            self._ready = True

        if (not self._movement_started and self._ready
                and left < 145 and right < 145):
            self._movement_started = True
            self._movement_started_at = now
            self._stage = "descending"
            self._rep_valid = not asymmetry_bad

        if self._movement_started and asymmetry_bad:
            self._rep_valid = False
        if (self._movement_started
                and self._depth_frames >= TARGET_STABLE_FRAMES):
            self._stage = "down"

        if (self._movement_started
                and self._standing_frames >= RETURN_STABLE_FRAMES):
            duration = now - (self._movement_started_at or now)
            if (self._stage == "down" and self._rep_valid
                    and duration >= 0.8):
                self.progress += 1
                completed_good_rep = True
            else:
                completed_good_rep = False
                if duration < 0.8:
                    fault = "movement_too_fast"
                else:
                    fault = ("uneven_squat" if not self._rep_valid
                             else "shallow_squat")
            self._stage = "up"
            self._movement_started = False
            self._movement_started_at = None
            self._ready = True
            self._rep_valid = True
            if completed_good_rep:
                return AnalysisResult(
                    f"Good rep! ({self.progress}/{self.target})", True,
                    "Knee", knee_angle)

        if asymmetry_bad and not self._bad_asymmetry:
            fault = fault or "uneven_squat"
        self._bad_asymmetry = asymmetry_bad

        if asymmetry_bad:
            return AnalysisResult(
                "Keep both knees moving evenly", False, "Knee", knee_angle,
                fault)
        if fault:
            if fault == "uneven_squat":
                message = "Keep both knees moving evenly"
            elif fault == "movement_too_fast":
                message = "Move slowly and with control"
            else:
                message = "Squat deeper before standing"
            return AnalysisResult(
                message, False, "Knee", knee_angle,
                fault)
        if self._stage == "down":
            status = "Good depth - stand up"
        elif not self._ready:
            status = "Stand fully upright before starting"
        else:
            status = f"Squat reps: {self.progress}/{self.target}"
        return AnalysisResult(status, True, "Knee", knee_angle)


class PushUpAnalyzer(RepetitionAnalyzer):
    """Count push-ups and check shoulder-hip-ankle alignment."""

    def __init__(self, target):
        super().__init__("Push-up", target)
        self._bad_alignment = False
        self._ready = False
        self._rep_valid = True
        self._extended_frames = 0
        self._depth_frames = 0

    def reset(self):
        super().reset()
        self._bad_alignment = False
        self._ready = False
        self._rep_valid = True
        self._extended_frames = 0
        self._depth_frames = 0

    def _reset_incomplete_rep(self):
        super()._reset_incomplete_rep()
        self._ready = False
        self._rep_valid = True
        self._bad_alignment = False
        self._extended_frames = 0
        self._depth_frames = 0

    def update(self, landmarks, now=None):
        required = (
            PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER,
            PoseLandmark.LEFT_ELBOW, PoseLandmark.RIGHT_ELBOW,
            PoseLandmark.LEFT_WRIST, PoseLandmark.RIGHT_WRIST,
            PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP,
            PoseLandmark.LEFT_ANKLE, PoseLandmark.RIGHT_ANKLE,
        )
        guard = self._guard_pose(
            landmarks,
            required,
            "Keep shoulders, arms, hips and ankles fully in view",
        )
        if guard:
            return guard
        now = time.monotonic() if now is None else now
        elbow_angle, left_elbow, right_elbow = _mean_joint_angle(
            landmarks,
            PoseLandmark.arm_joints("left"),
            PoseLandmark.arm_joints("right"),
        )
        alignment = _body_alignment(landmarks)
        horizontal = _is_horizontal(landmarks)
        supported = _has_upper_body_support(landmarks)
        posture_ok = (
            alignment >= 160
            and horizontal
            and supported
        )
        arms_extended = left_elbow >= 155 and right_elbow >= 155
        self._extended_frames = (
            self._extended_frames + 1
            if arms_extended and posture_ok else 0
        )
        self._depth_frames = (
            self._depth_frames + 1
            if left_elbow <= 100 and right_elbow <= 100 else 0
        )

        if (not self._movement_started
                and self._extended_frames >= READY_STABLE_FRAMES):
            self._ready = True
        if (not self._movement_started and self._ready
                and left_elbow < 145 and right_elbow < 145):
            self._movement_started = True
            self._movement_started_at = now
            self._stage = "descending"
            self._rep_valid = posture_ok
        if self._movement_started and not posture_ok:
            self._rep_valid = False
        if (self._movement_started
                and self._depth_frames >= TARGET_STABLE_FRAMES):
            self._stage = "down"

        fault = None
        if (self._movement_started
                and self._extended_frames >= RETURN_STABLE_FRAMES):
            duration = now - (self._movement_started_at or now)
            if (self._stage == "down" and self._rep_valid and posture_ok
                    and duration >= 0.8):
                self.progress += 1
                completed_good_rep = True
            else:
                completed_good_rep = False
                if duration < 0.8:
                    fault = "movement_too_fast"
                else:
                    fault = ("pushup_body_alignment" if not self._rep_valid
                             else "shallow_pushup")
            self._stage = "up"
            self._movement_started = False
            self._movement_started_at = None
            self._ready = posture_ok
            self._rep_valid = True
            if completed_good_rep:
                return AnalysisResult(
                    f"Good rep! ({self.progress}/{self.target})", True,
                    "Elbow", elbow_angle)

        posture_bad_during_rep = self._movement_started and not posture_ok
        if posture_bad_during_rep and not self._bad_alignment:
            fault = fault or "pushup_body_alignment"
        self._bad_alignment = posture_bad_during_rep

        if not horizontal:
            return AnalysisResult(
                "Turn side-on and take the top push-up position", False,
                "Elbow", elbow_angle, fault)
        if not supported:
            return AnalysisResult(
                "Place hands below the shoulders", False,
                "Elbow", elbow_angle, fault)
        if alignment < 160:
            return AnalysisResult(
                "Keep shoulders, hips and ankles in one line", False,
                "Elbow", elbow_angle, fault)
        if fault == "shallow_pushup":
            return AnalysisResult(
                "Lower until elbows reach about 90 degrees", False,
                "Elbow", elbow_angle, fault)
        if fault == "movement_too_fast":
            return AnalysisResult(
                "Move slowly and with control", False,
                "Elbow", elbow_angle, fault)
        status = ("Good depth - push up" if self._stage == "down"
                  else (f"Push-up reps: {self.progress}/{self.target}"
                        if self._ready
                        else "Hold a straight top position before starting"))
        return AnalysisResult(status, True, "Elbow", elbow_angle)


class PlankAnalyzer:
    """Accumulate only time spent in a straight, approximately horizontal pose."""

    name = "Plank"
    unit = "sec"

    def __init__(self, target_seconds):
        self.target = float(target_seconds)
        self.reset()

    @property
    def progress(self):
        return self._seconds

    @property
    def is_complete(self):
        return self._seconds >= self.target

    def reset(self):
        self._seconds = 0.0
        self._last_time = None
        self._bad_form = False
        self._invalid_seconds = 0.0

    def update(self, landmarks, now=None):
        now = time.monotonic() if now is None else now
        elapsed = 0.0 if self._last_time is None else min(now - self._last_time,
                                                          0.25)
        self._last_time = now

        required = (
            PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER,
            PoseLandmark.LEFT_ELBOW, PoseLandmark.RIGHT_ELBOW,
            PoseLandmark.LEFT_WRIST, PoseLandmark.RIGHT_WRIST,
            PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP,
            PoseLandmark.LEFT_ANKLE, PoseLandmark.RIGHT_ANKLE,
        )
        if not _landmarks_reliable(landmarks, required):
            self._invalid_seconds += elapsed
            if self._invalid_seconds >= 0.75:
                self._seconds = 0.0
            self._bad_form = False
            return AnalysisResult(
                "Keep shoulders, arms, hips and ankles fully in view",
                False, "", None)

        alignment = _body_alignment(landmarks)
        in_plank_position = _is_horizontal(landmarks)
        supported = _has_upper_body_support(landmarks)
        form_ok = in_plank_position and supported and alignment >= 160
        if form_ok:
            self._seconds = min(self.target, self._seconds + elapsed)
            self._invalid_seconds = 0.0
        else:
            self._invalid_seconds += elapsed
            if self._invalid_seconds >= 0.75:
                self._seconds = 0.0

        fault = None
        if in_plank_position and supported and not form_ok and not self._bad_form:
            fault = "plank_body_alignment"
        self._bad_form = in_plank_position and supported and not form_ok

        if not in_plank_position:
            status = "Turn side-on and hold a horizontal plank"
        elif not supported:
            status = "Place hands or forearms below the shoulders"
        elif alignment < 160:
            status = "Keep shoulders, hips and ankles in one line"
        else:
            status = f"Hold: {self._seconds:.1f}/{self.target:.0f} sec"
        return AnalysisResult(status, form_ok, "Body", alignment, fault)


def create_analyzer(exercise, cfg, arm=None):
    if exercise == "bicep_curl":
        return BicepCurlAnalyzer(
            cfg.target_reps,
            arm or cfg.track_arm,
            cfg.extended_angle,
            cfg.flexed_angle,
            cfg.partial_up_angle,
        )
    if exercise == "squat":
        return SquatAnalyzer(cfg.target_reps)
    if exercise == "plank":
        return PlankAnalyzer(cfg.plank_target_seconds)
    if exercise == "pushup":
        return PushUpAnalyzer(cfg.target_reps)
    raise ValueError(f"Unsupported exercise: {exercise}")
