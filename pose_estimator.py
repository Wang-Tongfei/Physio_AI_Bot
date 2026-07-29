import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


class PoseLandmark:
    """Named BlazePose indices used by exercise-analysis code."""
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
    LEFT_ELBOW, RIGHT_ELBOW = 13, 14
    LEFT_WRIST, RIGHT_WRIST = 15, 16
    LEFT_HIP, RIGHT_HIP = 23, 24
    LEFT_KNEE, RIGHT_KNEE = 25, 26
    LEFT_ANKLE, RIGHT_ANKLE = 27, 28

    @classmethod
    def arm_joints(cls, arm):
        """Return the (shoulder, elbow, wrist) indices for 'left' or 'right'."""
        if arm == "left":
            return cls.LEFT_SHOULDER, cls.LEFT_ELBOW, cls.LEFT_WRIST
        return cls.RIGHT_SHOULDER, cls.RIGHT_ELBOW, cls.RIGHT_WRIST

    @classmethod
    def leg_joints(cls, side):
        """Return the (hip, knee, ankle) indices for 'left' or 'right'."""
        if side == "left":
            return cls.LEFT_HIP, cls.LEFT_KNEE, cls.LEFT_ANKLE
        return cls.RIGHT_HIP, cls.RIGHT_KNEE, cls.RIGHT_ANKLE


class PoseEstimator:
    """Thin, single-responsibility wrapper around a MediaPipe PoseLandmarker."""

    def __init__(self, model_path):
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,   # ordered video frames
            num_poses=1,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def estimate(self, frame_bgr):
        """Return a list of 33 (x, y) normalised points, or None if no person."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(
            int(time.monotonic() * 1000),
            self._last_timestamp_ms + 1,
        )
        self._last_timestamp_ms = timestamp_ms
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.pose_landmarks:
            return None
        return [(lm.x, lm.y) for lm in result.pose_landmarks[0]]

    def close(self):
        self._landmarker.close()
