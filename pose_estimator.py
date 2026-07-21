"""
pose_estimator.py -- wraps MediaPipe so the rest of the app never touches it.

Turns a webcam frame into a plain list of (x, y) body points in [0, 1]. If we
ever swapped MediaPipe for another pose library (MoveNet, OpenPose, ...), only
THIS file would change -- nothing downstream depends on MediaPipe's own types.
"""
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


class PoseLandmark:
    """The BlazePose 33-point indices we care about (shoulders/elbows/wrists)."""
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
    LEFT_ELBOW, RIGHT_ELBOW = 13, 14
    LEFT_WRIST, RIGHT_WRIST = 15, 16

    @classmethod
    def arm_joints(cls, arm):
        """Return the (shoulder, elbow, wrist) indices for 'left' or 'right'."""
        if arm == "left":
            return cls.LEFT_SHOULDER, cls.LEFT_ELBOW, cls.LEFT_WRIST
        return cls.RIGHT_SHOULDER, cls.RIGHT_ELBOW, cls.RIGHT_WRIST


class PoseEstimator:
    """Thin, single-responsibility wrapper around a MediaPipe PoseLandmarker."""

    def __init__(self, model_path):
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,   # ordered video frames
            num_poses=1,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def estimate(self, frame_bgr):
        """Return a list of 33 (x, y) normalised points, or None if no person."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, int(time.time() * 1000))
        if not result.pose_landmarks:
            return None
        return [(lm.x, lm.y) for lm in result.pose_landmarks[0]]

    def close(self):
        self._landmarker.close()
