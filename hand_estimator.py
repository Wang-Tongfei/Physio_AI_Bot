"""Detailed two-hand tracking using MediaPipe's 21-point hand model."""
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


class HandEstimator:
    def __init__(self, model_path, num_hands=2):
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def estimate(self, frame_bgr):
        """Return zero to two hands, each containing 21 normalised (x, y) points."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(
            int(time.monotonic() * 1000),
            self._last_timestamp_ms + 1,
        )
        self._last_timestamp_ms = timestamp_ms
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        return [
            [(landmark.x, landmark.y) for landmark in hand]
            for hand in result.hand_landmarks
        ]

    def close(self):
        self._landmarker.close()
