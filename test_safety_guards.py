import unittest

from config import Config
from exercise_analysis import create_analyzer
from pose_estimator import PoseLandmark, PoseResult


def pose_with_confidence(confidence):
    points = [(0.5, 0.5) for _ in range(33)]
    return PoseResult(points, [confidence] * 33, [confidence] * 33)


def curl_pose(flexed=False):
    points = [(0.5, 0.5) for _ in range(33)]
    points[PoseLandmark.RIGHT_SHOULDER] = (0.5, 0.25)
    points[PoseLandmark.RIGHT_ELBOW] = (0.5, 0.50)
    points[PoseLandmark.RIGHT_HIP] = (0.5, 0.75)
    points[PoseLandmark.RIGHT_WRIST] = (
        (0.55, 0.27) if flexed else (0.5, 0.75)
    )
    return PoseResult(points, [1.0] * 33, [1.0] * 33)


class SafetyGuardTest(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()

    def test_inferred_low_confidence_body_never_advances_any_exercise(self):
        unreliable = pose_with_confidence(0.1)
        for exercise in ("bicep_curl", "squat", "plank", "pushup"):
            analyzer = create_analyzer(exercise, self.cfg)
            for frame in range(100):
                result = analyzer.update(unreliable, now=frame / 30)
            self.assertEqual(analyzer.progress, 0, exercise)
            self.assertFalse(result.form_ok, exercise)
            self.assertIn("view", result.status.lower(), exercise)

    def test_curl_rejects_implausibly_fast_angle_changes(self):
        analyzer = create_analyzer("bicep_curl", self.cfg)
        timestamp = 0.0
        for _ in range(5):
            analyzer.update(curl_pose(False), now=timestamp)
            timestamp += 0.01
        for _ in range(3):
            analyzer.update(curl_pose(True), now=timestamp)
            timestamp += 0.01
        for _ in range(4):
            result = analyzer.update(curl_pose(False), now=timestamp)
            timestamp += 0.01
        self.assertEqual(analyzer.progress, 0)
        self.assertEqual(result.fault, "movement_too_fast")

    def test_curl_counts_a_visible_controlled_repetition(self):
        analyzer = create_analyzer("bicep_curl", self.cfg)
        timestamp = 0.0
        for _ in range(5):
            analyzer.update(curl_pose(False), now=timestamp)
            timestamp += 0.1
        for _ in range(3):
            analyzer.update(curl_pose(True), now=timestamp)
            timestamp += 0.15
        for _ in range(4):
            analyzer.update(curl_pose(False), now=timestamp)
            timestamp += 0.15
        self.assertEqual(analyzer.progress, 1)


if __name__ == "__main__":
    unittest.main()
