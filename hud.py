"""
hud.py -- all on-screen drawing (the "therapist view"). Pure presentation.

It reads a HudState and paints it onto the frame. It contains NO pose, rep, or
notification logic -- swapping the whole UI would touch only this file.
"""
from dataclasses import dataclass
import math

import cv2

# Face landmarks (0-10) and Pose's coarse hand landmarks (17-22) are
# intentionally omitted. Hands are rendered separately by the 21-point model.
SKELETON_CONNECTIONS = [
    # torso and arms, ending at the wrists
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    # trunk and legs
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    # feet
    (27, 29), (29, 31), (27, 31),
    (28, 30), (30, 32), (28, 32),
]
BODY_LANDMARK_INDICES = set(range(11, 17)) | set(range(23, 33))

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

GREEN = (0, 200, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 215, 255)
CYAN = (255, 220, 0)


@dataclass
class HudState:
    """Everything the HUD needs to render one frame -- and nothing more."""
    exercise: str
    progress: float
    target: float
    unit: str
    arm: str
    metric_label: str
    metric_value: float | None
    form_status: str
    form_ok: bool
    complete: bool
    fault_active: bool
    landmarks: list | None
    hand_landmarks: list


def draw(frame, state):
    """Render the HUD for one frame, in place."""
    h, w = frame.shape[:2]
    if state.landmarks:
        _draw_skeleton(frame, state.landmarks)
    for hand in state.hand_landmarks:
        _draw_hand(frame, hand)

    # top panel: progress + exercise + live joint/body metric
    _panel(frame, 0, 0, w, 90)
    if state.unit == "sec":
        progress = f"{state.progress:.1f}/{state.target:.0f} SEC"
    else:
        progress = f"{int(state.progress)}/{int(state.target)} REPS"
    cv2.putText(frame, progress, (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
    exercise = state.exercise
    if state.exercise == "Bicep curl":
        exercise += f" ({state.arm} arm)"
    cv2.putText(frame, f"Exercise: {exercise}", (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
    if state.metric_value is not None:
        metric = f"{state.metric_label}: {int(state.metric_value)} deg"
        cv2.putText(frame, metric, (w - 230, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

    # bottom panel: form status (green = good, red = fault)
    _panel(frame, 0, h - 45, w, 45)
    colour = GREEN if state.form_ok else RED
    cv2.putText(frame, f"Form: {state.form_status}", (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

    # red border flash right after a fault clip is recorded
    if state.fault_active:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), RED, 8)

    if state.complete:
        _panel(frame, 0, h // 2 - 50, w, 100, colour=(0, 120, 0), alpha=0.7)
        cv2.putText(frame, "WORKOUT COMPLETE", (w // 2 - 220, h // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 3)


# private drawing helpers
def _panel(frame, x, y, w, h, colour=BLACK, alpha=0.55):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), colour, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _draw_skeleton(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for (x, y) in landmarks]
    for a, b in SKELETON_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], YELLOW, 2)
    for index in BODY_LANDMARK_INDICES:
        cv2.circle(frame, pts[index], 4, WHITE, -1)
    _draw_anonymous_head(frame, pts)


def _draw_anonymous_head(frame, pts):
    """Draw a rotatable anonymous oval inferred from ears and shoulders.

    Eye, nose, and mouth landmarks are neither displayed nor used.
    """
    left_ear, right_ear = pts[7], pts[8]
    left_shoulder, right_shoulder = pts[11], pts[12]
    left_hip, right_hip = pts[23], pts[24]
    shoulder_mid = (
        (left_shoulder[0] + right_shoulder[0]) / 2.0,
        (left_shoulder[1] + right_shoulder[1]) / 2.0,
    )
    hip_mid = (
        (left_hip[0] + right_hip[0]) / 2.0,
        (left_hip[1] + right_hip[1]) / 2.0,
    )
    shoulder_width = math.dist(left_shoulder, right_shoulder)
    ear_width = math.dist(left_ear, right_ear)
    if shoulder_width < 4:
        return

    ear_mid = (
        (left_ear[0] + right_ear[0]) / 2.0,
        (left_ear[1] + right_ear[1]) / 2.0,
    )
    head_up_x = ear_mid[0] - shoulder_mid[0]
    head_up_y = ear_mid[1] - shoulder_mid[1]
    head_up_length = math.hypot(head_up_x, head_up_y)
    if head_up_length < 1:
        head_up_x = shoulder_mid[0] - hip_mid[0]
        head_up_y = shoulder_mid[1] - hip_mid[1]
        head_up_length = math.hypot(head_up_x, head_up_y)
    if head_up_length < 1:
        return
    head_up_x /= head_up_length
    head_up_y /= head_up_length

    # Fall back to shoulder proportions if one/both ear landmarks are poor.
    ears_reliable = shoulder_width * 0.18 <= ear_width <= shoulder_width * 0.9
    face_width = ear_width if ears_reliable else shoulder_width * 0.48
    if ears_reliable:
        centre = (
            int(ear_mid[0] + head_up_x * face_width * 0.12),
            int(ear_mid[1] + head_up_y * face_width * 0.12),
        )
        angle = math.degrees(math.atan2(
            right_ear[1] - left_ear[1],
            right_ear[0] - left_ear[0],
        ))
    else:
        centre = (
            int(shoulder_mid[0] + head_up_x * shoulder_width * 0.72),
            int(shoulder_mid[1] + head_up_y * shoulder_width * 0.72),
        )
        angle = math.degrees(math.atan2(
            right_shoulder[1] - left_shoulder[1],
            right_shoulder[0] - left_shoulder[0],
        ))

    axes = (
        max(8, int(face_width * 0.62)),
        max(11, int(face_width * 0.82)),
    )
    neck = (
        int(shoulder_mid[0] + head_up_x * shoulder_width * 0.18),
        int(shoulder_mid[1] + head_up_y * shoulder_width * 0.18),
    )
    cv2.line(frame, left_shoulder, neck, YELLOW, 2)
    cv2.line(frame, right_shoulder, neck, YELLOW, 2)
    cv2.ellipse(frame, centre, axes, angle, 0, 360, YELLOW, 2)


def _draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for (x, y) in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], CYAN, 2)
    for point in pts:
        cv2.circle(frame, point, 3, WHITE, -1)
