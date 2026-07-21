"""
hud.py -- all on-screen drawing (the "therapist view"). Pure presentation.

It reads a HudState and paints it onto the frame. It contains NO pose, rep, or
notification logic -- swapping the whole UI would touch only this file.
"""
from dataclasses import dataclass

import cv2

# BlazePose stick-figure connections (torso + both arms) -- a drawing concern.
SKELETON_CONNECTIONS = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12)]

GREEN = (0, 200, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 215, 255)


@dataclass
class HudState:
    """Everything the HUD needs to render one frame -- and nothing more."""
    reps: int
    target_reps: int
    arm: str
    angle: float | None
    form_status: str
    form_ok: bool
    complete: bool
    fault_active: bool
    landmarks: list | None


def draw(frame, state):
    """Render the HUD for one frame, in place."""
    h, w = frame.shape[:2]
    if state.landmarks:
        _draw_skeleton(frame, state.landmarks)

    # top panel: reps + exercise + live elbow angle
    _panel(frame, 0, 0, w, 90)
    cv2.putText(frame, f"REPS: {state.reps}/{state.target_reps}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
    cv2.putText(frame, f"Exercise: Bicep curl ({state.arm} arm)", (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
    if state.angle is not None:
        cv2.putText(frame, f"Elbow: {int(state.angle)} deg", (w - 200, 35),
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
    for p in pts:
        cv2.circle(frame, p, 4, WHITE, -1)
