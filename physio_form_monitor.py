"""
CONTROLS (press key while the webcam window is focused):
    q  = quit
    r  = reset rep count / session
    a  = switch tracked arm (right <-> left)

Run:
    python physio_form_monitor.py
"""

import os
import sys
import time
import threading
from collections import deque

import numpy as np

# --- Import the vision libraries with a friendly message if they are missing ---
try:
    import cv2
except ImportError:
    sys.exit("OpenCV is not installed. Run:  pip install opencv-python")

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    sys.exit("MediaPipe is not installed. Run:  pip install mediapipe")

try:
    import requests            # used to send Telegram alerts + video clips
except ImportError:
    requests = None            # alerts fall back to console-only if missing


# CONFIGURATION
CAMERA_INDEX = 0            # 0 = default built-in laptop webcam

TARGET_REPS = 10            # session is "complete" after this many good reps
TRACK_ARM = "right"         # "right" or "left" — which arm to score

# Elbow-angle thresholds (degrees) for a bicep curl:
#   arm straight (down) has a LARGE angle; arm curled (up) has a SMALL angle.
EXTENDED_ANGLE = 160        # arm counts as fully extended above this angle
FLEXED_ANGLE = 45           # arm counts as fully curled below this angle
# A rep must reach BOTH ends of that range to be "good form". If the person
# comes up but does not pass FLEXED_ANGLE, it is a partial rep (bad form).
PARTIAL_UP_ANGLE = 80       # coming up to here but not to FLEXED = partial rep

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "models", "pose_landmarker_lite.task")
CLIPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clips")

# Rolling buffer length for the "evidence clip" (frames kept before a fault).
CLIP_BUFFER_SECONDS = 4

# Telegram alerts
# Secrets are read from ENVIRONMENT VARIABLES so no token is stored in this file.
# How to set them (Windows PowerShell):
#     $env:PHYSIO_TG_TOKEN = "123456:ABC-your-bot-token"
#     $env:PHYSIO_TG_CHAT  = "123456789"          # therapist's chat id
# If either is unset, alerts simply print to the console instead.
TELEGRAM_TOKEN = os.environ.get("PHYSIO_TG_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("PHYSIO_TG_CHAT", "").strip()
TELEGRAM_ENABLED = bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID and requests is not None)

# MediaPipe PoseLandmarker landmark indices (BlazePose 33-point model).
LM = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13,    "right_elbow": 14,
    "left_wrist": 15,    "right_wrist": 16,
}

# Colours (OpenCV uses BGR, not RGB).
GREEN = (0, 200, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 215, 255)


# HELPER FUNCTIONS
def calculate_angle(a, b, c):
    """Return the angle (degrees) at point b, formed by points a-b-c.

    Each point is an (x, y) pair. Used to measure a joint angle such as the
    elbow (shoulder -> elbow -> wrist).
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    if angle > 180.0:
        angle = 360.0 - angle
    return angle


def draw_skeleton(frame, landmarks):
    """Draw a simple stick-figure over the detected body landmarks."""
    h, w = frame.shape[:2]
    # Pairs of landmark indices to connect with lines (torso + both arms).
    connections = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12)]
    for start, end in connections:
        x1, y1 = int(landmarks[start].x * w), int(landmarks[start].y * h)
        x2, y2 = int(landmarks[end].x * w), int(landmarks[end].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), YELLOW, 2)
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, WHITE, -1)


def panel(frame, x, y, w, h, colour=BLACK, alpha=0.55):
    """Draw a semi-transparent rectangle to make overlaid text readable."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), colour, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def open_camera(index):
    """Open a camera and return (cap, backend_name) only if it yields a LIVE,
    non-black frame. A black-only camera (e.g. a virtual cam whose sensor is off)
    is rejected so the caller can try another index. Returns (None, None) on fail.
    """
    backends = [("DSHOW", cv2.CAP_DSHOW),    # DirectShow: most reliable on this laptop
                ("MSMF", cv2.CAP_MSMF),      # Media Foundation fallback
                ("default", cv2.CAP_ANY)]
    for name, backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        # Warm up: many webcams (and Poly) need a few frames to expose a real image.
        for _ in range(15):
            ok, frame = cap.read()
            if ok and frame is not None and float(frame.mean()) > 10:
                return cap, name            # fix for a non-black frame
            time.sleep(0.05)
        cap.release()                        # reject opened but only black frames
    return None, None


def _tg_post(method, data=None, files=None):
    """Low-level Telegram Bot API call (runs on a background thread)."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}"
    try:
        r = requests.post(url, data=data, files=files, timeout=30)
        if not r.ok:
            print(f"[Telegram] {method} failed: {r.status_code} {r.text[:150]}")
    except Exception as exc:                       # never let a network error crash the app
        print(f"[Telegram] {method} error: {exc}")


def send_telegram_message(text):
    """Send a text alert to the assigned therapist (non-blocking)."""
    if not TELEGRAM_ENABLED:
        return
    threading.Thread(
        target=_tg_post, args=("sendMessage",),
        kwargs={"data": {"chat_id": TELEGRAM_CHAT_ID, "text": text}},
        daemon=True,
    ).start()


def send_telegram_video(path, caption=""):
    """Send an evidence clip to the assigned therapist (non-blocking)."""
    if not TELEGRAM_ENABLED:
        return

    def _job():
        with open(path, "rb") as fh:
            _tg_post("sendVideo",
                     data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption},
                     files={"video": fh})

    threading.Thread(target=_job, daemon=True).start()


def save_clip(frames, fps, size, reason):
    """Write the buffered frames to an mp4 evidence clip and return its path."""
    os.makedirs(CLIPS_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(CLIPS_DIR, f"{reason}_{stamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    for f in frames:
        writer.write(f)
    writer.release()
    return path


def main():
    global TRACK_ARM

    if not os.path.exists(MODEL_PATH):
        sys.exit(
            f"Pose model not found at:\n  {MODEL_PATH}\n\n"
            "Download it once with:\n"
            "  curl -L -o models/pose_landmarker_lite.task \\\n"
            "    https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
        )

    # Build the MediaPipe PoseLandmarker (VIDEO mode = ordered frames)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    # Choose the video SOURCE
    # No argument: auto-detect the live laptop webcam.
    # An integer (0/1): force that camera index.
    # A file path: read from a recorded video (great for testing form detection without a working webcam).
    #   e.g.  python physio_form_monitor.py 1
    #         python physio_form_monitor.py my_curls.mp4
    source = sys.argv[1] if len(sys.argv) > 1 else None

    if source is not None and os.path.isfile(source):
        cap = cv2.VideoCapture(source)          # video-file test mode
        if not cap.isOpened():
            sys.exit(f"Could not open video file: {source}")
        backend, cam_index = "file", source
        print(f"Reading from video file: {source}")
    else:
        cam_index = CAMERA_INDEX
        if source is not None:
            try:
                cam_index = int(source)
            except ValueError:
                sys.exit(f"'{source}' is neither a camera index nor an existing file.")

        cap, backend = open_camera(cam_index)
        if cap is None:                         # requested index was black/unavailable -> scan others
            for alt in range(3):
                if alt == cam_index:
                    continue
                cap, backend = open_camera(alt)
                if cap is not None:
                    cam_index = alt
                    break
        if cap is None:
            sys.exit(
                "Could not get a LIVE (non-black) picture from any camera (indices 0-2).\n"
                " Check that your webcam is connected and not in use by another app."
            )
        print(f"Using camera index {cam_index} via the {backend} backend.")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps_guess = cap.get(cv2.CAP_PROP_FPS)
    fps = fps_guess if 1 < fps_guess < 120 else 20.0

    clip_buffer = deque(maxlen=int(CLIP_BUFFER_SECONDS * fps))

    # Session / rep-counting state
    reps = 0
    stage = "down"           # "down" = arm extended, "up" = arm curled
    reached_bottom = False   # did the arm fully extend at the bottom of this rep?
    angle_reached_top = False  # did the arm fully curl up during this rep?
    form_status = "Ready"
    form_ok = True
    complete = False
    notified_complete = False   # ensures the "complete" alert is sent only once
    last_fault_time = 0.0

    print("Physio Form Monitor running. Focus the webcam window.")
    print("Controls:  q=quit   r=reset   a=switch arm")
    if TELEGRAM_ENABLED:
        print(f"Telegram alerts: ENABLED (chat {TELEGRAM_CHAT_ID}).")
        send_telegram_message("🏃 Physio Form Monitor: session started.")
    elif requests is None:
        print("Telegram alerts: disabled ('requests' not installed).")
    else:
        print("Telegram alerts: disabled. Set PHYSIO_TG_TOKEN and PHYSIO_TG_CHAT "
              "env vars to enable.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Lost the camera feed.")
            break

        frame = cv2.flip(frame, 1)  # mirror so it feels like a mirror
        h, w = frame.shape[:2]

        # RGB Image; timestamp in milliseconds for VIDEO mode.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, ts_ms)

        angle = None
        person_present = bool(result.pose_landmarks)

        if person_present and not complete:
            lms = result.pose_landmarks[0]
            draw_skeleton(frame, lms)

            # Pick the tracked arm's three joints and compute the elbow angle.
            s = LM[f"{TRACK_ARM}_shoulder"]
            e = LM[f"{TRACK_ARM}_elbow"]
            wr = LM[f"{TRACK_ARM}_wrist"]
            shoulder = (lms[s].x, lms[s].y)
            elbow = (lms[e].x, lms[e].y)
            wrist = (lms[wr].x, lms[wr].y)
            angle = calculate_angle(shoulder, elbow, wrist)

            # Rep state machine
            if angle > EXTENDED_ANGLE:
                # Arm is fully straight = bottom of the curl.
                if stage == "up":
                    # We just came back down after going up -> evaluate the rep.
                    if reached_bottom and angle_reached_top:
                        reps += 1
                        form_status = f"Good rep!  ({reps}/{TARGET_REPS})"
                        form_ok = True
                        if reps >= TARGET_REPS:
                            complete = True
                    else:
                        # Came up but not far enough = partial rep / bad form.
                        form_status = "Partial rep - curl all the way up!"
                        form_ok = False
                        _flag_fault(clip_buffer, fps, (w, h), "partial_rep")
                        last_fault_time = time.time()
                stage = "down"
                reached_bottom = True
                angle_reached_top = False
            elif angle < FLEXED_ANGLE:
                # Arm fully curled = top of a good rep.
                stage = "up"
                angle_reached_top = True
            elif angle < PARTIAL_UP_ANGLE and stage == "down":
                # Started curling up from the bottom.
                stage = "up"
                angle_reached_top = False

        elif not person_present and not complete:
            form_status = "Step into view of the camera"
            form_ok = True

        # Keep the last few seconds of frames for the evidence clip.
        clip_buffer.append(frame.copy())

        #  DRAW THE HUD (mimics the clinic dashboard / therapist view)
        panel(frame, 0, 0, w, 90)
        cv2.putText(frame, f"REPS: {reps}/{TARGET_REPS}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
        cv2.putText(frame, f"Exercise: Bicep curl ({TRACK_ARM} arm)", (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
        if angle is not None:
            cv2.putText(frame, f"Elbow: {int(angle)} deg", (w - 200, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

        status_colour = GREEN if form_ok else RED
        panel(frame, 0, h - 45, w, 45)
        cv2.putText(frame, f"Form: {form_status}", (15, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_colour, 2)

        # Flash a red border briefly when a fault clip was just recorded.
        if time.time() - last_fault_time < 1.0:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), RED, 8)

        if complete:
            if not notified_complete:
                # The "workout complete" notification to the therapist.
                send_telegram_message(
                    f"✅ Workout complete: {reps}/{TARGET_REPS} good reps "
                    f"(bicep curl, {TRACK_ARM} arm).")
                print("[NOTIFY] Workout complete -> therapist notified.")
                notified_complete = True
            panel(frame, 0, h // 2 - 50, w, 100, colour=(0, 120, 0), alpha=0.7)
            cv2.putText(frame, "WORKOUT COMPLETE", (w // 2 - 220, h // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 3)

        cv2.imshow("Physio Form Monitor  (q=quit  r=reset  a=switch arm)", frame)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            reps, stage, complete, form_ok = 0, "down", False, True
            reached_bottom, angle_reached_top = False, False
            notified_complete = False
            form_status = "Session reset"
            print("Session reset.")
        elif key == ord("a"):
            TRACK_ARM = "left" if TRACK_ARM == "right" else "right"
            print(f"Now tracking the {TRACK_ARM} arm.")

    # Print a session summary (the clinic version would send a Telegram msg)
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("\n--- Session summary ---")
    print(f"Reps completed: {reps}/{TARGET_REPS}")
    print(f"Workout complete: {'YES' if complete else 'no'}")
    if os.path.isdir(CLIPS_DIR):
        clips = [f for f in os.listdir(CLIPS_DIR) if f.endswith('.mp4')]
        print(f"Evidence clips saved this run: see {CLIPS_DIR}")


def _flag_fault(clip_buffer, fps, size, reason):
    """Save an evidence clip for a bad-form event and log it to the console."""
    if len(clip_buffer) == 0:
        return
    path = save_clip(list(clip_buffer), fps, size, reason)
    print(f"[ALERT] Bad form ({reason}). Evidence clip saved: {path}")
    # Notify the assigned physiotherapist with the clip attached for review.
    send_telegram_video(
        path,
        caption=f"⚠️ Physio alert: bad form detected ({reason.replace('_', ' ')}). "
                "Please review this clip.",
    )


if __name__ == "__main__":
    main()
