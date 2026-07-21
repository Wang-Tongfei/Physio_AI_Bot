import os
import sys
import time

import cv2

import hud
from config import Config
from camera import VideoSource
from pose_estimator import PoseEstimator, PoseLandmark
from form_analysis import RepCounter, RepEvent, calculate_angle
from clip_recorder import ClipRecorder
from notifier import TelegramNotifier


def elbow_angle(landmarks, arm):
    """Elbow angle for the chosen arm, from the pose landmarks."""
    shoulder, elbow, wrist = PoseLandmark.arm_joints(arm)
    return calculate_angle(landmarks[shoulder], landmarks[elbow], landmarks[wrist])


def main():
    cfg = Config()

    if not os.path.exists(cfg.model_path):
        sys.exit(f"Pose model not found at {cfg.model_path}.\n"
                 "Download pose_landmarker_lite.task into the models/ folder.")

    # build the pieces (each gets only what it needs, from cfg)
    source_arg = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        source = VideoSource.open(source_arg, cfg.camera_index)
    except RuntimeError as exc:
        sys.exit(str(exc))

    pose = PoseEstimator(cfg.model_path)
    counter = RepCounter(cfg.target_reps, cfg.extended_angle,
                         cfg.flexed_angle, cfg.partial_up_angle)
    recorder = ClipRecorder(cfg.clips_dir, source.fps, cfg.clip_buffer_seconds)
    notifier = TelegramNotifier(cfg.telegram_token, cfg.telegram_chat_id)

    # session state that belongs to the orchestrator, not any component
    arm = cfg.track_arm
    form_status = "Ready"
    form_ok = True
    notified_complete = False
    last_fault_time = 0.0

    print(f"Physio Form Monitor running on {source.label}.")
    print("Controls:  q=quit   r=reset   a=switch arm")
    if notifier.enabled:
        print("Telegram alerts: ENABLED")
        notifier.send_message("Physio Form Monitor: session started.")
    else:
        print("Telegram alerts: disabled (set PHYSIO_TG_TOKEN and PHYSIO_TG_CHAT).")

    while True:
        ok, frame = source.read()
        if not ok:
            print("Video source ended / feed lost.")
            break
        frame = cv2.flip(frame, 1)          # mirror so it feels like a mirror

        landmarks = pose.estimate(frame)
        angle = None

        if landmarks is not None and not counter.is_complete:
            angle = elbow_angle(landmarks, arm)
            event = counter.update(angle)
            if event is RepEvent.GOOD_REP:
                form_ok = True
                form_status = f"Good rep!  ({counter.reps}/{counter.target_reps})"
            elif event is RepEvent.PARTIAL_REP:
                form_ok = False
                form_status = "Partial rep - curl all the way up!"
                path = recorder.save("partial_rep")
                if path:
                    print(f"[ALERT] Bad form (partial_rep). Clip saved: {path}")
                    notifier.send_video(
                        path, caption="Physio alert: bad form (partial rep). Please review.")
                last_fault_time = time.time()
        elif landmarks is None and not counter.is_complete:
            form_status = "Step into view of the camera"
            form_ok = True

        recorder.add(frame)                 # rolling buffer for the NEXT clip

        if counter.is_complete and not notified_complete:
            notifier.send_message(
                f"Workout complete: {counter.reps}/{counter.target_reps} good reps.")
            print("[NOTIFY] Workout complete -> therapist notified.")
            notified_complete = True

        # draw + show (view layer only reads state)
        state = hud.HudState(
            reps=counter.reps, target_reps=counter.target_reps, arm=arm,
            angle=angle, form_status=form_status, form_ok=form_ok,
            complete=counter.is_complete,
            fault_active=(time.time() - last_fault_time < 1.0),
            landmarks=landmarks)
        hud.draw(frame, state)
        cv2.imshow("Physio Form Monitor  (q=quit  r=reset  a=switch arm)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            counter.reset()
            notified_complete = False
            form_status, form_ok = "Session reset", True
            print("Session reset.")
        elif key == ord("a"):
            arm = "left" if arm == "right" else "right"
            print(f"Now tracking the {arm} arm.")

    # shutdown
    source.release()
    cv2.destroyAllWindows()
    pose.close()
    print("\n--- Session summary ---")
    print(f"Reps completed: {counter.reps}/{counter.target_reps}")
    print(f"Workout complete: {'YES' if counter.is_complete else 'no'}")


if __name__ == "__main__":
    main()
