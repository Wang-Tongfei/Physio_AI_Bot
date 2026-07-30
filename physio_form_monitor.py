import os
import sys
import time

import cv2

import hud
from config import Config
from camera import VideoSource
from pose_estimator import PoseEstimator
from hand_estimator import HandEstimator
from exercise_analysis import BicepCurlAnalyzer, create_analyzer
from clip_recorder import ClipRecorder
from notifier import TelegramNotifier


EXERCISE_KEYS = {
    ord("1"): "bicep_curl",
    ord("2"): "squat",
    ord("3"): "plank",
    ord("4"): "pushup",
}


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
    hand_pose = None
    if cfg.enable_hand_tracking:
        if os.path.exists(cfg.hand_model_path):
            hand_pose = HandEstimator(cfg.hand_model_path)
        else:
            print(f"Hand model not found at {cfg.hand_model_path}.")
            print("Detailed hand tracking is disabled; body tracking will continue.")
    exercise = cfg.exercise
    analyzer = create_analyzer(exercise, cfg)
    recorder = ClipRecorder(cfg.clips_dir, source.fps, cfg.clip_buffer_seconds)
    notifier = TelegramNotifier(cfg.telegram_token, cfg.telegram_chat_id)

    # session state that belongs to the orchestrator, not any component
    arm = cfg.track_arm
    form_status = "Ready"
    form_ok = True
    notified_complete = False
    last_fault_time = 0.0

    print(f"Physio Form Monitor running on {source.label}.")
    print("Controls: q=quit  r=reset  a=switch arm  "
          "1=curl  2=squat  3=plank  4=push-up")
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
        hand_landmarks = hand_pose.estimate(frame) if hand_pose else []
        metric_label = ""
        metric_value = None

        if landmarks is not None and not analyzer.is_complete:
            result = analyzer.update(landmarks)
            metric_label = result.metric_label
            metric_value = result.metric_value
            form_ok = result.form_ok
            form_status = result.status
            if result.fault:
                path = recorder.save(result.fault)
                if path:
                    print(f"[ALERT] Bad form ({result.fault}). Clip saved: {path}")
                    notifier.send_video(
                        path,
                        caption=f"Physio alert: {analyzer.name} - "
                                f"{result.status}. Please review.")
                last_fault_time = time.time()
        elif landmarks is None and not analyzer.is_complete:
            form_status = "Step into view of the camera"
            form_ok = True

        recorder.add(frame)                 # rolling buffer for the NEXT clip

        if analyzer.is_complete and not notified_complete:
            notifier.send_message(
                f"Workout complete: {analyzer.name} "
                f"{analyzer.progress:.1f}/{analyzer.target:g} {analyzer.unit}.")
            print("[NOTIFY] Workout complete -> therapist notified.")
            notified_complete = True

        # draw + show (view layer only reads state)
        state = hud.HudState(
            exercise=analyzer.name, progress=analyzer.progress,
            target=analyzer.target, unit=analyzer.unit, arm=arm,
            metric_label=metric_label, metric_value=metric_value,
            form_status=form_status, form_ok=form_ok,
            complete=analyzer.is_complete,
            fault_active=(time.time() - last_fault_time < 1.0),
            landmarks=landmarks, hand_landmarks=hand_landmarks)
        hud.draw(frame, state)
        cv2.imshow("Physio Monitor (1=curl 2=squat 3=plank 4=pushup)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            analyzer.reset()
            notified_complete = False
            form_status, form_ok = "Session reset", True
            print("Session reset.")
        elif key == ord("a"):
            arm = "left" if arm == "right" else "right"
            if isinstance(analyzer, BicepCurlAnalyzer):
                analyzer.arm = arm
            print(f"Now tracking the {arm} arm.")
        elif key in EXERCISE_KEYS:
            exercise = EXERCISE_KEYS[key]
            analyzer = create_analyzer(exercise, cfg, arm)
            notified_complete = False
            form_status, form_ok = f"Switched to {analyzer.name}", True
            print(f"Now tracking: {analyzer.name}.")

    # shutdown
    notifier.flush()            # let any in-flight Telegram uploads finish
    source.release()
    cv2.destroyAllWindows()
    pose.close()
    if hand_pose:
        hand_pose.close()
    print("\n--- Session summary ---")
    print(f"{analyzer.name}: {analyzer.progress:.1f}/{analyzer.target:g} "
          f"{analyzer.unit}")
    print(f"Workout complete: {'YES' if analyzer.is_complete else 'no'}")


if __name__ == "__main__":
    main()
