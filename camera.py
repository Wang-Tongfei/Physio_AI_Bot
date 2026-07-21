import os
import time

import cv2


def _open_camera(index):
    """Try several backends; return (cap, backend_name) only for a LIVE
    (non-black) camera, else (None, None)."""
    backends = [("DSHOW", cv2.CAP_DSHOW),    # DirectShow: most reliable on this laptop
                ("MSMF", cv2.CAP_MSMF),      # Media Foundation fallback
                ("default", cv2.CAP_ANY)]
    for name, backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        for _ in range(15):                  # warm-up: sensor may need a few frames
            ok, frame = cap.read()
            if ok and frame is not None and float(frame.mean()) > 10:
                return cap, name             # got a non-black frame -> use it
            time.sleep(0.05)
        cap.release()                         # opened but only black frames -> reject
    return None, None


class VideoSource:
    """Uniform read()/release() wrapper over a webcam OR a recorded video file."""

    def __init__(self, cap, label):
        self._cap = cap
        self.label = label
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if 1 < fps < 120 else 20.0

    def read(self):
        return self._cap.read()

    def release(self):
        self._cap.release()

    @classmethod
    def open(cls, source, default_index=0):
        # a recorded video file (test mode, no webcam needed)
        if source is not None and os.path.isfile(str(source)):
            cap = cv2.VideoCapture(str(source))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video file: {source}")
            return cls(cap, f"file:{source}")

        # a camera index (explicit, or auto-detect starting from the default)
        index = default_index
        if source is not None:
            try:
                index = int(source)
            except ValueError:
                raise RuntimeError(
                    f"'{source}' is neither a camera index nor an existing file.")

        cap, backend = _open_camera(index)
        if cap is None:                        # requested index was black: scan others
            for alt in range(3):
                if alt == index:
                    continue
                cap, backend = _open_camera(alt)
                if cap is not None:
                    index = alt
                    break
        if cap is None:
            raise RuntimeError(
                "No camera detected")
        return cls(cap, f"camera {index} ({backend})")
