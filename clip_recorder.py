import os
import time
from collections import deque

import cv2


class ClipRecorder:
    def __init__(self, clips_dir, fps, buffer_seconds=4):
        self._dir = clips_dir
        self._fps = fps
        self._buffer = deque(maxlen=max(1, int(buffer_seconds * fps)))

    def add(self, frame):
        """Call once per frame to keep the last few seconds available."""
        self._buffer.append(frame.copy())

    def save(self, reason):
        """Write the buffered frames to <clips_dir>/<reason>_<timestamp>.mp4.
        Returns the path, or None if nothing is buffered yet."""
        if not self._buffer:
            return None
        os.makedirs(self._dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self._dir, f"{reason}_{stamp}.mp4")
        height, width = self._buffer[0].shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 self._fps, (width, height))
        for frame in self._buffer:
            writer.write(frame)
        writer.release()
        return path
