"""
notifier.py -- sends alerts to the physiotherapist via Telegram.

Give it a token + chat id and call send_message / send_video. Sends run on
background threads so the video loop never blocks. IMPORTANT: call flush() before
the app exits so pending uploads (especially videos, which take a second or two)
finish instead of being killed with the process. If unconfigured, every call is a
silent no-op (enabled == False).
"""
import os
import threading

try:
    import requests
except ImportError:                    # alerts degrade to no-ops if requests is missing
    requests = None


class TelegramNotifier:
    def __init__(self, token, chat_id):
        self._token = token
        self._chat_id = chat_id
        self.enabled = bool(token and chat_id and requests is not None)
        self._threads = []

    def send_message(self, text):
        """Send a text alert (e.g. 'workout complete'). Non-blocking."""
        if not self.enabled:
            return
        self._spawn(lambda: self._post(
            "sendMessage", data={"chat_id": self._chat_id, "text": text}))

    def send_video(self, path, caption=""):
        """Send an evidence clip for the therapist to review. Non-blocking."""
        if not self.enabled:
            return

        def _job():
            with open(path, "rb") as fh:
                # Declare filename + MIME type so Telegram treats it as a video.
                files = {"video": (os.path.basename(path), fh, "video/mp4")}
                self._post("sendVideo",
                           data={"chat_id": self._chat_id, "caption": caption},
                           files=files)

        self._spawn(_job)

    def flush(self, timeout=60):
        """Wait for any in-flight sends to finish. Call before the app exits so
        a video upload isn't killed with the process."""
        for t in list(self._threads):
            t.join(timeout)

    # internals
    def _spawn(self, target):
        self._threads = [t for t in self._threads if t.is_alive()]   # prune finished
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        self._threads.append(thread)

    def _post(self, method, data=None, files=None):
        url = f"https://api.telegram.org/bot{self._token}/{method}"
        try:
            resp = requests.post(url, data=data, files=files, timeout=60)
            if not resp.ok:
                print(f"[Telegram] {method} failed: {resp.status_code} {resp.text[:200]}")
        except Exception as exc:
            print(f"[Telegram] {method} error: {exc}")
