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

    def send_message(self, text):
        """Send a text alert (e.g. 'workout complete'). Non-blocking."""
        if not self.enabled:
            return
        self._post_async("sendMessage", {"chat_id": self._chat_id, "text": text})

    def send_video(self, path, caption=""):
        """Send an evidence clip for the therapist to review. Non-blocking."""
        if not self.enabled:
            return

        def _job():
            with open(path, "rb") as fh:
                self._post("sendVideo",
                           data={"chat_id": self._chat_id, "caption": caption},
                           files={"video": fh})

        threading.Thread(target=_job, daemon=True).start()

    # internals
    def _post_async(self, method, data):
        threading.Thread(target=self._post, args=(method,),
                         kwargs={"data": data}, daemon=True).start()

    def _post(self, method, data=None, files=None):
        url = f"https://api.telegram.org/bot{self._token}/{method}"
        try:
            resp = requests.post(url, data=data, files=files, timeout=30)
            if not resp.ok:
                print(f"[Telegram] {method} failed: {resp.status_code} {resp.text[:150]}")
        except Exception as exc:
            print(f"[Telegram] {method} error: {exc}")
