"""
send_test_video.py -- diagnose why the Telegram bot isn't sending video clips.

Sends the MOST RECENT clip in clips/ SYNCHRONOUSLY (no background thread), so
any Telegram error is printed in full instead of being swallowed.

Usage (same window, after setting the env vars):
    $env:PHYSIO_TG_TOKEN = "8536638092:AA...token..."   # PowerShell
    $env:PHYSIO_TG_CHAT  = "123456789"
    python send_test_video.py                # newest clip in clips/
    python send_test_video.py clips\foo.mp4  # a specific file

On macOS/Linux use:  export PHYSIO_TG_TOKEN="..."  etc.
"""
import glob
import os
import sys

try:
    import requests
except ImportError:
    sys.exit("Install requests first:  pip install requests")

TOKEN = os.environ.get("PHYSIO_TG_TOKEN", "").strip()
CHAT = os.environ.get("PHYSIO_TG_CHAT", "").strip()

if not TOKEN or not CHAT:
    sys.exit("Set PHYSIO_TG_TOKEN and PHYSIO_TG_CHAT environment variables first.")

# Pick the clip to send: an explicit path, else the newest file in clips/.
if len(sys.argv) > 1:
    clip = sys.argv[1]
else:
    here = os.path.dirname(os.path.abspath(__file__))
    clips = sorted(glob.glob(os.path.join(here, "clips", "*.mp4")),
                   key=os.path.getmtime)
    if not clips:
        sys.exit("No clips found in clips/. Do a partial rep first, then re-run.")
    clip = clips[-1]

if not os.path.isfile(clip):
    sys.exit(f"File not found: {clip}")

size_mb = os.path.getsize(clip) / (1024 * 1024)
print(f"Token ...{TOKEN[-6:]}   Chat {CHAT}")
print(f"Clip: {clip}  ({size_mb:.2f} MB)\n")

# --- Step 1: confirm the credentials with a plain text message ---------------
print("Step 1: sendMessage (checks token + chat id) ...")
r1 = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                   data={"chat_id": CHAT, "text": "Video test: step 1 (text) OK"},
                   timeout=30)
print("  status:", r1.status_code, "| response:", r1.text[:200])
if not r1.ok:
    print("\n  Text send failed -> fix this first (401=token, 400=chat id, "
          "403=press Start on the bot). Not a video problem.")
    sys.exit(1)

# --- Step 2: send the video with an explicit filename + MIME type ------------
print("\nStep 2: sendVideo (the actual clip) ...")
with open(clip, "rb") as fh:
    files = {"video": (os.path.basename(clip), fh, "video/mp4")}
    r2 = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendVideo",
                       data={"chat_id": CHAT, "caption": "Video test: step 2 (clip)"},
                       files=files, timeout=120)
print("  status:", r2.status_code, "| response:", r2.text[:300])

if r2.ok:
    print("\nSUCCESS -- the clip was delivered. If the live app still fails, the")
    print("cause is the daemon thread being killed at exit (see notifier fix).")
else:
    print("\nsendVideo FAILED. Common causes:")
    print("  413 / 'Request Entity Too Large' -> clip over Telegram's limit.")
    print("  400 'wrong file identifier/HTTP URL' or codec complaint -> try")
    print("       sendDocument instead, or re-encode the clip.")
    print("  If step 1 worked but step 2 fails, it is video-specific, not creds.")
