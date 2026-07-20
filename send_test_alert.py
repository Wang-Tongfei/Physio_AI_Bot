"""
send_test_alert.py -- quick check that your Telegram bot can message you.

Sends ONE test message to your therapist chat and prints the exact Telegram
API response, so you can see WHY alerts do or don't arrive -- independently of
the camera or the main program.

Setup (run in the SAME PowerShell window, then run this script):
    $env:PHYSIO_TG_TOKEN = "8536638092:AA...your token..."
    $env:PHYSIO_TG_CHAT  = "123456789"
    python send_test_alert.py
"""
import os
import sys

try:
    import requests
except ImportError:
    sys.exit("Install requests first:  pip install requests")

TOKEN = os.environ.get("PHYSIO_TG_TOKEN", "").strip()
CHAT = os.environ.get("PHYSIO_TG_CHAT", "").strip()

if not TOKEN or not CHAT:
    sys.exit(
        "Environment variables not set. In THIS PowerShell window run:\n"
        '  $env:PHYSIO_TG_TOKEN = "8536638092:AA...your token..."\n'
        '  $env:PHYSIO_TG_CHAT  = "123456789"\n'
        "  python send_test_alert.py"
    )

print(f"Token ends with: ...{TOKEN[-6:]}    Chat id: {CHAT}")

url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
resp = requests.post(
    url,
    data={"chat_id": CHAT,
          "text": "Physio Form Monitor: TEST alert. If you can read this, alerts work."},
    timeout=30,
)

print("HTTP status:", resp.status_code)
print("Response   :", resp.text)

if resp.ok:
    print("\nSUCCESS -- check Telegram for the message.")
else:
    print("\nFAILED. Most common causes:")
    print("  401 Unauthorized      -> bot token is wrong; recopy it from BotFather.")
    print("  400 'chat not found'  -> PHYSIO_TG_CHAT is wrong (re-check via getUpdates).")
    print("  403 'bot can't...'    -> open @PhysioUpdateBot in Telegram and press START")
    print("                           (or send it any message) FIRST, then re-run.")
