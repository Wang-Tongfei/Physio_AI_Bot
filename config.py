import os
from dataclasses import dataclass, field

_HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Config:
    # video source
    camera_index: int = 0                 # default built-in webcam

    # exercise + form thresholds (degrees)
    target_reps: int = 10                 # session complete after this many good reps
    track_arm: str = "right"              # "right" or "left"
    extended_angle: float = 160           # arm counts as fully extended above this
    flexed_angle: float = 45              # arm counts as fully curled below this
    partial_up_angle: float = 80          # up to here but not to flexed = partial rep

    # flagged clips
    clip_buffer_seconds: float = 4
    clips_dir: str = os.path.join(_HERE, "clips")

    # pose model
    model_path: str = os.path.join(_HERE, "models", "pose_landmarker_lite.task")

    # telegram (read from OS environment variables so no secret is in the code).
    # Set them permanently for your Windows user with, e.g.:
    #   [Environment]::SetEnvironmentVariable("PHYSIO_TG_TOKEN", "...", "User")
    #   [Environment]::SetEnvironmentVariable("PHYSIO_TG_CHAT",  "...", "User")
    # then restart your terminal / VS Code so new processes pick them up.
    telegram_token: str = field(
        default_factory=lambda: os.environ.get("PHYSIO_TG_TOKEN", "").strip())
    telegram_chat_id: str = field(
        default_factory=lambda: os.environ.get("PHYSIO_TG_CHAT", "").strip())
