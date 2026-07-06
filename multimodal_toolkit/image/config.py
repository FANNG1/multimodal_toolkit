from __future__ import annotations

import os

from .. import config as _shared_config

_shared_config.load_env()

INSIGHTFACE_MODEL = os.getenv("INSIGHTFACE_MODEL", "buffalo_l")
INSIGHTFACE_ROOT = os.getenv("INSIGHTFACE_ROOT", "")  # "" = insightface default ~/.insightface
FACE_DET_SIZE = int(os.getenv("FACE_DET_SIZE", "640"))
FACE_DET_THRESH = float(os.getenv("FACE_DET_THRESH", "0.3"))
IMAGE_LONG_EDGE = int(os.getenv("IMAGE_LONG_EDGE", "1024"))
FACE_DET_SCORE_MIN = float(os.getenv("FACE_DET_SCORE_MIN", "0.5"))
MIN_FACE_RATIO = float(os.getenv("MIN_FACE_RATIO", "0.01"))
BLUR_THRESHOLD = float(os.getenv("BLUR_THRESHOLD", "100.0"))
FACE_BLUR_THRESHOLD = float(os.getenv("FACE_BLUR_THRESHOLD", "80.0"))

IMAGE_EMBED_MODEL = os.getenv("IMAGE_EMBED_MODEL", "OFA-Sys/chinese-clip-vit-base-patch16")
IMAGE_EMBED_DEVICE = os.getenv("IMAGE_EMBED_DEVICE", "cpu")
IMAGE_EMBED_DIM = int(os.getenv("IMAGE_EMBED_DIM", "512"))
