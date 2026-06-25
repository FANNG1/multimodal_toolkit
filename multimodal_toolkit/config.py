from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # Allows CLI --help before project dependencies are installed.
    load_dotenv = None


def _load_env() -> None:
    if load_dotenv is None:
        return
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)


_load_env()

S3_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://127.0.0.1:9000")
S3_KEY = os.getenv("MINIO_ROOT_USER", "minioadmin")
S3_SECRET = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
S3_REGION = os.getenv("MINIO_REGION", "us-east-1")
S3_USE_SSL = S3_ENDPOINT.startswith("https")

ASR_MODEL = os.getenv("ASR_MODEL", "iic/SenseVoiceSmall")
ASR_DEVICE = os.getenv("ASR_DEVICE", "cpu")

DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

MIN_DURATION_S = float(os.getenv("MIN_DURATION_S", "0"))
MAX_DURATION_S = float(os.getenv("MAX_DURATION_S", "1800"))

EMBED_BACKEND = os.getenv("EMBED_BACKEND", "signal")
EMBED_MODEL = os.getenv("EMBED_MODEL", "facebook/wav2vec2-base")
EMBED_DIM = int(os.getenv("EMBED_DIM", "128"))

USE_RAY = os.getenv("USE_RAY", "0").lower() in ("1", "true", "yes")
RAY_ADDRESS = os.getenv("RAY_ADDRESS") or None  # None = start/join local Ray

PRIMARY_REASONS = [
    "价格敏感",
    "套餐不匹配",
    "服务体验差",
    "竞品影响",
    "账户或设备变化",
    "非本人办理",
    "其他",
]

TEXT_EMOTIONS = ["平静", "不满", "焦急", "愤怒", "投诉倾向", "未知"]
