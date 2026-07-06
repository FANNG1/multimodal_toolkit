from __future__ import annotations

import os

from .. import config as _shared_config

_shared_config.load_env()

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
