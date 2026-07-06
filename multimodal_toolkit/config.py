from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # Allows CLI --help before project dependencies are installed.
    load_dotenv = None


def load_env() -> None:
    if load_dotenv is None:
        return
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)


load_env()

S3_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://127.0.0.1:9000")
S3_KEY = os.getenv("MINIO_ROOT_USER", "minioadmin")
S3_SECRET = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
S3_REGION = os.getenv("MINIO_REGION", "us-east-1")
S3_USE_SSL = S3_ENDPOINT.startswith("https")

USE_RAY = os.getenv("USE_RAY", "0").lower() in ("1", "true", "yes")
RAY_ADDRESS = os.getenv("RAY_ADDRESS") or None  # None = start/join local Ray
