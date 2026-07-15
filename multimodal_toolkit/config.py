from __future__ import annotations

import os
from pathlib import Path
from typing import TypeVar

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

_T = TypeVar("_T", int, float)


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


def _env_number(name: str, default: _T, parser: type[_T], *, minimum: _T) -> _T:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = parser(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a {parser.__name__}, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def env_int(name: str, default: int, *, minimum: int = 1) -> int:
    return _env_number(name, default, int, minimum=minimum)


def env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    return _env_number(name, default, float, minimum=minimum)


def env_optional_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip() or raw.strip().lower() in {"auto", "none"}:
        return None
    return env_int(name, 1)


def env_choice(name: str, default: str, choices: set[str]) -> str:
    value = os.getenv(name, default).strip().lower()
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
    return value

S3_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://127.0.0.1:9000")
S3_KEY = os.getenv("MINIO_ROOT_USER", "minioadmin")
S3_SECRET = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
S3_REGION = os.getenv("MINIO_REGION", "us-east-1")
S3_USE_SSL = S3_ENDPOINT.startswith("https")
S3_MAX_CONNECTIONS = env_int("S3_MAX_CONNECTIONS", 8)
S3_RETRY_INITIAL_BACKOFF_MS = env_int("S3_RETRY_INITIAL_BACKOFF_MS", 1000)
S3_CONNECT_TIMEOUT_MS = env_int("S3_CONNECT_TIMEOUT_MS", 10_000)
S3_READ_TIMEOUT_MS = env_int("S3_READ_TIMEOUT_MS", 60_000)
S3_NUM_TRIES = env_int("S3_NUM_TRIES", 5)
S3_RETRY_MODE = env_choice("S3_RETRY_MODE", "adaptive", {"adaptive", "standard"})

USE_RAY = env_bool("USE_RAY", False)
RAY_ADDRESS = os.getenv("RAY_ADDRESS") or None  # None = start/join local Ray

# Keep USE_RAY as a compatibility fallback. DAFT_RUNNER is the canonical switch.
DAFT_RUNNER = env_choice(
    "DAFT_RUNNER",
    "ray" if USE_RAY else "native",
    {"native", "ray"},
)
DAFT_NATIVE_NUM_THREADS = env_int("DAFT_NATIVE_NUM_THREADS", 4)
DAFT_DEFAULT_MORSEL_SIZE = env_int("DAFT_DEFAULT_MORSEL_SIZE", 32)
DAFT_ENABLE_DYNAMIC_BATCHING = env_bool("DAFT_ENABLE_DYNAMIC_BATCHING", True)
DAFT_ENABLE_SCAN_TASK_SPLIT_AND_MERGE = env_bool(
    "DAFT_ENABLE_SCAN_TASK_SPLIT_AND_MERGE", True
)
DAFT_SCAN_TASKS_MIN_SIZE_BYTES = env_int("DAFT_SCAN_TASKS_MIN_SIZE_BYTES", 32 * 1024 * 1024)
DAFT_SCAN_TASKS_MAX_SIZE_BYTES = env_int("DAFT_SCAN_TASKS_MAX_SIZE_BYTES", 128 * 1024 * 1024)
if DAFT_SCAN_TASKS_MAX_SIZE_BYTES < DAFT_SCAN_TASKS_MIN_SIZE_BYTES:
    raise ValueError("DAFT_SCAN_TASKS_MAX_SIZE_BYTES must be >= DAFT_SCAN_TASKS_MIN_SIZE_BYTES")
DAFT_MAX_SOURCES_PER_SCAN_TASK = env_int("DAFT_MAX_SOURCES_PER_SCAN_TASK", 16)
DAFT_SCANTASK_MAX_PARALLEL = env_int("DAFT_SCANTASK_MAX_PARALLEL", 4)
DAFT_JSON_TARGET_FILESIZE = env_int("DAFT_JSON_TARGET_FILESIZE", 128 * 1024 * 1024)
DAFT_MAINTAIN_ORDER = env_bool("DAFT_MAINTAIN_ORDER", False)
# 600s：Ray 上几十个 actor 同时冷启动 + 首次下载模型时，300s 会集体超时。
DAFT_ACTOR_UDF_READY_TIMEOUT = env_int("DAFT_ACTOR_UDF_READY_TIMEOUT", 600)

# analyze 阶段的分区数。manifest 只有几 MB，scan task 按字节切分不起作用，
# 不显式切分的话整条 download+分析链路在 Ray 上只有一个 task。
# 未设置（或 auto）时：Ray 模式按集群 CPU 数推导，native 模式不切分。
ANALYZE_NUM_PARTITIONS = env_optional_int("ANALYZE_NUM_PARTITIONS")

LANCE_MAX_ROWS_PER_FILE = env_int("LANCE_MAX_ROWS_PER_FILE", 100_000)
LANCE_MAX_BYTES_PER_FILE = env_int("LANCE_MAX_BYTES_PER_FILE", 512 * 1024 * 1024)
