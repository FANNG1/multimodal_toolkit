from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from multimodal_toolkit import config as project_config


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = REPO_ROOT / ".benchmarks"


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass(frozen=True)
class BenchmarkConfig:
    run_id: str
    manifest_uri: str
    lance_uri: str
    mock_url: str
    ray_address: str = "local"
    max_minutes: float = 15.0
    asr_actor_cpus: float = 4.0
    asr_actor_concurrency: int = 1
    asr_batch_size: int = 1
    llm_concurrency: int = 8
    llm_timeout_s: float = 15.0
    llm_max_attempts: int = 3
    sample_interval_s: float = 5.0
    num_partitions: int | None = None
    run_dir: str = ""

    @property
    def output_dir(self) -> Path:
        return Path(self.run_dir) if self.run_dir else DEFAULT_RUN_ROOT / self.run_id

    def worker_env(self) -> dict[str, str]:
        return {
            "BENCH_ASR_ACTOR_CPUS": str(self.asr_actor_cpus),
            "BENCH_ASR_ACTOR_CONCURRENCY": str(self.asr_actor_concurrency),
            "BENCH_ASR_BATCH_SIZE": str(self.asr_batch_size),
            "BENCH_LLM_CONCURRENCY": str(self.llm_concurrency),
            "BENCH_LLM_TIMEOUT_S": str(self.llm_timeout_s),
            "BENCH_LLM_MAX_ATTEMPTS": str(self.llm_max_attempts),
            "ASR_MODEL": os.getenv("ASR_MODEL", "iic/SenseVoiceSmall"),
            "ASR_DEVICE": os.getenv("ASR_DEVICE", "cpu"),
            "MINIO_ENDPOINT": project_config.S3_ENDPOINT,
            "MINIO_ROOT_USER": project_config.S3_KEY,
            "MINIO_ROOT_PASSWORD": project_config.S3_SECRET,
            "MINIO_REGION": project_config.S3_REGION,
        }


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def runtime_metadata(cfg: BenchmarkConfig, *, cluster: dict[str, Any] | None = None) -> dict[str, Any]:
    import daft
    import funasr
    import ray

    return {
        "config": asdict(cfg),
        "git_commit": git_commit(),
        "versions": {
            "python": sys.version.split()[0],
            "daft": daft.__version__,
            "ray": ray.__version__,
            "funasr": getattr(funasr, "__version__", "unknown"),
        },
        "platform": platform.platform(),
        "cluster": cluster or {},
        "started_at": datetime.now(timezone.utc).isoformat(),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
