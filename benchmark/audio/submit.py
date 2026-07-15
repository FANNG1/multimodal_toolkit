from __future__ import annotations

import asyncio
import shlex
from dataclasses import asdict

from ray.job_submission import JobSubmissionClient

from .config import BenchmarkConfig, REPO_ROOT


def build_entrypoint_args(cfg: BenchmarkConfig) -> list[str]:
    args = [
        "python", "-m", "benchmark.audio", "run",
        "--run-id", cfg.run_id,
        "--manifest", cfg.manifest_uri,
        "--lance-uri", cfg.lance_uri,
        "--mock-url", cfg.mock_url,
        "--ray-address", cfg.ray_address,
        "--max-minutes", str(cfg.max_minutes),
        "--asr-actor-cpus", str(cfg.asr_actor_cpus),
        "--asr-actor-concurrency", str(cfg.asr_actor_concurrency),
        "--asr-batch-size", str(cfg.asr_batch_size),
        "--llm-concurrency", str(cfg.llm_concurrency),
        "--llm-timeout-s", str(cfg.llm_timeout_s),
        "--llm-max-attempts", str(cfg.llm_max_attempts),
        "--sample-interval-s", str(cfg.sample_interval_s),
    ]
    if cfg.num_partitions is not None:
        args.extend(["--num-partitions", str(cfg.num_partitions)])
    if cfg.run_dir:
        args.extend(["--run-dir", cfg.run_dir])
    return args


def submit_job(cfg: BenchmarkConfig, dashboard_address: str, *, wait: bool = False) -> str:
    args = build_entrypoint_args(cfg)
    client = JobSubmissionClient(dashboard_address)
    job_id = client.submit_job(
        entrypoint=shlex.join(args),
        runtime_env={
            "working_dir": str(REPO_ROOT),
            "excludes": [".git", ".venv", ".benchmarks", "data", ".claude"],
            "env_vars": cfg.worker_env(),
        },
        metadata={"benchmark_run_id": cfg.run_id, "config": str(asdict(cfg))[:1024]},
    )
    if wait:
        async def tail() -> None:
            async for line in client.tail_job_logs(job_id):
                print(line, end="")

        asyncio.run(tail())
    return job_id
