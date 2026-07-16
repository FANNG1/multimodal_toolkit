from __future__ import annotations

from benchmark.audio.cli import build_parser
from benchmark.audio.config import BenchmarkConfig


def test_worker_env_contains_resource_controls():
    cfg = BenchmarkConfig(
        run_id="run",
        manifest_uri="s3://b/m.parquet",
        lance_uri="s3://b/o.lance",
        mock_url="http://mock:8010",
        asr_actor_cpus=2.0,
        asr_batch_size=3,
        llm_concurrency=7,
    )
    env = cfg.worker_env()
    assert env["BENCH_ASR_ACTOR_CPUS"] == "2.0"
    assert env["BENCH_ASR_BATCH_SIZE"] == "3"
    assert env["BENCH_LLM_CONCURRENCY"] == "7"


def test_cli_parses_run_controls():
    args = build_parser().parse_args(
        [
            "run", "--manifest", "s3://b/m.parquet", "--lance-uri", "s3://b/o.lance",
            "--mock-url", "http://mock", "--max-minutes", "60", "--asr-batch-size", "2",
            "--num-partitions", "64",
        ]
    )
    assert args.command == "run"
    assert args.max_minutes == 60
    assert args.asr_batch_size == 2
    assert args.num_partitions == 64


def test_local_baseline_defaults_to_two_fixed_sixty_second_runs():
    args = build_parser().parse_args(["local-baseline"])
    assert args.count == 50
    assert args.duration_s == 60
    assert args.repeats == 2
    assert args.mock_profile == "fast"
    assert args.max_minutes == 45
    assert args.skip_warmup is False
