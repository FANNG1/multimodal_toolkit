from __future__ import annotations

from benchmark.audio.config import BenchmarkConfig
from benchmark.audio.submit import build_entrypoint_args


def test_submit_forwards_all_run_controls():
    cfg = BenchmarkConfig(
        run_id="run",
        manifest_uri="s3://b/m.parquet",
        lance_uri="s3://b/o.lance",
        mock_url="http://mock",
        ray_address="auto",
        max_minutes=60,
        llm_timeout_s=22,
        llm_max_attempts=5,
        sample_interval_s=3,
        num_partitions=64,
        run_dir="/mnt/bench/run",
    )
    args = build_entrypoint_args(cfg)
    rendered = " ".join(args)
    assert "--ray-address auto" in rendered
    assert "--llm-timeout-s 22" in rendered
    assert "--llm-max-attempts 5" in rendered
    assert "--sample-interval-s 3" in rendered
    assert "--num-partitions 64" in rendered
    assert "--run-dir /mnt/bench/run" in rendered
