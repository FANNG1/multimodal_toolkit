from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .config import BenchmarkConfig, DEFAULT_RUN_ROOT, new_run_id


def _add_run_arguments(parser: argparse.ArgumentParser, *, ray_default: str = "local") -> None:
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lance-uri", required=True)
    parser.add_argument("--mock-url", required=True)
    parser.add_argument("--ray-address", default=ray_default)
    parser.add_argument("--max-minutes", type=float, default=15.0)
    parser.add_argument("--asr-actor-cpus", type=float, default=4.0)
    parser.add_argument("--asr-actor-concurrency", type=int, default=1)
    parser.add_argument("--asr-batch-size", type=int, default=1)
    parser.add_argument("--llm-concurrency", type=int, default=8)
    parser.add_argument("--llm-timeout-s", type=float, default=15.0)
    parser.add_argument("--llm-max-attempts", type=int, default=3)
    parser.add_argument("--sample-interval-s", type=float, default=5.0)
    parser.add_argument(
        "--num-partitions",
        type=int,
        default=None,
        help="manifest partitions before download; default is 2x cluster CPUs",
    )
    parser.add_argument("--run-dir", default=None)


def _config_from_args(args) -> BenchmarkConfig:
    run_id = args.run_id or new_run_id()
    run_dir = args.run_dir
    if run_dir is None and args.command != "submit":
        run_dir = str(DEFAULT_RUN_ROOT / run_id)
    return BenchmarkConfig(
        run_id=run_id,
        manifest_uri=args.manifest,
        lance_uri=args.lance_uri,
        mock_url=args.mock_url,
        ray_address=args.ray_address,
        max_minutes=args.max_minutes,
        asr_actor_cpus=args.asr_actor_cpus,
        asr_actor_concurrency=args.asr_actor_concurrency,
        asr_batch_size=args.asr_batch_size,
        llm_concurrency=args.llm_concurrency,
        llm_timeout_s=args.llm_timeout_s,
        llm_max_attempts=args.llm_max_attempts,
        sample_interval_s=args.sample_interval_s,
        num_partitions=args.num_partitions,
        run_dir=run_dir or "",
    )


def _cmd_generate(args) -> None:
    from .data import generate_dataset
    from .storage import preflight

    run_id = args.run_id or new_run_id()
    preflight(args.bucket)
    result = generate_dataset(
        run_id=run_id,
        bucket=args.bucket,
        prefix=args.prefix,
        profile=args.profile,
        count=args.count,
        fixed_duration_s=args.duration_s,
        source_dir=Path(args.source_dir) if args.source_dir else None,
        seed=args.seed,
    )
    print(json.dumps({"run_id": run_id, **asdict(result)}, ensure_ascii=False, indent=2))


def _cmd_run(args) -> None:
    from .pipeline import run_pipeline
    from .report import build_report

    cfg = _config_from_args(args)
    try:
        summary = run_pipeline(cfg)
    except Exception:
        if (cfg.output_dir / "summary.json").exists():
            build_report(cfg.output_dir)
        raise
    else:
        report = build_report(cfg.output_dir)
        print(json.dumps({"summary": summary, "report": report}, ensure_ascii=False, indent=2))


def _cmd_submit(args) -> None:
    from .submit import submit_job

    cfg = _config_from_args(args)
    job_id = submit_job(cfg, args.dashboard_address, wait=args.wait)
    print(job_id)


def _cmd_report(args) -> None:
    from .report import build_report

    print(json.dumps(build_report(Path(args.run_dir)), ensure_ascii=False, indent=2))


def _cmd_local_smoke(args) -> None:
    import ray

    from .data import generate_dataset
    from .mock_llm import start_in_thread
    from .pipeline import run_pipeline
    from .report import build_report
    from .storage import preflight

    run_id = args.run_id or new_run_id()
    print(f"[preflight] checking configured MinIO bucket {args.bucket}")
    print(json.dumps(preflight(args.bucket), ensure_ascii=False))
    print(f"[generate] creating {args.count} smoke objects")
    dataset = generate_dataset(
        run_id=run_id,
        bucket=args.bucket,
        prefix=args.prefix,
        profile="smoke",
        count=args.count,
        source_dir=Path(args.source_dir) if args.source_dir else None,
    )
    server, thread = start_in_thread(port=0, profile="fast")
    port = server.server_address[1]
    lance_uri = f"s3://{args.bucket}/{args.prefix.strip('/')}/{run_id}/output.lance"
    cfg = BenchmarkConfig(
        run_id=run_id,
        manifest_uri=dataset.manifest_uri,
        lance_uri=lance_uri,
        mock_url=f"http://127.0.0.1:{port}",
        ray_address="local",
        max_minutes=args.max_minutes,
        asr_actor_cpus=args.asr_actor_cpus,
        asr_actor_concurrency=1,
        asr_batch_size=1,
        llm_concurrency=4,
        llm_timeout_s=5.0,
        sample_interval_s=1.0,
        num_partitions=min(args.count, 2),
        run_dir=str(DEFAULT_RUN_ROOT / run_id),
    )
    try:
        print(f"[run] {dataset.manifest_uri} -> {lance_uri}")
        try:
            summary = run_pipeline(cfg)
        except Exception:
            if (cfg.output_dir / "summary.json").exists():
                build_report(cfg.output_dir)
            raise
        report = build_report(cfg.output_dir)
        if summary.get("rows") != dataset.rows:
            raise RuntimeError(f"row count mismatch: manifest={dataset.rows}, lance={summary.get('rows')}")
        if summary.get("unique_doc_ids") != dataset.rows:
            raise RuntimeError("output doc_id values are not unique")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"[ok] report: {cfg.output_dir / 'report.md'}")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        if ray.is_initialized():
            ray.shutdown()


def _cmd_serve_mock(args) -> None:
    from .mock_llm import serve

    serve(args.host, args.port, args.profile)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Daft audio stability benchmark")
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="create and upload benchmark audio")
    generate.add_argument("--run-id", default=None)
    generate.add_argument("--bucket", default="benchmark")
    generate.add_argument("--prefix", default="audio")
    generate.add_argument("--profile", choices=("smoke", "fixed", "mixed"), default="smoke")
    generate.add_argument("--count", type=int, default=4)
    generate.add_argument("--duration-s", type=float, default=30.0)
    generate.add_argument("--source-dir", default=None)
    generate.add_argument("--seed", type=int, default=20260715)
    generate.set_defaults(func=_cmd_generate)

    mock = commands.add_parser("serve-mock", help="run the OpenAI-compatible mock LLM")
    mock.add_argument("--host", default="0.0.0.0")
    mock.add_argument("--port", type=int, default=8010)
    mock.add_argument("--profile", choices=("fast", "standard"), default="fast")
    mock.set_defaults(func=_cmd_serve_mock)

    run = commands.add_parser("run", help="execute the benchmark directly")
    _add_run_arguments(run)
    run.set_defaults(func=_cmd_run)

    submit = commands.add_parser("submit", help="submit the benchmark with Ray Jobs")
    _add_run_arguments(submit, ray_default="auto")
    submit.add_argument("--dashboard-address", required=True)
    submit.add_argument("--wait", action="store_true")
    submit.set_defaults(func=_cmd_submit)

    report = commands.add_parser("report", help="rebuild reports for a run")
    report.add_argument("--run-dir", required=True)
    report.set_defaults(func=_cmd_report)

    smoke = commands.add_parser("local-smoke", help="run a real local end-to-end smoke test")
    smoke.add_argument("--run-id", default=None)
    smoke.add_argument("--bucket", default="benchmark")
    smoke.add_argument("--prefix", default="audio")
    smoke.add_argument("--count", type=int, default=4)
    smoke.add_argument("--source-dir", default=None)
    smoke.add_argument("--max-minutes", type=float, default=15.0)
    smoke.add_argument("--asr-actor-cpus", type=float, default=4.0)
    smoke.set_defaults(func=_cmd_local_smoke)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)
