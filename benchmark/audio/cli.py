from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .config import BenchmarkConfig, DEFAULT_RUN_ROOT, new_run_id


def _add_run_arguments(
    parser: argparse.ArgumentParser, *, ray_default: str = "local"
) -> None:
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
    print(
        json.dumps({"run_id": run_id, **asdict(result)}, ensure_ascii=False, indent=2)
    )


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
        print(
            json.dumps(
                {"summary": summary, "report": report}, ensure_ascii=False, indent=2
            )
        )


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
            raise RuntimeError(
                f"row count mismatch: manifest={dataset.rows}, lance={summary.get('rows')}"
            )
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


def _cmd_local_baseline(args) -> None:
    import ray

    from .data import generate_dataset
    from .mock_llm import start_in_thread
    from .pipeline import run_pipeline
    from .report import build_baseline_report, build_report
    from .storage import preflight

    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2")
    standard_baseline = (
        40 <= args.count <= 60
        and args.duration_s == 60.0
        and 0 < args.max_minutes <= 45.0
    )
    if not standard_baseline:
        print(
            "[warning] non-standard diagnostic baseline: the comparable baseline uses "
            "40-60 files, 60s duration, and a per-repeat limit no greater than 45 minutes"
        )
    base_run_id = args.run_id or f"baseline-{new_run_id()}"
    base_dir = DEFAULT_RUN_ROOT / base_run_id
    print(f"[preflight] checking configured MinIO bucket {args.bucket}")
    print(json.dumps(preflight(args.bucket), ensure_ascii=False))
    print(
        f"[generate] creating {args.count} fixed-duration objects "
        f"({args.duration_s:.0f}s each)"
    )
    dataset = generate_dataset(
        run_id=base_run_id,
        bucket=args.bucket,
        prefix=args.prefix,
        profile="fixed",
        count=args.count,
        fixed_duration_s=args.duration_s,
        source_dir=Path(args.source_dir) if args.source_dir else None,
        seed=args.seed,
    )
    warmup_dataset = None
    if not args.skip_warmup:
        print("[generate] creating one warm-up object")
        warmup_dataset = generate_dataset(
            run_id=f"{base_run_id}-warmup",
            bucket=args.bucket,
            prefix=args.prefix,
            profile="smoke",
            count=1,
            source_dir=Path(args.source_dir) if args.source_dir else None,
            seed=args.seed,
        )
    server, thread = start_in_thread(port=0, profile=args.mock_profile)
    port = server.server_address[1]
    reports = []
    repeat_error: Exception | None = None
    try:
        if warmup_dataset is not None:
            warmup_cfg = BenchmarkConfig(
                run_id=f"{base_run_id}-warmup",
                manifest_uri=warmup_dataset.manifest_uri,
                lance_uri=(
                    f"s3://{args.bucket}/{args.prefix.strip('/')}/{base_run_id}/warmup.lance"
                ),
                mock_url=f"http://127.0.0.1:{port}",
                ray_address="local",
                max_minutes=min(args.max_minutes, 15.0),
                asr_actor_cpus=args.asr_actor_cpus,
                asr_actor_concurrency=args.asr_actor_concurrency,
                asr_batch_size=args.asr_batch_size,
                llm_concurrency=args.llm_concurrency,
                llm_timeout_s=15.0,
                sample_interval_s=args.sample_interval_s,
                num_partitions=1,
                run_dir=str(base_dir / "warmup"),
            )
            print(f"[warmup] {warmup_dataset.manifest_uri} -> {warmup_cfg.lance_uri}")
            warmup_summary = run_pipeline(warmup_cfg)
            build_report(warmup_cfg.output_dir)
            if warmup_summary.get("rows") != 1 or warmup_summary.get(
                "status_counts"
            ) != {"ok": 1}:
                raise RuntimeError(f"warm-up validation failed: {warmup_summary}")
        for repeat in range(1, args.repeats + 1):
            repeat_name = f"repeat-{repeat:02d}"
            repeat_run_id = f"{base_run_id}-{repeat_name}"
            lance_uri = (
                f"s3://{args.bucket}/{args.prefix.strip('/')}/{base_run_id}/"
                f"output-{repeat_name}.lance"
            )
            cfg = BenchmarkConfig(
                run_id=repeat_run_id,
                manifest_uri=dataset.manifest_uri,
                lance_uri=lance_uri,
                mock_url=f"http://127.0.0.1:{port}",
                ray_address="local",
                max_minutes=args.max_minutes,
                asr_actor_cpus=args.asr_actor_cpus,
                asr_actor_concurrency=args.asr_actor_concurrency,
                asr_batch_size=args.asr_batch_size,
                llm_concurrency=args.llm_concurrency,
                llm_timeout_s=15.0,
                sample_interval_s=args.sample_interval_s,
                num_partitions=args.num_partitions,
                run_dir=str(base_dir / repeat_name),
            )
            print(
                f"[run {repeat}/{args.repeats}] {dataset.manifest_uri} -> {lance_uri}"
            )
            try:
                summary = run_pipeline(cfg)
            except Exception as exc:
                if (cfg.output_dir / "summary.json").exists():
                    reports.append(build_report(cfg.output_dir))
                else:
                    reports.append(
                        {
                            "run_id": repeat_run_id,
                            "state": "failed",
                            "summary": {"elapsed_s": 0.0, "error": repr(exc)},
                        }
                    )
                repeat_error = exc
                break
            report = build_report(cfg.output_dir)
            if (
                summary.get("rows") != dataset.rows
                or summary.get("unique_doc_ids") != dataset.rows
            ):
                repeat_error = RuntimeError(
                    f"repeat {repeat} row validation failed: expected={dataset.rows}, "
                    f"rows={summary.get('rows')}, unique={summary.get('unique_doc_ids')}"
                )
            reports.append(report)
            if repeat_error is not None:
                break
        comparison = build_baseline_report(
            base_dir,
            reports,
            {
                "run_id": base_run_id,
                "manifest_uri": dataset.manifest_uri,
                "rows": dataset.rows,
                "duration_s": args.duration_s,
                "total_bytes": dataset.total_bytes,
                "total_audio_seconds": dataset.total_audio_seconds,
                "planned_repeats": args.repeats,
                "seed": args.seed,
                "source_dir": args.source_dir,
                "mock_profile": args.mock_profile,
                "warmup_enabled": not args.skip_warmup,
                "standard_baseline": standard_baseline,
                "max_minutes_per_repeat": args.max_minutes,
                "asr_actor_cpus": args.asr_actor_cpus,
                "asr_actor_concurrency": args.asr_actor_concurrency,
                "asr_batch_size": args.asr_batch_size,
                "llm_concurrency": args.llm_concurrency,
                "sample_interval_s": args.sample_interval_s,
                "num_partitions": args.num_partitions,
            },
        )
        print(json.dumps(comparison, ensure_ascii=False, indent=2))
        print(f"[summary] comparison: {base_dir / 'baseline-summary.md'}")
        if repeat_error is not None:
            raise repeat_error
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
    generate.add_argument(
        "--profile", choices=("smoke", "fixed", "mixed"), default="smoke"
    )
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

    smoke = commands.add_parser(
        "local-smoke", help="run a real local end-to-end smoke test"
    )
    smoke.add_argument("--run-id", default=None)
    smoke.add_argument("--bucket", default="benchmark")
    smoke.add_argument("--prefix", default="audio")
    smoke.add_argument("--count", type=int, default=4)
    smoke.add_argument("--source-dir", default=None)
    smoke.add_argument("--max-minutes", type=float, default=15.0)
    smoke.add_argument("--asr-actor-cpus", type=float, default=4.0)
    smoke.set_defaults(func=_cmd_local_smoke)

    baseline = commands.add_parser(
        "local-baseline",
        help="run repeated fixed-duration local performance baselines",
    )
    baseline.add_argument("--run-id", default=None)
    baseline.add_argument("--bucket", default="benchmark")
    baseline.add_argument("--prefix", default="audio")
    baseline.add_argument("--count", type=int, default=50)
    baseline.add_argument("--duration-s", type=float, default=60.0)
    baseline.add_argument("--repeats", type=int, default=2)
    baseline.add_argument("--source-dir", default=None)
    baseline.add_argument("--seed", type=int, default=20260715)
    baseline.add_argument(
        "--mock-profile", choices=("fast", "standard"), default="fast"
    )
    baseline.add_argument(
        "--skip-warmup",
        action="store_true",
        help="skip the one-row model/cache warm-up before measured repeats",
    )
    baseline.add_argument("--max-minutes", type=float, default=45.0)
    baseline.add_argument("--asr-actor-cpus", type=float, default=4.0)
    baseline.add_argument("--asr-actor-concurrency", type=int, default=1)
    baseline.add_argument("--asr-batch-size", type=int, default=1)
    baseline.add_argument("--llm-concurrency", type=int, default=8)
    baseline.add_argument("--sample-interval-s", type=float, default=5.0)
    baseline.add_argument("--num-partitions", type=int, default=None)
    baseline.set_defaults(func=_cmd_local_baseline)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)
