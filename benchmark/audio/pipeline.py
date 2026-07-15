from __future__ import annotations

import io
import json
import os
import signal
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import daft
from daft import col, lit
from daft.functions import download, when

from multimodal_toolkit import config as project_config
from multimodal_toolkit.storage.blob import validate_blob_v2
from multimodal_toolkit.storage.io import daft_io_config, lance_storage_options

from .config import BenchmarkConfig, runtime_metadata, write_json
from .metrics import ResourceSampler


ASR_DTYPE = daft.DataType.struct(
    {
        "input_bytes": daft.DataType.int64(),
        "duration_s": daft.DataType.float64(),
        "transcript": daft.DataType.string(),
        "acoustic_emotion": daft.DataType.string(),
        "asr_ms": daft.DataType.float64(),
        "error": daft.DataType.string(),
    }
)

LLM_DTYPE = daft.DataType.struct(
    {
        "content": daft.DataType.string(),
        "llm_ms": daft.DataType.float64(),
        "attempts": daft.DataType.int64(),
        "error": daft.DataType.string(),
    }
)


def _asr_udf(cfg: BenchmarkConfig):
    class AsrBenchmarkUDF:
        def __init__(self) -> None:
            from multimodal_toolkit.audio.asr import SenseVoiceASR

            self._asr = SenseVoiceASR()

        @daft.method.batch(return_dtype=ASR_DTYPE, batch_size=cfg.asr_batch_size)
        def __call__(self, audio_bytes_col, doc_ids):
            import soundfile as sf

            rows = []
            for audio_bytes, doc_id in zip(audio_bytes_col.to_pylist(), doc_ids.to_pylist()):
                started = time.perf_counter()
                duration_s = 0.0
                if not audio_bytes:
                    rows.append(
                        {
                            "input_bytes": 0,
                            "duration_s": 0.0,
                            "transcript": None,
                            "acoustic_emotion": None,
                            "asr_ms": 0.0,
                            "error": "download_failed",
                        }
                    )
                    continue
                try:
                    info = sf.info(io.BytesIO(audio_bytes))
                    duration_s = float(info.frames) / info.samplerate if info.samplerate else 0.0
                    suffix = Path(doc_id or "audio.wav").suffix or ".wav"
                    result = self._asr.transcribe_bytes(audio_bytes, suffix)
                    rows.append(
                        {
                            "input_bytes": len(audio_bytes),
                            "duration_s": duration_s,
                            "transcript": result["transcript"],
                            "acoustic_emotion": result["acoustic_emotion"],
                            "asr_ms": (time.perf_counter() - started) * 1000,
                            "error": None,
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "input_bytes": len(audio_bytes),
                            "duration_s": duration_s,
                            "transcript": None,
                            "acoustic_emotion": None,
                            "asr_ms": (time.perf_counter() - started) * 1000,
                            "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                        }
                    )
            return rows

    decorated = daft.cls(
        cpus=cfg.asr_actor_cpus,
        max_concurrency=cfg.asr_actor_concurrency,
        max_retries=1,
        name_override="SenseVoiceBenchmark",
        ray_options={"scheduling_strategy": "SPREAD"},
    )(AsrBenchmarkUDF)
    return decorated()


def _llm_udf(cfg: BenchmarkConfig):
    class MockLLMBenchmarkUDF:
        def __init__(self) -> None:
            import httpx

            self._client = httpx.AsyncClient(timeout=cfg.llm_timeout_s)

        @daft.method(return_dtype=LLM_DTYPE)
        async def __call__(self, transcript: str | None, doc_id: str, asr_error: str | None):
            import asyncio
            import httpx

            if asr_error:
                return {"content": None, "llm_ms": 0.0, "attempts": 0, "error": "skipped_after_asr"}
            started = time.perf_counter()
            error = None
            for attempt in range(1, cfg.llm_max_attempts + 1):
                try:
                    response = await self._client.post(
                        f"{cfg.mock_url.rstrip('/')}/v1/chat/completions",
                        headers={
                            "X-Benchmark-Doc-Id": doc_id,
                            "X-Benchmark-Attempt": str(attempt),
                        },
                        json={
                            "model": "benchmark-mock",
                            "messages": [{"role": "user", "content": transcript or ""}],
                            "temperature": 0,
                        },
                    )
                    response.raise_for_status()
                    content = response.json()["choices"][0]["message"]["content"]
                    return {
                        "content": content,
                        "llm_ms": (time.perf_counter() - started) * 1000,
                        "attempts": attempt,
                        "error": None,
                    }
                except (httpx.HTTPError, KeyError, ValueError) as exc:
                    error = f"{type(exc).__name__}: {str(exc)[:500]}"
                    if attempt < cfg.llm_max_attempts:
                        await asyncio.sleep(0.5 * (2 ** (attempt - 1)))
            return {
                "content": None,
                "llm_ms": (time.perf_counter() - started) * 1000,
                "attempts": cfg.llm_max_attempts,
                "error": error,
            }

    decorated = daft.cls(
        cpus=0.1,
        max_concurrency=cfg.llm_concurrency,
        name_override="MockLLMBenchmark",
    )(MockLLMBenchmarkUDF)
    return decorated()


def _configure_daft(cfg: BenchmarkConfig) -> dict[str, Any]:
    import ray

    if not ray.is_initialized():
        ray.init(
            address=cfg.ray_address,
            ignore_reinit_error=True,
            log_to_driver=False,
            runtime_env={"env_vars": cfg.worker_env()},
        )
    daft.set_runner_ray(address=cfg.ray_address, noop_if_initialized=True)
    daft.set_execution_config(
        default_morsel_size=project_config.DAFT_DEFAULT_MORSEL_SIZE,
        enable_dynamic_batching=project_config.DAFT_ENABLE_DYNAMIC_BATCHING,
        enable_scan_task_split_and_merge=project_config.DAFT_ENABLE_SCAN_TASK_SPLIT_AND_MERGE,
        scan_tasks_min_size_bytes=project_config.DAFT_SCAN_TASKS_MIN_SIZE_BYTES,
        scan_tasks_max_size_bytes=project_config.DAFT_SCAN_TASKS_MAX_SIZE_BYTES,
        max_sources_per_scan_task=project_config.DAFT_MAX_SOURCES_PER_SCAN_TASK,
        scantask_max_parallel=project_config.DAFT_SCANTASK_MAX_PARALLEL,
        maintain_order=project_config.DAFT_MAINTAIN_ORDER,
        actor_udf_ready_timeout=project_config.DAFT_ACTOR_UDF_READY_TIMEOUT,
    )
    return {
        "resources": ray.cluster_resources(),
        "nodes": [
            {"node_id": n["NodeID"], "address": n["NodeManagerAddress"], "alive": n["Alive"]}
            for n in ray.nodes()
        ],
    }


@contextmanager
def _deadline(minutes: float):
    if minutes <= 0 or not hasattr(signal, "setitimer"):
        yield
        return

    def expired(_signum, _frame):
        raise TimeoutError(f"benchmark exceeded {minutes} minute limit")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, minutes * 60)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _build_dataframe(cfg: BenchmarkConfig):
    import ray

    io_config = daft_io_config()
    df = daft.read_parquet(cfg.manifest_uri, io_config=io_config).select("doc_id", "s3_url")
    num_partitions = cfg.num_partitions
    if num_partitions is None:
        num_partitions = max(1, 2 * int(ray.cluster_resources().get("CPU", 1)))
    if num_partitions > 1:
        df = df.into_partitions(num_partitions)
    df = df.with_column("audio_blob", download(col("s3_url"), on_error="null", io_config=io_config))
    asr = _asr_udf(cfg)
    df = df.with_column("asr", asr(col("audio_blob"), col("doc_id")))
    for name in ("input_bytes", "duration_s", "transcript", "acoustic_emotion", "asr_ms"):
        df = df.with_column(name, col("asr")[name])
    df = df.with_column("asr_error", col("asr")["error"])

    llm = _llm_udf(cfg)
    df = df.with_column("llm", llm(col("transcript"), col("doc_id"), col("asr_error")))
    df = df.with_column("llm_content", col("llm")["content"])
    df = df.with_column("llm_ms", col("llm")["llm_ms"])
    df = df.with_column("llm_attempts", col("llm")["attempts"])
    df = df.with_column("llm_error", col("llm")["error"])
    df = df.with_column(
        "status",
        when(col("audio_blob").is_null(), "download_failed")
        .when(~col("asr_error").is_null(), "asr_failed")
        .when(~col("llm_error").is_null(), "llm_failed")
        .otherwise("ok"),
    )
    df = df.with_column(
        "error_code",
        when(col("audio_blob").is_null(), "download_failed")
        .when(~col("asr_error").is_null(), "asr_error")
        .when(~col("llm_error").is_null(), "llm_error"),
    )
    df = df.with_column("run_id", lit(cfg.run_id))
    df = df.with_column(
        "processed_at",
        lit(datetime.now(timezone.utc)).cast(daft.DataType.timestamp("us", "UTC")),
    )
    return df.select(
        "run_id", "doc_id", "s3_url", "input_bytes", "duration_s", "transcript",
        "acoustic_emotion", "llm_content", "asr_ms", "llm_ms", "llm_attempts",
        "status", "error_code", "asr_error", "llm_error", "audio_blob", "processed_at",
    )


def run_pipeline(cfg: BenchmarkConfig) -> dict[str, Any]:
    import lance
    from daft.subscribers.event_log import disable_event_log, enable_event_log

    run_dir = cfg.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    cluster = _configure_daft(cfg)
    metadata = runtime_metadata(cfg, cluster=cluster)
    write_json(run_dir / "metadata.json", metadata)
    enable_event_log(run_dir / "daft-events")

    df = _build_dataframe(cfg)
    with (run_dir / "plan.txt").open("w") as plan:
        df.explain(show_all=True, file=plan)

    sampler = ResourceSampler(run_dir / "resources.csv", cfg.sample_interval_s)
    sampler.start()
    started = time.perf_counter()
    state = "success"
    error = None
    try:
        with _deadline(cfg.max_minutes):
            df.write_lance(
                cfg.lance_uri,
                mode="overwrite",
                io_config=daft_io_config(),
                blob_columns=["audio_blob"],
                max_rows_per_file=project_config.LANCE_MAX_ROWS_PER_FILE,
                max_bytes_per_file=project_config.LANCE_MAX_BYTES_PER_FILE,
            )
        validate_blob_v2(cfg.lance_uri, "audio_blob")
    except Exception as exc:
        state = "timeout" if isinstance(exc, TimeoutError) else "failed"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        sampler.stop()
        disable_event_log()
        elapsed = time.perf_counter() - started
        summary: dict[str, Any] = {
            "run_id": cfg.run_id,
            "state": state,
            "error": error,
            "elapsed_s": elapsed,
            "lance_uri": cfg.lance_uri,
        }
        if state == "success":
            ds = lance.dataset(cfg.lance_uri, storage_options=lance_storage_options(cfg.lance_uri))
            table = ds.to_table(columns=["doc_id", "status", "input_bytes", "duration_s"])
            summary.update(
                {
                    "rows": table.num_rows,
                    "unique_doc_ids": len(set(table.column("doc_id").to_pylist())),
                    "status_counts": {
                        status: table.column("status").to_pylist().count(status)
                        for status in sorted(set(table.column("status").to_pylist()))
                    },
                    "input_bytes": sum(x or 0 for x in table.column("input_bytes").to_pylist()),
                    "audio_seconds": sum(x or 0 for x in table.column("duration_s").to_pylist()),
                }
            )
        write_json(run_dir / "summary.json", summary)
    return summary
