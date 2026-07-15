from __future__ import annotations

from .. import config


def daft_io_config():
    from daft.io import IOConfig, S3Config

    return IOConfig(
        s3=S3Config(
            endpoint_url=config.S3_ENDPOINT,
            key_id=config.S3_KEY,
            access_key=config.S3_SECRET,
            region_name=config.S3_REGION,
            use_ssl=config.S3_USE_SSL,
            force_virtual_addressing=False,
            max_connections=config.S3_MAX_CONNECTIONS,
            num_tries=config.S3_NUM_TRIES,
            retry_initial_backoff_ms=config.S3_RETRY_INITIAL_BACKOFF_MS,
            connect_timeout_ms=config.S3_CONNECT_TIMEOUT_MS,
            read_timeout_ms=config.S3_READ_TIMEOUT_MS,
            retry_mode=config.S3_RETRY_MODE,
        )
    )


def lance_storage_options(uri: str) -> dict:
    """Return storage_options for lance.dataset() when uri is an S3 path.

    For local paths returns an empty dict (no-op).
    MinIO requires path-style access (aws_virtual_hosted_style_access=false).
    """
    if not uri.startswith("s3://"):
        return {}
    opts = {
        "aws_access_key_id": config.S3_KEY,
        "aws_secret_access_key": config.S3_SECRET,
        "aws_endpoint": config.S3_ENDPOINT,
        "aws_virtual_hosted_style_access": "false",
    }
    if not config.S3_USE_SSL:
        opts["allow_http"] = "true"
    return opts


def lance_write_mode(uri: str) -> str:
    """Return append for existing Lance datasets, create for missing datasets."""
    import lance

    try:
        lance.dataset(uri, storage_options=lance_storage_options(uri))
    except ValueError as exc:
        message = str(exc)
        if "was not found" in message and "_versions" in message:
            return "create"
        raise
    return "append"


def read_analysis_output(path: str, io_config):
    """Read Stage 1 analysis output as JSON or a Lance staging table.

    JSON extensions are unambiguous. Everything else is treated as Lance so
    S3 staging URIs do not need a `.lance` suffix. For local no-extension
    paths, detect Lance by the `_versions` directory and otherwise preserve
    existing JSON directory behavior.
    """
    from pathlib import Path

    import daft

    low = path.rstrip("/").lower()
    if low.endswith(".jsonl") or low.endswith(".ndjson") or low.endswith(".json"):
        return daft.read_json(path, io_config=io_config)
    if low.endswith(".lance"):
        return daft.read_lance(path, io_config=io_config)
    if not low.startswith("s3://") and not Path(path, "_versions").exists():
        return daft.read_json(path, io_config=io_config)
    return daft.read_lance(path, io_config=io_config)


def configure_daft_runner() -> None:
    """Switch Daft to the Ray runner when USE_RAY=1 and apply execution config.

    Daft only reads a handful of DAFT_* environment variables natively; the
    rest of the knobs in config.py take effect exclusively through this
    set_execution_config call, so it runs for both runners.
    """
    from .. import config

    import daft

    if config.USE_RAY:
        daft.set_runner_ray(address=config.RAY_ADDRESS, noop_if_initialized=True)

    daft.set_execution_config(
        default_morsel_size=config.DAFT_DEFAULT_MORSEL_SIZE,
        enable_dynamic_batching=config.DAFT_ENABLE_DYNAMIC_BATCHING,
        enable_scan_task_split_and_merge=config.DAFT_ENABLE_SCAN_TASK_SPLIT_AND_MERGE,
        scan_tasks_min_size_bytes=config.DAFT_SCAN_TASKS_MIN_SIZE_BYTES,
        scan_tasks_max_size_bytes=config.DAFT_SCAN_TASKS_MAX_SIZE_BYTES,
        max_sources_per_scan_task=config.DAFT_MAX_SOURCES_PER_SCAN_TASK,
        scantask_max_parallel=config.DAFT_SCANTASK_MAX_PARALLEL,
        json_target_filesize=config.DAFT_JSON_TARGET_FILESIZE,
        maintain_order=config.DAFT_MAINTAIN_ORDER,
        actor_udf_ready_timeout=config.DAFT_ACTOR_UDF_READY_TIMEOUT,
    )


def analysis_num_partitions() -> int | None:
    """Partition count for analyze stages, or None to keep the input layout.

    The manifests are tiny files, so Daft's byte-based scan task splitting
    never kicks in and the whole download+analysis chain would otherwise run
    as a single Ray task. Explicit ANALYZE_NUM_PARTITIONS wins; on Ray we
    derive 2 tasks per cluster CPU so the scheduler can keep workers busy.
    """
    from .. import config

    if config.ANALYZE_NUM_PARTITIONS is not None:
        return config.ANALYZE_NUM_PARTITIONS
    if not config.USE_RAY:
        return None
    import ray

    if not ray.is_initialized():
        # Same address the Ray runner uses; None starts/joins a local cluster.
        ray.init(address=config.RAY_ADDRESS, ignore_reinit_error=True)
    cpus = ray.cluster_resources().get("CPU", 0)
    return max(1, 2 * int(cpus))


def spread_partitions(df):
    """Split the manifest across the cluster before download/analysis."""
    n = analysis_num_partitions()
    return df if n is None else df.into_partitions(n)


def coalesce_for_write(df):
    """Merge partitions before write_json so high analysis parallelism does
    not translate 1:1 into tiny output files."""
    n = analysis_num_partitions()
    if n is None or n <= 8:
        return df
    return df.into_partitions(max(1, n // 8))


def read_manifest(manifest_uri: str):
    import daft

    io_config = daft_io_config()
    lower = manifest_uri.lower()
    if lower.endswith(".parquet"):
        return daft.read_parquet(manifest_uri, io_config=io_config).select("doc_id", "s3_url")
    if lower.endswith(".jsonl") or lower.endswith(".ndjson"):
        return daft.read_json(manifest_uri, io_config=io_config).select("doc_id", "s3_url")
    if lower.endswith(".csv"):
        return daft.read_csv(manifest_uri, io_config=io_config).select("doc_id", "s3_url")
    raise ValueError(f"Unsupported manifest format: {manifest_uri}. Use parquet/jsonl/csv.")
