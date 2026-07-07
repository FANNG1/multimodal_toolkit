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
    """Switch Daft to the Ray runner when USE_RAY=1, otherwise leave the default (native)."""
    from .. import config

    if not config.USE_RAY:
        return
    import daft

    daft.set_runner_ray(address=config.RAY_ADDRESS, noop_if_initialized=True)


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
