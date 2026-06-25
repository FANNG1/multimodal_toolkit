from __future__ import annotations

from . import config


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
    return {
        "aws_access_key_id": config.S3_KEY,
        "aws_secret_access_key": config.S3_SECRET,
        "aws_endpoint": config.S3_ENDPOINT,
        "aws_virtual_hosted_style_access": "false",
    }


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


def write_jsonl(rows: list[dict], out_uri: str) -> None:
    import daft

    daft.from_pylist(rows).write_json(out_uri, write_mode="overwrite", io_config=daft_io_config())
