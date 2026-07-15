from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import pyarrow.fs as pafs

from multimodal_toolkit import config


def s3_filesystem() -> pafs.S3FileSystem:
    parsed = urlparse(config.S3_ENDPOINT)
    endpoint = parsed.netloc or parsed.path
    return pafs.S3FileSystem(
        access_key=config.S3_KEY,
        secret_key=config.S3_SECRET,
        endpoint_override=endpoint,
        scheme="https" if config.S3_USE_SSL else "http",
        region=config.S3_REGION,
        allow_bucket_creation=True,
    )


def split_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected s3:// URI, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def arrow_path(uri: str) -> str:
    bucket, key = split_s3_uri(uri)
    return f"{bucket}/{key}" if key else bucket


def ensure_bucket(fs: pafs.S3FileSystem, bucket: str) -> None:
    if fs.get_file_info(bucket).type == pafs.FileType.NotFound:
        fs.create_dir(bucket)


def upload_file(fs: pafs.S3FileSystem, source: Path, uri: str) -> int:
    path = arrow_path(uri)
    with source.open("rb") as src, fs.open_output_stream(path) as dst:
        while chunk := src.read(8 * 1024 * 1024):
            dst.write(chunk)
    return source.stat().st_size


def preflight(bucket: str) -> dict[str, object]:
    """Validate the configured MinIO without modifying existing user objects."""
    import uuid

    fs = s3_filesystem()
    ensure_bucket(fs, bucket)
    key = f"{bucket}/_benchmark_preflight/{uuid.uuid4().hex}.txt"
    payload = b"multimodal-toolkit benchmark preflight"
    with fs.open_output_stream(key) as out:
        out.write(payload)
    with fs.open_input_file(key) as inp:
        actual = inp.read()
    fs.delete_file(key)
    if actual != payload:
        raise RuntimeError("MinIO preflight read did not match the written payload")
    return {"endpoint": config.S3_ENDPOINT, "bucket": bucket, "read_write": True}
