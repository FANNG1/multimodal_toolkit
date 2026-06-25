from __future__ import annotations

from pathlib import Path

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
    import json

    if out_uri.startswith("s3://"):
        raise ValueError("S3 JSONL output is not supported in this POC. Use a local --out-jsonl path.")

    data = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    path = Path(out_uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")
