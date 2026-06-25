#!/usr/bin/env python3
"""Upload local audio files to MinIO/S3 and write the manifest.

Usage:
    python scripts/init_s3.py
    python scripts/init_s3.py --data-dir data/audio --bucket contacts \
        --raw-prefix raw/calls --manifest-key audio_poc/manifest.parquet

Place .wav / .mp3 / .m4a / .flac / .ogg files in data/audio/ before running.
The manifest is written to s3://<bucket>/<manifest-key> in Parquet format.

Dependencies: pyarrow (already required by the project), python-dotenv (optional).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for candidate in (_REPO_ROOT / ".env", Path.cwd() / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)
            break


def _s3_config() -> dict:
    endpoint = os.getenv("MINIO_ENDPOINT", "http://127.0.0.1:9000")
    return {
        "access_key": os.getenv("MINIO_ROOT_USER", "minioadmin"),
        "secret_key": os.getenv("MINIO_ROOT_PASSWORD", "minioadmin"),
        "endpoint": endpoint,
        "use_ssl": endpoint.startswith("https"),
    }


def _s3fs(cfg: dict):
    from pyarrow.fs import S3FileSystem

    host = cfg["endpoint"].replace("https://", "").replace("http://", "")
    return S3FileSystem(
        access_key=cfg["access_key"],
        secret_key=cfg["secret_key"],
        endpoint_override=host,
        scheme="https" if cfg["use_ssl"] else "http",
    )


def _ensure_bucket(s3, bucket: str) -> None:
    from pyarrow.fs import FileType

    if s3.get_file_info(bucket).type == FileType.NotFound:
        s3.create_dir(bucket)
        print(f"[created] bucket: {bucket}")
    else:
        print(f"[exists]  bucket: {bucket}")


def _upload_files(s3, data_dir: Path, bucket: str, prefix: str) -> list[dict]:
    files = sorted(f for f in data_dir.iterdir() if f.suffix.lower() in AUDIO_SUFFIXES)
    if not files:
        print(f"[warn] no audio files found in {data_dir}", file=sys.stderr)
        return []

    rows = []
    for f in files:
        s3_key = f"{bucket}/{prefix}/{f.name}"
        with f.open("rb") as src, s3.open_output_stream(s3_key) as dst:
            dst.write(src.read())
        print(f"[upload] {f.name} → s3://{s3_key}")
        rows.append({"doc_id": f.name, "s3_url": f"s3://{s3_key}"})
    return rows


def _write_manifest(rows: list[dict], s3, bucket: str, manifest_key: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {"doc_id": [r["doc_id"] for r in rows], "s3_url": [r["s3_url"] for r in rows]},
        schema=pa.schema([pa.field("doc_id", pa.utf8()), pa.field("s3_url", pa.utf8())]),
    )
    s3_key = f"{bucket}/{manifest_key}"
    with s3.open_output_stream(s3_key) as f:
        pq.write_table(table, f)
    print(f"[manifest] s3://{s3_key}  ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/audio", help="Local directory with audio files")
    parser.add_argument("--bucket", default="contacts", help="S3 bucket name")
    parser.add_argument("--raw-prefix", default="raw/calls", help="S3 prefix for audio files")
    parser.add_argument("--manifest-key", default="audio_poc/manifest.parquet", help="S3 key for the manifest")
    args = parser.parse_args()

    _load_env()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = _REPO_ROOT / data_dir
    if not data_dir.exists():
        sys.exit(f"[error] data dir not found: {data_dir}")

    s3 = _s3fs(_s3_config())

    _ensure_bucket(s3, args.bucket)
    rows = _upload_files(s3, data_dir, args.bucket, args.raw_prefix)
    if not rows:
        sys.exit(1)
    _write_manifest(rows, s3, args.bucket, args.manifest_key)
    print(f"\n[ok] {len(rows)} file(s) uploaded. Run the pipeline with:")
    print(f"  mmt-ingest --manifest s3://{args.bucket}/{args.manifest_key} \\")
    print(f"             --lance-uri s3://{args.bucket}/audio_poc/calls.lance")


if __name__ == "__main__":
    main()
