#!/usr/bin/env python3
"""Upload local audio files to MinIO/S3 and write the manifest.

Usage:
    python scripts/init_s3.py
    python scripts/init_s3.py --data-dir data/audio --bucket contacts \
        --raw-prefix raw/calls --manifest-key audio_poc/manifest.parquet

Place .wav / .mp3 / .m4a / .flac / .ogg files in data/audio/ before running.
The manifest is written to s3://<bucket>/<manifest-key> in Parquet format.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


def _s3fs(config):
    from pyarrow.fs import S3FileSystem

    endpoint = config.S3_ENDPOINT
    scheme = "https" if config.S3_USE_SSL else "http"
    host = endpoint.replace("https://", "").replace("http://", "")
    return S3FileSystem(
        access_key=config.S3_KEY,
        secret_key=config.S3_SECRET,
        endpoint_override=host,
        scheme=scheme,
    )


def _ensure_bucket(s3, bucket: str) -> None:
    from pyarrow.fs import FileType

    info = s3.get_file_info(bucket)
    if info.type == FileType.NotFound:
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
        s3_url = f"s3://{s3_key}"
        with f.open("rb") as src, s3.open_output_stream(s3_key) as dst:
            dst.write(src.read())
        print(f"[upload] {f.name} → {s3_url}")
        rows.append({"doc_id": f.name, "s3_url": s3_url})
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
    parser.add_argument("--data-dir", default="data/audio", help="Local directory containing audio files")
    parser.add_argument("--bucket", default="contacts", help="S3 bucket name")
    parser.add_argument("--raw-prefix", default="raw/calls", help="S3 prefix for audio files inside the bucket")
    parser.add_argument("--manifest-key", default="audio_poc/manifest.parquet", help="S3 key for the manifest file")
    args = parser.parse_args()

    # resolve data_dir relative to repo root (script may be run from anywhere)
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = repo_root / data_dir
    if not data_dir.exists():
        sys.exit(f"[error] data dir not found: {data_dir}")

    from multimodal_toolkit import config

    s3 = _s3fs(config)

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
