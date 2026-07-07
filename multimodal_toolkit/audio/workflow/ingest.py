"""Stage 2: analysis output → lance asset table (append, blob v2, with ingest_time).

Reads all fields from Stage 1 output (JSONL or lance staging table).
The input must include s3_url so audio blobs can be downloaded.
Appends to the lance asset table — never overwrites.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import daft
from daft import col, lit
from daft.functions import download

from ...storage.blob import validate_blob_v2
from ...storage.io import configure_daft_runner, daft_io_config, lance_write_mode, read_analysis_output


def run(analysis_path: str, lance_uri: str) -> None:
    configure_daft_runner()
    io_config = daft_io_config()

    now = datetime.now(timezone.utc)

    df = read_analysis_output(analysis_path, io_config)

    df = df.with_column(
        "audio_blob", download(col("s3_url"), on_error="null", io_config=io_config)
    )
    df = df.where(~col("audio_blob").is_null())
    df = df.with_column(
        "ingest_time",
        lit(now).cast(daft.DataType.timestamp("us", "UTC")),
    )

    mode = lance_write_mode(lance_uri)
    df.write_lance(lance_uri, mode=mode, io_config=io_config, blob_columns=["audio_blob"])
    validate_blob_v2(lance_uri, "audio_blob")

    result = daft.read_lance(lance_uri, io_config=io_config)
    print(f"[ok] appended to lance asset table: {lance_uri}")
    print(f"[ok] total rows: {result.count_rows()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis", required=True,
        help="Stage 1 output: S3 JSONL path or lance staging table URI",
    )
    parser.add_argument("--lance-uri", required=True, help="lance asset table URI (S3)")
    args = parser.parse_args()
    run(args.analysis, args.lance_uri)


if __name__ == "__main__":
    main()
