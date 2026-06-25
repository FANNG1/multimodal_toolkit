from __future__ import annotations

import argparse

from ..storage.blob import validate_blob_v2
from ..storage.io import configure_daft_runner, daft_io_config, read_manifest


def _build_with_daft_download(manifest: str):
    from daft import col
    from daft.functions import download

    io_config = daft_io_config()
    df = read_manifest(manifest)
    df = df.with_column("audio_blob", download(col("s3_url"), on_error="null", io_config=io_config))
    return df.where(~col("audio_blob").is_null()).select("doc_id", "s3_url", "audio_blob")


def run(manifest: str, lance_uri: str, overwrite: bool = True) -> None:
    configure_daft_runner()
    io_config = daft_io_config()
    df = _build_with_daft_download(manifest)

    write_mode = "overwrite" if overwrite else "create"
    df.write_lance(lance_uri, mode=write_mode, io_config=io_config, blob_columns=["audio_blob"])
    validate_blob_v2(lance_uri, "audio_blob")

    import daft

    result = daft.read_lance(lance_uri, io_config=io_config)
    print(f"[ok] wrote Lance blob v2 table: {lance_uri}")
    print(f"[ok] rows: {result.count_rows()}")
    print(f"[ok] schema: {result.schema()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lance-uri", required=True)
    parser.add_argument("--no-overwrite", action="store_true")
    args = parser.parse_args()
    run(args.manifest, args.lance_uri, overwrite=not args.no_overwrite)


if __name__ == "__main__":
    main()
