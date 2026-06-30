"""Distributed embed step using daft + daft_lance.merge_columns_df.

NOTE: This file depends on three bug fixes in daft-lance that are pending review:
  - PR #44: fixed_size_list<float32>[N] type erased to list<float64> in FastPathFragmentWriter
    https://github.com/daft-engine/daft-lance/pull/44
  - PR #45: df.collect() in _can_use_fast_path exhausts one-shot BlobFile objects
    https://github.com/daft-engine/daft-lance/pull/45
  - PR #46: next_fid collides with nested struct child field IDs
    https://github.com/daft-engine/daft-lance/pull/46

Do NOT use this file until all three PRs are merged and released.
Use embed.py (single-machine) in the meantime.
"""
from __future__ import annotations

import argparse

import daft
import daft_lance
import lance
from daft import col

from .. import config
from ..storage.blob import validate_blob_v2
from ..storage.io import configure_daft_runner, daft_io_config, lance_storage_options


@daft.func.batch(return_dtype=daft.DataType.binary())
def _read_bytes(audio_blobs):
    return [blob.read() if blob is not None else None for blob in audio_blobs.to_pylist()]


@daft.cls(cpus=1)
class _EmbedUDF:
    def __init__(self) -> None:
        from multimodal_toolkit.audio.embedding import get_embedder

        self._embedder = get_embedder()

    @daft.method.batch(
        return_dtype=daft.DataType.fixed_size_list(daft.DataType.float32(), config.EMBED_DIM)
    )
    def __call__(self, audio_bytes_col):
        return [
            self._embedder.embed_bytes(b) if b is not None else None
            for b in audio_bytes_col.to_pylist()
        ]


def run(lance_uri: str) -> None:
    configure_daft_runner()

    validate_blob_v2(lance_uri, "audio_blob")

    io_config = daft_io_config()
    storage_options = lance_storage_options(lance_uri)
    ds = lance.dataset(lance_uri, storage_options=storage_options)

    if "audio_embedding" in ds.schema.names:
        raise ValueError(
            "audio_embedding already exists. Recreate the table or delete the column before recomputing."
        )

    df = daft.read_lance(
        lance_uri,
        io_config=io_config,
        include_fragment_id=True,
        default_scan_options={"with_row_id": True, "with_row_address": True},
    )
    df = daft_lance.take_blobs(df, ds, "audio_blob")
    # Materialize blob bytes once — BlobFile is a one-shot stream
    df = df.with_column("audio_bytes", _read_bytes(col("audio_blob")))
    df = df.with_column("audio_embedding", _EmbedUDF()(col("audio_bytes")))

    daft_lance.merge_columns_df(
        df.select("fragment_id", "_rowaddr", "audio_embedding"),
        lance_uri,
        io_config=io_config,
        storage_options=storage_options or None,
    )

    print(f"[ok] appended audio_embedding to: {lance_uri}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lance-uri", required=True)
    args = parser.parse_args()
    run(args.lance_uri)


if __name__ == "__main__":
    main()
