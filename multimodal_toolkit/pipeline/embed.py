from __future__ import annotations

import argparse

import daft_lance
import lance
import pyarrow as pa

from .. import config
from ..storage.blob import validate_blob_v2
from ..storage.io import configure_daft_runner, daft_io_config, lance_storage_options


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

    from multimodal_toolkit.audio.embedding import get_embedder

    embedder = get_embedder()

    def _embed_transform(batch: pa.RecordBatch) -> pa.RecordBatch:
        row_ids = batch["_rowid"].to_pylist()
        blobs = ds.take_blobs("audio_blob", ids=row_ids)
        embeddings = [
            embedder.embed_bytes(blob.read()) if blob else None
            for blob in blobs
        ]
        return pa.record_batch(
            {
                "audio_embedding": pa.array(
                    [e.tolist() if e is not None else None for e in embeddings],
                    type=pa.list_(pa.float32(), config.EMBED_DIM),
                )
            }
        )

    daft_lance.merge_columns(
        lance_uri,
        io_config=io_config,
        transform=_embed_transform,
        read_columns=["_rowid"],
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
