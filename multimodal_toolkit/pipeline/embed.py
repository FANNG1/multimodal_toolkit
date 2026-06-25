from __future__ import annotations

import argparse

import lance
import pyarrow as pa

from .. import config
from ..storage.blob import validate_blob_v2
from ..storage.io import lance_storage_options


_EMBED_SCHEMA = pa.schema(
    [
        pa.field("audio_embedding", pa.list_(pa.float32(), config.EMBED_DIM)),
    ]
)


def _compute_embeddings(audio_bytes: list[bytes | None]) -> pa.Table:
    from multimodal_toolkit.audio.embedding import get_embedder

    embedder = get_embedder()
    embeddings = [embedder.embed_bytes(b) if b is not None else None for b in audio_bytes]
    return pa.table(
        {
            "audio_embedding": pa.array(
                embeddings,
                type=_EMBED_SCHEMA.field("audio_embedding").type,
            )
        },
        schema=_EMBED_SCHEMA,
    )


def run(lance_uri: str) -> None:
    validate_blob_v2(lance_uri, "audio_blob")

    storage_options = lance_storage_options(lance_uri)
    ds = lance.dataset(lance_uri, storage_options=storage_options)

    if "audio_embedding" in ds.schema.names:
        raise ValueError("audio_embedding already exists. Recreate the table or delete the column before recomputing.")

    # TODO: Switch this step back to a distributed path when the dependencies
    # are stable enough:
    # - daft_lance.merge_columns_df currently has correctness/compatibility
    #   issues in the blob-v2 embedding path.
    # - lance-ray can read blob bytes through read_lance, but the write-back
    #   APIs needed for this POC require a newer/stable release.
    #
    # For now, match the verified Guangdong Daft POC: materialize blob v2 bytes
    # with Lance native read_blobs, compute embeddings locally, and append the
    # new column with Lance native add_columns.
    row_count = ds.count_rows()
    blob_rows = ds.read_blobs("audio_blob", indices=list(range(row_count)), preserve_order=True)
    audio_bytes = [blob for _idx, blob in sorted(blob_rows, key=lambda x: x[0])]

    table = _compute_embeddings(audio_bytes)
    reader = pa.RecordBatchReader.from_batches(_EMBED_SCHEMA, table.to_batches())
    ds.add_columns(reader)

    print(f"[ok] appended audio_embedding to: {lance_uri}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lance-uri", required=True)
    args = parser.parse_args()
    run(args.lance_uri)


if __name__ == "__main__":
    main()
