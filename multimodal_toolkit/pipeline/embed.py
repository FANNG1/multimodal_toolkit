from __future__ import annotations

import argparse

import pyarrow as pa

from .. import config
from ..storage.blob import validate_blob_v2
from ..storage.io import lance_storage_options

_EMBED_TYPE = pa.list_(pa.float32(), config.EMBED_DIM)
_EMBED_SCHEMA = pa.schema(
    [
        pa.field("audio_embedding", _EMBED_TYPE),
    ]
)

_embedder = None


def _embed_transform(batch: pa.Table) -> pa.Table:
    global _embedder
    if _embedder is None:
        from multimodal_toolkit.audio.embedding import get_embedder
        _embedder = get_embedder()
    audio_blobs = batch.column("audio_blob").to_pylist()
    results = [_embedder.embed_bytes(b) if b is not None else None for b in audio_blobs]
    return pa.table(
        {
            "audio_embedding": pa.array(results, type=_EMBED_TYPE),
        },
        schema=_EMBED_SCHEMA,
    )


def run(lance_uri: str) -> None:
    import lance

    validate_blob_v2(lance_uri, "audio_blob")

    storage_options = lance_storage_options(lance_uri) or None
    ds = lance.dataset(lance_uri, storage_options=storage_options)
    if "audio_embedding" in ds.schema.names:
        raise ValueError("audio_embedding already exists. Recreate the table or delete the column before recomputing embeddings.")

    doc_ids = ds.to_table(columns=["doc_id"])["doc_id"].to_pylist()
    blob_rows = ds.read_blobs("audio_blob", indices=list(range(len(doc_ids))), preserve_order=True)
    audio_bytes = [b for _row_idx, b in sorted(blob_rows, key=lambda x: x[0])]
    table = _embed_transform(pa.table({"audio_blob": audio_bytes}))
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
