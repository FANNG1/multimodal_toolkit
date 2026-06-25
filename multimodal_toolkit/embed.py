from __future__ import annotations

import argparse

import daft
import daft_lance
import lance
from daft import col

from . import config
from .audio_embedding import cosine_score, get_embedder
from .blob import append_columns_by_doc_id, validate_blob_v2
from .io import daft_io_config, lance_storage_options


@daft.cls(cpus=1)
class _EmbedUDF:
    def __init__(self) -> None:
        self._embedder = get_embedder()

    @daft.method.batch(
        return_dtype=daft.DataType.fixed_size_list(daft.DataType.float32(), config.EMBED_DIM)
    )
    def __call__(self, audio_blobs):
        results = []
        for blob in audio_blobs.to_pylist():
            if blob is None:
                results.append(None)
                continue
            results.append(self._embedder.embed_bytes(blob.read()))
        return results


def run(lance_uri: str, seed_doc_ids: list[str], threshold: float) -> None:
    import pyarrow as pa

    if not seed_doc_ids:
        raise ValueError("--seed-doc-ids is required for similar_complaint classification")
    validate_blob_v2(lance_uri, "audio_blob")

    io_config = daft_io_config()
    ds = lance.dataset(lance_uri, storage_options=lance_storage_options(lance_uri))

    df = daft.read_lance(lance_uri, io_config=io_config, default_scan_options={"with_row_id": True})
    df = df.select("doc_id", "audio_blob", "_rowid")
    df = daft_lance.take_blobs(df, ds, "audio_blob")

    embed_udf = _EmbedUDF()
    df = df.with_column("audio_embedding", embed_udf(col("audio_blob")))
    df = df.select("doc_id", "audio_embedding")

    # Collect all embeddings then run O(N*M) seed classification in Python.
    # N (call records) is small in this POC; avoids a second full pipeline pass.
    data = df.collect().to_pydict()
    doc_ids = [str(x) for x in data["doc_id"]]
    emb_map = dict(zip(doc_ids, data["audio_embedding"]))

    missing_seeds = [x for x in seed_doc_ids if x not in emb_map]
    if missing_seeds:
        raise ValueError(f"seed doc_id not found in Lance table: {missing_seeds}")

    rows = []
    for doc_id, emb in emb_map.items():
        best_seed, best_score = "", 0.0
        for sid in seed_doc_ids:
            if sid == doc_id:
                continue
            score = cosine_score(emb, emb_map[sid])
            if score > best_score:
                best_score, best_seed = score, sid
        rows.append(
            {
                "doc_id": doc_id,
                "audio_embedding": emb,
                "similar_complaint": best_score >= threshold,
                "nearest_seed_doc_id": best_seed,
                "nearest_seed_score": best_score,
            }
        )

    table = pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                pa.field("doc_id", pa.utf8()),
                pa.field("audio_embedding", pa.list_(pa.float32(), config.EMBED_DIM)),
                pa.field("similar_complaint", pa.bool_()),
                pa.field("nearest_seed_doc_id", pa.utf8()),
                pa.field("nearest_seed_score", pa.float64()),
            ]
        ),
    )
    append_columns_by_doc_id(lance_uri, table)
    print(f"[ok] appended audio_embedding and similar_complaint columns to: {lance_uri}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lance-uri", required=True)
    parser.add_argument("--seed-doc-ids", required=True, help="Comma-separated complaint seed doc_id list")
    parser.add_argument("--threshold", type=float, default=0.80)
    args = parser.parse_args()
    seed_doc_ids = [x.strip() for x in args.seed_doc_ids.split(",") if x.strip()]
    run(args.lance_uri, seed_doc_ids, args.threshold)


if __name__ == "__main__":
    main()
