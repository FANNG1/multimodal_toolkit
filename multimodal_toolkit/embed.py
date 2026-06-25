from __future__ import annotations

import argparse

from . import config
from .audio_embedding import cosine_score, get_embedder
from .blob import append_columns_by_doc_id, read_audio_blobs, validate_blob_v2


def run(lance_uri: str, seed_doc_ids: list[str], threshold: float) -> None:
    import pyarrow as pa

    if not seed_doc_ids:
        raise ValueError("--seed-doc-ids is required for similar_complaint classification")
    validate_blob_v2(lance_uri, "audio_blob")

    blobs = read_audio_blobs(lance_uri)
    embedder = get_embedder()
    embeddings: dict[str, list[float] | None] = {}
    for doc_id, audio_bytes in blobs.items():
        embeddings[doc_id] = embedder.embed_bytes(audio_bytes)

    missing_seeds = [x for x in seed_doc_ids if x not in embeddings]
    if missing_seeds:
        raise ValueError(f"seed doc_id not found in Lance table: {missing_seeds}")

    rows = []
    for doc_id, emb in embeddings.items():
        best_seed = ""
        best_score = 0.0
        for seed_id in seed_doc_ids:
            if seed_id == doc_id:
                continue
            score = cosine_score(emb, embeddings[seed_id])
            if score > best_score:
                best_score = score
                best_seed = seed_id
        rows.append(
            {
                "doc_id": doc_id,
                "audio_embedding": emb,
                "similar_complaint": bool(best_score >= threshold),
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
