"""Stage 3: build indexes on the lance asset table.

API selection:
  Scalar index (ZONEMAP) → pylance ds.create_scalar_index()
  Vector index (IVF_PQ)  → lance_ray.create_index()

  --embedding   IVF index on an embedding column (default: audio_embedding)
  --time        ZONEMAP index on ingest_time (for fast time range queries)

Note: IVF_PQ requires at least num_partitions * 256 rows by default.
      For small tables use --num-partitions 1 --sample-rate 2 --index-type IVF_FLAT.
"""
from __future__ import annotations

import argparse

import lance
import lance_ray

from ..storage.io import lance_storage_options


def build_embedding_index(
    lance_uri: str,
    column: str = "audio_embedding",
    num_partitions: int = 16,
    num_sub_vectors: int = 16,
    sample_rate: int = 256,
    index_type: str = "IVF_PQ",
) -> None:
    ds = lance.dataset(lance_uri, storage_options=lance_storage_options(lance_uri))
    if column not in ds.schema.names:
        raise ValueError(f"{column} column not found; run Stage 1 with --embed first.")
    storage_options = lance_storage_options(lance_uri) or None
    kwargs: dict = dict(
        column=column,
        index_type=index_type,
        num_partitions=num_partitions,
        sample_rate=sample_rate,
        replace=True,
        storage_options=storage_options,
    )
    if index_type in ("IVF_PQ", "IVF_HNSW_PQ"):
        kwargs["num_sub_vectors"] = num_sub_vectors
    try:
        lance_ray.create_index(lance_uri, **kwargs)
    except Exception as exc:
        print(f"[warn] lance_ray index build failed; falling back to pylance: {exc}")
        fallback_kwargs: dict = dict(
            index_type=index_type,
            replace=True,
            num_partitions=num_partitions,
            sample_rate=sample_rate,
            storage_options=storage_options,
        )
        if index_type in ("IVF_PQ", "IVF_HNSW_PQ"):
            fallback_kwargs["num_sub_vectors"] = num_sub_vectors
        ds.create_index(column, **fallback_kwargs)
    print(f"[ok] built {index_type} index on {column} ({num_partitions} partitions): {lance_uri}")


def build_time_index(lance_uri: str) -> None:
    ds = lance.dataset(lance_uri, storage_options=lance_storage_options(lance_uri))
    ds.create_scalar_index("ingest_time", index_type="ZONEMAP", replace=True)
    print(f"[ok] built ZONEMAP index on ingest_time: {lance_uri}")


def run(
    lance_uri: str,
    embedding: bool = True,
    time: bool = True,
    embedding_column: str = "audio_embedding",
    num_partitions: int = 16,
    num_sub_vectors: int = 16,
    sample_rate: int = 256,
    index_type: str = "IVF_PQ",
) -> None:
    if embedding:
        build_embedding_index(
            lance_uri,
            embedding_column,
            num_partitions,
            num_sub_vectors,
            sample_rate,
            index_type,
        )
    if time:
        build_time_index(lance_uri)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lance-uri", required=True, help="lance asset table URI (S3)")
    parser.add_argument("--embedding", action="store_true", default=True, help="build IVF index on the embedding column (default: on)")
    parser.add_argument("--no-embedding", dest="embedding", action="store_false")
    parser.add_argument("--embedding-column", default="audio_embedding", help="embedding column to index (default: audio_embedding)")
    parser.add_argument("--time", action="store_true", default=True, help="build ZONEMAP index on ingest_time (default: on)")
    parser.add_argument("--no-time", dest="time", action="store_false")
    parser.add_argument("--num-partitions", type=int, default=16)
    parser.add_argument("--num-sub-vectors", type=int, default=16)
    parser.add_argument("--sample-rate", type=int, default=256,
                        help="rows sampled per IVF partition; lower for small tables (default: 256)")
    parser.add_argument("--index-type", default="IVF_PQ",
                        choices=["IVF_PQ", "IVF_FLAT", "IVF_SQ", "IVF_HNSW_PQ", "IVF_HNSW_FLAT"],
                        help="vector index type (default: IVF_PQ; use IVF_FLAT for small tables)")
    args = parser.parse_args()
    run(
        args.lance_uri,
        args.embedding,
        args.time,
        args.embedding_column,
        args.num_partitions,
        args.num_sub_vectors,
        args.sample_rate,
        args.index_type,
    )


if __name__ == "__main__":
    main()
