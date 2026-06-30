"""Stage 3: build indexes on the lance asset table.

API selection (lance_ray preferred for all Lance table management):
  Scalar index (ZONEMAP) → lance_ray.create_scalar_index()
  Vector index (IVF_PQ)  → lance_ray.create_index()

  --embedding   IVF_PQ index on audio_embedding (for ANN vector search)
  --time        ZONEMAP index on ingest_time (for fast time range queries)

Note: IVF_PQ requires at least num_partitions * 256 rows by default.
      For small tables use --num-partitions 1 or wait until the table has enough rows.
"""
from __future__ import annotations

import argparse

import lance
import lance_ray

from ..storage.io import lance_storage_options


def build_embedding_index(
    lance_uri: str,
    num_partitions: int = 16,
    num_sub_vectors: int = 16,
) -> None:
    ds = lance.dataset(lance_uri, storage_options=lance_storage_options(lance_uri))
    if "audio_embedding" not in ds.schema.names:
        raise ValueError("audio_embedding column not found; run Stage 1 with --embed first.")
    storage_options = lance_storage_options(lance_uri) or None
    lance_ray.create_index(
        lance_uri,
        column="audio_embedding",
        index_type="IVF_PQ",
        num_partitions=num_partitions,
        num_sub_vectors=num_sub_vectors,
        replace=True,
        storage_options=storage_options,
    )
    print(f"[ok] built IVF_PQ index on audio_embedding ({num_partitions} partitions, {num_sub_vectors} sub-vecs): {lance_uri}")


def build_time_index(lance_uri: str) -> None:
    storage_options = lance_storage_options(lance_uri) or None
    lance_ray.create_scalar_index(
        lance_uri,
        column="ingest_time",
        index_type="ZONEMAP",
        replace=True,
        storage_options=storage_options,
    )
    print(f"[ok] built ZONEMAP index on ingest_time: {lance_uri}")


def run(
    lance_uri: str,
    embedding: bool = True,
    time: bool = True,
    num_partitions: int = 16,
    num_sub_vectors: int = 16,
) -> None:
    if embedding:
        build_embedding_index(lance_uri, num_partitions, num_sub_vectors)
    if time:
        build_time_index(lance_uri)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lance-uri", required=True, help="lance asset table URI (S3)")
    parser.add_argument("--embedding", action="store_true", default=True, help="build IVF_PQ index on audio_embedding (default: on)")
    parser.add_argument("--no-embedding", dest="embedding", action="store_false")
    parser.add_argument("--time", action="store_true", default=True, help="build ZONEMAP index on ingest_time (default: on)")
    parser.add_argument("--no-time", dest="time", action="store_false")
    parser.add_argument("--num-partitions", type=int, default=16)
    parser.add_argument("--num-sub-vectors", type=int, default=16)
    args = parser.parse_args()
    run(args.lance_uri, args.embedding, args.time, args.num_partitions, args.num_sub_vectors)


if __name__ == "__main__":
    main()
