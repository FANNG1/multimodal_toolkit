"""Stage 5: manage the lance asset table — delete rows by ingest_time range.

API priority (lance_ray preferred for table management):
  delete        → pylance ds.delete()              (no Daft/lance-ray equivalent)
  compact       → lance_ray.compact_files()        (preferred for table management)
  cleanup       → pylance ds.cleanup_old_versions() (no alternative)

  --before DATE   delete rows where ingest_time < DATE
  --after  DATE   delete rows where ingest_time > DATE

DATE format: ISO 8601, e.g. 2025-01-01 or 2025-01-01T00:00:00
At least one bound must be provided; both can be combined for a date range.
"""
from __future__ import annotations

import argparse

import lance
import lance_ray

from ..storage.io import lance_storage_options


def delete_by_date(
    lance_uri: str,
    before: str | None = None,
    after: str | None = None,
) -> None:
    if not before and not after:
        raise ValueError("Provide at least one of: --before or --after")

    clauses = []
    if before:
        clauses.append(f"ingest_time < timestamp '{before}'")
    if after:
        clauses.append(f"ingest_time > timestamp '{after}'")
    filter_str = " AND ".join(clauses)

    # delete: pylance only (no Daft/lance-ray equivalent)
    ds = lance.dataset(lance_uri, storage_options=lance_storage_options(lance_uri))
    ds.delete(filter_str)

    print(f"[ok] deleted rows where: {filter_str}")

    # compact: lance_ray (preferred for table management).
    # compaction_options must be an explicit dict: lance_ray 0.4.x passes the
    # default None straight into Compaction.plan(), which rejects it
    # (lance-format/lance-ray#5224; drop once the fix ships).
    lance_ray.compact_files(
        lance_uri,
        compaction_options={},
        storage_options=lance_storage_options(lance_uri),
    )
    print(f"[ok] compacted: {lance_uri}")
    ds.cleanup_old_versions()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lance-uri", required=True, help="lance asset table URI (S3)")
    parser.add_argument("--before", help="delete rows with ingest_time before this date (ISO 8601)")
    parser.add_argument("--after", help="delete rows with ingest_time after this date (ISO 8601)")
    args = parser.parse_args()
    delete_by_date(args.lance_uri, args.before, args.after)


if __name__ == "__main__":
    main()
