"""Stage 4: query the lance asset table.

  --where       scalar filter via Daft pushdown
  --sql         full Daft SQL SELECT statement (overrides --where; table name: calls)
  --vector-from doc_id to use as ANN query vector

ANN search (IVF index) uses Daft's Lance scanner and is triggered by --vector-from.
--vector-from and --where can be combined for pre-filtered ANN.
"""
from __future__ import annotations

import argparse

from ...storage.io import daft_io_config

DEFAULT_COLUMNS = [
    "doc_id",
    "ingest_time",
    "text_emotion",
    "bad_tone",
    "emotion_score",
    "downgrade_related",
    "primary_reason",
    "secondary_reason",
]


def _rows_from_pydict(rows: dict) -> list[dict]:
    n = len(next(iter(rows.values()), []))
    return [{k: rows[k][i] for k in rows} for i in range(n)]


def _doc_id_filter(doc_id: str) -> str:
    escaped = doc_id.replace("'", "''")
    return f"doc_id = '{escaped}'"


def scalar_query(lance_uri: str, where: str | None = None, top_k: int = 100) -> list[dict]:
    """Filter query via Daft (pushes filter to Lance scanner)."""
    import daft

    kwargs: dict = {}
    if where:
        kwargs["default_scan_options"] = {"filter": where}
    df = daft.read_lance(lance_uri, io_config=daft_io_config(), **kwargs)
    names = set(df.schema().column_names())
    cols = [c for c in DEFAULT_COLUMNS if c in names]
    rows = df.select(*cols).limit(top_k).collect().to_pydict()
    return _rows_from_pydict(rows)


def sql_query(lance_uri: str, sql: str, top_k: int = 100) -> list[dict]:
    """Arbitrary Daft SQL SELECT against the Lance table.

    The table is available as ``calls`` in the SQL scope. Supports scalar
    filters, projections, aggregations, ORDER BY, etc.

    Examples::

        SELECT doc_id, primary_reason, emotion_score
        FROM calls
        WHERE bad_tone = true AND emotion_score > 0.5
        ORDER BY emotion_score DESC

        SELECT primary_reason, COUNT(*) AS cnt, AVG(emotion_score) AS avg_score
        FROM calls
        GROUP BY primary_reason
        ORDER BY cnt DESC
    """
    import daft

    calls = daft.read_lance(lance_uri, io_config=daft_io_config())
    rows = daft.sql(sql, calls=calls).limit(top_k).collect().to_pydict()
    return _rows_from_pydict(rows)


def vector_query(
    lance_uri: str,
    query_doc_id: str,
    top_k: int = 10,
    where: str | None = None,
    distance_range: tuple[float, float] | None = None,
) -> list[dict]:
    """ANN similarity search via Daft's Lance scanner."""
    import daft
    import pyarrow as pa

    query_rows = (
        daft.read_lance(
            lance_uri,
            io_config=daft_io_config(),
            default_scan_options={"filter": _doc_id_filter(query_doc_id)},
        )
        .select("audio_embedding")
        .limit(1)
        .collect()
        .to_pydict()
    )
    if not query_rows.get("audio_embedding") or query_rows["audio_embedding"][0] is None:
        raise ValueError(f"query_doc_id not found: {query_doc_id}")
    q_vec = query_rows["audio_embedding"][0]

    nearest: dict = {
        "column": "audio_embedding",
        "q": pa.array(q_vec, type=pa.float32()),
        "k": top_k,
    }
    if distance_range is not None:
        nearest["distance_range"] = distance_range

    scan_options: dict = {"nearest": nearest, "disable_scoring_autoprojection": True}
    if where:
        scan_options["filter"] = where
        scan_options["prefilter"] = True

    df = daft.read_lance(lance_uri, io_config=daft_io_config(), default_scan_options=scan_options)
    names = set(df.schema().column_names())
    cols = [c for c in DEFAULT_COLUMNS if c in names]
    rows = df.select(*cols).limit(top_k).collect().to_pydict()
    return _rows_from_pydict(rows)


def run(
    lance_uri: str,
    where: str | None,
    sql: str | None,
    top_k: int,
    query_doc_id: str | None,
    distance_range: tuple[float, float] | None = None,
) -> None:
    if sql:
        results = sql_query(lance_uri, sql, top_k)
    elif query_doc_id:
        results = vector_query(lance_uri, query_doc_id, top_k, where, distance_range)
    else:
        results = scalar_query(lance_uri, where, top_k)
    for row in results:
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lance-uri", required=True, help="lance asset table URI (S3)")
    parser.add_argument("--where", help="SQL WHERE clause pushed down to Lance scanner")
    parser.add_argument("--sql", help="full Daft SQL SELECT (table name: calls)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--vector-from", dest="query_doc_id", help="doc_id to use as ANN query vector")
    parser.add_argument("--distance-min", type=float, help="minimum vector distance for ANN results")
    parser.add_argument("--distance-max", type=float, help="maximum vector distance for ANN results")
    args = parser.parse_args()
    distance_range = None
    if args.distance_min is not None or args.distance_max is not None:
        if args.distance_min is None or args.distance_max is None:
            parser.error("--distance-min and --distance-max must be provided together")
        distance_range = (args.distance_min, args.distance_max)
    run(args.lance_uri, args.where, args.sql, args.top_k, args.query_doc_id, distance_range)


if __name__ == "__main__":
    main()
