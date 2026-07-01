"""Stage 4: query the lance asset table.

  --where       scalar filter via Daft pushdown
  --sql         full Daft SQL SELECT statement (overrides --where; table name: calls)
  --vector-from doc_id to use as ANN query vector (pylance native scanner)

ANN search (IVF index) requires pylance scanner and is triggered by --vector-from.
--vector-from and --where can be combined for pre-filtered ANN.
"""
from __future__ import annotations

import argparse

from ..storage.io import daft_io_config, lance_storage_options

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
    n = len(next(iter(rows.values()), []))
    return [{k: rows[k][i] for k in rows} for i in range(n)]


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
    n = len(next(iter(rows.values()), []))
    return [{k: rows[k][i] for k in rows} for i in range(n)]


def vector_query(
    lance_uri: str,
    query_doc_id: str,
    top_k: int = 10,
    where: str | None = None,
) -> list[dict]:
    """ANN similarity search via pylance (exposes _distance column unavailable in Daft)."""
    import lance
    import pyarrow as pa

    ds = lance.dataset(lance_uri, storage_options=lance_storage_options(lance_uri))
    q_table = ds.scanner(
        columns=["doc_id", "audio_embedding"],
        filter=f"doc_id = '{query_doc_id}'",
    ).to_table()
    if q_table.num_rows == 0:
        raise ValueError(f"query_doc_id not found: {query_doc_id}")
    q_vec = q_table["audio_embedding"][0].as_py()

    names = set(ds.schema.names)
    cols = [c for c in DEFAULT_COLUMNS if c in names]
    scanner_kwargs: dict = {
        "columns": cols,
        "nearest": {
            "column": "audio_embedding",
            "q": pa.array(q_vec, type=pa.float32()),
            "k": top_k,
        },
        "disable_scoring_autoprojection": True,
    }
    if where:
        scanner_kwargs["filter"] = where
    table = ds.scanner(**scanner_kwargs).to_table()
    rows = table.to_pydict()
    return [{k: rows[k][i] for k in rows} for i in range(table.num_rows)]


def run(
    lance_uri: str,
    where: str | None,
    sql: str | None,
    top_k: int,
    query_doc_id: str | None,
) -> None:
    if sql:
        results = sql_query(lance_uri, sql, top_k)
    elif query_doc_id:
        results = vector_query(lance_uri, query_doc_id, top_k, where)
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
    args = parser.parse_args()
    run(args.lance_uri, args.where, args.sql, args.top_k, args.query_doc_id)


if __name__ == "__main__":
    main()
