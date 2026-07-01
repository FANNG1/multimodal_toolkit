from __future__ import annotations

import argparse
from pathlib import Path

from ..storage.io import daft_io_config, lance_storage_options


DEFAULT_COLUMNS = [
    "doc_id",
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


def _scalar_query(lance_uri: str, where: str | None, top_k: int) -> list[dict]:
    import daft

    kwargs = {}
    if where:
        kwargs["default_scan_options"] = {"filter": where}
    df = daft.read_lance(lance_uri, io_config=daft_io_config(), **kwargs)
    names = set(df.schema().column_names())
    cols = [c for c in DEFAULT_COLUMNS if c in names]
    rows = df.select(*cols).limit(top_k).collect().to_pydict()
    return _rows_from_pydict(rows)


def _ann_query_daft(
    lance_uri: str,
    query_doc_id: str,
    top_k: int,
    where: str | None,
    distance_range: tuple[float, float] | None,
) -> list[dict]:
    """ANN via Daft's Lance scanner(nearest=...)."""
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
    if not query_rows.get("audio_embedding"):
        raise ValueError(f"query_doc_id not found in Lance table: {query_doc_id}")
    query_vec = query_rows["audio_embedding"][0]

    nearest: dict = {
        "column": "audio_embedding",
        "q": pa.array(query_vec, type=pa.float32()),
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
    top_k: int,
    query_doc_id: str | None,
    export_audio_dir: str | None,
    distance_range: tuple[float, float] | None = None,
) -> None:
    if query_doc_id:
        selected = _ann_query_daft(lance_uri, query_doc_id, top_k, where, distance_range)
    else:
        selected = _scalar_query(lance_uri, where, top_k)

    for row in selected:
        print(row)

    if export_audio_dir and selected:
        import daft
        import daft_lance

        out_dir = Path(export_audio_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        selected_ids = [str(row["doc_id"]) for row in selected]

        import lance

        ds = lance.dataset(lance_uri, storage_options=lance_storage_options(lance_uri))
        df = (
            daft.read_lance(lance_uri, io_config=daft_io_config(), default_scan_options={"with_row_id": True})
            .where(daft.col("doc_id").is_in(selected_ids))
            .select("doc_id", "audio_blob", "_rowid")
        )
        df = daft_lance.take_blobs(df, ds, "audio_blob")
        rows = df.collect().to_pydict()
        for doc_id, blob in zip(rows["doc_id"], rows["audio_blob"]):
            if blob:
                (out_dir / doc_id).write_bytes(blob.read())
        print(f"[ok] exported {len(rows['doc_id'])} audio blobs to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lance-uri", required=True)
    parser.add_argument("--where")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query-doc-id")
    parser.add_argument("--export-audio-dir")
    parser.add_argument("--distance-min", type=float)
    parser.add_argument("--distance-max", type=float)
    args = parser.parse_args()
    distance_range = None
    if args.distance_min is not None or args.distance_max is not None:
        if args.distance_min is None or args.distance_max is None:
            parser.error("--distance-min and --distance-max must be provided together")
        distance_range = (args.distance_min, args.distance_max)
    run(args.lance_uri, args.where, args.top_k, args.query_doc_id, args.export_audio_dir, distance_range)


if __name__ == "__main__":
    main()
