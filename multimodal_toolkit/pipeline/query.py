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


def _scalar_query(lance_uri: str, where: str | None, top_k: int) -> list[dict]:
    import daft

    kwargs = {}
    if where:
        kwargs["default_scan_options"] = {"filter": where}
    df = daft.read_lance(lance_uri, io_config=daft_io_config(), **kwargs)
    names = set(df.schema().column_names())
    cols = [c for c in DEFAULT_COLUMNS if c in names]
    rows = df.select(*cols).limit(top_k).collect().to_pydict()
    n = len(next(iter(rows.values()), []))
    return [{k: rows[k][i] for k in rows} for i in range(n)]


def _ann_query_lance(lance_uri: str, query_doc_id: str, top_k: int, where: str | None) -> list[dict]:
    """ANN via Lance native scanner(nearest=...).

    Lance native is used intentionally: daft.read_lance hides _distance, making
    it impossible to rank ANN results by similarity score inside Daft.
    """
    import lance

    ds = lance.dataset(lance_uri, storage_options=lance_storage_options(lance_uri))
    query_table = ds.scanner(columns=["doc_id", "audio_embedding"], filter=f"doc_id = '{query_doc_id}'").to_table()
    if query_table.num_rows == 0:
        raise ValueError(f"query_doc_id not found in Lance table: {query_doc_id}")
    query_vec = query_table["audio_embedding"][0].as_py()

    import pyarrow as pa

    names = set(ds.schema.names)
    cols = [c for c in DEFAULT_COLUMNS if c in names]
    scanner_kwargs: dict = {
        "columns": cols,
        "nearest": {"column": "audio_embedding", "q": pa.array(query_vec, type=pa.float32()), "k": top_k},
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
    top_k: int,
    query_doc_id: str | None,
    export_audio_dir: str | None,
) -> None:
    if query_doc_id:
        selected = _ann_query_lance(lance_uri, query_doc_id, top_k, where)
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
    args = parser.parse_args()
    run(args.lance_uri, args.where, args.top_k, args.query_doc_id, args.export_audio_dir)


if __name__ == "__main__":
    main()
