from __future__ import annotations

import argparse
from pathlib import Path

from .blob import read_audio_blobs
from .io import daft_io_config, lance_storage_options


DEFAULT_COLUMNS = [
    "doc_id",
    "s3_url",
    "text_emotion",
    "bad_tone",
    "emotion_score",
    "downgrade_related",
    "primary_reason",
    "secondary_reason",
    "similar_complaint",
    "nearest_seed_doc_id",
    "nearest_seed_score",
]


def _schema_names(df) -> set[str]:
    schema = df.schema()
    if hasattr(schema, "column_names"):
        return set(schema.column_names())
    return {field.name for field in schema}


def _read_rows_daft(lance_uri: str, where: str | None):
    import daft

    opts = {"filter": where} if where else None
    if opts:
        df = daft.read_lance(lance_uri, default_scan_options=opts, io_config=daft_io_config())
    else:
        df = daft.read_lance(lance_uri, io_config=daft_io_config())
    names = _schema_names(df)
    cols = [c for c in DEFAULT_COLUMNS if c in names]
    return df.select(*cols).collect().to_pydict()


def _read_rows_lance(lance_uri: str, where: str | None):
    import lance

    ds = lance.dataset(lance_uri, storage_options=lance_storage_options(lance_uri))
    names = set(ds.schema.names)
    cols = [c for c in DEFAULT_COLUMNS if c in names]
    scanner = ds.scanner(columns=cols, filter=where) if where else ds.scanner(columns=cols)
    return scanner.to_table().to_pydict()


def _ann_query_lance(lance_uri: str, query_doc_id: str, top_k: int, where: str | None) -> list[dict]:
    """ANN query via Lance native scanner(nearest=...).

    Uses Lance ANN instead of collecting all embeddings into memory.
    _distance is not schema-hidden here (unlike daft.read_lance).
    """
    import lance

    ds = lance.dataset(lance_uri, storage_options=lance_storage_options(lance_uri))
    query_table = ds.scanner(columns=["doc_id", "audio_embedding"], filter=f"doc_id = '{query_doc_id}'").to_table()
    if query_table.num_rows == 0:
        raise ValueError(f"query_doc_id not found in Lance table: {query_doc_id}")
    query_vec = query_table["audio_embedding"][0].as_py()

    names = set(ds.schema.names)
    cols = [c for c in DEFAULT_COLUMNS if c in names]
    nearest = {"column": "audio_embedding", "key": query_vec, "k": top_k}
    scanner_kwargs: dict = {"columns": cols, "nearest": nearest}
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
    engine: str = "daft",
) -> None:
    if query_doc_id:
        selected = _ann_query_lance(lance_uri, query_doc_id, top_k, where)
    elif engine == "daft":
        rows = _read_rows_daft(lance_uri, where)
        doc_ids = [str(x) for x in rows.get("doc_id", [])]
        selected = [{k: v[i] for k, v in rows.items()} for i in range(min(top_k, len(doc_ids)))]
    elif engine == "lance":
        rows = _read_rows_lance(lance_uri, where)
        doc_ids = [str(x) for x in rows.get("doc_id", [])]
        selected = [{k: v[i] for k, v in rows.items()} for i in range(min(top_k, len(doc_ids)))]
    else:
        raise ValueError(f"unsupported query engine: {engine}")

    for row in selected:
        print(row)

    if export_audio_dir and selected:
        out_dir = Path(export_audio_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        blobs = read_audio_blobs(lance_uri, [row["doc_id"] for row in selected])
        for doc_id, blob in blobs.items():
            if blob:
                (out_dir / f"{doc_id}.audio").write_bytes(blob)
        print(f"[ok] exported {len(blobs)} audio blobs to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lance-uri", required=True)
    parser.add_argument("--where")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query-doc-id")
    parser.add_argument("--export-audio-dir")
    parser.add_argument("--engine", choices=["daft", "lance"], default="daft")
    args = parser.parse_args()
    run(args.lance_uri, args.where, args.top_k, args.query_doc_id, args.export_audio_dir, args.engine)


if __name__ == "__main__":
    main()
