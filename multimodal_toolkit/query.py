from __future__ import annotations

import argparse
from pathlib import Path

from .audio_embedding import cosine_score
from .blob import read_audio_blobs
from .io import daft_io_config


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

    ds = lance.dataset(lance_uri)
    names = set(ds.schema.names)
    cols = [c for c in DEFAULT_COLUMNS if c in names]
    scanner = ds.scanner(columns=cols, filter=where) if where else ds.scanner(columns=cols)
    return scanner.to_table().to_pydict()


def _read_embeddings(lance_uri: str) -> dict[str, list[float] | None]:
    import daft

    df = daft.read_lance(lance_uri, io_config=daft_io_config()).select("doc_id", "audio_embedding")
    data = df.collect().to_pydict()
    return {str(doc_id): emb for doc_id, emb in zip(data["doc_id"], data["audio_embedding"], strict=False)}


def run(
    lance_uri: str,
    where: str | None,
    top_k: int,
    query_doc_id: str | None,
    export_audio_dir: str | None,
    engine: str = "daft",
) -> None:
    if engine == "daft":
        rows = _read_rows_daft(lance_uri, where)
    elif engine == "lance":
        rows = _read_rows_lance(lance_uri, where)
    else:
        raise ValueError(f"unsupported query engine: {engine}")
    doc_ids = [str(x) for x in rows.get("doc_id", [])]

    if query_doc_id:
        embeddings = _read_embeddings(lance_uri)
        query_emb = embeddings.get(query_doc_id)
        scored = [(doc_id, cosine_score(query_emb, embeddings.get(doc_id))) for doc_id in doc_ids if doc_id != query_doc_id]
        keep = {doc_id for doc_id, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]}
        keep.add(query_doc_id)
        mask = [doc_id in keep for doc_id in doc_ids]
    else:
        mask = [True] * len(doc_ids)

    selected = []
    for i, ok in enumerate(mask):
        if ok:
            selected.append({k: v[i] for k, v in rows.items()})

    limit = len(selected) if query_doc_id else top_k
    for row in selected[:limit]:
        print(row)

    if export_audio_dir and selected:
        out_dir = Path(export_audio_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        blobs = read_audio_blobs(lance_uri, [row["doc_id"] for row in selected[:limit]])
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
