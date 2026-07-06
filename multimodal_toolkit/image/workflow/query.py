"""图片 Stage 4：查询 lance 图片资产表。

  --where      标量过滤，经 Daft 下推到 Lance scanner
  --sql        完整的 Daft SQL SELECT 语句（优先于 --where；表名：images）
  --text       文本搜图（ChineseCLIP 文本向量 → image_embedding ANN）
  --image-path 本地图片搜相似图
  --image-from 表内 doc_id 搜相似图
"""
from __future__ import annotations

import argparse

from ...storage.io import daft_io_config

# 查询默认返回的列（不含 image_blob——查询结果里不需要拖着原始字节）。
# 实际返回时会和表的真实 schema 求交集，缺列不会报错。
DEFAULT_COLUMNS = [
    "doc_id",
    "ingest_time",
    "status",
    "width",
    "height",
    "face_count",
    "face_score",
    "blur_score",
    "face_blur_score",
    "has_face",
    "is_blurry",
    "is_face_blurry",
]


def _rows_from_pydict(rows: dict) -> list[dict]:
    """把 Daft 的列式结果 {列名: [值...]} 转成行式 [{列名: 值}...]，方便打印。"""
    n = len(next(iter(rows.values()), []))
    return [{k: rows[k][i] for k in rows} for i in range(n)]


def _doc_id_filter(doc_id: str) -> str:
    escaped = doc_id.replace("'", "''")
    return f"doc_id = '{escaped}'"


def scalar_query(lance_uri: str, where: str | None = None, top_k: int = 100) -> list[dict]:
    """标量过滤查询（过滤条件经 Daft 下推到 Lance scanner，不全表扫描）。"""
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
    """对图片表执行任意 Daft SQL SELECT（表在 SQL 里叫 ``images``）。

    示例::

        SELECT doc_id, blur_score, face_count
        FROM images
        WHERE has_face = true AND is_blurry = false
        ORDER BY blur_score ASC

        SELECT has_face, COUNT(*) AS cnt, AVG(blur_score) AS avg_blur
        FROM images
        GROUP BY has_face
    """
    import daft

    images = daft.read_lance(lance_uri, io_config=daft_io_config())
    rows = daft.sql(sql, images=images).limit(top_k).collect().to_pydict()
    return _rows_from_pydict(rows)


def _vector_query(
    lance_uri: str,
    q_vec: list[float],
    top_k: int = 10,
    where: str | None = None,
    distance_range: tuple[float, float] | None = None,
) -> list[dict]:
    import daft
    import pyarrow as pa

    nearest: dict = {
        "column": "image_embedding",
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


def text_query(
    lance_uri: str,
    text: str,
    top_k: int = 10,
    where: str | None = None,
    distance_range: tuple[float, float] | None = None,
) -> list[dict]:
    from multimodal_toolkit.image.embedding import get_embedder

    q_vec = get_embedder().embed_text(text)
    if q_vec is None:
        raise ValueError("text query is empty")
    return _vector_query(lance_uri, q_vec, top_k, where, distance_range)


def image_path_query(
    lance_uri: str,
    image_path: str,
    top_k: int = 10,
    where: str | None = None,
    distance_range: tuple[float, float] | None = None,
) -> list[dict]:
    from multimodal_toolkit.image.embedding import get_embedder

    with open(image_path, "rb") as fp:
        q_vec = get_embedder().embed_image_bytes(fp.read())
    if q_vec is None:
        raise ValueError(f"image query cannot be embedded: {image_path}")
    return _vector_query(lance_uri, q_vec, top_k, where, distance_range)


def image_doc_query(
    lance_uri: str,
    query_doc_id: str,
    top_k: int = 10,
    where: str | None = None,
    distance_range: tuple[float, float] | None = None,
) -> list[dict]:
    import daft

    query_rows = (
        daft.read_lance(
            lance_uri,
            io_config=daft_io_config(),
            default_scan_options={"filter": _doc_id_filter(query_doc_id)},
        )
        .select("image_embedding")
        .limit(1)
        .collect()
        .to_pydict()
    )
    if not query_rows.get("image_embedding") or query_rows["image_embedding"][0] is None:
        raise ValueError(f"query_doc_id not found or has null image_embedding: {query_doc_id}")
    return _vector_query(lance_uri, query_rows["image_embedding"][0], top_k, where, distance_range)


def run(
    lance_uri: str,
    where: str | None,
    sql: str | None,
    top_k: int,
    text: str | None = None,
    image_path: str | None = None,
    image_from: str | None = None,
    distance_range: tuple[float, float] | None = None,
) -> None:
    if sql:
        results = sql_query(lance_uri, sql, top_k)
    elif text:
        results = text_query(lance_uri, text, top_k, where, distance_range)
    elif image_path:
        results = image_path_query(lance_uri, image_path, top_k, where, distance_range)
    elif image_from:
        results = image_doc_query(lance_uri, image_from, top_k, where, distance_range)
    else:
        results = scalar_query(lance_uri, where, top_k)
    for row in results:
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lance-uri", required=True, help="lance image asset table URI (S3)")
    parser.add_argument("--where", help="SQL WHERE clause pushed down to Lance scanner")
    parser.add_argument("--sql", help="full Daft SQL SELECT (table name: images)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--text", help="text-to-image query")
    parser.add_argument("--image-path", help="local image file to use as a similarity query")
    parser.add_argument("--image-from", help="doc_id in the image table to use as a similarity query")
    parser.add_argument("--distance-min", type=float, help="minimum vector distance for ANN results")
    parser.add_argument("--distance-max", type=float, help="maximum vector distance for ANN results")
    args = parser.parse_args()
    vector_modes = [bool(args.text), bool(args.image_path), bool(args.image_from)]
    if sum(vector_modes) > 1:
        parser.error("provide only one of --text, --image-path, or --image-from")
    distance_range = None
    if args.distance_min is not None or args.distance_max is not None:
        if args.distance_min is None or args.distance_max is None:
            parser.error("--distance-min and --distance-max must be provided together")
        distance_range = (args.distance_min, args.distance_max)
    run(
        args.lance_uri,
        args.where,
        args.sql,
        args.top_k,
        args.text,
        args.image_path,
        args.image_from,
        distance_range,
    )


if __name__ == "__main__":
    main()
