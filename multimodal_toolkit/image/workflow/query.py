"""图片 Stage 4：查询 lance 图片资产表。

  --where  标量过滤，经 Daft 下推到 Lance scanner
  --sql    完整的 Daft SQL SELECT 语句（优先于 --where；表名：images）

v1 不支持向量检索（图片表没有 embedding 列）。
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


def run(lance_uri: str, where: str | None, sql: str | None, top_k: int) -> None:
    if sql:
        results = sql_query(lance_uri, sql, top_k)
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
    args = parser.parse_args()
    run(args.lance_uri, args.where, args.sql, args.top_k)


if __name__ == "__main__":
    main()
