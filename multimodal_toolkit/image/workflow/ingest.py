"""图片 Stage 2：分析结果 → lance 图片资产表（追加写入，blob v2，带 ingest_time）。

读取图片 Stage 1 的 JSONL 输出（必须包含 s3_url，用于下载图片 blob），
把图片字节和分析元数据一起追加进 lance 资产表——只追加，不覆盖。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import daft
from daft import col, lit
from daft.functions import download

from ...storage.blob import validate_blob_v2
from ...storage.io import (
    configure_daft_runner,
    daft_io_config,
    lance_storage_options,
    lance_write_mode,
    read_analysis_output,
)


def _validate_embedding_schema(lance_uri: str, mode: str, analysis_has_embedding: bool) -> None:
    if mode != "append":
        return

    import lance

    ds = lance.dataset(lance_uri, storage_options=lance_storage_options(lance_uri))
    table_has_embedding = "image_embedding" in ds.schema.names
    if table_has_embedding == analysis_has_embedding:
        return

    existing = "with" if table_has_embedding else "without"
    incoming = "with" if analysis_has_embedding else "without"
    raise ValueError(
        "Cannot append image analysis "
        f"{incoming} image_embedding to an existing table {existing} image_embedding. "
        "Use a separate Lance table, or keep all batches consistent about Stage 1 --embed."
    )


def run(analysis_path: str, lance_uri: str) -> None:
    configure_daft_runner()
    io_config = daft_io_config()

    now = datetime.now(timezone.utc)

    df = read_analysis_output(analysis_path, io_config)
    analysis_has_embedding = "image_embedding" in df.schema().column_names()
    mode = lance_write_mode(lance_uri)
    _validate_embedding_schema(lance_uri, mode, analysis_has_embedding)

    # 重新按 s3_url 下载图片字节作为 blob 列。Stage 1 的 JSONL 里不带
    # 原始字节（JSON 存不了大二进制），所以这里是第二次下载。
    # 下载失败的行（含 Stage 1 标记为 download_failed 的）blob 为 null，
    # 但行本身保留——Lance 表是完整台账，status 列记录了失败原因；
    # 解码失败的坏图字节也照常入库留证。
    df = df.with_column(
        "image_blob", download(col("s3_url"), on_error="null", io_config=io_config)
    )
    df = df.with_column(
        "ingest_time",
        lit(now).cast(daft.DataType.timestamp("us", "UTC")),
    )

    df.write_lance(lance_uri, mode=mode, io_config=io_config, blob_columns=["image_blob"])
    # 校验 image_blob 确实以 lance blob v2 编码落盘（而不是被静默降级成
    # 普通 large_binary），库版本升级时这是最容易出问题的地方。
    validate_blob_v2(lance_uri, "image_blob")

    result = daft.read_lance(lance_uri, io_config=io_config)
    print(f"[ok] appended to lance image asset table: {lance_uri}")
    print(f"[ok] total rows: {result.count_rows()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, help="image Stage 1 output: S3 JSONL path")
    parser.add_argument("--lance-uri", required=True, help="lance image asset table URI (S3)")
    args = parser.parse_args()
    run(args.analysis, args.lance_uri)


if __name__ == "__main__":
    main()
