"""Stage 2: analysis output → lance asset table (append, blob v2, with ingest_time).

Reads all fields from Stage 1 output (JSONL or lance staging table).
The input must include s3_url so audio blobs can be downloaded.
Appends to the lance asset table — never overwrites.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import daft
from daft import col, lit
from daft.functions import download, when

from ... import config
from ...storage.blob import validate_blob_v2
from ...storage.io import configure_daft_runner, daft_io_config, lance_write_mode, read_analysis_output

# JSONL 分析结果靠类型推断读入：某列在整个批次里全为 null 时（比如整批
# 没配 DEEPSEEK_API_KEY 导致打标列全空，或全批次 download_failed），推断
# 出来的是 null 类型。直接落 Lance 会建出 null 类型的列，后续正常批次
# append 时类型冲突、永远写不进去。这里记录每列的规范类型（即正常批次
# JSON 推断会得到的类型），把全 null 的列显式 cast 回去。
_ANALYSIS_COLUMN_DTYPES = {
    "doc_id": daft.DataType.string(),
    "s3_url": daft.DataType.string(),
    "status": daft.DataType.string(),
    "duration_s": daft.DataType.float64(),
    "transcript": daft.DataType.string(),
    "acoustic_emotion": daft.DataType.string(),
    "downgrade_related": daft.DataType.bool(),
    "primary_reason": daft.DataType.string(),
    "secondary_reason": daft.DataType.string(),
    "summary": daft.DataType.string(),
    "confidence": daft.DataType.float64(),
    "text_emotion": daft.DataType.string(),
    "bad_tone": daft.DataType.bool(),
    "emotion_score": daft.DataType.float64(),
}


def _cast_all_null_columns(df: daft.DataFrame) -> daft.DataFrame:
    for field in df.schema():
        dtype = _ANALYSIS_COLUMN_DTYPES.get(field.name)
        if dtype is not None and field.dtype == daft.DataType.null():
            df = df.with_column(field.name, col(field.name).cast(dtype))
    return df


def run(analysis_path: str, lance_uri: str) -> None:
    configure_daft_runner()
    io_config = daft_io_config()

    now = datetime.now(timezone.utc)

    df = read_analysis_output(analysis_path, io_config)
    df = _cast_all_null_columns(df)

    # 重新按 s3_url 下载音频字节作为 blob 列（Stage 1 输出不带原始字节）。
    # 下载失败的行（含 Stage 1 标记为 download_failed 的）blob 为 null，
    # 但行本身保留——Lance 表是完整台账，status 列记录了失败原因。
    df = df.with_column(
        "audio_blob", download(col("s3_url"), on_error="null", io_config=io_config)
    )
    # Stage 1 到 Stage 2 之间对象可能被删或失效：Stage 1 判为 ok 的行这次
    # 下载不到字节时，改标 blob_download_failed，不能让台账里出现
    # "status=ok 但没有 blob"的行。转写和打标列保持 Stage 1 的值——
    # 分析本身是成功的，缺的只是归档字节。
    df = df.with_column(
        "status",
        when(
            (col("status") == "ok") & col("audio_blob").is_null(),
            "blob_download_failed",
        ).otherwise(col("status")),
    )
    df = df.with_column(
        "ingest_time",
        lit(now).cast(daft.DataType.timestamp("us", "UTC")),
    )

    mode = lance_write_mode(lance_uri)
    df.write_lance(
        lance_uri,
        mode=mode,
        io_config=io_config,
        blob_columns=["audio_blob"],
        max_rows_per_file=config.LANCE_MAX_ROWS_PER_FILE,
        max_bytes_per_file=config.LANCE_MAX_BYTES_PER_FILE,
    )
    validate_blob_v2(lance_uri, "audio_blob")

    result = daft.read_lance(lance_uri, io_config=io_config)
    print(f"[ok] appended to lance asset table: {lance_uri}")
    print(f"[ok] total rows: {result.count_rows()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis", required=True,
        help="Stage 1 output: S3 JSONL path or lance staging table URI",
    )
    parser.add_argument("--lance-uri", required=True, help="lance asset table URI (S3)")
    args = parser.parse_args()
    run(args.analysis, args.lance_uri)


if __name__ == "__main__":
    main()
