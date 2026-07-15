"""图片 Stage 1：manifest → 人脸存在 + 清晰度分数 + 规则结论 → JSONL 或 Lance staging。

流程：逐张下载图片 → ImageQualityUDF 计算原始分数（SCRFD 人脸框、
Laplacian 方差清晰度——各字段含义见 image/udfs.py）→
rules.add_rule_columns 按 config 阈值把分数转成布尔结论
（has_face / is_blurry / is_face_blurry）。

manifest 里的每个条目都对应输出中的一行——下载失败、解码失败的图片
不会被丢弃，而是通过 status 列标记（ok / download_failed / decode_failed），
分数和结论为 null。合规场景必须能区分"图有问题"和"根本没处理"，
且"图片打不开"本身往往就是需要上报的结论。

输出始终带 s3_url，Stage 2（图片 ingest）靠它下载图片 blob。默认写 JSONL；
带 --embed 时写 Lance staging 表，因为 JSON 不适合承载 fixed-size-list 向量列。
"""
from __future__ import annotations

import argparse

import daft
from daft import col
from daft.functions import download, when

from .. import config
from ...storage.io import (
    coalesce_for_write,
    configure_daft_runner,
    daft_io_config,
    read_manifest,
    spread_partitions,
)
from ... import config as shared_config
from ..rules import add_rule_columns
from ..udfs import ImageQualityUDF

# Stage 1 落盘的全部列：标识列（doc_id/s3_url）+ 处理状态 + 原始分数 +
# 布尔结论。分数和结论都保留，后续调阈值只需重算结论，不用重跑模型。
_OUTPUT_COLS = [
    "doc_id",
    "s3_url",
    "status",
    "width",
    "height",
    "face_count",
    "face_score",
    "face_area_ratio",
    "blur_score",
    "face_blur_score",
    "has_face",
    "is_blurry",
    "is_face_blurry",
]

# ImageQualityUDF 返回的 struct 里需要展开为顶层列的字段。
_SCORE_FIELDS = [
    "width",
    "height",
    "face_count",
    "face_score",
    "face_area_ratio",
    "blur_score",
    "face_blur_score",
]


@daft.cls(cpus=1, max_retries=2)
class _ImageEmbedUDF:
    def __init__(self) -> None:
        from multimodal_toolkit.image.embedding import get_embedder

        self._embedder = get_embedder()

    @daft.method.batch(
        return_dtype=daft.DataType.fixed_size_list(daft.DataType.float32(), config.IMAGE_EMBED_DIM)
    )
    def __call__(self, image_bytes_col, status_col):
        # TODO: Batch CLIP inference here instead of calling one forward pass
        # per row; keep nulls aligned with failed rows when adding the batch API.
        return [
            self._embedder.embed_image_bytes(image_bytes) if status == "ok" and image_bytes else None
            for image_bytes, status in zip(image_bytes_col.to_pylist(), status_col.to_pylist())
        ]


def run(manifest: str, out_path: str, embed: bool = False) -> None:
    configure_daft_runner()
    io_config = daft_io_config()

    low_out = out_path.rstrip("/").lower()
    if embed and (low_out.endswith(".json") or low_out.endswith(".jsonl") or low_out.endswith(".ndjson")):
        raise ValueError("--embed writes a Lance staging table; use a .lance output URI")
    if not embed and low_out.endswith(".lance"):
        raise ValueError(".lance output requires --embed; use a JSONL output URI without --embed")

    # manifest 只有 doc_id + s3_url 两列；按 s3_url 下载图片字节，
    # 失败的行 image_bytes 为 null（on_error="null"），保留不丢。
    # 下载前先切分区：manifest 文件太小，不切的话 Ray 上整条链路只有一个 task。
    df = read_manifest(manifest)
    df = spread_partitions(df)
    df = df.with_column(
        "image_bytes", download(col("s3_url"), on_error="null", io_config=io_config)
    )

    # 核心分析：一个 UDF 里同时算人脸和清晰度（图片只解码/缩放一次），
    # 结果是 struct 列，随后展开成顶层列方便过滤和落盘。
    # 输入为 null 或解码失败时 UDF 返回全 null 行。
    # TODO: Avoid decoding images a second time in the embedding UDF when
    # --embed is enabled, either by sharing decoded arrays or by fusing UDFs.
    iq_udf = ImageQualityUDF()
    df = df.with_column("iq", iq_udf(col("image_bytes")))
    for field in _SCORE_FIELDS:
        df = df.with_column(field, col("iq")[field])

    # 处理状态：没下到字节 → download_failed；下到了但 UDF 全 null
    # （blur_score 为 null 只有解码失败一种可能）→ decode_failed。
    df = df.with_column(
        "status",
        when(col("image_bytes").is_null(), "download_failed")
        .when(col("blur_score").is_null(), "decode_failed")
        .otherwise("ok"),
    )

    # 分数 → 布尔结论（阈值来自 config，可用环境变量调整）；
    # 非 ok 行的结论为 null。
    df = add_rule_columns(df)

    if embed:
        embed_udf = _ImageEmbedUDF()
        df = df.with_column("image_embedding", embed_udf(col("image_bytes"), col("status")))
        output = df.select(*_OUTPUT_COLS, "image_embedding")
        output.write_lance(
            out_path,
            mode="overwrite",
            io_config=io_config,
            max_rows_per_file=shared_config.LANCE_MAX_ROWS_PER_FILE,
            max_bytes_per_file=shared_config.LANCE_MAX_BYTES_PER_FILE,
        )
        print(f"[ok] wrote image analysis+embedding lance staging table: {out_path}")
    else:
        output = coalesce_for_write(df.select(*_OUTPUT_COLS))
        output.write_json(out_path, write_mode="overwrite", io_config=io_config)
        print(f"[ok] wrote image analysis JSONL: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="parquet/jsonl/csv manifest with doc_id, s3_url")
    parser.add_argument("--out", required=True, help="S3 output .jsonl path or .lance URI when --embed")
    parser.add_argument("--embed", action="store_true", help="compute image_embedding (output becomes lance table)")
    args = parser.parse_args()
    run(args.manifest, args.out, embed=args.embed)


if __name__ == "__main__":
    main()
