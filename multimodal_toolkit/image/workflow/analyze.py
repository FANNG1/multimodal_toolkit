"""图片 Stage 1：manifest → 本地或 VLM 图片分析 → JSONL 或 Lance staging。

默认流程：逐张下载图片 → ImageQualityUDF 计算原始分数（SCRFD 人脸框、
Laplacian 方差清晰度）→ rules.add_rule_columns 产生布尔结论。

带 --use-llm 时切换为视觉大模型后端：图片解码并缩到固定长边后，通过
OpenAI-compatible Chat Completions 判断整图是否模糊、是否为真人单人头像。
该模式不会加载本地人脸检测模型，且必须写入单独的 Lance 资产表。

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
from daft import col, lit
from daft.functions import download, when

from .. import config
from ...storage.io import configure_daft_runner, daft_io_config, read_manifest
from ..rules import add_rule_columns
from ..udfs import ImageQualityUDF, prepare_image_for_vlm

LLM_ANALYSIS_DTYPE = daft.DataType.struct(
    {
        "is_blurry": daft.DataType.bool(),
        "is_avatar": daft.DataType.bool(),
        "clarity_confidence": daft.DataType.float64(),
        "avatar_confidence": daft.DataType.float64(),
        "reason": daft.DataType.string(),
    }
)

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

_LLM_OUTPUT_COLS = [
    "is_avatar",
    "clarity_confidence",
    "avatar_confidence",
    "llm_reason",
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


@daft.cls(cpus=1)
class _ImageVLMUDF:
    def __init__(self) -> None:
        from multimodal_toolkit.image.vlm import ImageVLMClient

        self._client = ImageVLMClient()

    @daft.method.batch(return_dtype=daft.DataType.string())
    def __call__(self, image_bytes_col):
        return [self._client.analyze(image_bytes) for image_bytes in image_bytes_col.to_pylist()]


@daft.cls(cpus=1)
class _ImageEmbedUDF:
    def __init__(self) -> None:
        from multimodal_toolkit.image.embedding import get_embedder

        self._embedder = get_embedder()

    @daft.method.batch(
        return_dtype=daft.DataType.fixed_size_list(daft.DataType.float32(), config.IMAGE_EMBED_DIM)
    )
    def __call__(self, image_bytes_col):
        # TODO: Batch CLIP inference here instead of calling one forward pass
        # per row; keep nulls aligned with failed rows when adding the batch API.
        return [
            self._embedder.embed_image_bytes(image_bytes) if image_bytes else None
            for image_bytes in image_bytes_col.to_pylist()
        ]


def _require_vlm_config() -> None:
    missing = []
    if not config.IMAGE_VLM_API_KEY:
        missing.append("IMAGE_VLM_API_KEY")
    if not config.IMAGE_VLM_MODEL:
        missing.append("IMAGE_VLM_MODEL")
    if not config.IMAGE_VLM_BASE_URL:
        missing.append("IMAGE_VLM_BASE_URL")
    if missing:
        raise ValueError("--use-llm requires: " + ", ".join(missing))


def _add_local_analysis(df: daft.DataFrame) -> daft.DataFrame:
    # 核心分析：一个 UDF 里同时算人脸和清晰度（图片只解码/缩放一次），
    # 结果是 struct 列，随后展开成顶层列方便过滤和落盘。
    iq_udf = ImageQualityUDF()
    df = df.with_column("iq", iq_udf(col("image_bytes")))
    for field in _SCORE_FIELDS:
        df = df.with_column(field, col("iq")[field])

    df = df.with_column(
        "status",
        when(col("image_bytes").is_null(), "download_failed")
        .when(col("blur_score").is_null(), "decode_failed")
        .otherwise("ok"),
    )
    return add_rule_columns(df)


def _add_llm_analysis(df: daft.DataFrame) -> daft.DataFrame:
    prep = prepare_image_for_vlm(col("image_bytes"))
    df = df.with_column("vlm_prep", prep)
    df = df.with_column("width", col("vlm_prep")["width"])
    df = df.with_column("height", col("vlm_prep")["height"])
    df = df.with_column("vlm_image_bytes", col("vlm_prep")["vlm_image_bytes"])

    vlm_udf = _ImageVLMUDF()
    df = df.with_column("llm_json", vlm_udf(col("vlm_image_bytes")))
    df = df.with_column("llm_analysis", col("llm_json").try_deserialize("json", LLM_ANALYSIS_DTYPE))

    clarity_confidence = col("llm_analysis")["clarity_confidence"]
    avatar_confidence = col("llm_analysis")["avatar_confidence"]
    valid = (
        ~col("llm_analysis")["is_blurry"].is_null()
        & ~col("llm_analysis")["is_avatar"].is_null()
        & ~clarity_confidence.is_null()
        & (clarity_confidence >= 0.0)
        & (clarity_confidence <= 1.0)
        & ~avatar_confidence.is_null()
        & (avatar_confidence >= 0.0)
        & (avatar_confidence <= 1.0)
        & ~col("llm_analysis")["reason"].is_null()
    )
    df = df.with_column(
        "status",
        when(col("image_bytes").is_null(), "download_failed")
        .when(col("vlm_image_bytes").is_null(), "decode_failed")
        .when(~valid.fill_null(False), "llm_failed")
        .otherwise("ok"),
    )
    ok = col("status") == "ok"
    df = df.with_column("is_blurry", when(ok, col("llm_analysis")["is_blurry"]))
    df = df.with_column("is_avatar", when(ok, col("llm_analysis")["is_avatar"]))
    df = df.with_column("clarity_confidence", when(ok, clarity_confidence))
    df = df.with_column("avatar_confidence", when(ok, avatar_confidence))
    df = df.with_column("llm_reason", when(ok, col("llm_analysis")["reason"]))

    # These fields belong to the local detector/threshold backend. Keep their
    # canonical types in LLM JSON output without inventing incompatible scores.
    null_types = {
        "face_count": daft.DataType.int64(),
        "face_score": daft.DataType.float64(),
        "face_area_ratio": daft.DataType.float64(),
        "blur_score": daft.DataType.float64(),
        "face_blur_score": daft.DataType.float64(),
        "has_face": daft.DataType.bool(),
        "is_face_blurry": daft.DataType.bool(),
    }
    for name, dtype in null_types.items():
        df = df.with_column(name, lit(None).cast(dtype))
    return df


def run(manifest: str, out_path: str, embed: bool = False, use_llm: bool = False) -> None:
    configure_daft_runner()
    io_config = daft_io_config()

    if use_llm:
        _require_vlm_config()

    low_out = out_path.rstrip("/").lower()
    if embed and (low_out.endswith(".json") or low_out.endswith(".jsonl") or low_out.endswith(".ndjson")):
        raise ValueError("--embed writes a Lance staging table; use a .lance output URI")
    if not embed and low_out.endswith(".lance"):
        raise ValueError(".lance output requires --embed; use a JSONL output URI without --embed")

    # manifest 只有 doc_id + s3_url 两列；按 s3_url 下载图片字节，
    # 失败的行 image_bytes 为 null（on_error="null"），保留不丢。
    df = read_manifest(manifest)
    df = df.with_column(
        "image_bytes", download(col("s3_url"), on_error="null", io_config=io_config)
    )

    # TODO: Avoid decoding images a second time in the embedding UDF when
    # --embed is enabled, either by sharing decoded arrays or by fusing UDFs.
    df = _add_llm_analysis(df) if use_llm else _add_local_analysis(df)

    output_cols = [*_OUTPUT_COLS, *(_LLM_OUTPUT_COLS if use_llm else [])]

    if embed:
        embed_udf = _ImageEmbedUDF()
        df = df.with_column("image_embedding", embed_udf(col("image_bytes")))
        output = df.select(*output_cols, "image_embedding")
        output.write_lance(out_path, mode="overwrite", io_config=io_config)
        print(f"[ok] wrote image analysis+embedding lance staging table: {out_path}")
    else:
        output = df.select(*output_cols)
        output.write_json(out_path, write_mode="overwrite", io_config=io_config)
        print(f"[ok] wrote image analysis JSONL: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="parquet/jsonl/csv manifest with doc_id, s3_url")
    parser.add_argument("--out", required=True, help="S3 output .jsonl path or .lance URI when --embed")
    parser.add_argument("--embed", action="store_true", help="compute image_embedding (output becomes lance table)")
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="use an OpenAI-compatible vision model instead of local face/blur analysis",
    )
    args = parser.parse_args()
    run(args.manifest, args.out, embed=args.embed, use_llm=args.use_llm)


if __name__ == "__main__":
    main()
