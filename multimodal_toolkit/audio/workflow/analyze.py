"""Stage 1: manifest → analysis tags (+ optional embedding) → S3 output.

Flow: download each recording → duration gate → ASR (speech-to-text via
SenseVoice, which also labels the speaker's acoustic emotion from tone of
voice) → redact PII from the transcript → DeepSeek LLM reads the transcript
and tags business fields (downgrade intent, complaint reason, emotion, ...)
as JSON → optionally append an acoustic embedding (a fixed-length vector
summarizing how the audio *sounds*, used later for similarity search).

Output format depends on --embed flag:
  default (no embed)  →  JSONL on S3  (scalar fields, embeddable in downstream JSON pipelines)
  --embed             →  Lance staging table on S3  (fixed-size vector cannot be stored in JSON)

manifest 里的每个条目都对应输出中的一行——下载失败、解码失败、被时长过滤、
LLM 打标失败的通话不会被丢弃，而是通过 status 列标记（ok / download_failed /
decode_failed / duration_filtered / llm_failed），转写和打标结论为 null。
合规场景必须能区分"通话有问题"和"根本没处理"。

The output always includes s3_url so Stage 2 (ingest.py) can download the audio blob.
"""
from __future__ import annotations

import argparse

import daft
from daft import col
from daft.functions import download, regexp_replace, when
from daft.functions.ai import prompt as llm_prompt

from .. import config
from ..llm import validate_llm_responses
from ..udfs import (
    AsrUDF,
    duration_udf,
    prompt_udf,
)
from ...storage.io import (
    coalesce_for_write,
    configure_daft_runner,
    daft_io_config,
    read_manifest,
    spread_partitions,
)
from ... import config as shared_config

# PII patterns (Chinese mainland): 18-digit resident ID and 11-digit mobile
# number. Redacted from transcripts before they are sent to the LLM or stored.
_ID_CARD_PAT = r"\d{17}[\dXx]"
_PHONE_PAT = r"1[3-9]\d{9}"

_BASE_OUTPUT_COLS = [
    "doc_id",
    "s3_url",
    "status",
    "duration_s",
    "transcript",
    "acoustic_emotion",
    "downgrade_related",
    "primary_reason",
    "secondary_reason",
    "summary",
    "confidence",
    "text_emotion",
    "bad_tone",
    "emotion_score",
]


@daft.cls(cpus=1, max_retries=2)
class _EmbedUDF:
    """Turns raw audio into a fixed-length float vector (the "embedding").

    Recordings that sound alike get vectors that are close together, which is
    what makes nearest-neighbour search in Stage 4 possible. The backend
    (config.EMBED_BACKEND) is either cheap signal statistics or wav2vec2.
    """

    def __init__(self) -> None:
        # Loaded once per worker process, not per row (same pattern as AsrUDF).
        from multimodal_toolkit.audio.embedding import get_embedder

        self._embedder = get_embedder()

    @daft.method.batch(
        return_dtype=daft.DataType.fixed_size_list(daft.DataType.float32(), config.EMBED_DIM)
    )
    def __call__(self, audio_bytes_col):
        return [
            self._embedder.embed_bytes(b) if b else None
            for b in audio_bytes_col.to_pylist()
        ]


def _build_analysis_df(manifest: str, io_config) -> daft.DataFrame:
    """Download audio from S3 and run the full analysis pipeline."""
    # 下载前先切分区：manifest 文件太小，不切的话 Ray 上整条链路只有一个 task。
    df = read_manifest(manifest)
    df = spread_partitions(df)
    df = df.with_column(
        "audio_bytes", download(col("s3_url"), on_error="null", io_config=io_config)
    )

    # manifest 里的每个条目都对应输出中的一行——下载失败、解码失败、被时长
    # 过滤的音频不会被丢弃，而是通过 status 列标记，转写和打标结论为 null。
    # 合规场景必须能区分"通话有问题"和"根本没处理"。duration_udf 对坏字节
    # 返回 null，所以 duration 为 null 即 decode_failed；时长超出
    # [MIN_DURATION_S, MAX_DURATION_S] 的行标 duration_filtered（太短没有
    # 分析价值，太长成本不可控），duration_s 本身保留。
    df = df.with_column("duration_s", duration_udf(col("audio_bytes")))
    df = df.with_column(
        "status",
        when(col("audio_bytes").is_null(), "download_failed")
        .when(col("duration_s").is_null(), "decode_failed")
        .when(
            (col("duration_s") < config.MIN_DURATION_S)
            | (col("duration_s") > config.MAX_DURATION_S),
            "duration_filtered",
        )
        .otherwise("ok"),
    )

    # ASR = automatic speech recognition. SenseVoice returns the transcript
    # plus an acoustic emotion label (angry/sad/... inferred from tone, not words).
    # 非 ok 行传 null 字节进 UDF，ASR 返回 null struct，昂贵的模型推理只花在
    # 真正要分析的行上。
    asr = AsrUDF()
    df = df.with_column(
        "asr", asr(when(col("status") == "ok", col("audio_bytes")), col("doc_id"))
    )
    df = df.with_column("transcript_raw", col("asr")["transcript"])
    df = df.with_column("acoustic_emotion", col("asr")["acoustic_emotion"])

    # Redact PII before the transcript leaves this stage. ID card first: a
    # phone-number match inside an ID number would otherwise break it apart.
    df = df.with_column(
        "transcript",
        regexp_replace(col("transcript_raw"), _ID_CARD_PAT, "[ID_REDACTED]"),
    )
    df = df.with_column(
        "transcript",
        regexp_replace(col("transcript"), _PHONE_PAT, "[PHONE_REDACTED]"),
    )

    # LLM tagging: build the instruction (audio/prompt.py), ask DeepSeek for a
    # JSON object with the business fields. Skipped when no API key is set —
    # those columns then stay null（"没打标"，不是"判定为否"）。
    # 非 ok 行 transcript 为 null，prompt_udf 返回 null，LLM 不会收到请求。
    df = df.with_column("prompt", prompt_udf(col("transcript"), col("acoustic_emotion")))
    if config.DEEPSEEK_API_KEY:
        # 不直接用 OpenAIProvider：Daft 0.7.15 会把 on_error/concurrency 泄漏进
        # OpenAI 请求 kwargs 导致每行 TypeError，适配层见 audio/llm.py。
        from ..llm import get_audio_llm_provider

        df = df.with_column(
            "analysis_json",
            llm_prompt(
                col("prompt"),
                provider=get_audio_llm_provider(),
                model=config.DEEPSEEK_MODEL,
                use_chat_completions=True,
                response_format={"type": "json_object"},
                temperature=0,
                concurrency=config.DEEPSEEK_CONCURRENCY,
                on_error="ignore",
            ),
        )
    else:
        df = df.with_column("analysis_json", daft.lit(None).cast(daft.DataType.string()))

    # 在 Python batch UDF 里严格校验 LLM JSON，再生成 typed struct。不能用
    # try_deserialize：Daft 0.7.15 遇到合法 JSON 中的类型错误会抛批次级异常，
    # 而缺字段对象会生成非 null struct，两个行为都会破坏 llm_failed 语义。
    # 结论列不做 fill_null：null 就是"LLM 没给出结论"（没配 key、请求失败、
    # JSON 不合法），填 False/'其他'/0.0 会把"没处理"伪装成"判定为否"，
    # 真实降档通话会被静默漏报。
    df = df.with_column("analysis", validate_llm_responses(col("analysis_json")))
    df = (
        df.with_column("downgrade_related", col("analysis")["downgrade_related"])
        .with_column("primary_reason", col("analysis")["primary_reason"])
        .with_column("secondary_reason", col("analysis")["secondary_reason"])
        .with_column("summary", col("analysis")["summary"])
        .with_column("confidence", col("analysis")["confidence"])
        .with_column("text_emotion", col("analysis")["text_emotion"])
        .with_column("bad_tone", col("analysis")["bad_tone"])
        .with_column("emotion_score", col("analysis")["emotion_score"])
    )

    # 配了 key 却没有得到可解析结果的 ok 行标 llm_failed——区别于"没配 key
    # 整批跳过"（status 保持 ok、结论列全 null）。只升级 ok 行：download/
    # decode/duration 状态是更早的失败原因，不能被覆盖。
    if config.DEEPSEEK_API_KEY:
        df = df.with_column(
            "status",
            when(
                (col("status") == "ok")
                & col("analysis")["downgrade_related"].is_null(),
                "llm_failed",
            ).otherwise(col("status")),
        )
    return df


def _embedding_input(audio_bytes, status):
    """只按音频本身是否可分析决定 embedding 输入。

    llm_failed 表示音频下载、解码和时长门控均已通过，只是外部 LLM 没有给出
    合法结论；声学 embedding 与 LLM 独立，必须继续计算，避免可用音频从 ANN
    索引中静默消失。
    """
    media_ready = (status == "ok") | (status == "llm_failed")
    return when(media_ready, audio_bytes)


def run(manifest: str, out_path: str, embed: bool = False) -> None:
    configure_daft_runner()
    io_config = daft_io_config()

    df = _build_analysis_df(manifest, io_config)

    if embed:
        embed_udf = _EmbedUDF()
        # 下载/解码失败和时长过滤行不做 embedding；llm_failed 的音频本身
        # 已通过媒体门控，声学 embedding 不依赖 LLM，仍应正常计算。
        df = df.with_column(
            "audio_embedding",
            embed_udf(_embedding_input(col("audio_bytes"), col("status"))),
        )
        output = df.select(*_BASE_OUTPUT_COLS, "audio_embedding")
        output.write_lance(
            out_path,
            mode="overwrite",
            io_config=io_config,
            max_rows_per_file=shared_config.LANCE_MAX_ROWS_PER_FILE,
            max_bytes_per_file=shared_config.LANCE_MAX_BYTES_PER_FILE,
        )
        print(f"[ok] wrote analysis+embedding lance staging table: {out_path}")
    else:
        output = coalesce_for_write(df.select(*_BASE_OUTPUT_COLS))
        output.write_json(out_path, write_mode="overwrite", io_config=io_config)
        print(f"[ok] wrote analysis JSONL: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="parquet/jsonl/csv manifest with doc_id, s3_url")
    parser.add_argument("--out", required=True, help="S3 output: .jsonl path (no embed) or .lance URI (embed)")
    parser.add_argument("--embed", action="store_true", help="compute audio_embedding (output becomes lance table)")
    args = parser.parse_args()
    run(args.manifest, args.out, embed=args.embed)


if __name__ == "__main__":
    main()
