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

The output always includes s3_url so Stage 2 (ingest.py) can download the audio blob.
"""
from __future__ import annotations

import argparse

import daft
from daft import col
from daft.functions import download, regexp_replace
from daft.functions.ai import prompt as llm_prompt

from .. import config
from ..udfs import (
    ANALYSIS_DTYPE,
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
    df = df.where(~col("audio_bytes").is_null())

    # Duration gate: drop clips too short to analyse or too long to afford.
    df = df.with_column("duration_s", duration_udf(col("audio_bytes")))
    df = df.where(
        (col("duration_s") >= config.MIN_DURATION_S)
        & (col("duration_s") <= config.MAX_DURATION_S)
    )

    # ASR = automatic speech recognition. SenseVoice returns the transcript
    # plus an acoustic emotion label (angry/sad/... inferred from tone, not words).
    asr = AsrUDF()
    df = df.with_column("asr", asr(col("audio_bytes"), col("doc_id")))
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
    # those columns then fall back to the fill_null defaults below.
    df = df.with_column("prompt", prompt_udf(col("transcript"), col("acoustic_emotion")))
    if config.DEEPSEEK_API_KEY:
        from daft.ai.openai.provider import OpenAIProvider

        provider = OpenAIProvider(
            base_url=config.DEEPSEEK_BASE_URL,
            api_key=config.DEEPSEEK_API_KEY,
        )
        df = df.with_column(
            "analysis_json",
            llm_prompt(
                col("prompt"),
                provider=provider,
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

    # Parse the LLM's JSON into typed columns. try_deserialize yields null on
    # malformed JSON, and fill_null supplies a safe default per field, so one
    # bad LLM response never fails the batch.
    df = df.with_column("analysis", col("analysis_json").try_deserialize("json", ANALYSIS_DTYPE))
    df = (
        df.with_column("downgrade_related", col("analysis")["downgrade_related"].fill_null(False))
        .with_column("primary_reason", col("analysis")["primary_reason"].fill_null("其他"))
        .with_column("secondary_reason", col("analysis")["secondary_reason"].fill_null(""))
        .with_column("summary", col("analysis")["summary"].fill_null(""))
        .with_column("confidence", col("analysis")["confidence"].fill_null(0.0))
        .with_column("text_emotion", col("analysis")["text_emotion"].fill_null("未知"))
        .with_column("bad_tone", col("analysis")["bad_tone"].fill_null(False))
        .with_column("emotion_score", col("analysis")["emotion_score"].fill_null(0.0))
    )
    return df


def run(manifest: str, out_path: str, embed: bool = False) -> None:
    configure_daft_runner()
    io_config = daft_io_config()

    df = _build_analysis_df(manifest, io_config)

    if embed:
        embed_udf = _EmbedUDF()
        df = df.with_column("audio_embedding", embed_udf(col("audio_bytes")))
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
