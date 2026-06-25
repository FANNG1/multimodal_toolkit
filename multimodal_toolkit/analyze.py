from __future__ import annotations

import argparse

import daft
import daft_lance
import lance
from daft import col
from daft.functions import llm_generate, regexp_replace

from . import config
from .blob import append_columns_by_doc_id, validate_blob_v2
from .io import daft_io_config, lance_storage_options

# Rust regex (no look-around): match ID card before phone to avoid partial overlap
_ID_CARD_PAT = r"\d{17}[\dXx]"
_PHONE_PAT = r"1[3-9]\d{9}"

_ANALYSIS_DTYPE = daft.DataType.struct(
    {
        "downgrade_related": daft.DataType.bool_(),
        "primary_reason": daft.DataType.string(),
        "secondary_reason": daft.DataType.string(),
        "summary": daft.DataType.string(),
        "confidence": daft.DataType.float64(),
        "text_emotion": daft.DataType.string(),
        "bad_tone": daft.DataType.bool_(),
        "emotion_score": daft.DataType.float64(),
    }
)

_OUTPUT_COLS = [
    "doc_id",
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


@daft.udf(return_dtype=daft.DataType.float64())
def _duration_udf(audio_blobs):
    import io as _io

    import soundfile as sf

    results = []
    for blob in audio_blobs.to_pylist():
        if blob is None:
            results.append(0.0)
            continue
        try:
            info = sf.info(_io.BytesIO(blob.read()))
            results.append(float(info.frames) / info.samplerate if info.samplerate else 0.0)
        except Exception:
            results.append(0.0)
    return results


@daft.udf(return_dtype=daft.DataType.string())
def _prompt_udf(transcripts, acoustic_emotions):
    from .prompt import build_prompt

    return [
        build_prompt(t or "", e or "NEUTRAL")
        for t, e in zip(transcripts.to_pylist(), acoustic_emotions.to_pylist())
    ]


@daft.cls(cpus=1)
class _AsrUDF:
    def __init__(self) -> None:
        from .asr import SenseVoiceASR

        self._asr = SenseVoiceASR()

    @daft.method.batch(
        return_dtype=daft.DataType.struct(
            {
                "transcript": daft.DataType.string(),
                "acoustic_emotion": daft.DataType.string(),
            }
        )
    )
    def __call__(self, audio_blobs, s3_urls):
        from pathlib import Path

        results = []
        for blob, url in zip(audio_blobs.to_pylist(), s3_urls.to_pylist()):
            suffix = Path(url).suffix if url else ".wav"
            if not suffix:
                suffix = ".wav"
            audio_bytes = blob.read() if blob is not None else None
            results.append(self._asr.transcribe_bytes(audio_bytes, suffix))
        return results


def run(lance_uri: str, out_jsonl: str) -> None:
    io_config = daft_io_config()

    validate_blob_v2(lance_uri, "audio_blob")

    ds = lance.dataset(lance_uri, storage_options=lance_storage_options(lance_uri))
    df = daft.read_lance(lance_uri, io_config=io_config, default_scan_options={"with_row_id": True})
    df = df.select("doc_id", "s3_url", "audio_blob", "_rowid")
    df = daft_lance.take_blobs(df, ds, "audio_blob")
    df = df.where(~col("audio_blob").is_null())

    # Duration quality gate
    df = df.with_column("duration_s", _duration_udf(col("audio_blob")))
    df = df.where((col("duration_s") >= config.MIN_DURATION_S) & (col("duration_s") <= config.MAX_DURATION_S))

    # ASR (stateful: model loads once per worker)
    asr = _AsrUDF()
    df = df.with_column("asr", asr(col("audio_blob"), col("s3_url")))
    df = df.with_column("transcript_raw", col("asr")["transcript"])
    df = df.with_column("acoustic_emotion", col("asr")["acoustic_emotion"])

    # PII desensitization (ID card before phone to avoid partial digit overlap)
    df = df.with_column("transcript", regexp_replace(col("transcript_raw"), _ID_CARD_PAT, "[ID_REDACTED]"))
    df = df.with_column("transcript", regexp_replace(col("transcript"), _PHONE_PAT, "[PHONE_REDACTED]"))

    # LLM analysis
    df = df.with_column("prompt", _prompt_udf(col("transcript"), col("acoustic_emotion")))
    if config.DEEPSEEK_API_KEY:
        df = df.with_column(
            "analysis_json",
            llm_generate(
                col("prompt"),
                model=config.DEEPSEEK_MODEL,
                provider="openai",
                base_url=config.DEEPSEEK_BASE_URL,
                api_key=config.DEEPSEEK_API_KEY,
                response_format={"type": "json_object"},
                temperature=0,
            ),
        )
    else:
        df = df.with_column("analysis_json", daft.lit(None))

    # Parse JSON → struct → unnest; try_deserialize returns null on failure
    df = df.with_column("analysis", col("analysis_json").try_deserialize("json", _ANALYSIS_DTYPE))
    df = df.unnest("analysis")

    # Fill nulls for rows where LLM returned unparseable JSON or no API key
    df = (
        df.with_column("downgrade_related", col("downgrade_related").fill_null(False))
        .with_column("primary_reason", col("primary_reason").fill_null("其他"))
        .with_column("secondary_reason", col("secondary_reason").fill_null(""))
        .with_column("summary", col("summary").fill_null(""))
        .with_column("confidence", col("confidence").fill_null(0.0))
        .with_column("text_emotion", col("text_emotion").fill_null("未知"))
        .with_column("bad_tone", col("bad_tone").fill_null(False))
        .with_column("emotion_score", col("emotion_score").fill_null(0.0))
    )

    output = df.select(*_OUTPUT_COLS)

    output.write_json(out_jsonl, write_mode="overwrite", io_config=io_config)

    table = output.collect().to_arrow()
    append_columns_by_doc_id(lance_uri, table)

    print(f"[ok] wrote analysis to: {out_jsonl}")
    print(f"[ok] appended analysis columns to: {lance_uri}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lance-uri", required=True)
    parser.add_argument("--out-jsonl", required=True)
    args = parser.parse_args()
    run(args.lance_uri, args.out_jsonl)


if __name__ == "__main__":
    main()
