from __future__ import annotations

import argparse

import daft
import daft_lance
import lance
from daft import col
from daft.functions import regexp_replace
from daft.functions.ai import prompt as llm_prompt

from .. import config
from ..storage.blob import validate_blob_v2
from ..storage.io import configure_daft_runner, daft_io_config, lance_storage_options

# Rust regex (no look-around): match ID card before phone to avoid partial overlap
_ID_CARD_PAT = r"\d{17}[\dXx]"
_PHONE_PAT = r"1[3-9]\d{9}"

_ANALYSIS_DTYPE = daft.DataType.struct(
    {
        "downgrade_related": daft.DataType.bool(),
        "primary_reason": daft.DataType.string(),
        "secondary_reason": daft.DataType.string(),
        "summary": daft.DataType.string(),
        "confidence": daft.DataType.float64(),
        "text_emotion": daft.DataType.string(),
        "bad_tone": daft.DataType.bool(),
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


@daft.func.batch(return_dtype=daft.DataType.binary())
def _read_bytes(audio_blobs):
    return [blob.read() if blob is not None else None for blob in audio_blobs.to_pylist()]


@daft.func.batch(return_dtype=daft.DataType.float64())
def _duration_udf(audio_bytes_col):
    import io as _io

    import soundfile as sf

    results = []
    for b in audio_bytes_col.to_pylist():
        if not b:
            results.append(0.0)
            continue
        try:
            info = sf.info(_io.BytesIO(b))
            results.append(float(info.frames) / info.samplerate if info.samplerate else 0.0)
        except Exception:
            results.append(0.0)
    return results


@daft.func.batch(return_dtype=daft.DataType.string())
def _prompt_udf(transcripts, acoustic_emotions):
    from multimodal_toolkit.audio.prompt import build_prompt

    return [
        build_prompt(t or "", e or "NEUTRAL")
        for t, e in zip(transcripts.to_pylist(), acoustic_emotions.to_pylist())
    ]


@daft.cls(cpus=1)
class _AsrUDF:
    def __init__(self) -> None:
        from multimodal_toolkit.audio.asr import SenseVoiceASR

        self._asr = SenseVoiceASR()

    @daft.method.batch(
        return_dtype=daft.DataType.struct(
            {
                "transcript": daft.DataType.string(),
                "acoustic_emotion": daft.DataType.string(),
            }
        )
    )
    def __call__(self, audio_bytes_col, doc_ids):
        from pathlib import Path

        results = []
        for audio_bytes, doc_id in zip(audio_bytes_col.to_pylist(), doc_ids.to_pylist()):
            suffix = Path(doc_id).suffix if doc_id else ".wav"
            if not suffix:
                suffix = ".wav"
            results.append(self._asr.transcribe_bytes(audio_bytes, suffix))
        return results


def run(lance_uri: str, out_jsonl: str) -> None:
    configure_daft_runner()
    io_config = daft_io_config()

    validate_blob_v2(lance_uri, "audio_blob")

    ds = lance.dataset(lance_uri, storage_options=lance_storage_options(lance_uri))
    df = daft.read_lance(lance_uri, io_config=io_config, default_scan_options={"with_row_id": True})
    df = df.select("doc_id", "audio_blob", "_rowid")
    df = daft_lance.take_blobs(df, ds, "audio_blob")
    df = df.where(~col("audio_blob").is_null())

    # Materialize blob bytes once — BlobFile is a one-shot stream
    df = df.with_column("audio_bytes", _read_bytes(col("audio_blob")))

    # Duration quality gate
    df = df.with_column("duration_s", _duration_udf(col("audio_bytes")))
    df = df.where((col("duration_s") >= config.MIN_DURATION_S) & (col("duration_s") <= config.MAX_DURATION_S))

    # ASR (stateful: model loads once per worker)
    asr = _AsrUDF()
    df = df.with_column("asr", asr(col("audio_bytes"), col("doc_id")))
    df = df.with_column("transcript_raw", col("asr")["transcript"])
    df = df.with_column("acoustic_emotion", col("asr")["acoustic_emotion"])

    # PII desensitization (ID card before phone to avoid partial digit overlap)
    df = df.with_column("transcript", regexp_replace(col("transcript_raw"), _ID_CARD_PAT, "[ID_REDACTED]"))
    df = df.with_column("transcript", regexp_replace(col("transcript"), _PHONE_PAT, "[PHONE_REDACTED]"))

    # LLM analysis
    df = df.with_column("prompt", _prompt_udf(col("transcript"), col("acoustic_emotion")))
    if config.DEEPSEEK_API_KEY:
        from daft.ai.openai.provider import OpenAIProvider

        _provider = OpenAIProvider(
            base_url=config.DEEPSEEK_BASE_URL,
            api_key=config.DEEPSEEK_API_KEY,
        )
        df = df.with_column(
            "analysis_json",
            llm_prompt(
                col("prompt"),
                provider=_provider,
                model=config.DEEPSEEK_MODEL,
                response_format={"type": "json_object"},
                temperature=0,
            ),
        )
    else:
        df = df.with_column("analysis_json", daft.lit(None).cast(daft.DataType.string()))

    # Parse JSON → struct, then expand each field; try_deserialize returns null on parse failure
    df = df.with_column("analysis", col("analysis_json").try_deserialize("json", _ANALYSIS_DTYPE))
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

    output = df.select(*_OUTPUT_COLS)

    output.write_json(out_jsonl, write_mode="overwrite", io_config=io_config)
    print(f"[ok] wrote analysis to: {out_jsonl}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lance-uri", required=True)
    parser.add_argument("--out-jsonl", required=True)
    args = parser.parse_args()
    run(args.lance_uri, args.out_jsonl)


if __name__ == "__main__":
    main()
