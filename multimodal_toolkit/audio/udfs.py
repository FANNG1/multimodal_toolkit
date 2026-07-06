"""Shared Daft UDFs for the audio analysis pipeline (Stage 1)."""
from __future__ import annotations

import daft

ANALYSIS_DTYPE = daft.DataType.struct(
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


@daft.func.batch(return_dtype=daft.DataType.float64())
def duration_udf(audio_bytes_col):
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
def prompt_udf(transcripts, acoustic_emotions):
    from multimodal_toolkit.audio.prompt import build_prompt

    return [
        build_prompt(t or "", e or "NEUTRAL")
        for t, e in zip(transcripts.to_pylist(), acoustic_emotions.to_pylist())
    ]


@daft.cls(cpus=1)
class AsrUDF:
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
