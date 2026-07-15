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
    """音频时长（秒）。字节缺失或解码失败返回 null 而不是 0.0——
    下游靠 duration 是否为 null 判定 decode_failed，0.0 会和
    "合法的零长音频"混在一起，无法区分"坏文件"和"空录音"。
    """
    import io as _io

    import soundfile as sf

    results = []
    for b in audio_bytes_col.to_pylist():
        if not b:
            results.append(None)
            continue
        try:
            info = sf.info(_io.BytesIO(b))
            results.append(float(info.frames) / info.samplerate if info.samplerate else None)
        except Exception:
            results.append(None)
    return results


@daft.func.batch(return_dtype=daft.DataType.string())
def prompt_udf(transcripts, acoustic_emotions):
    """构造 LLM 打标提示词。transcript 为 null 说明该行下载/解码/ASR
    没有成功，返回 null 让 LLM 跳过——空字符串转写（真实的无语音录音）
    仍然正常构造提示词。
    """
    from multimodal_toolkit.audio.prompt import build_prompt

    return [
        build_prompt(t, e or "NEUTRAL") if t is not None else None
        for t, e in zip(transcripts.to_pylist(), acoustic_emotions.to_pylist())
    ]


@daft.cls(cpus=1, max_retries=2)
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
            # 字节为 null（下载/解码失败或被时长过滤的行）→ 整个 struct 置
            # null，transcript 保持"没处理"语义，不能落成空字符串。
            if not audio_bytes:
                results.append(None)
                continue
            suffix = Path(doc_id).suffix if doc_id else ".wav"
            if not suffix:
                suffix = ".wav"
            results.append(self._asr.transcribe_bytes(audio_bytes, suffix))
        return results
