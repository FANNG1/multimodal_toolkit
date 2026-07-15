"""音频 Stage 1 UDF 的空值语义测试（不加载 ASR/嵌入模型，纯本地）。

行不丢弃约定的基石：duration_udf 对坏字节返回 null（下游据此判定
decode_failed），prompt_udf 对 null 转写返回 null（LLM 跳过没处理的行）。
"""
from __future__ import annotations

import io

import daft
import soundfile as sf


def _wav_bytes(seconds: float = 1.0, samplerate: int = 16000) -> bytes:
    import numpy as np

    buf = io.BytesIO()
    samples = np.zeros(int(seconds * samplerate), dtype="float32")
    sf.write(buf, samples, samplerate, format="WAV")
    return buf.getvalue()


def test_duration_udf_null_for_bad_bytes():
    """字节缺失或解码失败必须返回 null 而不是 0.0，否则无法区分坏文件和空录音。"""
    from multimodal_toolkit.audio.udfs import duration_udf

    df = daft.from_pydict({"b": [None, b"not-audio-at-all", _wav_bytes(1.0)]})
    out = df.select(duration_udf(df["b"]).alias("d")).to_pydict()["d"]
    assert out[0] is None
    assert out[1] is None
    assert out[2] is not None and abs(out[2] - 1.0) < 0.01


def test_prompt_udf_null_transcript_skips_llm():
    """null 转写（没处理的行）→ null 提示词；空字符串转写（无语音）仍构造提示词。"""
    from multimodal_toolkit.audio.udfs import prompt_udf

    df = daft.from_pydict(
        {"t": [None, "", "想改成8元套餐"], "e": [None, "NEUTRAL", "ANGRY"]}
    )
    out = df.select(prompt_udf(df["t"], df["e"]).alias("p")).to_pydict()["p"]
    assert out[0] is None
    assert out[1] is not None
    assert out[2] is not None and "8元套餐" in out[2]


def test_embedding_input_keeps_llm_failed_audio():
    """LLM 失败不影响独立的声学 embedding，只跳过媒体本身不可用的行。"""
    from multimodal_toolkit.audio.workflow.analyze import _embedding_input

    df = daft.from_pydict(
        {
            "audio_bytes": [b"ok", b"llm-failed", b"missing", b"bad", b"filtered"],
            "status": [
                "ok",
                "llm_failed",
                "download_failed",
                "decode_failed",
                "duration_filtered",
            ],
        }
    )
    out = df.select(
        _embedding_input(df["audio_bytes"], df["status"]).alias("embedding_input")
    ).to_pydict()["embedding_input"]

    assert out == [b"ok", b"llm-failed", None, None, None]
