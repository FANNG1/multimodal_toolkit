from __future__ import annotations

import os
import re
import tempfile
from collections import Counter

from .. import config

_EMO_RE = re.compile(r"<\|(HAPPY|SAD|ANGRY|NEUTRAL|FEARFUL|DISGUSTED|SURPRISED|EMO_UNKNOWN)\|>")
_EMO_NOISE = {"NEUTRAL", "EMO_UNKNOWN"}


def _aggregate_emotion(labels: list[str]) -> str:
    meaningful = [x for x in labels if x not in _EMO_NOISE]
    if not meaningful:
        return "NEUTRAL"
    return Counter(meaningful).most_common(1)[0][0]


class SenseVoiceASR:
    def __init__(self) -> None:
        from funasr import AutoModel

        self._model = AutoModel(
            model=config.ASR_MODEL,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device=config.ASR_DEVICE,
            disable_update=True,
            disable_pbar=True,
            log_level="ERROR",
        )

    def transcribe_file(self, path: str) -> dict:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        res = self._model.generate(
            input=path,
            cache={},
            language="zh",
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )
        if not res:
            return {"transcript": "", "acoustic_emotion": "NEUTRAL"}
        raw = "".join(seg.get("text", "") for seg in res)
        transcript = "".join(rich_transcription_postprocess(seg.get("text", "")) for seg in res)
        return {"transcript": transcript, "acoustic_emotion": _aggregate_emotion(_EMO_RE.findall(raw))}

    def transcribe_bytes(self, audio_bytes: bytes | None, suffix: str = ".wav") -> dict:
        if not audio_bytes:
            return {"transcript": "", "acoustic_emotion": "NEUTRAL"}
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fp:
            fp.write(audio_bytes)
            name = fp.name
        try:
            return self.transcribe_file(name)
        finally:
            os.unlink(name)


_ASR: SenseVoiceASR | None = None


def get_asr() -> SenseVoiceASR:
    global _ASR
    if _ASR is None:
        _ASR = SenseVoiceASR()
    return _ASR

