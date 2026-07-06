from __future__ import annotations

import io

import numpy as np

from . import config


class SignalEmbedder:
    """Small acoustic embedding for POC.

    Produces 128 dimensions: 64 bucketed RMS-energy features plus 64 bucketed
    zero-crossing-rate features. This is intentionally acoustic-only and avoids
    text/semantic models.
    """

    dim = 128

    def embed_bytes(self, audio_bytes: bytes | None, suffix: str = ".wav") -> list[float] | None:
        if not audio_bytes:
            return None
        import soundfile as sf

        wav, _sr = sf.read(io.BytesIO(audio_bytes), always_2d=True, dtype="float32")
        mono = wav.mean(axis=1)
        if mono.size == 0:
            return None
        peak = float(np.max(np.abs(mono)))
        if peak > 0:
            mono = mono / peak

        buckets = np.array_split(mono, 64)
        rms = np.array([float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0 for chunk in buckets], dtype="float32")
        zcr = np.array(
            [
                float(np.mean(np.abs(np.diff(np.signbit(chunk).astype("int8"))))) if chunk.size > 1 else 0.0
                for chunk in buckets
            ],
            dtype="float32",
        )
        vec = np.concatenate([rms, zcr]).astype("float32")
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


class Wav2Vec2Embedder:
    def __init__(self) -> None:
        raise RuntimeError("Wav2Vec2 backend is disabled in this POC run; use EMBED_BACKEND=signal.")

    def embed_bytes(self, audio_bytes: bytes | None, suffix: str = ".wav") -> list[float] | None:
        raise RuntimeError("Wav2Vec2 backend is disabled in this POC run; use EMBED_BACKEND=signal.")


_EMBEDDER = None


def get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        if config.EMBED_BACKEND == "signal":
            _EMBEDDER = SignalEmbedder()
        elif config.EMBED_BACKEND == "wav2vec2":
            _EMBEDDER = Wav2Vec2Embedder()
        else:
            raise ValueError(f"unsupported EMBED_BACKEND={config.EMBED_BACKEND}")
    return _EMBEDDER
