from __future__ import annotations

import io

import numpy as np

from . import config


def _normalize(vec: np.ndarray) -> list[float]:
    vec = np.asarray(vec, dtype="float32").reshape(-1)
    if vec.size != config.IMAGE_EMBED_DIM:
        raise ValueError(
            f"Embedding dimension mismatch: got {vec.size}, expected {config.IMAGE_EMBED_DIM}"
        )
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def _bytes_to_pil(image_bytes: bytes):
    from PIL import Image

    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


class ChineseClipEmbedder:
    dim = config.IMAGE_EMBED_DIM

    def __init__(self) -> None:
        import torch
        from transformers import ChineseCLIPModel, ChineseCLIPProcessor

        self._torch = torch
        self._device = config.IMAGE_EMBED_DEVICE
        self._processor = ChineseCLIPProcessor.from_pretrained(config.IMAGE_EMBED_MODEL)
        self._model = ChineseCLIPModel.from_pretrained(config.IMAGE_EMBED_MODEL)
        model_dim = getattr(self._model.config, "projection_dim", None)
        if model_dim is not None and model_dim != config.IMAGE_EMBED_DIM:
            raise ValueError(
                f"IMAGE_EMBED_DIM={config.IMAGE_EMBED_DIM} does not match "
                f"{config.IMAGE_EMBED_MODEL} projection_dim={model_dim}"
            )
        self._model.to(self._device)
        self._model.eval()

    def embed_image_bytes(self, image_bytes: bytes | None) -> list[float] | None:
        if not image_bytes:
            return None
        # cv2 能解码的图 PIL 不一定能打开（典型：截断的 JPEG），所以上游的
        # status="ok" 不保证这里解码成功。解不开返回 None（embedding 落 null），
        # 与"坏图不丢行"的管道约定一致，不让一张坏图打断整个批次。
        # 只包解码这一步：模型/设备层面的错误仍然照常抛出。
        try:
            image = _bytes_to_pil(image_bytes)
        except Exception:
            return None
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with self._torch.no_grad():
            features = self._model.get_image_features(**inputs)
        return _normalize(features.pooler_output[0].detach().cpu().numpy())

    def embed_text(self, text: str | None) -> list[float] | None:
        if not text:
            return None
        inputs = self._processor(text=[text], padding=True, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with self._torch.no_grad():
            features = self._model.get_text_features(**inputs)
        return _normalize(features.pooler_output[0].detach().cpu().numpy())


_EMBEDDER: ChineseClipEmbedder | None = None


def get_embedder() -> ChineseClipEmbedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = ChineseClipEmbedder()
    return _EMBEDDER
