"""Shared helpers for image tests."""
from __future__ import annotations

from pathlib import Path

from multimodal_toolkit import config


def scrfd_model_dir() -> Path:
    """Local path of the InsightFace model pack (may not exist)."""
    root = Path(config.INSIGHTFACE_ROOT) if config.INSIGHTFACE_ROOT else Path.home() / ".insightface"
    return root / "models" / config.INSIGHTFACE_MODEL


def scrfd_model_available() -> bool:
    return scrfd_model_dir().exists()
