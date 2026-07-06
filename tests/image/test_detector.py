"""Integration test for image/detector.py — needs the InsightFace model pack.

Skipped unless the model is already available locally (no network download in CI).
"""
from __future__ import annotations

import os

import pytest

from .conftest import scrfd_model_available

pytestmark = pytest.mark.skipif(
    not scrfd_model_available() or os.getenv("SKIP_DETECTOR_TESTS"),
    reason="InsightFace model pack not available locally",
)


def test_detector_no_face_on_blank_image():
    import numpy as np

    from multimodal_toolkit.image.detector import get_detector

    blank = np.full((640, 640, 3), 128, dtype=np.uint8)
    faces = get_detector().detect(blank)
    assert faces == []
