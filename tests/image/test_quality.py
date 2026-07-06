"""Tests for image/quality.py — synthetic images, no model needed."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from multimodal_toolkit.image.quality import (
    crop_bbox,
    decode_image,
    laplacian_variance,
    resize_long_edge,
)


def _checkerboard(size: int = 256, cell: int = 8) -> np.ndarray:
    tile = np.kron(
        np.indices((size // cell, size // cell)).sum(axis=0) % 2, np.ones((cell, cell))
    )
    return cv2.cvtColor((tile * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def _encode(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_decode_image_roundtrip():
    img = _checkerboard()
    decoded = decode_image(_encode(img))
    assert decoded is not None
    assert decoded.shape == img.shape


def test_decode_image_garbage_returns_none():
    assert decode_image(b"not an image") is None
    assert decode_image(b"") is None
    assert decode_image(None) is None


def test_laplacian_variance_sharp_greater_than_blurred():
    sharp = _checkerboard()
    blurred = cv2.GaussianBlur(sharp, (15, 15), 5)
    assert laplacian_variance(sharp) > laplacian_variance(blurred)


def test_resize_long_edge_downscales():
    img = np.zeros((500, 2000, 3), dtype=np.uint8)
    resized, scale = resize_long_edge(img, long_edge=1000)
    assert scale == pytest.approx(0.5)
    assert resized.shape[:2] == (250, 1000)


def test_resize_long_edge_never_upscales():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    resized, scale = resize_long_edge(img, long_edge=1000)
    assert scale == 1.0
    assert resized.shape[:2] == (100, 200)


def test_crop_bbox_clamps_to_bounds():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = crop_bbox(img, (-10, -10, 50, 200))
    assert crop is not None
    assert crop.shape[:2] == (100, 50)


def test_crop_bbox_degenerate_returns_none():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    assert crop_bbox(img, (50, 50, 50, 60)) is None
    assert crop_bbox(img, (200, 200, 300, 300)) is None
