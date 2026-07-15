"""Tests for image/rules.py — threshold verdicts over raw score columns."""
from __future__ import annotations

import daft
import pytest

from multimodal_toolkit.image.rules import add_rule_columns
from multimodal_toolkit.image import config


@pytest.fixture(autouse=True)
def _fixed_thresholds(monkeypatch):
    monkeypatch.setattr(config, "FACE_DET_SCORE_MIN", 0.5)
    monkeypatch.setattr(config, "MIN_FACE_RATIO", 0.01)
    monkeypatch.setattr(config, "BLUR_THRESHOLD", 100.0)
    monkeypatch.setattr(config, "FACE_BLUR_THRESHOLD", 80.0)
    monkeypatch.setattr(config, "AVATAR_MIN_FACE_RATIO", 0.03)


def _rows(df: daft.DataFrame) -> dict:
    data = df.collect().to_pydict()
    return data


def test_rule_columns():
    df = daft.from_pydict(
        {
            "doc_id": ["clear_face", "low_score", "tiny_face", "no_face", "blurry_face", "bad_download", "bad_image"],
            "status": ["ok", "ok", "ok", "ok", "ok", "download_failed", "decode_failed"],
            "face_count": [1, 1, 1, 0, 1, None, None],
            "face_score": [0.9, 0.3, 0.9, 0.0, 0.9, None, None],
            "face_area_ratio": [0.2, 0.2, 0.001, 0.0, 0.2, None, None],
            "blur_score": [500.0, 500.0, 500.0, 50.0, 500.0, None, None],
            "face_blur_score": [200.0, 200.0, 200.0, None, 30.0, None, None],
        }
    )
    out = _rows(add_rule_columns(df))
    by_id = {d: i for i, d in enumerate(out["doc_id"])}

    assert out["has_face"][by_id["clear_face"]] is True
    assert out["has_face"][by_id["low_score"]] is False  # largest face's det score below threshold
    assert out["has_face"][by_id["tiny_face"]] is False  # face too small
    assert out["has_face"][by_id["no_face"]] is False

    assert out["is_blurry"][by_id["no_face"]] is True  # blur_score 50 < 100
    assert out["is_blurry"][by_id["clear_face"]] is False

    assert out["is_face_blurry"][by_id["blurry_face"]] is True  # face_blur 30 < 80
    assert out["is_face_blurry"][by_id["clear_face"]] is False
    assert out["is_face_blurry"][by_id["no_face"]] is False  # null face_blur → False

    assert out["is_avatar"][by_id["clear_face"]] is True
    assert out["is_avatar"][by_id["low_score"]] is False
    assert out["is_avatar"][by_id["tiny_face"]] is False
    assert out["is_avatar"][by_id["blurry_face"]] is False
    assert out["is_avatar"][by_id["no_face"]] is False

    # Failed rows get null verdicts, never False — "unknown" must stay
    # distinguishable from "judged no".
    for doc in ("bad_download", "bad_image"):
        assert out["has_face"][by_id[doc]] is None
        assert out["is_blurry"][by_id[doc]] is None
        assert out["is_face_blurry"][by_id[doc]] is None
        assert out["is_avatar"][by_id[doc]] is None
