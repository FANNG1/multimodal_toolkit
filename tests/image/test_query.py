"""Tests for image/workflow/query.py — runs against a temporary local lance table."""
from __future__ import annotations

import pathlib
import tempfile

import lance
import pyarrow as pa
import pytest

from multimodal_toolkit.image.workflow.query import scalar_query, sql_query

ROWS = {
    "doc_id": ["img_a", "img_b", "img_c", "img_d", "img_bad"],
    "ingest_time": [
        "2024-01-01T00:00:00",
        "2024-01-02T00:00:00",
        "2024-01-03T00:00:00",
        "2024-01-04T00:00:00",
        "2024-01-05T00:00:00",
    ],
    "status": ["ok", "ok", "ok", "ok", "decode_failed"],
    "width": [1920, 640, 1024, 800, None],
    "height": [1080, 480, 768, 600, None],
    "face_count": [1, 0, 2, 1, None],
    "face_score": [0.95, 0.0, 0.88, 0.6, None],
    "blur_score": [500.0, 50.0, 300.0, 80.0, None],
    "face_blur_score": [200.0, None, 150.0, 30.0, None],
    "has_face": [True, False, True, True, None],
    "is_blurry": [False, True, False, True, None],
    "is_face_blurry": [False, False, False, True, None],
}


@pytest.fixture(scope="module")
def lance_uri() -> str:
    tmp = tempfile.mkdtemp()
    uri = str(pathlib.Path(tmp) / "test_images.lance")
    lance.write_dataset(pa.table(ROWS), uri)
    return uri


def test_scalar_query_no_filter(lance_uri):
    rows = scalar_query(lance_uri, top_k=10)
    assert len(rows) == 5
    assert "has_face" in rows[0]
    assert "blur_score" in rows[0]
    assert "status" in rows[0]


def test_scalar_query_where(lance_uri):
    rows = scalar_query(lance_uri, where="has_face = true AND is_blurry = false")
    assert sorted(r["doc_id"] for r in rows) == ["img_a", "img_c"]


def test_sql_query_filter(lance_uri):
    rows = sql_query(lance_uri, "SELECT doc_id, blur_score FROM images WHERE is_blurry = true")
    assert sorted(r["doc_id"] for r in rows) == ["img_b", "img_d"]


def test_sql_query_aggregation(lance_uri):
    rows = sql_query(
        lance_uri,
        "SELECT has_face, COUNT(*) AS cnt FROM images GROUP BY has_face ORDER BY cnt DESC",
    )
    counts = {r["has_face"]: r["cnt"] for r in rows}
    assert counts[True] == 3
    assert counts[False] == 1
    assert counts[None] == 1  # failed row keeps null verdict


def test_sql_query_status_filter(lance_uri):
    rows = sql_query(lance_uri, "SELECT doc_id, status FROM images WHERE status != 'ok'")
    assert [r["doc_id"] for r in rows] == ["img_bad"]
