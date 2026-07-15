"""Tests for image/workflow/query.py — runs against a temporary local lance table."""
from __future__ import annotations

import pathlib
import tempfile

import lance
import pyarrow as pa
import pytest

from multimodal_toolkit.image.workflow.query import (
    image_doc_query,
    image_path_query,
    main,
    scalar_query,
    sql_query,
    text_query,
)

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
    "is_avatar": [True, False, False, False, None],
    "analysis_backend": ["local", "local", "llm", "llm", "local"],
    "clarity_confidence": [None, None, 0.9, 0.8, None],
    "avatar_confidence": [None, None, 0.2, 0.1, None],
    "llm_reason": [None, None, "多人合影", "人脸模糊", None],
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
    assert "is_avatar" in rows[0]
    assert "analysis_backend" in rows[0]


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


@pytest.fixture()
def lance_uri_with_embedding() -> str:
    uri = str(pathlib.Path(tempfile.mkdtemp()) / "test_images_emb.lance")
    dim = 4
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.9, 0.1, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        None,
    ]
    tbl = pa.table(
        {
            **ROWS,
            "image_embedding": pa.array(embeddings, type=pa.list_(pa.float32(), dim)),
        }
    )
    lance.write_dataset(tbl, uri)
    return uri


class _FakeImageEmbedder:
    def embed_text(self, text):
        if not text:
            return None
        return [0.0, 1.0, 0.0, 0.0]

    def embed_image_bytes(self, image_bytes):
        if not image_bytes:
            return None
        return [1.0, 0.0, 0.0, 0.0]


def test_text_query(monkeypatch, lance_uri_with_embedding):
    import multimodal_toolkit.image.embedding as embedding

    monkeypatch.setattr(embedding, "get_embedder", lambda: _FakeImageEmbedder())
    rows = text_query(lance_uri_with_embedding, "头像", top_k=2)
    assert len(rows) == 2
    assert rows[0]["doc_id"] in {"img_b", "img_c"}


def test_image_path_query(monkeypatch, tmp_path, lance_uri_with_embedding):
    import multimodal_toolkit.image.embedding as embedding

    monkeypatch.setattr(embedding, "get_embedder", lambda: _FakeImageEmbedder())
    query_path = tmp_path / "query.jpg"
    query_path.write_bytes(b"fake image bytes")
    rows = image_path_query(lance_uri_with_embedding, str(query_path), top_k=1, where="status = 'ok'")
    assert [r["doc_id"] for r in rows] == ["img_a"]


def test_image_doc_query(lance_uri_with_embedding):
    rows = image_doc_query(lance_uri_with_embedding, "img_a", top_k=1, where="status = 'ok'")
    assert [r["doc_id"] for r in rows] == ["img_a"]


def test_image_doc_query_null_embedding(lance_uri_with_embedding):
    with pytest.raises(ValueError, match="null image_embedding"):
        image_doc_query(lance_uri_with_embedding, "img_bad")


def test_cli_rejects_sql_with_vector_mode(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["query", "--lance-uri", "images.lance", "--sql", "SELECT 1", "--text", "头像"],
    )
    with pytest.raises(SystemExit):
        main()


def test_cli_rejects_distance_without_vector_mode(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["query", "--lance-uri", "images.lance", "--where", "status = 'ok'", "--distance-min", "0", "--distance-max", "1"],
    )
    with pytest.raises(SystemExit):
        main()
