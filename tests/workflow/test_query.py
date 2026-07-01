"""Tests for workflow/query.py — runs against a temporary local lance table."""
from __future__ import annotations

import pathlib
import tempfile

import lance
import pyarrow as pa
import pytest

from multimodal_toolkit.workflow.query import scalar_query, sql_query, vector_query

ROWS = {
    "doc_id": ["call_a", "call_b", "call_c", "call_d"],
    "ingest_time": [
        "2024-01-01T00:00:00",
        "2024-01-02T00:00:00",
        "2024-01-03T00:00:00",
        "2024-01-04T00:00:00",
    ],
    "bad_tone": [False, True, False, True],
    "emotion_score": [0.1, 0.9, 0.5, 0.7],
    "primary_reason": ["投诉", "降级", "投诉", "降级"],
    "secondary_reason": ["", "", "", ""],
    "text_emotion": ["neutral", "angry", "neutral", "angry"],
    "downgrade_related": [False, True, False, True],
}


@pytest.fixture(scope="module")
def lance_uri() -> str:
    tmp = tempfile.mkdtemp()
    uri = str(pathlib.Path(tmp) / "test_calls.lance")
    lance.write_dataset(pa.table(ROWS), uri)
    return uri


# ---------------------------------------------------------------------------
# scalar_query
# ---------------------------------------------------------------------------

def test_scalar_query_no_filter(lance_uri):
    rows = scalar_query(lance_uri, top_k=10)
    assert len(rows) == 4


def test_scalar_query_where(lance_uri):
    rows = scalar_query(lance_uri, where="bad_tone = true")
    assert len(rows) == 2
    assert all(r["bad_tone"] for r in rows)


def test_scalar_query_top_k(lance_uri):
    rows = scalar_query(lance_uri, top_k=2)
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# sql_query
# ---------------------------------------------------------------------------

def test_sql_query_scalar_filter(lance_uri):
    rows = sql_query(lance_uri, "SELECT doc_id, emotion_score FROM calls WHERE bad_tone = true")
    assert len(rows) == 2
    assert all(r["doc_id"] in ("call_b", "call_d") for r in rows)


def test_sql_query_aggregation(lance_uri):
    rows = sql_query(
        lance_uri,
        "SELECT primary_reason, COUNT(*) AS cnt FROM calls GROUP BY primary_reason ORDER BY cnt DESC",
    )
    assert len(rows) == 2
    # "投诉" and "降级" each appear twice
    counts = {r["primary_reason"]: r["cnt"] for r in rows}
    assert counts["投诉"] == 2
    assert counts["降级"] == 2


def test_sql_query_order_by(lance_uri):
    rows = sql_query(lance_uri, "SELECT doc_id, emotion_score FROM calls ORDER BY emotion_score DESC")
    scores = [r["emotion_score"] for r in rows]
    assert scores == sorted(scores, reverse=True)


def test_sql_query_top_k_limits_results(lance_uri):
    rows = sql_query(lance_uri, "SELECT doc_id FROM calls", top_k=2)
    assert len(rows) == 2


def test_sql_query_projection(lance_uri):
    rows = sql_query(lance_uri, "SELECT doc_id FROM calls")
    assert all(set(r.keys()) == {"doc_id"} for r in rows)


# ---------------------------------------------------------------------------
# vector_query — requires audio_embedding column; skip if absent
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lance_uri_with_embedding() -> str:
    import numpy as np

    tmp = tempfile.mkdtemp()
    uri = str(pathlib.Path(tmp) / "test_calls_emb.lance")
    n = 4
    dim = 16
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((n, dim)).astype("float32")
    tbl = pa.table(
        {
            **ROWS,
            "audio_embedding": pa.FixedSizeListArray.from_arrays(
                pa.array(embeddings.ravel().tolist(), type=pa.float32()), dim
            ),
        }
    )
    lance.write_dataset(tbl, uri)
    return uri


def test_vector_query_returns_top_k(lance_uri_with_embedding):
    rows = vector_query(lance_uri_with_embedding, "call_a", top_k=2)
    assert len(rows) == 2


def test_vector_query_missing_doc_id(lance_uri_with_embedding):
    with pytest.raises(ValueError, match="not found"):
        vector_query(lance_uri_with_embedding, "nonexistent")


def test_vector_query_with_where(lance_uri_with_embedding):
    rows = vector_query(lance_uri_with_embedding, "call_a", top_k=4, where="bad_tone = true")
    assert all(r["bad_tone"] for r in rows)


def test_vector_query_with_distance_range(lance_uri_with_embedding):
    rows = vector_query(lance_uri_with_embedding, "call_a", top_k=4, distance_range=(0.0, 0.0001))
    assert [r["doc_id"] for r in rows] == ["call_a"]
