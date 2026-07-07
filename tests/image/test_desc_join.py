"""场景测试：以文/以图搜图时，按 doc_id left join 描述表输出 description。

描述表在代码中生成为 parquet 文件（doc_id → description），覆盖三种情况：
正常映射、图片无描述（left join 后为 null）、描述表中多余的 doc_id（被忽略）。
"""
from __future__ import annotations

import pathlib
import tempfile

import lance
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from multimodal_toolkit.image.workflow.query import (
    image_doc_query,
    scalar_query,
    sql_query,
    text_query,
)

DIM = 4

ROWS = {
    "doc_id": ["img_a", "img_b", "img_c"],
    "status": ["ok", "ok", "ok"],
    "has_face": [True, False, True],
    "blur_score": [500.0, 50.0, 300.0],
    "is_blurry": [False, True, False],
}

EMBEDDINGS = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.9, 0.1, 0.0],
]

# img_b 故意没有描述；img_zzz 是表里不存在的图片。
DESCRIPTIONS = {
    "doc_id": ["img_a", "img_c", "img_zzz"],
    "description": ["清晰正面人像", "两人合影", "不存在的图"],
}


@pytest.fixture(scope="module")
def scenario() -> dict:
    tmp = pathlib.Path(tempfile.mkdtemp())
    lance_uri = str(tmp / "images.lance")
    tbl = pa.table(
        {
            **ROWS,
            "image_embedding": pa.array(EMBEDDINGS, type=pa.list_(pa.float32(), DIM)),
        }
    )
    lance.write_dataset(tbl, lance_uri)

    # 描述表：代码中生成 parquet，文件本身即可当表参与 join
    desc_path = str(tmp / "descriptions.parquet")
    pq.write_table(pa.table(DESCRIPTIONS), desc_path)

    return {"lance_uri": lance_uri, "desc_table": desc_path}


class _FakeImageEmbedder:
    def embed_text(self, text):
        return [0.0, 1.0, 0.0, 0.0] if text else None

    def embed_image_bytes(self, image_bytes):
        return [1.0, 0.0, 0.0, 0.0] if image_bytes else None


def test_text_query_joins_description(monkeypatch, scenario):
    import multimodal_toolkit.image.embedding as embedding

    monkeypatch.setattr(embedding, "get_embedder", lambda: _FakeImageEmbedder())
    rows = text_query(scenario["lance_uri"], "合影", top_k=3, desc_table=scenario["desc_table"])
    by_id = {r["doc_id"]: r for r in rows}
    assert by_id["img_c"]["description"] == "两人合影"
    assert by_id["img_a"]["description"] == "清晰正面人像"
    # left join：没有描述的图片不丢，description 为 null
    assert by_id["img_b"]["description"] is None


def test_image_doc_query_joins_description(scenario):
    rows = image_doc_query(scenario["lance_uri"], "img_a", top_k=1, desc_table=scenario["desc_table"])
    assert rows[0]["doc_id"] == "img_a"
    assert rows[0]["description"] == "清晰正面人像"


def test_extra_doc_id_in_desc_table_is_ignored(scenario):
    rows = scalar_query(scenario["lance_uri"], top_k=10, desc_table=scenario["desc_table"])
    assert sorted(r["doc_id"] for r in rows) == ["img_a", "img_b", "img_c"]  # img_zzz 不出现


def test_sql_query_manual_join(scenario):
    rows = sql_query(
        scenario["lance_uri"],
        "SELECT i.doc_id, d.description FROM images i "
        "LEFT JOIN descriptions d ON i.doc_id = d.doc_id "
        "WHERE i.has_face = true ORDER BY i.doc_id",
        desc_table=scenario["desc_table"],
    )
    assert [(r["doc_id"], r["description"]) for r in rows] == [
        ("img_a", "清晰正面人像"),
        ("img_c", "两人合影"),
    ]


def test_no_desc_table_keeps_existing_behavior(monkeypatch, scenario):
    import multimodal_toolkit.image.embedding as embedding

    monkeypatch.setattr(embedding, "get_embedder", lambda: _FakeImageEmbedder())
    rows = text_query(scenario["lance_uri"], "合影", top_k=3)
    assert all("description" not in r for r in rows)
