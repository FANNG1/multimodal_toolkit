"""Image embedding pipeline tests that do not require local model downloads."""
from __future__ import annotations

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


class _FakeDetector:
    def detect(self, image):
        return []


class _FakeEmbedder:
    def __init__(self, vectors_by_bytes: dict[bytes, list[float]]):
        self._vectors_by_bytes = vectors_by_bytes

    def embed_text(self, text):
        if not text:
            return None
        return [1.0] + [0.0] * 511

    def embed_image_bytes(self, image_bytes):
        if not image_bytes:
            return None
        return self._vectors_by_bytes[bytes(image_bytes)]


def test_embedding_pipeline_from_analyze_to_similarity_query(monkeypatch, tmp_path):
    import multimodal_toolkit.image.detector as detector
    import multimodal_toolkit.image.embedding as embedding
    from multimodal_toolkit.image.workflow.analyze import run as analyze_run
    from multimodal_toolkit.image.workflow.ingest import run as ingest_run
    from multimodal_toolkit.image.workflow.query import image_path_query, text_query
    from multimodal_toolkit.workflow.index import build_embedding_index

    img_a = tmp_path / "avatar.jpg"
    img_b = tmp_path / "landscape.jpg"
    cv2.imwrite(str(img_a), np.full((32, 32, 3), (255, 255, 255), dtype=np.uint8))
    cv2.imwrite(str(img_b), np.full((32, 32, 3), (0, 0, 0), dtype=np.uint8))

    vectors_by_bytes = {
        img_a.read_bytes(): [1.0] + [0.0] * 511,
        img_b.read_bytes(): [0.0, 1.0] + [0.0] * 510,
    }
    fake_embedder = _FakeEmbedder(vectors_by_bytes)
    create_index_calls = []

    def fake_create_index(uri, **kwargs):
        create_index_calls.append((uri, kwargs))

    monkeypatch.setattr(detector, "get_detector", lambda: _FakeDetector())
    monkeypatch.setattr(embedding, "get_embedder", lambda: fake_embedder)
    monkeypatch.setattr("multimodal_toolkit.workflow.index.lance_ray.create_index", fake_create_index)

    manifest = tmp_path / "manifest.parquet"
    pq.write_table(
        pa.table(
            {
                "doc_id": ["avatar.jpg", "landscape.jpg"],
                "s3_url": [str(img_a), str(img_b)],
            }
        ),
        manifest,
    )

    staging_uri = str(tmp_path / "staging")
    assets_uri = str(tmp_path / "assets.lance")
    analyze_run(str(manifest), staging_uri, embed=True)
    ingest_run(staging_uri, assets_uri)
    build_embedding_index(
        assets_uri,
        column="image_embedding",
        num_partitions=1,
        sample_rate=2,
        index_type="IVF_FLAT",
    )

    assert create_index_calls == [
        (
            assets_uri,
            {
                "column": "image_embedding",
                "index_type": "IVF_FLAT",
                "num_partitions": 1,
                "sample_rate": 2,
                "replace": True,
                "storage_options": None,
            },
        )
    ]

    assert [r["doc_id"] for r in text_query(assets_uri, "头像", top_k=1)] == ["avatar.jpg"]
    assert [r["doc_id"] for r in image_path_query(assets_uri, str(img_a), top_k=1)] == [
        "avatar.jpg"
    ]


def test_embed_image_bytes_tolerates_undecodable_bytes():
    """cv2 能解、PIL 解不开的字节不能打挂批次：返回 None 而不是抛异常。"""
    from multimodal_toolkit.image.embedding import ChineseClipEmbedder

    # 跳过 __init__（不加载模型）：解码失败发生在触碰任何模型属性之前
    embedder = object.__new__(ChineseClipEmbedder)
    assert embedder.embed_image_bytes(b"not an image") is None


def test_analyze_without_embed_rejects_lance_output(tmp_path):
    from multimodal_toolkit.image.workflow.analyze import run as analyze_run

    with pytest.raises(ValueError, match="requires --embed"):
        analyze_run(str(tmp_path / "manifest.parquet"), str(tmp_path / "analysis.lance"), embed=False)


def test_ingest_rejects_mixed_embedding_schema(tmp_path):
    import lance

    from multimodal_toolkit.image.workflow.ingest import run as ingest_run

    assets_uri = str(tmp_path / "assets.lance")
    lance.write_dataset(pa.table({"doc_id": ["old"], "s3_url": ["old.jpg"]}), assets_uri)

    staging_uri = str(tmp_path / "staging")
    lance.write_dataset(
        pa.table(
            {
                "doc_id": ["new"],
                "s3_url": ["new.jpg"],
                "image_embedding": pa.array(
                    [[1.0] + [0.0] * 511],
                    type=pa.list_(pa.float32(), 512),
                ),
            }
        ),
        staging_uri,
    )

    with pytest.raises(ValueError, match="consistent about Stage 1 --embed"):
        ingest_run(staging_uri, assets_uri)
