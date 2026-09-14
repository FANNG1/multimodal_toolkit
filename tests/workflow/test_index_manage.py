"""在本地 Lance 表上验证 workflow/index.py 和 workflow/manage.py。

build_embedding_index 的真实 Ray 路径不属于默认测试；需要时用 `pytest -m ray`
显式执行。delete_by_date 会执行分布式 compaction，因此删除测试使用共享的 local_ray
fixture，并同时覆盖普通表和 Blob v2 表。
"""
from __future__ import annotations

import pathlib
import tempfile
from datetime import datetime, timezone

import lance
import numpy as np
import pyarrow as pa
import pytest

from multimodal_toolkit.workflow.index import build_embedding_index, build_time_index
from multimodal_toolkit.workflow.manage import delete_by_date

N_ROWS = 300
DIM = 16
DAYS = ["2024-01-01", "2024-06-01", "2024-12-01"]  # 100 rows per day


def _make_table(with_embedding: bool = True) -> pa.Table:
    rng = np.random.default_rng(7)
    times = [
        datetime.fromisoformat(DAYS[i % len(DAYS)]).replace(tzinfo=timezone.utc)
        for i in range(N_ROWS)
    ]
    cols: dict = {
        "doc_id": pa.array([f"doc_{i:04d}" for i in range(N_ROWS)]),
        "ingest_time": pa.array(times, type=pa.timestamp("us", tz="UTC")),
    }
    if with_embedding:
        emb = rng.standard_normal((N_ROWS, DIM)).astype("float32")
        cols["audio_embedding"] = pa.FixedSizeListArray.from_arrays(
            pa.array(emb.ravel().tolist(), type=pa.float32()), DIM
        )
    return pa.table(cols)


@pytest.fixture()
def lance_uri() -> str:
    tmp = tempfile.mkdtemp()
    uri = str(pathlib.Path(tmp) / "table.lance")
    lance.write_dataset(_make_table(), uri)
    return uri


@pytest.fixture()
def lance_uri_blob() -> str:
    """按资产表的写入方式创建多 fragment 的 Blob v2 表。

    每批 100 行分开 append，确保删除后 compaction 至少有多个 fragment 可以合并；
    这能覆盖 pylance 9 与 lance-ray 0.5 的真实 Blob v2 表维护链路。
    """
    import daft
    from daft import col

    tmp = tempfile.mkdtemp()
    uri = str(pathlib.Path(tmp) / "table_blob.lance")
    for batch in range(3):
        times = [datetime(2024, 1, batch + 1, tzinfo=timezone.utc)] * 100
        df = daft.from_pydict(
            {
                "doc_id": [f"blob_{batch}_{row:04d}" for row in range(100)],
                "ingest_time": times,
                "blob": [b"x" * 100] * 100,
            }
        ).with_column(
            "ingest_time", col("ingest_time").cast(daft.DataType.timestamp("us", "UTC"))
        )
        df.write_lance(
            uri,
            mode="create" if batch == 0 else "append",
            blob_columns=["blob"],
        )
    return uri


@pytest.fixture()
def lance_uri_no_embedding() -> str:
    tmp = tempfile.mkdtemp()
    uri = str(pathlib.Path(tmp) / "table_noemb.lance")
    lance.write_dataset(_make_table(with_embedding=False), uri)
    return uri


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------

def test_build_time_index(lance_uri):
    build_time_index(lance_uri)
    indices = lance.dataset(lance_uri).list_indices()
    assert any(idx["fields"] == ["ingest_time"] for idx in indices)


def test_build_embedding_index_uses_lance_ray(monkeypatch, lance_uri):
    calls = []

    def fake_create_index(uri, **kwargs):
        calls.append((uri, kwargs))

    monkeypatch.setattr("multimodal_toolkit.workflow.index.lance_ray.create_index", fake_create_index)
    build_embedding_index(lance_uri, num_partitions=1, sample_rate=2, index_type="IVF_FLAT")

    assert calls == [
        (
            lance_uri,
            {
                "column": "audio_embedding",
                "index_type": "IVF_FLAT",
                "num_partitions": 1,
                "sample_rate": 2,
                "replace": True,
                "storage_options": None,
            },
        )
    ]


def test_build_embedding_index_propagates_lance_ray_failure(monkeypatch, lance_uri):
    def fake_create_index(uri, **kwargs):
        raise RuntimeError("distributed index failed")

    monkeypatch.setattr("multimodal_toolkit.workflow.index.lance_ray.create_index", fake_create_index)

    with pytest.raises(RuntimeError, match="distributed index failed"):
        build_embedding_index(lance_uri, num_partitions=1, sample_rate=2, index_type="IVF_FLAT")


def test_build_embedding_index_missing_column(lance_uri_no_embedding):
    with pytest.raises(ValueError, match="audio_embedding column not found"):
        build_embedding_index(lance_uri_no_embedding)


def test_build_embedding_index_missing_custom_column(lance_uri_no_embedding):
    with pytest.raises(ValueError, match="image_embedding column not found"):
        build_embedding_index(lance_uri_no_embedding, column="image_embedding")


# ---------------------------------------------------------------------------
# manage
# ---------------------------------------------------------------------------

def test_delete_requires_a_bound(lance_uri):
    with pytest.raises(ValueError, match="at least one"):
        delete_by_date(lance_uri)


def test_delete_before(lance_uri, local_ray):
    delete_by_date(lance_uri, before="2024-03-01")
    assert lance.dataset(lance_uri).count_rows() == 200  # 2024-01-01 rows gone


def test_delete_after(lance_uri, local_ray):
    delete_by_date(lance_uri, after="2024-09-01")
    assert lance.dataset(lance_uri).count_rows() == 200  # 2024-12-01 rows gone


def test_delete_window(lance_uri, local_ray):
    # Outside 2024-03-01 .. 2024-09-01 survives: keeps Jan and Dec rows.
    delete_by_date(lance_uri, before="2024-09-01", after="2024-03-01")
    remaining = lance.dataset(lance_uri).count_rows()
    assert remaining == 200


def test_delete_and_compact_blob_v2_table(lance_uri_blob, local_ray):
    """删除后必须能压实 Blob v2 表，并保持剩余 blob 可读。"""
    from multimodal_toolkit.storage.blob import validate_blob_v2

    # Daft 的执行配置会影响每次 append 实际写出的 fragment 数；这里只约束
    # compaction 的核心语义：压实前确实有多个 fragment，压实后数量必须减少。
    fragments_before = len(lance.dataset(lance_uri_blob).get_fragments())
    assert fragments_before > 1
    delete_by_date(lance_uri_blob, before="2024-01-02")

    dataset = lance.dataset(lance_uri_blob)
    assert dataset.count_rows() == 200
    assert len(dataset.get_fragments()) < fragments_before
    assert len(dataset.take_blobs("blob", indices=[0])[0].read()) == 100
    validate_blob_v2(lance_uri_blob, "blob")
