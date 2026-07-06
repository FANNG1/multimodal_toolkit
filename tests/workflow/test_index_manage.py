"""Tests for workflow/index.py and workflow/manage.py — local lance tables.

Ray-touching tests use the shared `local_ray` fixture (tests/conftest.py) so
they always run against a hermetic local cluster from this venv.
build_embedding_index is additionally marked `ray` and excluded from the
default run (see pyproject) — run it explicitly with `pytest -m ray`.
delete_by_date tests stay in the default suite: since pylance 8 removed the
blob compaction limitation, they verify the full delete + compact + cleanup
path, including on a blob v2 table.
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
    """A blob v2 table written the way the asset tables are (daft-lance).

    This layout (extension type lance.blob.v2, storage 2.2) could not be
    compacted before pylance 8.0.0 (lance-format/lance#7071).
    """
    import daft
    from daft import col

    tmp = tempfile.mkdtemp()
    uri = str(pathlib.Path(tmp) / "table_blob.lance")
    times = [
        datetime.fromisoformat(DAYS[i % len(DAYS)]).replace(tzinfo=timezone.utc)
        for i in range(N_ROWS)
    ]
    df = daft.from_pydict(
        {
            "doc_id": [f"doc_{i:04d}" for i in range(N_ROWS)],
            "ingest_time": times,
            "blob": [b"x" * 100] * N_ROWS,
        }
    )
    df = df.with_column("ingest_time", col("ingest_time").cast(daft.DataType.timestamp("us", "UTC")))
    df.write_lance(uri, mode="create", blob_columns=["blob"])
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


@pytest.mark.ray
def test_build_embedding_index_small_table(lance_uri, local_ray):
    # Small-table parameters recommended by index.py's own docstring.
    build_embedding_index(lance_uri, num_partitions=1, sample_rate=2, index_type="IVF_FLAT")
    indices = lance.dataset(lance_uri).list_indices()
    assert any(idx["fields"] == ["audio_embedding"] for idx in indices)


def test_build_embedding_index_missing_column(lance_uri_no_embedding):
    with pytest.raises(ValueError, match="audio_embedding column not found"):
        build_embedding_index(lance_uri_no_embedding)


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


def test_delete_and_compact_blob_table(lance_uri_blob, local_ray):
    """Regression for lance-format/lance#7071: blob v2 tables can be
    compacted since pylance 8.0.0. On 7.x this raised inside lance's decoder
    and manage.py had to treat compaction as best-effort."""
    delete_by_date(lance_uri_blob, before="2024-03-01")
    ds = lance.dataset(lance_uri_blob)
    assert ds.count_rows() == 200
    # Reaching here means compact_files succeeded (manage.py no longer
    # swallows compaction errors); verify blobs still readable afterwards.
    blob = ds.take_blobs("blob", indices=[0])[0]
    assert len(blob.read()) == 100
