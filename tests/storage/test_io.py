"""Tests for storage/io.py — manifest reading and lance storage options."""
from __future__ import annotations

import json

import lance
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from multimodal_toolkit.storage.io import lance_storage_options, lance_write_mode, read_manifest

ROWS = {"doc_id": ["a.jpg", "b.jpg"], "s3_url": ["s3://bkt/a.jpg", "s3://bkt/b.jpg"]}


def _assert_manifest(df) -> None:
    out = df.collect().to_pydict()
    assert out["doc_id"] == ROWS["doc_id"]
    assert out["s3_url"] == ROWS["s3_url"]
    assert set(out.keys()) == {"doc_id", "s3_url"}


def test_read_manifest_parquet(tmp_path):
    path = tmp_path / "m.parquet"
    # Extra column must be dropped by the doc_id/s3_url projection.
    pq.write_table(pa.table({**ROWS, "extra": [1, 2]}), path)
    _assert_manifest(read_manifest(str(path)))


def test_read_manifest_jsonl(tmp_path):
    path = tmp_path / "m.jsonl"
    lines = [
        json.dumps({"doc_id": d, "s3_url": u})
        for d, u in zip(ROWS["doc_id"], ROWS["s3_url"])
    ]
    path.write_text("\n".join(lines))
    _assert_manifest(read_manifest(str(path)))


def test_read_manifest_csv(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text(
        "doc_id,s3_url\n"
        + "\n".join(f"{d},{u}" for d, u in zip(ROWS["doc_id"], ROWS["s3_url"]))
    )
    _assert_manifest(read_manifest(str(path)))


def test_read_manifest_unsupported_format():
    with pytest.raises(ValueError, match="Unsupported manifest format"):
        read_manifest("manifest.xlsx")


def test_lance_storage_options_local_is_noop():
    assert lance_storage_options("/tmp/table.lance") == {}


def test_lance_storage_options_s3():
    opts = lance_storage_options("s3://bucket/table.lance")
    assert opts["aws_endpoint"]
    assert opts["aws_virtual_hosted_style_access"] == "false"


def test_lance_write_mode_create_for_missing_dataset(tmp_path):
    assert lance_write_mode(str(tmp_path / "missing.lance")) == "create"


def test_lance_write_mode_append_for_existing_dataset(tmp_path):
    uri = str(tmp_path / "existing.lance")
    lance.write_dataset(pa.table({"doc_id": ["a"]}), uri)
    assert lance_write_mode(uri) == "append"


def test_lance_write_mode_reraises_non_missing_value_error(monkeypatch):
    import multimodal_toolkit.storage.io as storage_io

    def _raise_value_error(*args, **kwargs):
        raise ValueError("credential refresh failed")

    monkeypatch.setattr(lance, "dataset", _raise_value_error)
    with pytest.raises(ValueError, match="credential refresh failed"):
        storage_io.lance_write_mode("s3://bucket/table.lance")
