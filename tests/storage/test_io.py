"""Tests for storage/io.py — manifest reading and lance storage options."""
from __future__ import annotations

import json

import lance
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from multimodal_toolkit.storage.io import (
    analysis_num_partitions,
    coalesce_for_write,
    daft_io_config,
    lance_storage_options,
    lance_write_mode,
    read_analysis_output,
    read_manifest,
    spread_partitions,
)

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


def test_daft_io_config_applies_s3_tuning():
    from multimodal_toolkit import config

    s3 = daft_io_config().s3
    assert s3.max_connections == config.S3_MAX_CONNECTIONS
    assert s3.num_tries == config.S3_NUM_TRIES
    assert s3.retry_initial_backoff_ms == config.S3_RETRY_INITIAL_BACKOFF_MS
    assert s3.connect_timeout_ms == config.S3_CONNECT_TIMEOUT_MS
    assert s3.read_timeout_ms == config.S3_READ_TIMEOUT_MS
    assert s3.retry_mode == config.S3_RETRY_MODE


class _FakePartitionedDf:
    def __init__(self):
        self.partitions = None

    def into_partitions(self, n):
        self.partitions = n
        return self


def test_spread_partitions_explicit_setting(monkeypatch):
    from multimodal_toolkit import config

    monkeypatch.setattr(config, "ANALYZE_NUM_PARTITIONS", 7)
    df = _FakePartitionedDf()
    assert spread_partitions(df) is df
    assert df.partitions == 7


def test_spread_partitions_native_default_is_noop(monkeypatch):
    from multimodal_toolkit import config

    monkeypatch.setattr(config, "ANALYZE_NUM_PARTITIONS", None)
    monkeypatch.setattr(config, "USE_RAY", False)
    df = _FakePartitionedDf()
    assert spread_partitions(df) is df
    assert df.partitions is None


def test_analysis_num_partitions_derives_from_ray_cpus(monkeypatch):
    import sys
    import types

    from multimodal_toolkit import config

    fake_ray = types.SimpleNamespace(
        is_initialized=lambda: True,
        cluster_resources=lambda: {"CPU": 5.0},
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(config, "ANALYZE_NUM_PARTITIONS", None)
    monkeypatch.setattr(config, "USE_RAY", True)
    assert analysis_num_partitions() == 10


def test_coalesce_for_write_merges_large_partition_counts(monkeypatch):
    from multimodal_toolkit import config

    monkeypatch.setattr(config, "ANALYZE_NUM_PARTITIONS", 64)
    df = _FakePartitionedDf()
    assert coalesce_for_write(df) is df
    assert df.partitions == 8

    monkeypatch.setattr(config, "ANALYZE_NUM_PARTITIONS", 4)
    small = _FakePartitionedDf()
    assert coalesce_for_write(small) is small
    assert small.partitions is None


def test_read_analysis_output_lance_suffix_uses_lance_reader(monkeypatch, tmp_path):
    import daft

    calls = []

    def fake_read_lance(path, io_config=None):
        calls.append(("lance", path, io_config))
        return "lance-df"

    def fake_read_json(path, io_config=None):
        calls.append(("json", path, io_config))
        return "json-df"

    monkeypatch.setattr(daft, "read_lance", fake_read_lance)
    monkeypatch.setattr(daft, "read_json", fake_read_json)

    io_config = daft_io_config()
    path = str(tmp_path / "missing.lance")
    assert read_analysis_output(path, io_config) == "lance-df"
    assert calls == [("lance", path, io_config)]
