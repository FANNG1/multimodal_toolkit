"""Tests for storage/blob.py — blob v2 validation failure path.

The success path is covered end-to-end in tests/image/test_pipeline_e2e.py.
"""
from __future__ import annotations

import pathlib
import tempfile

import lance
import pyarrow as pa
import pytest

from multimodal_toolkit.storage.blob import BlobV2Error, validate_blob_v2


def test_validate_blob_v2_rejects_plain_binary():
    tmp = tempfile.mkdtemp()
    uri = str(pathlib.Path(tmp) / "plain.lance")
    lance.write_dataset(
        pa.table({"doc_id": ["a"], "audio_blob": pa.array([b"bytes"], type=pa.large_binary())}),
        uri,
    )
    with pytest.raises(BlobV2Error, match="not Lance blob v2"):
        validate_blob_v2(uri, "audio_blob")
