"""End-to-end test of the image pipeline, fully local (no MinIO).

Runs the real Stage 1 → 2 → 3 → 4 → 5 chain against a temp directory:
synthetic images + a corrupt file + a missing path in the manifest, then
asserts statuses, verdicts, blob v2 storage, indexing, querying, deletion.
"""
from __future__ import annotations

import os
import pathlib
import tempfile

import cv2
import pytest

from .conftest import scrfd_model_available

pytestmark = pytest.mark.skipif(
    not scrfd_model_available() or os.getenv("SKIP_DETECTOR_TESTS"),
    reason="InsightFace model pack not available locally",
)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory) -> dict:
    """Build local test images + manifest, run analyze + ingest once."""
    from skimage import data

    from multimodal_toolkit.image.workflow.analyze import run as analyze_run
    from multimodal_toolkit.image.workflow.ingest import run as ingest_run

    tmp = tmp_path_factory.mktemp("image_e2e")
    img_dir = tmp / "images"
    img_dir.mkdir()

    face = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(img_dir / "face_sharp.jpg"), face)
    cv2.imwrite(str(img_dir / "face_blurred.jpg"), cv2.GaussianBlur(face, (31, 31), 12))
    cv2.imwrite(str(img_dir / "no_face.jpg"), cv2.cvtColor(data.coffee(), cv2.COLOR_RGB2BGR))
    (img_dir / "corrupt.jpg").write_bytes(b"this is not a valid image")

    import pyarrow as pa
    import pyarrow.parquet as pq

    doc_ids = ["face_sharp.jpg", "face_blurred.jpg", "no_face.jpg", "corrupt.jpg", "missing.jpg"]
    manifest = tmp / "manifest.parquet"
    pq.write_table(
        pa.table({"doc_id": doc_ids, "s3_url": [str(img_dir / d) for d in doc_ids]}),
        manifest,
    )

    analysis_out = str(tmp / "analysis_jsonl")
    lance_uri = str(tmp / "assets.lance")
    analyze_run(str(manifest), analysis_out)
    ingest_run(analysis_out, lance_uri)

    import daft

    rows_dict = daft.read_json(analysis_out).collect().to_pydict()
    n = len(rows_dict["doc_id"])
    analysis_rows = {rows_dict["doc_id"][i]: {k: rows_dict[k][i] for k in rows_dict} for i in range(n)}

    return {"lance_uri": lance_uri, "analysis": analysis_rows}


# ---------------------------------------------------------------------------
# Stage 1 — analyze output
# ---------------------------------------------------------------------------

def test_analyze_keeps_one_row_per_manifest_entry(pipeline):
    assert len(pipeline["analysis"]) == 5


def test_analyze_statuses(pipeline):
    statuses = {d: r["status"] for d, r in pipeline["analysis"].items()}
    assert statuses == {
        "face_sharp.jpg": "ok",
        "face_blurred.jpg": "ok",
        "no_face.jpg": "ok",
        "corrupt.jpg": "decode_failed",
        "missing.jpg": "download_failed",
    }


def test_analyze_verdicts(pipeline):
    rows = pipeline["analysis"]
    assert rows["face_sharp.jpg"]["has_face"] is True
    assert rows["face_sharp.jpg"]["is_blurry"] is False
    assert rows["face_blurred.jpg"]["is_face_blurry"] is True
    assert rows["no_face.jpg"]["has_face"] is False
    assert rows["face_sharp.jpg"]["blur_score"] > rows["face_blurred.jpg"]["blur_score"]


def test_analyze_failed_rows_have_null_verdicts(pipeline):
    rows = pipeline["analysis"]
    for doc in ("corrupt.jpg", "missing.jpg"):
        assert rows[doc]["has_face"] is None
        assert rows[doc]["is_blurry"] is None
        assert rows[doc]["is_face_blurry"] is None
        assert rows[doc]["blur_score"] is None


# ---------------------------------------------------------------------------
# Stage 2 — ingest into lance
# ---------------------------------------------------------------------------

def test_ingest_row_count_and_blob_v2(pipeline):
    import lance

    from multimodal_toolkit.storage.blob import validate_blob_v2

    assert lance.dataset(pipeline["lance_uri"]).count_rows() == 5
    validate_blob_v2(pipeline["lance_uri"], "image_blob")


def test_ingest_blob_presence_matches_status(pipeline):
    import lance

    ds = lance.dataset(pipeline["lance_uri"])
    tbl = ds.to_table(columns=["doc_id", "status"]).to_pydict()
    row_index = {d: i for i, d in enumerate(tbl["doc_id"])}

    ok_blob = ds.take_blobs("image_blob", indices=[row_index["face_sharp.jpg"]])[0]
    assert len(ok_blob.read()) > 0

    # take_blobs skips null blobs entirely — a failed row yields an empty list.
    assert ds.take_blobs("image_blob", indices=[row_index["missing.jpg"]]) == []


# ---------------------------------------------------------------------------
# Stage 3 — time index
# ---------------------------------------------------------------------------

def test_time_index(pipeline):
    import lance

    from multimodal_toolkit.workflow.index import build_time_index

    build_time_index(pipeline["lance_uri"])
    indices = lance.dataset(pipeline["lance_uri"]).list_indices()
    assert any(idx["fields"] == ["ingest_time"] for idx in indices)


# ---------------------------------------------------------------------------
# Stage 4 — query the real pipeline output
# ---------------------------------------------------------------------------

def test_query_scalar_on_pipeline_table(pipeline):
    from multimodal_toolkit.image.workflow.query import scalar_query

    rows = scalar_query(pipeline["lance_uri"], where="has_face = true AND is_blurry = false")
    assert [r["doc_id"] for r in rows] == ["face_sharp.jpg"]


def test_query_sql_status_breakdown(pipeline):
    from multimodal_toolkit.image.workflow.query import sql_query

    rows = sql_query(
        pipeline["lance_uri"],
        "SELECT status, COUNT(*) AS cnt FROM images GROUP BY status",
    )
    counts = {r["status"]: r["cnt"] for r in rows}
    assert counts == {"ok": 3, "decode_failed": 1, "download_failed": 1}


# ---------------------------------------------------------------------------
# Stage 5 — manage (runs last: mutates the table)
# ---------------------------------------------------------------------------

def test_zz_manage_delete_by_date(pipeline, local_ray):
    import lance

    from multimodal_toolkit.workflow.manage import delete_by_date

    # All rows share one ingest stamp; a past bound deletes nothing but still
    # exercises the delete + compact + cleanup path (on a real blob v2 table).
    delete_by_date(pipeline["lance_uri"], before="1970-01-01")
    assert lance.dataset(pipeline["lance_uri"]).count_rows() == 5

    delete_by_date(pipeline["lance_uri"], before="2100-01-01")
    assert lance.dataset(pipeline["lance_uri"]).count_rows() == 0
