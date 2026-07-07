"""Stage 2 ingest tests that run fully local without model downloads."""
from __future__ import annotations

import json
from pathlib import Path

import lance

_SCORE_NULLS = {
    "width": None,
    "height": None,
    "face_count": None,
    "face_score": None,
    "face_area_ratio": None,
    "blur_score": None,
    "face_blur_score": None,
    "has_face": None,
    "is_blurry": None,
    "is_face_blurry": None,
}

_SCORE_OK = {
    "width": 32,
    "height": 32,
    "face_count": 0,
    "face_score": 0.0,
    "face_area_ratio": 0.0,
    "blur_score": 12.5,
    "face_blur_score": None,
    "has_face": False,
    "is_blurry": True,
    "is_face_blurry": False,
}


def _analysis_row(doc_id: str, s3_url: str, status: str = "ok") -> dict:
    scores = _SCORE_OK if status == "ok" else _SCORE_NULLS
    return {"doc_id": doc_id, "s3_url": s3_url, "status": status, **scores}


def _write_analysis_jsonl(path: Path, rows: list[dict]) -> str:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(path)


def _table_by_doc_id(lance_uri: str, columns: list[str]) -> dict[str, dict]:
    tbl = lance.dataset(lance_uri).to_table(columns=["doc_id", *columns]).to_pydict()
    return {
        doc_id: {c: tbl[c][i] for c in columns} for i, doc_id in enumerate(tbl["doc_id"])
    }


def test_ingest_marks_ok_rows_with_missing_blob(tmp_path):
    """Stage 1 判 ok 但 Stage 2 下载不到字节的行必须改标 blob_download_failed。"""
    from multimodal_toolkit.image.workflow.ingest import run as ingest_run

    present = tmp_path / "present.jpg"
    present.write_bytes(b"\xff\xd8fake-jpeg-bytes")
    gone = tmp_path / "gone.jpg"  # 从不落盘：模拟两阶段之间对象被删

    analysis = _write_analysis_jsonl(
        tmp_path / "analysis.jsonl",
        [
            _analysis_row("present.jpg", str(present)),
            _analysis_row("gone.jpg", str(gone)),
            _analysis_row("missing.jpg", str(tmp_path / "missing.jpg"), status="download_failed"),
        ],
    )
    assets_uri = str(tmp_path / "assets.lance")
    ingest_run(analysis, assets_uri)

    rows = _table_by_doc_id(assets_uri, ["status", "is_blurry"])
    assert rows["present.jpg"]["status"] == "ok"
    assert rows["gone.jpg"]["status"] == "blob_download_failed"
    # Stage 1 的失败状态原样保留，不被覆盖
    assert rows["missing.jpg"]["status"] == "download_failed"
    # 分析结论是 Stage 1 算出来的，blob 缺失不应抹掉它
    assert rows["gone.jpg"]["is_blurry"] is True


def test_ingest_all_failed_batch_does_not_poison_schema(tmp_path):
    """全失败批次（分数列全 null，JSON 推断为 null 类型）建表后，正常批次仍能 append。"""
    from multimodal_toolkit.image.workflow.ingest import run as ingest_run

    assets_uri = str(tmp_path / "assets.lance")

    batch1 = _write_analysis_jsonl(
        tmp_path / "batch1.jsonl",
        [
            _analysis_row("a.jpg", str(tmp_path / "a.jpg"), status="download_failed"),
            _analysis_row("b.jpg", str(tmp_path / "b.jpg"), status="download_failed"),
        ],
    )
    ingest_run(batch1, assets_uri)

    img = tmp_path / "c.jpg"
    img.write_bytes(b"\xff\xd8fake-jpeg-bytes")
    batch2 = _write_analysis_jsonl(
        tmp_path / "batch2.jsonl", [_analysis_row("c.jpg", str(img))]
    )
    ingest_run(batch2, assets_uri)

    rows = _table_by_doc_id(assets_uri, ["status", "blur_score", "has_face"])
    assert len(rows) == 3
    assert rows["c.jpg"]["status"] == "ok"
    assert rows["c.jpg"]["blur_score"] == 12.5
    assert rows["c.jpg"]["has_face"] is False
    assert rows["a.jpg"]["blur_score"] is None
