"""音频 Stage 2 ingest 的台账语义测试（不下载模型，全本地）。

与图片侧 tests/image/test_ingest.py 对应：blob 缺失的行改标
blob_download_failed 而不是丢弃；全 null 批次（没配 DEEPSEEK_API_KEY 时
打标列整批为 null）不能把 Lance 表的 schema 毒化成 null 类型。
"""
from __future__ import annotations

import json
from pathlib import Path

import lance

# 没配 DEEPSEEK_API_KEY 的典型 Stage 1 输出：ASR 字段有值，LLM 打标字段全 null。
_ANALYSIS_OK = {
    "duration_s": 12.5,
    "transcript": "想把128元套餐改成8元套餐",
    "acoustic_emotion": "NEUTRAL",
    "downgrade_related": None,
    "primary_reason": None,
    "secondary_reason": None,
    "summary": None,
    "confidence": None,
    "text_emotion": None,
    "bad_tone": None,
    "emotion_score": None,
}

_ANALYSIS_NULLS = {key: None for key in _ANALYSIS_OK}


def _analysis_row(doc_id: str, s3_url: str, status: str = "ok") -> dict:
    fields = _ANALYSIS_OK if status == "ok" else _ANALYSIS_NULLS
    return {"doc_id": doc_id, "s3_url": s3_url, "status": status, **fields}


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
    from multimodal_toolkit.audio.workflow.ingest import run as ingest_run

    present = tmp_path / "present.wav"
    present.write_bytes(b"RIFFfake-wav-bytes")
    gone = tmp_path / "gone.wav"  # 从不落盘：模拟两阶段之间对象被删

    analysis = _write_analysis_jsonl(
        tmp_path / "analysis.jsonl",
        [
            _analysis_row("present.wav", str(present)),
            _analysis_row("gone.wav", str(gone)),
            _analysis_row("missing.wav", str(tmp_path / "missing.wav"), status="download_failed"),
        ],
    )
    calls_uri = str(tmp_path / "calls.lance")
    ingest_run(analysis, calls_uri)

    rows = _table_by_doc_id(calls_uri, ["status", "transcript"])
    assert len(rows) == 3
    assert rows["present.wav"]["status"] == "ok"
    assert rows["gone.wav"]["status"] == "blob_download_failed"
    # Stage 1 的失败状态原样保留，不被覆盖
    assert rows["missing.wav"]["status"] == "download_failed"
    # 转写是 Stage 1 算出来的，blob 缺失不应抹掉它
    assert rows["gone.wav"]["transcript"] == _ANALYSIS_OK["transcript"]


def test_ingest_all_null_llm_batch_does_not_poison_schema(tmp_path):
    """整批打标列全 null（没配 key）建表后，正常批次仍能 append。"""
    from multimodal_toolkit.audio.workflow.ingest import run as ingest_run

    calls_uri = str(tmp_path / "calls.lance")

    batch1 = _write_analysis_jsonl(
        tmp_path / "batch1.jsonl",
        [
            _analysis_row("a.wav", str(tmp_path / "a.wav"), status="download_failed"),
            _analysis_row("b.wav", str(tmp_path / "b.wav"), status="download_failed"),
        ],
    )
    ingest_run(batch1, calls_uri)

    audio = tmp_path / "c.wav"
    audio.write_bytes(b"RIFFfake-wav-bytes")
    batch2 = _write_analysis_jsonl(
        tmp_path / "batch2.jsonl", [_analysis_row("c.wav", str(audio))]
    )
    ingest_run(batch2, calls_uri)

    rows = _table_by_doc_id(calls_uri, ["status", "transcript", "downgrade_related"])
    assert len(rows) == 3
    assert rows["c.wav"]["status"] == "ok"
    assert rows["c.wav"]["transcript"] == _ANALYSIS_OK["transcript"]
    # 没配 key 时打标结论必须保持 null（"没打标"），不能被填成 False
    assert rows["c.wav"]["downgrade_related"] is None
    assert rows["a.wav"]["transcript"] is None
