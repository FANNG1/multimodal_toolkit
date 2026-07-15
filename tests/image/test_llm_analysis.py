"""Tests for the optional OpenAI-compatible image analysis backend."""
from __future__ import annotations

import json

import cv2
import daft
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from multimodal_toolkit.image import config
from multimodal_toolkit.image.prompt import build_image_analysis_prompt
from multimodal_toolkit.image.udfs import prepare_image_for_vlm


def _image_bytes(height: int = 100, width: int = 200) -> bytes:
    image = np.full((height, width, 3), 180, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def _manifest(tmp_path, rows: list[tuple[str, str]]) -> str:
    path = tmp_path / "manifest.parquet"
    pq.write_table(
        pa.table(
            {
                "doc_id": [row[0] for row in rows],
                "s3_url": [row[1] for row in rows],
            }
        ),
        path,
    )
    return str(path)


def _fake_prompt(monkeypatch, payload: str) -> None:
    def fake_prompt(messages, **kwargs):
        return daft.lit(payload)

    monkeypatch.setattr(
        "multimodal_toolkit.image.workflow.analyze.llm_prompt", fake_prompt
    )


@pytest.fixture(autouse=True)
def _vlm_config(monkeypatch):
    monkeypatch.setattr(config, "IMAGE_VLM_API_KEY", "test-key")
    monkeypatch.setattr(config, "IMAGE_VLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setattr(config, "IMAGE_VLM_MODEL", "test-vision-model")
    monkeypatch.setattr(config, "IMAGE_VLM_TIMEOUT_S", 1.0)
    monkeypatch.setattr(config, "IMAGE_VLM_MAX_RETRIES", 0)
    monkeypatch.setattr(config, "IMAGE_VLM_CONCURRENCY", 1)


def test_prompt_defines_avatar_and_clarity_contract():
    prompt = build_image_analysis_prompt()
    assert "真人单人" in prompt
    assert "整张图片" in prompt
    assert "has_face" in prompt
    assert "is_blurry" in prompt
    assert "is_face_blurry" in prompt
    assert "is_avatar" in prompt
    assert "clarity_confidence" in prompt
    assert "avatar_confidence" in prompt


def test_prepare_image_for_vlm_downscales_and_rejects_bad_bytes(monkeypatch):
    monkeypatch.setattr(config, "IMAGE_LONG_EDGE", 1024)
    df = daft.from_pydict(
        {"image_bytes": [_image_bytes(500, 2000), _image_bytes(100, 200), b"bad"]}
    )
    df = df.with_column("prep", prepare_image_for_vlm(daft.col("image_bytes")))
    rows = df.collect().to_pydict()["prep"]

    assert (rows[0]["width"], rows[0]["height"]) == (2000, 500)
    resized = cv2.imdecode(np.frombuffer(rows[0]["vlm_image_bytes"], np.uint8), cv2.IMREAD_COLOR)
    assert resized.shape[:2] == (256, 1024)

    small = cv2.imdecode(np.frombuffer(rows[1]["vlm_image_bytes"], np.uint8), cv2.IMREAD_COLOR)
    assert small.shape[:2] == (100, 200)
    assert rows[2] == {"width": None, "height": None, "vlm_image_bytes": None}


def test_llm_analyze_writes_typed_results_and_preserves_failed_rows(monkeypatch, tmp_path):
    from multimodal_toolkit.image.workflow.analyze import run

    image = tmp_path / "avatar.png"
    image.write_bytes(_image_bytes())
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"not an image")
    missing = tmp_path / "missing.jpg"
    manifest = _manifest(
        tmp_path,
        [("avatar", str(image)), ("corrupt", str(corrupt)), ("missing", str(missing))],
    )
    _fake_prompt(
        monkeypatch,
        json.dumps(
            {
                "has_face": True,
                "is_blurry": False,
                "is_face_blurry": False,
                "is_avatar": True,
                "clarity_confidence": 0.96,
                "avatar_confidence": 0.91,
                "reason": "图片清晰，单个真人为主体",
            }
        ),
    )

    # Proves that --use-llm does not initialize the local SCRFD detector.
    monkeypatch.setattr(
        "multimodal_toolkit.image.detector.get_detector",
        lambda: (_ for _ in ()).throw(AssertionError("local detector loaded")),
    )

    out = tmp_path / "analysis.jsonl"
    run(manifest, str(out), use_llm=True)
    data = daft.read_json(str(out)).collect().to_pydict()
    by_id = {doc_id: i for i, doc_id in enumerate(data["doc_id"])}

    avatar = by_id["avatar"]
    assert data["status"][avatar] == "ok"
    assert data["has_face"][avatar] is True
    assert data["is_blurry"][avatar] is False
    assert data["is_face_blurry"][avatar] is False
    assert data["is_avatar"][avatar] is True
    assert data["analysis_backend"][avatar] == "llm"
    assert data["clarity_confidence"][avatar] == pytest.approx(0.96)
    assert data["avatar_confidence"][avatar] == pytest.approx(0.91)
    assert data["llm_reason"][avatar] == "图片清晰，单个真人为主体"
    assert data["face_count"][avatar] is None
    assert data["blur_score"][avatar] is None

    for doc_id, expected in (("corrupt", "decode_failed"), ("missing", "download_failed")):
        i = by_id[doc_id]
        assert data["status"][i] == expected
        assert data["is_avatar"][i] is None
        assert data["is_blurry"][i] is None

@pytest.mark.parametrize(
    "payload",
    [
        "not-json",
        json.dumps(
            {
                "has_face": True,
                "is_blurry": False,
                "is_face_blurry": False,
                "is_avatar": True,
                "clarity_confidence": 1.5,
                "avatar_confidence": 0.9,
                "reason": "invalid confidence",
            }
        ),
        json.dumps(
            {
                "has_face": True,
                "is_blurry": "false",
                "is_face_blurry": False,
                "is_avatar": True,
                "clarity_confidence": 0.9,
                "avatar_confidence": 0.9,
                "reason": "wrong boolean type",
            }
        ),
    ],
)
def test_invalid_llm_response_marks_row_failed(monkeypatch, tmp_path, payload):
    from multimodal_toolkit.image.workflow.analyze import run

    image = tmp_path / "image.png"
    image.write_bytes(_image_bytes())
    manifest = _manifest(tmp_path, [("image", str(image))])
    _fake_prompt(monkeypatch, payload)

    out = tmp_path / "analysis.jsonl"
    run(manifest, str(out), use_llm=True)
    row = daft.read_json(str(out)).collect().to_pydict()
    assert row["status"] == ["llm_failed"]
    assert row["is_blurry"] == [None]
    assert row["is_avatar"] == [None]
    assert row["llm_reason"] == [None]


def test_llm_mode_requires_configuration(monkeypatch, tmp_path):
    from multimodal_toolkit.image.workflow.analyze import run

    monkeypatch.setattr(config, "IMAGE_VLM_API_KEY", "")
    monkeypatch.setattr(config, "IMAGE_VLM_MODEL", "")
    with pytest.raises(ValueError, match="IMAGE_VLM_API_KEY.*IMAGE_VLM_MODEL"):
        run(str(tmp_path / "missing.parquet"), str(tmp_path / "out.jsonl"), use_llm=True)


def test_native_provider_keeps_udf_on_error_out_of_openai_request():
    from multimodal_toolkit.image.vlm import get_image_vlm_provider

    descriptor = get_image_vlm_provider().get_prompter(
        "test-vision-model",
        use_chat_completions=True,
        concurrency=1,
        on_error="ignore",
        response_format={"type": "json_object"},
        temperature=0,
    )
    assert descriptor.get_udf_options().on_error == "ignore"
    assert descriptor.get_udf_options().concurrency == 1
    prompter = descriptor.instantiate()
    assert "on_error" not in prompter.generation_config
    assert "concurrency" not in prompter.generation_config
    assert prompter.generation_config["response_format"] == {"type": "json_object"}
    assert prompter.generation_config["temperature"] == 0


def test_llm_validator_rejects_malformed_response_envelopes():
    from multimodal_toolkit.image.vlm import validate_llm_response

    for raw in (None, "", "[]", "{}", '{"has_face": "true"}'):
        assert validate_llm_response(raw)["has_face"] is None


def test_local_and_llm_batches_append_to_one_unified_table(tmp_path):
    import lance

    from multimodal_toolkit.image.workflow.ingest import run as ingest_run

    image = tmp_path / "image.jpg"
    image.write_bytes(b"\xff\xd8fake-jpeg-bytes")
    common = {
        "s3_url": str(image),
        "status": "ok",
        "width": 100,
        "height": 100,
        "face_count": 1,
        "face_score": 0.9,
        "face_area_ratio": 0.2,
        "blur_score": 200.0,
        "face_blur_score": 150.0,
        "has_face": True,
        "is_blurry": False,
        "is_face_blurry": False,
        "is_avatar": True,
    }
    local = {
        **common,
        "doc_id": "local",
        "analysis_backend": "local",
        "clarity_confidence": None,
        "avatar_confidence": None,
        "llm_reason": None,
    }
    llm = {
        **common,
        "doc_id": "llm",
        "face_count": None,
        "face_score": None,
        "face_area_ratio": None,
        "blur_score": None,
        "face_blur_score": None,
        "analysis_backend": "llm",
        "clarity_confidence": 0.95,
        "avatar_confidence": 0.9,
        "llm_reason": "清晰的单人头像",
    }
    local_path = tmp_path / "local.jsonl"
    llm_path = tmp_path / "llm.jsonl"
    local_path.write_text(json.dumps(local) + "\n")
    llm_path.write_text(json.dumps(llm) + "\n")

    assets = str(tmp_path / "assets.lance")
    ingest_run(str(local_path), assets)
    ingest_run(str(llm_path), assets)
    rows = lance.dataset(assets).to_table(
        columns=["doc_id", "analysis_backend", "is_avatar", "llm_reason"]
    ).to_pylist()
    assert {row["analysis_backend"] for row in rows} == {"local", "llm"}
    assert {row["doc_id"] for row in rows} == {"local", "llm"}
