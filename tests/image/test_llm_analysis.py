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
    class FakeVLMUDF:
        def __call__(self, image_bytes):
            return daft.lit(payload)

    monkeypatch.setattr(
        "multimodal_toolkit.image.workflow.analyze._ImageVLMUDF", FakeVLMUDF
    )


@pytest.fixture(autouse=True)
def _vlm_config(monkeypatch):
    monkeypatch.setattr(config, "IMAGE_VLM_API_KEY", "test-key")
    monkeypatch.setattr(config, "IMAGE_VLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setattr(config, "IMAGE_VLM_MODEL", "test-vision-model")
    monkeypatch.setattr(config, "IMAGE_VLM_TIMEOUT_S", 1.0)
    monkeypatch.setattr(config, "IMAGE_VLM_MAX_RETRIES", 0)


def test_prompt_defines_avatar_and_clarity_contract():
    prompt = build_image_analysis_prompt()
    assert "真人单人" in prompt
    assert "整张图片" in prompt
    assert "is_blurry" in prompt
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
                "is_blurry": False,
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
    assert data["is_blurry"][avatar] is False
    assert data["is_avatar"][avatar] is True
    assert data["clarity_confidence"][avatar] == pytest.approx(0.96)
    assert data["avatar_confidence"][avatar] == pytest.approx(0.91)
    assert data["llm_reason"][avatar] == "图片清晰，单个真人为主体"
    assert data["face_count"][avatar] is None
    assert data["blur_score"][avatar] is None
    assert data["has_face"][avatar] is None

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
                "is_blurry": False,
                "is_avatar": True,
                "clarity_confidence": 1.5,
                "avatar_confidence": 0.9,
                "reason": "invalid confidence",
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


def test_vlm_client_builds_vision_request_and_handles_errors(monkeypatch):
    from multimodal_toolkit.image.vlm import ImageVLMClient

    calls = []

    class Message:
        content = '{"is_blurry": false}'

    class Response:
        choices = [type("Choice", (), {"message": Message()})()]

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return Response()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": Completions()})()

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    client = ImageVLMClient()
    assert client.analyze(b"jpeg") == '{"is_blurry": false}'
    request = calls[0]
    assert request["model"] == "test-vision-model"
    assert request["response_format"] == {"type": "json_object"}
    assert request["messages"][0]["content"][1]["image_url"]["url"].startswith(
        "data:image/jpeg;base64,"
    )

    def fail(**kwargs):
        raise RuntimeError("API unavailable")

    client._client.chat.completions.create = fail
    assert client.analyze(b"jpeg") is None


def test_ingest_rejects_mixing_local_and_llm_tables(tmp_path):
    import lance

    from multimodal_toolkit.image.workflow.ingest import _validate_analysis_backend_schema

    local_uri = str(tmp_path / "local.lance")
    lance.write_dataset(pa.table({"doc_id": ["old"]}), local_uri)
    llm_columns = {
        "doc_id",
        "is_avatar",
        "clarity_confidence",
        "avatar_confidence",
        "llm_reason",
    }
    with pytest.raises(ValueError, match="Use a new Lance table"):
        _validate_analysis_backend_schema(local_uri, "append", llm_columns)

    llm_uri = str(tmp_path / "llm.lance")
    lance.write_dataset(
        pa.table(
            {
                "doc_id": ["old"],
                "is_avatar": [True],
                "clarity_confidence": [0.9],
                "avatar_confidence": [0.9],
                "llm_reason": ["ok"],
            }
        ),
        llm_uri,
    )
    with pytest.raises(ValueError, match="Use a new Lance table"):
        _validate_analysis_backend_schema(llm_uri, "append", {"doc_id"})
