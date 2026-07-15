"""audio/llm.py 适配层测试（不发真实请求），与 tests/image/test_llm_analysis.py
的适配器测试对应：UDF 选项不得泄漏进 OpenAI 请求参数，null prompt 不得
触发 API 调用。
"""
from __future__ import annotations

import asyncio
import json

import daft


def _make_descriptor():
    from multimodal_toolkit.audio.llm import AudioLLMProvider

    provider = AudioLLMProvider(
        name="audio-llm-test", base_url="https://example.invalid/v1", api_key="test-key"
    )
    return provider.get_prompter(
        "test-chat-model",
        use_chat_completions=True,
        concurrency=8,
        on_error="ignore",
        response_format={"type": "json_object"},
        temperature=0,
    )


def test_udf_options_kept_out_of_openai_request():
    descriptor = _make_descriptor()
    # UDF 层的选项 Daft 仍要读到
    assert descriptor.get_udf_options().on_error == "ignore"
    assert descriptor.get_udf_options().concurrency == 8
    # 但不能进入发给 OpenAI SDK 的请求参数
    prompter = descriptor.instantiate()
    assert "on_error" not in prompter.generation_config
    assert "concurrency" not in prompter.generation_config
    assert prompter.generation_config["response_format"] == {"type": "json_object"}
    assert prompter.generation_config["temperature"] == 0


def test_null_prompt_skipped_without_api_call():
    prompter = _make_descriptor().instantiate()
    assert asyncio.run(prompter.prompt((None,))) is None


def _valid_analysis_payload() -> dict:
    return {
        "downgrade_related": True,
        "primary_reason": "价格敏感",
        "secondary_reason": "套餐费用过高",
        "summary": "客户希望降低套餐费用",
        "confidence": 0.92,
        "text_emotion": "不满",
        "bad_tone": True,
        "emotion_score": 0.75,
    }


def test_llm_validator_accepts_complete_typed_response():
    """完整且满足业务约束的响应必须保留类型和值。"""
    from multimodal_toolkit.audio.llm import validate_llm_response

    expected = _valid_analysis_payload()
    assert validate_llm_response(json.dumps(expected)) == expected


def test_llm_validator_rejects_bad_envelopes_without_batch_failure():
    """合法 JSON 中的错类型/缺字段也只能让该行失败，不能终止整个 morsel。"""
    from multimodal_toolkit.audio.llm import validate_llm_responses

    wrong_type = _valid_analysis_payload()
    wrong_type["downgrade_related"] = "yes"
    missing_field = _valid_analysis_payload()
    missing_field.pop("summary")
    invalid_enum = _valid_analysis_payload()
    invalid_enum["primary_reason"] = "模型自造原因"
    invalid_score = _valid_analysis_payload()
    invalid_score["confidence"] = 1.5

    raw = [
        "not-json",
        "[]",
        "{}",
        json.dumps(wrong_type),
        json.dumps(missing_field),
        json.dumps(invalid_enum),
        json.dumps(invalid_score),
    ]
    df = daft.from_pydict({"raw": raw})
    rows = df.select(validate_llm_responses(df["raw"]).alias("analysis")).to_pydict()

    assert len(rows["analysis"]) == len(raw)
    assert all(
        analysis is not None and all(value is None for value in analysis.values())
        for analysis in rows["analysis"]
    )
