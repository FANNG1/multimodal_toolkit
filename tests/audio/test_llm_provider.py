"""audio/llm.py 适配层测试（不发真实请求），与 tests/image/test_llm_analysis.py
的适配器测试对应：UDF 选项不得泄漏进 OpenAI 请求参数，null prompt 不得
触发 API 调用。
"""
from __future__ import annotations

import asyncio


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
