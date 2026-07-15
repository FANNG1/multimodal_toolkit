"""音频 LLM 打标的响应校验与 OpenAI-compatible provider 适配层。

与 image/vlm.py 的适配器解决同一个 Daft 0.7.15 问题（媒体隔离约定下各自
持有一份）：

1. Daft 的 prompt UDF 会把 UDF 级选项（on_error / concurrency / ...）原样
   留在 OpenAIPrompter.generation_config 里，OpenAI SDK 收到未知 kwarg 直接
   抛 TypeError——每一行都失败，整批打标全军覆没。descriptor 适配器在
   实例化 prompter 前剔除这些 UDF-only 选项，其余请求参数原样保留。
2. 非 ok 行（下载/解码失败、时长过滤）的 prompt 为 null，不能进入 Daft 的
   消息分发器（0.7.15 里一个 None 能带崩整个 morsel），也不该浪费一次 API
   调用；prompter 对含 null 的输入直接返回 null。
3. JSON mode 不提供字段类型和业务约束保证，而 Daft 0.7.15 的
   try_deserialize 遇到合法 JSON 中的错类型会终止整个批次。因此本模块在
   Python batch UDF 中逐行严格校验，坏响应降级为全 null，由 workflow 标记
   llm_failed，确保单行模型异常不会破坏“行不丢弃”语义。

历史教训：llm_prompt 的 concurrency/on_error 参数是 #18 引入的，引入即
触发泄漏、每行失败，但当时结论列有 fill_null 缺省值兜底——输出全是
downgrade_related=False，看起来像"LLM 判定为否"，回归完全不可见。
结论列保持 null + status=llm_failed 之后它才第一次暴露出来。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import daft
from daft.ai.openai.protocols.prompter import OpenAIPrompter, OpenAIPrompterDescriptor
from daft.ai.openai.provider import OpenAIProvider
from daft.ai.typing import UDFOptions

from . import config
from .udfs import ANALYSIS_DTYPE


# 校验失败时返回字段齐全、值全为 null 的 struct。不能直接依赖 Daft 的
# try_deserialize：0.7.15 遇到“JSON 语法合法但字段类型错误”的响应会抛异常，
# 一个坏响应就能终止整个批次；缺字段对象又会被解析成非 null struct，无法
# 正确标记 llm_failed。
_NULL_ANALYSIS = {
    "downgrade_related": None,
    "primary_reason": None,
    "secondary_reason": None,
    "summary": None,
    "confidence": None,
    "text_emotion": None,
    "bad_tone": None,
    "emotion_score": None,
}
_EXPECTED_FIELDS = set(_NULL_ANALYSIS)


def _valid_score(value) -> bool:
    """只接受 0 到 1 的有限实数；bool 虽是 int 子类，但不是合法分数。"""
    return (
        type(value) in (int, float)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def validate_llm_response(raw: str | None) -> dict:
    """把单条模型响应校验为规范结论，任何异常都降级为全 null。

    JSON mode 只能保证响应大概率是 JSON，不能保证字段完整、类型正确或枚举/
    分数满足业务约束。因此这里严格要求字段集合与 prompt 契约一致，避免半条
    结论被当成成功，也确保坏响应不会逃逸成 Daft 的批次级异常。
    """
    try:
        payload = json.loads(raw) if raw is not None else None
        if not isinstance(payload, dict) or set(payload) != _EXPECTED_FIELDS:
            return dict(_NULL_ANALYSIS)

        if type(payload["downgrade_related"]) is not bool:
            return dict(_NULL_ANALYSIS)
        if type(payload["bad_tone"]) is not bool:
            return dict(_NULL_ANALYSIS)
        if payload["primary_reason"] not in config.PRIMARY_REASONS:
            return dict(_NULL_ANALYSIS)
        if payload["text_emotion"] not in config.TEXT_EMOTIONS:
            return dict(_NULL_ANALYSIS)

        secondary_reason = payload["secondary_reason"]
        summary = payload["summary"]
        if not isinstance(secondary_reason, str) or not isinstance(summary, str):
            return dict(_NULL_ANALYSIS)
        if not summary.strip():
            return dict(_NULL_ANALYSIS)
        if not _valid_score(payload["confidence"]):
            return dict(_NULL_ANALYSIS)
        if not _valid_score(payload["emotion_score"]):
            return dict(_NULL_ANALYSIS)

        return {
            "downgrade_related": payload["downgrade_related"],
            "primary_reason": payload["primary_reason"],
            "secondary_reason": secondary_reason.strip(),
            "summary": summary.strip(),
            "confidence": float(payload["confidence"]),
            "text_emotion": payload["text_emotion"],
            "bad_tone": payload["bad_tone"],
            "emotion_score": float(payload["emotion_score"]),
        }
    except (TypeError, ValueError, OverflowError):
        return dict(_NULL_ANALYSIS)


@daft.func.batch(return_dtype=ANALYSIS_DTYPE)
def validate_llm_responses(raw_responses):
    """逐行校验响应，保持输入输出行数和 null 对齐。"""
    return [validate_llm_response(raw) for raw in raw_responses.to_pylist()]


@dataclass
class _OnErrorSafePrompterDescriptor(OpenAIPrompterDescriptor):
    def instantiate(self):
        prompt_options = dict(self.prompt_options)
        for name in UDFOptions.__dataclass_fields__:
            prompt_options.pop(name, None)
        return _NullSafeOpenAIPrompter(
            provider_name=self.provider_name,
            provider_options=self.provider_options,
            model=self.model_name,
            system_message=self.system_message,
            return_format=self.return_format,
            prompt_options=prompt_options,
        )


class _NullSafeOpenAIPrompter(OpenAIPrompter):
    async def prompt(self, messages):
        if any(message is None for message in messages):
            return None
        return await super().prompt(messages)


class AudioLLMProvider(OpenAIProvider):
    def get_prompter(self, model=None, return_format=None, system_message=None, **options):
        descriptor = super().get_prompter(
            model,
            return_format=return_format,
            system_message=system_message,
            **options,
        )
        return _OnErrorSafePrompterDescriptor(
            provider_name=descriptor.provider_name,
            provider_options=descriptor.provider_options,
            model_name=descriptor.model_name,
            prompt_options=descriptor.prompt_options,
            system_message=descriptor.system_message,
            return_format=descriptor.return_format,
        )


def get_audio_llm_provider() -> AudioLLMProvider:
    return AudioLLMProvider(
        name="audio-llm",
        base_url=config.DEEPSEEK_BASE_URL,
        api_key=config.DEEPSEEK_API_KEY,
    )
