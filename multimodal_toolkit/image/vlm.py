from __future__ import annotations

import json
import math
from dataclasses import dataclass

import daft
from daft.ai.openai.protocols.prompter import OpenAIPrompter, OpenAIPrompterDescriptor
from daft.ai.openai.provider import OpenAIProvider
from daft.ai.typing import UDFOptions

from . import config

LLM_ANALYSIS_DTYPE = daft.DataType.struct(
    {
        "has_face": daft.DataType.bool(),
        "is_blurry": daft.DataType.bool(),
        "is_face_blurry": daft.DataType.bool(),
        "is_avatar": daft.DataType.bool(),
        "clarity_confidence": daft.DataType.float64(),
        "avatar_confidence": daft.DataType.float64(),
        "reason": daft.DataType.string(),
    }
)

_NULL_ANALYSIS = {
    "has_face": None,
    "is_blurry": None,
    "is_face_blurry": None,
    "is_avatar": None,
    "clarity_confidence": None,
    "avatar_confidence": None,
    "reason": None,
}
_EXPECTED_FIELDS = set(_NULL_ANALYSIS)


def _valid_confidence(value) -> bool:
    return (
        type(value) in (int, float)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def validate_llm_response(raw: str | None) -> dict:
    """Parse one model response without allowing malformed data to escape."""
    try:
        payload = json.loads(raw) if raw is not None else None
        if not isinstance(payload, dict):
            return dict(_NULL_ANALYSIS)
        if set(payload) != _EXPECTED_FIELDS:
            return dict(_NULL_ANALYSIS)
        bool_fields = ("has_face", "is_blurry", "is_face_blurry", "is_avatar")
        if any(type(payload.get(name)) is not bool for name in bool_fields):
            return dict(_NULL_ANALYSIS)
        if payload["is_face_blurry"] and not payload["has_face"]:
            return dict(_NULL_ANALYSIS)
        if payload["is_avatar"] and (
            not payload["has_face"]
            or payload["is_blurry"]
            or payload["is_face_blurry"]
        ):
            return dict(_NULL_ANALYSIS)
        if not _valid_confidence(payload.get("clarity_confidence")):
            return dict(_NULL_ANALYSIS)
        if not _valid_confidence(payload.get("avatar_confidence")):
            return dict(_NULL_ANALYSIS)
        reason = payload.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            return dict(_NULL_ANALYSIS)
        return {
            **{name: payload[name] for name in bool_fields},
            "clarity_confidence": float(payload["clarity_confidence"]),
            "avatar_confidence": float(payload["avatar_confidence"]),
            "reason": reason.strip(),
        }
    except (TypeError, ValueError, OverflowError):
        return dict(_NULL_ANALYSIS)


@daft.func.batch(return_dtype=LLM_ANALYSIS_DTYPE)
def validate_llm_responses(raw_responses):
    return [validate_llm_response(raw) for raw in raw_responses.to_pylist()]


# Daft 0.7.15 correctly uses on_error for its UDF wrapper, but also leaves it
# in OpenAIPrompter.generation_config, where the OpenAI SDK rejects it. This
# descriptor removes UDF-only options from request kwargs while retaining the
# original descriptor options used by Daft's native prompt UDF.


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
        # Missing/download-failed/undecodable images must not enter Daft's
        # MIME dispatcher: in 0.7.15 one None can fail the entire prompt morsel.
        if any(message is None for message in messages):
            return None
        return await super().prompt(messages)


class ImageVLMProvider(OpenAIProvider):
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


def get_image_vlm_provider() -> ImageVLMProvider:
    return ImageVLMProvider(
        name="image-vlm",
        base_url=config.IMAGE_VLM_BASE_URL,
        api_key=config.IMAGE_VLM_API_KEY,
        timeout=config.IMAGE_VLM_TIMEOUT_S,
        max_retries=config.IMAGE_VLM_MAX_RETRIES,
    )
