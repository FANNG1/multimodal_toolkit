from __future__ import annotations

import base64

from . import config
from .prompt import build_image_analysis_prompt


class ImageVLMClient:
    """Small synchronous client for one-row-at-a-time VLM classification.

    Daft 0.7.15 forwards its UDF ``on_error`` option to Chat Completions,
    which makes compatible endpoints reject the unexpected argument. Keeping
    the OpenAI client behind this wrapper lets us catch failures per image and
    preserve the manifest row as ``llm_failed``.
    """

    def __init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            api_key=config.IMAGE_VLM_API_KEY,
            base_url=config.IMAGE_VLM_BASE_URL,
            timeout=config.IMAGE_VLM_TIMEOUT_S,
            max_retries=config.IMAGE_VLM_MAX_RETRIES,
        )

    def analyze(self, jpeg_bytes: bytes | None) -> str | None:
        if not jpeg_bytes:
            return None
        encoded = base64.b64encode(jpeg_bytes).decode("ascii")
        try:
            response = self._client.chat.completions.create(
                model=config.IMAGE_VLM_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": build_image_analysis_prompt()},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded}"
                                },
                            },
                        ],
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=256,
            )
        except Exception:
            return None
        return response.choices[0].message.content
