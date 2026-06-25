from __future__ import annotations

import json

from . import config

DEFAULT_ANALYSIS = {
    "downgrade_related": False,
    "primary_reason": "其他",
    "secondary_reason": "",
    "summary": "",
    "confidence": 0.0,
    "text_emotion": "未知",
    "bad_tone": False,
    "emotion_score": 0.0,
}


def build_prompt(transcript: str, acoustic_emotion: str) -> str:
    reasons = "、".join(config.PRIMARY_REASONS)
    emotions = "、".join(config.TEXT_EMOTIONS)
    return "\n".join(
        [
            "你是电信客服通话质检分析助手。请只输出严格 JSON，不要解释。",
            "",
            "字段：",
            "- downgrade_related(bool): 是否在谈套餐降档/降费/改小套餐",
            f"- primary_reason(str): 一级原因，必须从以下枚举选择：{reasons}",
            "- secondary_reason(str): 二级原因，可为空",
            "- summary(str): 一句话摘要",
            "- confidence(float): 0 到 1",
            f"- text_emotion(str): 客户情绪，必须从以下枚举选择：{emotions}",
            "- bad_tone(bool): 是否存在不满、焦急、愤怒、投诉倾向或服务风险",
            "- emotion_score(float): 0 到 1，越高越负面",
            "",
            f"SenseVoice 声学情绪标签：{acoustic_emotion or 'NEUTRAL'}",
            "",
            "通话转写：",
            transcript or "",
        ]
    )


def normalize_analysis(raw: str | dict | None) -> dict:
    data: dict = {}
    if isinstance(raw, dict):
        data = raw
    elif isinstance(raw, str) and raw.strip():
        text = raw.strip()
        fence = chr(96) * 3
        if text.startswith(fence):
            text = text[text.find("{") : text.rfind("}") + 1]
        try:
            data = json.loads(text)
        except Exception:
            data = {}

    out = dict(DEFAULT_ANALYSIS)
    for key in out:
        if key in data:
            out[key] = data[key]

    out["downgrade_related"] = bool(out["downgrade_related"])
    out["bad_tone"] = bool(out["bad_tone"])
    if out["primary_reason"] not in config.PRIMARY_REASONS:
        out["primary_reason"] = "其他"
    if out["text_emotion"] not in config.TEXT_EMOTIONS:
        out["text_emotion"] = "未知"
    for k in ("confidence", "emotion_score"):
        try:
            out[k] = max(0.0, min(1.0, float(out[k])))
        except Exception:
            out[k] = 0.0
    out["secondary_reason"] = str(out["secondary_reason"] or "")
    out["summary"] = str(out["summary"] or "")
    return out


def analyze_with_llm(transcript: str, acoustic_emotion: str) -> dict:
    if not config.DEEPSEEK_API_KEY:
        return normalize_analysis(None)

    from openai import OpenAI

    client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)
    resp = client.chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": build_prompt(transcript, acoustic_emotion)}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return normalize_analysis(resp.choices[0].message.content)
