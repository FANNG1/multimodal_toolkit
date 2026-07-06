from __future__ import annotations

from . import config


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
