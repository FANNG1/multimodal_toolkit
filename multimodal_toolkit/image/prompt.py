from __future__ import annotations


def build_image_analysis_prompt() -> str:
    """Build the fixed contract used by the image VLM analysis backend."""
    return "\n".join(
        [
            "你是图片质量与头像合规分析助手。请观察图片并只输出严格 JSON，不要解释或使用 Markdown。",
            "",
            "判断标准：",
            "- is_blurry(bool)：整张图片的主要内容是否明显模糊、失焦，导致细节难以辨认。",
            "  正常压缩、轻微噪点或背景虚化不算整图模糊。",
            "- is_avatar(bool)：是否为适合作为个人头像的真人单人图片。必须只有一个真人作为主要主体，",
            "  脸部清楚可见且占画面合理比例。多人照、卡通、Logo、动物、风景、产品、背景小脸均为 false。",
            "- clarity_confidence(float)：对 is_blurry 判断的置信度，范围 0 到 1。",
            "- avatar_confidence(float)：对 is_avatar 判断的置信度，范围 0 到 1。",
            "- reason(str)：用一句简短中文同时说明清晰度与头像判断依据。",
            "",
            "JSON 必须恰好包含以下字段：is_blurry、is_avatar、clarity_confidence、",
            "avatar_confidence、reason。",
        ]
    )
