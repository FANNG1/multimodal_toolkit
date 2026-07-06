"""Tests for audio/prompt.py — LLM instruction builder."""
from __future__ import annotations

from multimodal_toolkit.audio.prompt import build_prompt
from multimodal_toolkit.audio import config


def test_prompt_contains_transcript_and_emotion():
    prompt = build_prompt("客户想改成小套餐", "ANGRY")
    assert "客户想改成小套餐" in prompt
    assert "ANGRY" in prompt


def test_prompt_contains_enums():
    prompt = build_prompt("你好", "NEUTRAL")
    for reason in config.PRIMARY_REASONS:
        assert reason in prompt
    for emotion in config.TEXT_EMOTIONS:
        assert emotion in prompt


def test_prompt_defaults_for_empty_inputs():
    prompt = build_prompt("", "")
    assert "NEUTRAL" in prompt  # empty emotion falls back to NEUTRAL
