from __future__ import annotations

import random

from benchmark.audio.data import _mixed_duration


def test_mixed_duration_stays_in_supported_range():
    rng = random.Random(42)
    values = [_mixed_duration(rng) for _ in range(1000)]
    assert min(values) >= 15
    assert max(values) <= 1800
    assert any(value <= 90 for value in values)
    assert any(90 < value <= 300 for value in values)
    assert any(value > 300 for value in values)
