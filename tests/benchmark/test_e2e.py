from __future__ import annotations

import pytest


@pytest.mark.benchmark_e2e
def test_real_local_smoke():
    """Run explicitly: pytest -m benchmark_e2e tests/benchmark/test_e2e.py -s."""
    from benchmark.audio.cli import build_parser

    args = build_parser().parse_args(["local-smoke", "--count", "2", "--max-minutes", "15"])
    args.func(args)
