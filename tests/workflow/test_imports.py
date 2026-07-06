"""Smoke test: workflow entry points must be importable on a clean checkout.

Guards against regressions like the removed pipeline/ module leaving dangling imports.
"""
from __future__ import annotations


def test_audio_analyze_importable():
    from multimodal_toolkit.workflow.analyze import run  # noqa: F401


def test_image_workflow_importable():
    from multimodal_toolkit.image.workflow.analyze import run as image_analyze  # noqa: F401
    from multimodal_toolkit.image.workflow.ingest import run as image_ingest  # noqa: F401
    from multimodal_toolkit.image.workflow.query import scalar_query  # noqa: F401
