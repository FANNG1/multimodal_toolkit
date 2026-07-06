"""Shared fixtures for the test suite."""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def local_ray():
    """Guarantee a hermetic local Ray cluster started from this venv.

    Ray's default init prefers joining an existing cluster (RAY_ADDRESS env
    or a stray `ray start` on the machine). A foreign cluster's workers run
    a different Python env and cannot deserialize this venv's lance_ray
    functions — tests would then fail for environment reasons, not code.
    Module-scoped so Ray starts once per module.
    """
    import os

    import ray

    os.environ.pop("RAY_ADDRESS", None)
    if ray.is_initialized():
        ray.shutdown()
    ray.init(address="local", include_dashboard=False, log_to_driver=False)
    yield
    ray.shutdown()
