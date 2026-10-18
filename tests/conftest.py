"""Shared fixtures for the pihm test suite."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest


@pytest.fixture
def rng(request: pytest.FixtureRequest) -> np.random.Generator:
    """Return a generator seeded from the test's node id.

    The seed is reproducible from run to run, and independent of test order and of how
    pytest-xdist distributes the tests.
    """
    digest = hashlib.sha256(request.node.nodeid.encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))
