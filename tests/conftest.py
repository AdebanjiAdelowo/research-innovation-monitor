"""Shared fixtures. The toy dataset is synthetic and committed, so no test needs a download."""

from __future__ import annotations

import pytest

from rim.data.schema import RetrievalDataset
from rim.demo import build_demo_dataset


@pytest.fixture
def toy_dataset() -> RetrievalDataset:
    return build_demo_dataset()
