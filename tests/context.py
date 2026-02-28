# pyre-unsafe
"""Test context for importing bootstrap_stat modules."""

import os
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

import bootstrap_stat  # noqa: F401, E402
from bootstrap_stat import datasets  # noqa: F401, E402
