"""pytest configuration for the project's test suite.

Placing this file in the ``tests/`` directory has a single effect:
pytest, when collecting tests, executes it before importing any test
module. We use that hook to put the project root on ``sys.path`` so
that every test in this directory can write ``from src.xxx import ...``
without needing its own per-file ``sys.path`` manipulation.

This is the standard pytest idiom for projects whose ``src`` package
is a sibling of ``tests`` rather than an installed distribution. It
also makes the test suite robust to being launched from any working
directory (``pytest tests/`` from the project root, ``pytest`` from
``tests/`` itself, IDE-driven runs, etc.) -- anywhere the test
collector finds this file, ``src`` becomes importable.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
