"""Pytest configuration.

Ensures the local `src/` tree is importable so tests run against the working
copy instead of an unrelated installed package.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

