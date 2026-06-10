"""Pytest configuration.

The source modules use flat, script-style imports (``from config import ...``),
so we put ``src/`` on the import path for the test session. This keeps the
existing code unchanged while still making it importable from tests/.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
