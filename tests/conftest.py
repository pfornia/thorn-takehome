"""Make the repo modules and Thorn's provided `model` importable from tests.

`pyproject.toml` sets `pythonpath = [".", "homework"]`, which covers the normal pytest run.
This file makes direct invocations (e.g. `pytest tests/test_autosearch.py` from a subdirectory)
work too, and keeps a single definition of where the repo root and homework directory are.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HOMEWORK_DIR = REPO_ROOT / "homework"

for d in (REPO_ROOT, HOMEWORK_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
