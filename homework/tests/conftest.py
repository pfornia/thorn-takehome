"""Make `autosearch` and `model` importable from tests.

Thorn's provided `pyproject.toml` sets `pythonpath = [".."]`, which is what lets their own
`tests/test_false_positives.py` do `from homework.model import MobileNetSmall`. That puts the
*parent* of this directory on the path, not `homework/` itself, so `import autosearch` fails.

Adding `homework/` here keeps both import styles working without editing their config.
"""

import sys
from pathlib import Path

HOMEWORK_DIR = Path(__file__).resolve().parent.parent
if str(HOMEWORK_DIR) not in sys.path:
    sys.path.insert(0, str(HOMEWORK_DIR))
