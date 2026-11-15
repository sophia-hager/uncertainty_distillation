"""Make the ``uncertainty_distillation`` package importable from ``scripts/``.

Scripts can be run directly (``python scripts/train.py``) from any working
directory without installing the package first.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
