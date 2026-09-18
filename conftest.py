"""Root conftest: ensure the repo root is on sys.path.

``seed_db.py`` sits at the repo root and is imported by
``tests/test_nk_database.py``. pytest only prepends the test directory to
``sys.path``, and PEP 660 editable installs do not expose the project root,
so make the root importable explicitly.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
