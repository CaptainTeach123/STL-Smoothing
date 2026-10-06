"""Make ``src`` and the tests directory importable without installing the package."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "..", "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
