import sys
from pathlib import Path

# Make `src` importable when pytest is invoked from anywhere (repo root,
# this directory, or an IDE test runner) without needing an installed
# package or a PYTHONPATH env var.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
