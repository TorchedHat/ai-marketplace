"""Pytest configuration for analyzer tests."""

import sys
from pathlib import Path

# Add project root (two levels up from this file)
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
