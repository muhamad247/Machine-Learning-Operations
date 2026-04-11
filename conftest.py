"""
Makes src/ importable when pytest runs from the project root.
Automatically discovered by pytest — never called directly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
