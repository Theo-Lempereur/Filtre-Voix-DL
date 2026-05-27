"""Lance la GUI d'orchestration des entraînements.

Usage :
    python scripts/gui.py
    # équivalent : python -m gui (depuis la racine du repo)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Permet ``python scripts/gui.py`` depuis n'importe où.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gui.app import main

if __name__ == "__main__":
    raise SystemExit(main())
