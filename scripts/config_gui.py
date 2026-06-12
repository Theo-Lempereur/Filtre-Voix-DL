"""Launch the modern visual dataset configuration editor."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from gui.config_editor_modern import main


if __name__ == "__main__":
    main()
