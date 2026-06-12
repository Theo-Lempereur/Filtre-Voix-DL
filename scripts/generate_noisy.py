"""Run the final noisy/clean dataset generation stage from the active YAML config.

The script is intentionally small: it loads the ``generation`` and
``augmentations`` sections, then delegates all heavy work to
``src.dataset_builder.dataset_generation``. It also passes the YAML path so the
generator can save ``config_used.yaml`` next to the generated dataset.
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.dataset_builder.config import load_generation_config
from src.dataset_builder.dataset_generation import (
    load_augmentation_section,
    run_dataset_generation,
)


def main() -> None:
    """Load generation and augmentation settings, then build the generated dataset.

    Returns:
        None. Generated WAV files, metadata CSV files, errors CSV files, logs,
        and ``config_used.yaml`` are written as side effects.
    """
    config_path = PROJECT_ROOT / "configs" / "dataset_config.yaml"

    cfg = load_generation_config(config_path)
    augment_cfg = load_augmentation_section(config_path)

    run_dataset_generation(
        cfg=cfg,
        augment_cfg=augment_cfg,
        config_path=config_path,
    )


if __name__ == "__main__":
    main()
