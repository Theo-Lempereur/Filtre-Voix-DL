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
    config_path = PROJECT_ROOT / "configs" / "dataset_config.yaml"

    cfg = load_generation_config(config_path)
    augment_cfg = load_augmentation_section(config_path)

    run_dataset_generation(
        cfg=cfg,
        augment_cfg=augment_cfg,
    )


if __name__ == "__main__":
    main()