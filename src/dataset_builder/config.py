"""Typed YAML configuration loaders for every dataset pipeline stage.

The project stores user-editable settings in ``configs/dataset_config.yaml``.
This module converts those YAML dictionaries into typed dataclasses used by the
preprocessing and generation scripts. Relative paths are resolved from the
repository root so commands can be launched from a consistent project context.
"""

from dataclasses import dataclass
from pathlib import Path
import yaml

from src import config as project_config


def _expand_drive_project(value: str) -> str:
    """Substitue ``${DRIVE_PROJECT}`` par ``src.config.DRIVE_PROJECT``.

    Permet de pointer les sources brutes dans le YAML sans coder en dur de
    chemin spécifique à une machine (Drive Desktop Win/Mac/Linux ou Colab
    sont résolus automatiquement par ``src.config``).
    """
    return value.replace("${DRIVE_PROJECT}", project_config.DRIVE_PROJECT)


@dataclass
class CleanPreprocessingConfig:
    """Settings for converting raw clean speech files into prepared chunks.

    Attributes:
        input_dir: Folder containing raw clean speech recordings.
        output_dir: Folder where prepared clean chunks are written.
        metadata_dir: Folder where clean metadata and errors are written.
        logs_dir: Folder where preprocessing logs are written.
        sample_rate: Target sample rate in Hz.
        mono: Whether multichannel files should be downmixed to mono.
        chunk_duration_sec: Duration of each fixed chunk in seconds.
        max_files: Optional cap on the number of source files to process.
        shuffle_files: Whether to shuffle source files before applying max_files.
        random_seed: Seed used for deterministic file ordering.
        pad_short_files: Whether short speech files should be zero-padded.
        normalize_rms: Whether to normalize clean speech loudness.
        target_rms_db: Target RMS level for clean speech normalization.
        max_gain_db: Maximum gain allowed during normalization.
        peak_limit: Maximum accepted absolute amplitude.
        min_duration_sec: Minimum source duration unless padding is enabled.
        silence_threshold_db: Threshold used by silence detection.
        min_non_silent_ratio: Minimum active-sample ratio accepted per chunk.
        max_workers: Number of preprocessing worker processes.
        skip_existing: Whether already written chunks should be skipped.
        allowed_extensions: Source file extensions accepted by discovery.
    """
    input_dir: Path
    output_dir: Path
    metadata_dir: Path
    logs_dir: Path

    sample_rate: int = 16000
    mono: bool = True
    chunk_duration_sec: float = 3.0

    max_files: int | None = None
    shuffle_files: bool = True
    random_seed: int = 2204

    pad_short_files: bool = False

    normalize_rms: bool = True
    target_rms_db: float = -25.0
    max_gain_db: float = 20.0
    peak_limit: float = 0.98

    min_duration_sec: float = 3.0
    silence_threshold_db: float = -50.0
    min_non_silent_ratio: float = 0.05

    max_workers: int = 2
    skip_existing: bool = True

    allowed_extensions: tuple = (
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".m4a",
        ".aac",
    )

    @property
    def chunk_samples(self) -> int:
        """Return the configured chunk duration as a number of samples.

        Returns:
            Integer sample count computed from ``sample_rate`` and
            ``chunk_duration_sec``.
        """
        return int(self.sample_rate * self.chunk_duration_sec)

    def to_dict(self) -> dict:
        """Return a multiprocessing-friendly dictionary representation.

        Returns:
            Plain dictionary containing serializable values. Paths are converted
            to strings so the object can be passed to subprocess workers.
        """
        return {
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "metadata_dir": str(self.metadata_dir),
            "logs_dir": str(self.logs_dir),
            "sample_rate": self.sample_rate,
            "mono": self.mono,
            "chunk_duration_sec": self.chunk_duration_sec,
            "max_files": self.max_files,
            "shuffle_files": self.shuffle_files,
            "random_seed": self.random_seed,
            "pad_short_files": self.pad_short_files,
            "normalize_rms": self.normalize_rms,
            "target_rms_db": self.target_rms_db,
            "max_gain_db": self.max_gain_db,
            "peak_limit": self.peak_limit,
            "min_duration_sec": self.min_duration_sec,
            "silence_threshold_db": self.silence_threshold_db,
            "min_non_silent_ratio": self.min_non_silent_ratio,
            "max_workers": self.max_workers,
            "skip_existing": self.skip_existing,
            "allowed_extensions": list(self.allowed_extensions),
        }


def load_clean_config(config_path: str | Path) -> CleanPreprocessingConfig:
    """Load clean speech preprocessing settings from YAML.

    Args:
        config_path: Path to ``dataset_config.yaml``.

    Returns:
        ``CleanPreprocessingConfig`` with resolved paths and typed values.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        KeyError: If the required ``clean_preprocessing`` section or required
            path keys are missing.
    """
    config_path = Path(config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    project_root = config_path.parents[1]

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = raw["clean_preprocessing"]

    def resolve_path(path_value: str) -> Path:
        """Resolve one clean preprocessing path from YAML.

        Args:
            path_value: Absolute or project-relative path string.

        Returns:
            Resolved ``Path`` instance.
        """
        path = Path(_expand_drive_project(path_value))
        if path.is_absolute():
            return path
        return project_root / path

    return CleanPreprocessingConfig(
        input_dir=resolve_path(cfg["input_dir"]),
        output_dir=resolve_path(cfg["output_dir"]),
        metadata_dir=resolve_path(cfg["metadata_dir"]),
        logs_dir=resolve_path(cfg["logs_dir"]),

        sample_rate=int(cfg.get("sample_rate", 16000)),
        mono=bool(cfg.get("mono", True)),
        chunk_duration_sec=float(cfg.get("chunk_duration_sec", 3.0)),

        max_files=cfg.get("max_files", None),
        shuffle_files=bool(cfg.get("shuffle_files", True)),
        random_seed=int(cfg.get("random_seed", 42)),

        pad_short_files=bool(cfg.get("pad_short_files", False)),

        normalize_rms=bool(cfg.get("normalize_rms", True)),
        target_rms_db=float(cfg.get("target_rms_db", -25.0)),
        max_gain_db=float(cfg.get("max_gain_db", 20.0)),
        peak_limit=float(cfg.get("peak_limit", 0.98)),

        min_duration_sec=float(cfg.get("min_duration_sec", 3.0)),
        silence_threshold_db=float(cfg.get("silence_threshold_db", -50.0)),
        min_non_silent_ratio=float(cfg.get("min_non_silent_ratio", 0.05)),

        max_workers=int(cfg.get("max_workers", 2)),
        skip_existing=bool(cfg.get("skip_existing", True)),

        allowed_extensions=tuple(
            ext.lower() for ext in cfg.get(
                "allowed_extensions",
                [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"]
            )
        ),
    )

@dataclass
class NoisePreprocessingConfig:
    """Settings for converting raw noise files into prepared chunks.

    Attributes:
        input_dir: Folder containing raw background noise recordings.
        output_dir: Folder where prepared noise chunks are written.
        metadata_dir: Folder where noise metadata and errors are written.
        logs_dir: Folder where preprocessing logs are written.
        sample_rate: Target sample rate in Hz.
        mono: Whether multichannel noise should be downmixed to mono.
        chunk_duration_sec: Duration of each fixed chunk in seconds.
        max_files: Optional cap on the number of source files to process.
        shuffle_files: Whether to shuffle source files before applying max_files.
        random_seed: Seed used for deterministic file ordering.
        repeat_short_files: Whether short noise files can be tiled to one chunk.
        min_duration_sec: Minimum source duration before rejection.
        silence_threshold_db: Threshold used by silence detection.
        min_non_silent_ratio: Minimum active-sample ratio accepted per chunk.
        peak_limit: Maximum accepted absolute amplitude.
        max_workers: Number of preprocessing worker processes.
        skip_existing: Whether already written chunks should be skipped.
        allowed_extensions: Source file extensions accepted by discovery.
    """
    input_dir: Path
    output_dir: Path
    metadata_dir: Path
    logs_dir: Path

    sample_rate: int = 16000
    mono: bool = True
    chunk_duration_sec: float = 3.0

    max_files: int | None = None
    shuffle_files: bool = True
    random_seed: int = 2204

    repeat_short_files: bool = True

    min_duration_sec: float = 0.5
    silence_threshold_db: float = -60.0
    min_non_silent_ratio: float = 0.03

    peak_limit: float = 0.98

    max_workers: int = 1
    skip_existing: bool = True

    allowed_extensions: tuple = (
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".m4a",
        ".aac",
    )

    @property
    def chunk_samples(self) -> int:
        """Return the configured chunk duration as a number of samples.

        Returns:
            Integer sample count computed from ``sample_rate`` and
            ``chunk_duration_sec``.
        """
        return int(self.sample_rate * self.chunk_duration_sec)

    def to_dict(self) -> dict:
        """Return a multiprocessing-friendly dictionary representation.

        Returns:
            Plain dictionary containing serializable values. Paths are converted
            to strings so the object can be passed to subprocess workers.
        """
        return {
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "metadata_dir": str(self.metadata_dir),
            "logs_dir": str(self.logs_dir),
            "sample_rate": self.sample_rate,
            "mono": self.mono,
            "chunk_duration_sec": self.chunk_duration_sec,
            "max_files": self.max_files,
            "shuffle_files": self.shuffle_files,
            "random_seed": self.random_seed,
            "repeat_short_files": self.repeat_short_files,
            "min_duration_sec": self.min_duration_sec,
            "silence_threshold_db": self.silence_threshold_db,
            "min_non_silent_ratio": self.min_non_silent_ratio,
            "peak_limit": self.peak_limit,
            "max_workers": self.max_workers,
            "skip_existing": self.skip_existing,
            "allowed_extensions": list(self.allowed_extensions),
        }


def load_noise_config(config_path: str | Path) -> NoisePreprocessingConfig:
    """Load noise preprocessing settings from YAML.

    Args:
        config_path: Path to ``dataset_config.yaml``.

    Returns:
        ``NoisePreprocessingConfig`` with resolved paths and typed values.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        KeyError: If the required ``noise_preprocessing`` section or required
            path keys are missing.
    """
    config_path = Path(config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    project_root = config_path.parents[1]

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = raw["noise_preprocessing"]

    def resolve_path(path_value: str) -> Path:
        """Resolve one noise preprocessing path from YAML.

        Args:
            path_value: Absolute or project-relative path string.

        Returns:
            Resolved ``Path`` instance.
        """
        path = Path(_expand_drive_project(path_value))
        if path.is_absolute():
            return path
        return project_root / path

    return NoisePreprocessingConfig(
        input_dir=resolve_path(cfg["input_dir"]),
        output_dir=resolve_path(cfg["output_dir"]),
        metadata_dir=resolve_path(cfg["metadata_dir"]),
        logs_dir=resolve_path(cfg["logs_dir"]),

        sample_rate=int(cfg.get("sample_rate", 16000)),
        mono=bool(cfg.get("mono", True)),
        chunk_duration_sec=float(cfg.get("chunk_duration_sec", 3.0)),

        max_files=cfg.get("max_files", None),
        shuffle_files=bool(cfg.get("shuffle_files", True)),
        random_seed=int(cfg.get("random_seed", 42)),

        repeat_short_files=bool(cfg.get("repeat_short_files", True)),

        min_duration_sec=float(cfg.get("min_duration_sec", 0.5)),
        silence_threshold_db=float(cfg.get("silence_threshold_db", -60.0)),
        min_non_silent_ratio=float(cfg.get("min_non_silent_ratio", 0.03)),

        peak_limit=float(cfg.get("peak_limit", 0.98)),

        max_workers=int(cfg.get("max_workers", 1)),
        skip_existing=bool(cfg.get("skip_existing", True)),

        allowed_extensions=tuple(
            ext.lower() for ext in cfg.get(
                "allowed_extensions",
                [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"]
            )
        ),
    )


@dataclass
class DatasetGenerationConfig:
    """Settings for generating final noisy/clean paired samples.

    Attributes:
        clean_dir: Folder containing prepared clean chunks.
        noise_dir: Folder containing prepared noise chunks.
        output_dir: Folder where generated split folders are written.
        metadata_dir: Folder where generation metadata and errors are written.
        logs_dir: Folder where generation logs are written.
        sample_rate: Expected sample rate in Hz.
        duration_sec: Duration of each generated pair.
        num_train_samples: Number of training pairs to generate.
        num_val_samples: Number of validation pairs to generate.
        num_test_samples: Number of test pairs to generate.
        snr_min_db: Lower bound for random SNR selection.
        snr_max_db: Upper bound for random SNR selection.
        min_clean_rms_db: Minimum accepted RMS for selected clean chunks.
        min_noise_rms_db: Minimum accepted RMS for selected noise chunks.
        peak_limit: Maximum accepted absolute amplitude after mixing.
        avoid_clipping: Whether mixes exceeding peak_limit are attenuated.
        apply_gain_to_target: Whether anti-clipping gain is applied to targets.
        batch_size: Number of generated samples flushed per metadata batch.
        max_workers: Number of generation worker processes.
        seed: Base deterministic seed.
        deterministic: Whether sample selection is reproducible.
        skip_existing: Whether existing output pairs should be skipped.
        save_noise: Whether to store the exact added noise in split folders.
        apply_clean_augment: Whether to augment clean speech before mixing.
        apply_noise_augment: Whether to augment noise before mixing.
        apply_post_noisy_augment: Whether to augment final noisy audio.
        post_noisy_augment_probability: Probability for post-mix augmentation.
        template_mix: Optional novice-mode template schedule.
        allowed_extensions: Generated input extensions accepted by discovery.
    """
    clean_dir: Path
    noise_dir: Path
    output_dir: Path
    metadata_dir: Path
    logs_dir: Path

    sample_rate: int = 16000
    duration_sec: float = 3.0

    num_train_samples: int = 1000
    num_val_samples: int = 100
    num_test_samples: int = 100

    snr_min_db: float = -5.0
    snr_max_db: float = 20.0

    min_clean_rms_db: float = -45.0
    min_noise_rms_db: float = -60.0

    peak_limit: float = 0.98
    avoid_clipping: bool = True
    apply_gain_to_target: bool = True

    batch_size: int = 64
    max_workers: int = 1

    seed: int = 2205
    deterministic: bool = True
    skip_existing: bool = True

    save_noise: bool = False

    apply_clean_augment: bool = True
    apply_noise_augment: bool = True
    apply_post_noisy_augment: bool = False
    post_noisy_augment_probability: float = 0.10

    template_mix: list[dict] | None = None

    # Isolation des sources entre splits : si True, partitionne les chunks
    # clean et noise sources en 3 sous-pools disjoints (train/val/test) selon
    # ``split_ratios`` avant la génération. Évite la fuite train→test.
    enforce_split_isolation: bool = True
    split_ratios: tuple = (0.8, 0.1, 0.1)

    allowed_extensions: tuple = (".wav",)

    @property
    def chunk_samples(self) -> int:
        """Return the generated pair duration as a number of samples.

        Returns:
            Integer sample count computed from ``sample_rate`` and
            ``duration_sec``.
        """
        return int(self.sample_rate * self.duration_sec)

    def to_dict(self) -> dict:
        """Return a multiprocessing-friendly dictionary representation.

        Returns:
            Plain dictionary containing serializable values used to initialize
            generation worker processes.
        """
        return {
            "clean_dir": str(self.clean_dir),
            "noise_dir": str(self.noise_dir),
            "output_dir": str(self.output_dir),
            "metadata_dir": str(self.metadata_dir),
            "logs_dir": str(self.logs_dir),
            "sample_rate": self.sample_rate,
            "duration_sec": self.duration_sec,
            "num_train_samples": self.num_train_samples,
            "num_val_samples": self.num_val_samples,
            "num_test_samples": self.num_test_samples,
            "snr_min_db": self.snr_min_db,
            "snr_max_db": self.snr_max_db,
            "min_clean_rms_db": self.min_clean_rms_db,
            "min_noise_rms_db": self.min_noise_rms_db,
            "peak_limit": self.peak_limit,
            "avoid_clipping": self.avoid_clipping,
            "apply_gain_to_target": self.apply_gain_to_target,
            "batch_size": self.batch_size,
            "max_workers": self.max_workers,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "skip_existing": self.skip_existing,
            "save_noise": self.save_noise,
            "apply_clean_augment": self.apply_clean_augment,
            "apply_noise_augment": self.apply_noise_augment,
            "apply_post_noisy_augment": self.apply_post_noisy_augment,
            "post_noisy_augment_probability": self.post_noisy_augment_probability,
            "template_mix": self.template_mix or [],
            "enforce_split_isolation": self.enforce_split_isolation,
            "split_ratios": list(self.split_ratios),
            "allowed_extensions": list(self.allowed_extensions),
        }


def load_generation_config(config_path: str | Path) -> DatasetGenerationConfig:
    """Load noisy/clean generation settings from YAML.

    Args:
        config_path: Path to ``dataset_config.yaml``.

    Returns:
        ``DatasetGenerationConfig`` with resolved paths and typed values.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        KeyError: If the required ``generation`` section or required path keys
            are missing.
    """
    config_path = Path(config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    project_root = config_path.parents[1]

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = raw["generation"]

    def resolve_path(path_value: str) -> Path:
        """Resolve one generation path from YAML.

        Args:
            path_value: Absolute or project-relative path string.

        Returns:
            Resolved ``Path`` instance.
        """
        path = Path(_expand_drive_project(path_value))
        if path.is_absolute():
            return path
        return project_root / path

    return DatasetGenerationConfig(
        clean_dir=resolve_path(cfg["clean_dir"]),
        noise_dir=resolve_path(cfg["noise_dir"]),
        output_dir=resolve_path(cfg["output_dir"]),
        metadata_dir=resolve_path(cfg["metadata_dir"]),
        logs_dir=resolve_path(cfg["logs_dir"]),

        sample_rate=int(cfg.get("sample_rate", 16000)),
        duration_sec=float(cfg.get("duration_sec", 3.0)),

        num_train_samples=int(cfg.get("num_train_samples", 1000)),
        num_val_samples=int(cfg.get("num_val_samples", 100)),
        num_test_samples=int(cfg.get("num_test_samples", 100)),

        snr_min_db=float(cfg.get("snr_min_db", -5.0)),
        snr_max_db=float(cfg.get("snr_max_db", 20.0)),

        min_clean_rms_db=float(cfg.get("min_clean_rms_db", -45.0)),
        min_noise_rms_db=float(cfg.get("min_noise_rms_db", -60.0)),

        peak_limit=float(cfg.get("peak_limit", 0.98)),
        avoid_clipping=bool(cfg.get("avoid_clipping", True)),
        apply_gain_to_target=bool(cfg.get("apply_gain_to_target", True)),

        batch_size=int(cfg.get("batch_size", 64)),
        max_workers=int(cfg.get("max_workers", 1)),

        seed=int(cfg.get("seed", 42)),
        deterministic=bool(cfg.get("deterministic", True)),
        skip_existing=bool(cfg.get("skip_existing", True)),

        save_noise=bool(cfg.get("save_noise", False)),

        apply_clean_augment=bool(cfg.get("apply_clean_augment", True)),
        apply_noise_augment=bool(cfg.get("apply_noise_augment", True)),
        apply_post_noisy_augment=bool(cfg.get("apply_post_noisy_augment", False)),
        post_noisy_augment_probability=float(
            cfg.get("post_noisy_augment_probability", 0.10)
        ),

        template_mix=cfg.get("template_mix", []) or [],

        enforce_split_isolation=bool(cfg.get("enforce_split_isolation", True)),
        split_ratios=tuple(
            float(r) for r in cfg.get("split_ratios", (0.8, 0.1, 0.1))
        ),

        allowed_extensions=tuple(
            ext.lower() for ext in cfg.get("allowed_extensions", [".wav"])
        ),
    )
