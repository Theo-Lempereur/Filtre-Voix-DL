from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class CleanPreprocessingConfig:
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
        return int(self.sample_rate * self.chunk_duration_sec)

    def to_dict(self) -> dict:
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
    config_path = Path(config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config introuvable : {config_path}")

    project_root = config_path.parents[1]

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = raw["clean_preprocessing"]

    def resolve_path(path_value: str) -> Path:
        path = Path(path_value)
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
        return int(self.sample_rate * self.chunk_duration_sec)

    def to_dict(self) -> dict:
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
    config_path = Path(config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config introuvable : {config_path}")

    project_root = config_path.parents[1]

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = raw["noise_preprocessing"]

    def resolve_path(path_value: str) -> Path:
        path = Path(path_value)
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

    allowed_extensions: tuple = (".wav",)

    @property
    def chunk_samples(self) -> int:
        return int(self.sample_rate * self.duration_sec)

    def to_dict(self) -> dict:
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
            "allowed_extensions": list(self.allowed_extensions),
        }


def load_generation_config(config_path: str | Path) -> DatasetGenerationConfig:
    config_path = Path(config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config introuvable : {config_path}")

    project_root = config_path.parents[1]

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = raw["generation"]

    def resolve_path(path_value: str) -> Path:
        path = Path(path_value)
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

        allowed_extensions=tuple(
            ext.lower() for ext in cfg.get("allowed_extensions", [".wav"])
        ),
    )