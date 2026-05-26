from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields
from datetime import datetime
from pathlib import Path
import csv
import json
import logging
import random
import traceback

import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm

from src.dataset_builder.config import DatasetGenerationConfig
from src.dataset_builder.chunking import (
    random_crop_wav_file,
    stable_seed_from_text,
    make_rng,
)
from src.dataset_builder.mixing import (
    MixConfig,
    mix_with_random_snr,
    rms_db as compute_rms_db,
    peak as compute_peak,
)
from src.dataset_builder.augment import (
    AugmentConfig,
    apply_random_augmentations,
)
from src.dataset_builder.utils import discover_audio_files


GEN_METADATA_FIELDS = [
    "sample_id",
    "split",
    "seed",
    "noisy_file",
    "clean_file",
    "noise_file",
    "source_clean_file",
    "source_noise_file",
    "clean_start_sample",
    "noise_start_sample",
    "sample_rate",
    "duration_sec",
    "num_samples",
    "snr_db",
    "clean_rms_db",
    "noise_rms_db_before",
    "noise_rms_db_after",
    "noisy_rms_db",
    "peak_clean",
    "peak_noise",
    "peak_noisy",
    "global_gain_db",
    "clean_augmentations",
    "noise_augmentations",
    "post_noisy_augmentations",
    "clean_aug_metadata",
    "noise_aug_metadata",
    "post_noisy_aug_metadata",
]


GEN_ERROR_FIELDS = [
    "sample_id",
    "split",
    "stage",
    "source_clean_file",
    "source_noise_file",
    "error",
]


_CLEAN_FILES: list[Path] = []
_NOISE_FILES: list[Path] = []
_GEN_CFG: dict = {}
_AUGMENT_CFG: dict = {}


def setup_generation_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("dataset_generation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def init_csv_if_missing(path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    if not rows:
        return

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        for row in rows:
            safe_row = {key: row.get(key, "") for key in fieldnames}
            writer.writerow(safe_row)


def load_augmentation_section(config_path: str | Path) -> dict:
    config_path = Path(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return raw.get("augmentations", {"enabled": False})


def _init_worker(
    clean_files: list[str],
    noise_files: list[str],
    cfg_dict: dict,
    augment_cfg: dict,
) -> None:
    global _CLEAN_FILES
    global _NOISE_FILES
    global _GEN_CFG
    global _AUGMENT_CFG

    _CLEAN_FILES = [Path(p) for p in clean_files]
    _NOISE_FILES = [Path(p) for p in noise_files]
    _GEN_CFG = cfg_dict
    _AUGMENT_CFG = augment_cfg or {"enabled": False}


def _make_augment_config() -> AugmentConfig:
    valid_keys = {field.name for field in fields(AugmentConfig)}

    data = {
        key: value
        for key, value in _AUGMENT_CFG.items()
        if key in valid_keys
    }

    data["sample_rate"] = int(_GEN_CFG["sample_rate"])
    data["duration_sec"] = float(_GEN_CFG["duration_sec"])

    if "codec_choices" in data:
        data["codec_choices"] = tuple(data["codec_choices"])

    return AugmentConfig(**data)


def _make_mix_config() -> MixConfig:
    return MixConfig(
        sample_rate=int(_GEN_CFG["sample_rate"]),
        duration_sec=float(_GEN_CFG["duration_sec"]),
        snr_min_db=float(_GEN_CFG["snr_min_db"]),
        snr_max_db=float(_GEN_CFG["snr_max_db"]),
        min_clean_rms_db=float(_GEN_CFG["min_clean_rms_db"]),
        min_noise_rms_db=float(_GEN_CFG["min_noise_rms_db"]),
        peak_limit=float(_GEN_CFG["peak_limit"]),
        avoid_clipping=bool(_GEN_CFG["avoid_clipping"]),
        apply_gain_to_target=bool(_GEN_CFG["apply_gain_to_target"]),
    )


def _make_sample_seed(split: str, sample_index: int) -> int:
    base_seed = int(_GEN_CFG["seed"])

    if bool(_GEN_CFG["deterministic"]):
        text = f"{base_seed}|{split}|{sample_index}"
        return stable_seed_from_text(text, base_seed=base_seed)

    return random.SystemRandom().randint(0, 2**32 - 1)


def _output_paths(split: str, sample_id: str) -> dict[str, Path]:
    output_dir = Path(_GEN_CFG["output_dir"])
    split_dir = output_dir / split

    return {
        "noisy": split_dir / "noisy" / f"{sample_id}.wav",
        "clean": split_dir / "clean" / f"{sample_id}.wav",
        "noise": split_dir / "noise" / f"{sample_id}.wav",
    }


def _outputs_already_exist(paths: dict[str, Path]) -> bool:
    if not bool(_GEN_CFG["skip_existing"]):
        return False

    required = [
        paths["noisy"],
        paths["clean"],
    ]

    if bool(_GEN_CFG["save_noise"]):
        required.append(paths["noise"])

    return all(path.exists() for path in required)


def _save_wav_atomic(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim != 1:
        raise ValueError(f"Audio must be mono before saving. Got shape {audio.shape}")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains NaN or Inf before saving.")

    tmp_path = path.with_name(path.stem + ".tmp" + path.suffix)

    if tmp_path.exists():
        tmp_path.unlink()

    sf.write(tmp_path, audio, sample_rate, subtype="PCM_16")
    tmp_path.replace(path)


def _validate_final_pair(
    noisy: np.ndarray,
    clean: np.ndarray,
    expected_samples: int,
) -> None:
    if len(noisy) != expected_samples:
        raise ValueError(
            f"Invalid noisy length: {len(noisy)}, expected {expected_samples}"
        )

    if len(clean) != expected_samples:
        raise ValueError(
            f"Invalid clean length: {len(clean)}, expected {expected_samples}"
        )

    if noisy.ndim != 1 or clean.ndim != 1:
        raise ValueError("Final noisy and clean must be mono.")

    if not np.all(np.isfinite(noisy)):
        raise ValueError("Final noisy contains NaN or Inf.")

    if not np.all(np.isfinite(clean)):
        raise ValueError("Final clean contains NaN or Inf.")


def _generate_one_sample(split: str, sample_index: int) -> dict:
    sample_id = f"{split}_{sample_index:08d}"

    clean_path = ""
    noise_path = ""

    try:
        paths = _output_paths(split, sample_id)

        if _outputs_already_exist(paths):
            return {
                "metadata": [],
                "errors": [],
                "generated": 0,
                "skipped": 1,
            }

        sample_rate = int(_GEN_CFG["sample_rate"])
        duration_sec = float(_GEN_CFG["duration_sec"])
        expected_samples = int(sample_rate * duration_sec)

        sample_seed = _make_sample_seed(split, sample_index)

        py_rng = random.Random(sample_seed)
        np_rng = make_rng(sample_seed)

        clean_path = py_rng.choice(_CLEAN_FILES)
        noise_path = py_rng.choice(_NOISE_FILES)

        clean_crop = random_crop_wav_file(
            path=clean_path,
            crop_samples=expected_samples,
            expected_sample_rate=sample_rate,
            rng=np_rng,
            pad_if_short=False,
            pad_mode="zero",
        )

        noise_crop = random_crop_wav_file(
            path=noise_path,
            crop_samples=expected_samples,
            expected_sample_rate=sample_rate,
            rng=np_rng,
            pad_if_short=True,
            pad_mode="repeat",
        )

        clean_audio = clean_crop.audio
        noise_audio = noise_crop.audio

        augment_cfg = _make_augment_config()

        clean_aug_applied = []
        noise_aug_applied = []
        post_aug_applied = []

        clean_aug_metadata = {}
        noise_aug_metadata = {}
        post_aug_metadata = {}

        if bool(_GEN_CFG["apply_clean_augment"]):
            clean_aug = apply_random_augmentations(
                audio=clean_audio,
                cfg=augment_cfg,
                rng=py_rng,
                allow_codec=False,
                name="clean",
            )

            clean_audio = clean_aug.audio
            clean_aug_applied = clean_aug.applied
            clean_aug_metadata = clean_aug.metadata

        if bool(_GEN_CFG["apply_noise_augment"]):
            noise_aug = apply_random_augmentations(
                audio=noise_audio,
                cfg=augment_cfg,
                rng=py_rng,
                allow_codec=False,
                name="noise",
            )

            noise_audio = noise_aug.audio
            noise_aug_applied = noise_aug.applied
            noise_aug_metadata = noise_aug.metadata

        mix_cfg = _make_mix_config()

        mix_result = mix_with_random_snr(
            clean=clean_audio,
            noise=noise_audio,
            cfg=mix_cfg,
            rng=py_rng,
        )

        noisy_audio = mix_result.noisy
        clean_target = mix_result.clean_target
        noise_scaled = mix_result.noise_scaled

        if bool(_GEN_CFG["apply_post_noisy_augment"]):
            p_post = float(_GEN_CFG["post_noisy_augment_probability"])

            if py_rng.random() < p_post:
                post_aug = apply_random_augmentations(
                    audio=noisy_audio,
                    cfg=augment_cfg,
                    rng=py_rng,
                    allow_codec=True,
                    name="post_noisy",
                )

                noisy_audio = post_aug.audio
                post_aug_applied = post_aug.applied
                post_aug_metadata = post_aug.metadata

        _validate_final_pair(
            noisy=noisy_audio,
            clean=clean_target,
            expected_samples=expected_samples,
        )

        _save_wav_atomic(paths["noisy"], noisy_audio, sample_rate)
        _save_wav_atomic(paths["clean"], clean_target, sample_rate)

        noise_file_value = ""

        if bool(_GEN_CFG["save_noise"]):
            _save_wav_atomic(paths["noise"], noise_scaled, sample_rate)
            noise_file_value = str(paths["noise"])

        metadata_row = {
            "sample_id": sample_id,
            "split": split,
            "seed": sample_seed,

            "noisy_file": str(paths["noisy"]),
            "clean_file": str(paths["clean"]),
            "noise_file": noise_file_value,

            "source_clean_file": str(clean_path),
            "source_noise_file": str(noise_path),

            "clean_start_sample": clean_crop.start_sample,
            "noise_start_sample": noise_crop.start_sample,

            "sample_rate": sample_rate,
            "duration_sec": duration_sec,
            "num_samples": expected_samples,

            "snr_db": round(mix_result.snr_db, 4),
            "clean_rms_db": round(compute_rms_db(clean_target), 4),
            "noise_rms_db_before": round(mix_result.noise_rms_db_before, 4),
            "noise_rms_db_after": round(mix_result.noise_rms_db_after, 4),
            "noisy_rms_db": round(compute_rms_db(noisy_audio), 4),

            "peak_clean": round(compute_peak(clean_target), 6),
            "peak_noise": round(compute_peak(noise_scaled), 6),
            "peak_noisy": round(compute_peak(noisy_audio), 6),
            "global_gain_db": round(mix_result.global_gain_db, 4),

            "clean_augmentations": "|".join(clean_aug_applied),
            "noise_augmentations": "|".join(noise_aug_applied),
            "post_noisy_augmentations": "|".join(post_aug_applied),

            "clean_aug_metadata": json.dumps(clean_aug_metadata, ensure_ascii=False),
            "noise_aug_metadata": json.dumps(noise_aug_metadata, ensure_ascii=False),
            "post_noisy_aug_metadata": json.dumps(post_aug_metadata, ensure_ascii=False),
        }

        return {
            "metadata": [metadata_row],
            "errors": [],
            "generated": 1,
            "skipped": 0,
        }

    except Exception as e:
        error_row = {
            "sample_id": sample_id,
            "split": split,
            "stage": "generation",
            "source_clean_file": str(clean_path),
            "source_noise_file": str(noise_path),
            "error": repr(e) + "\n" + traceback.format_exc(limit=2),
        }

        return {
            "metadata": [],
            "errors": [error_row],
            "generated": 0,
            "skipped": 0,
        }


def _split_counts(cfg: DatasetGenerationConfig) -> dict[str, int]:
    return {
        "train": cfg.num_train_samples,
        "val": cfg.num_val_samples,
        "test": cfg.num_test_samples,
    }


def _ensure_output_dirs(cfg: DatasetGenerationConfig) -> None:
    for split in ["train", "val", "test"]:
        (cfg.output_dir / split / "noisy").mkdir(parents=True, exist_ok=True)
        (cfg.output_dir / split / "clean").mkdir(parents=True, exist_ok=True)

        if cfg.save_noise:
            (cfg.output_dir / split / "noise").mkdir(parents=True, exist_ok=True)


def run_dataset_generation(
    cfg: DatasetGenerationConfig,
    augment_cfg: dict | None = None,
) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.metadata_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)

    _ensure_output_dirs(cfg)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = cfg.logs_dir / f"generate_noisy_{timestamp}.log"
    metadata_file = cfg.metadata_dir / "generated_metadata.csv"
    errors_file = cfg.metadata_dir / "generated_errors.csv"

    logger = setup_generation_logger(log_file)

    init_csv_if_missing(metadata_file, GEN_METADATA_FIELDS)
    init_csv_if_missing(errors_file, GEN_ERROR_FIELDS)

    logger.info("Démarrage génération dataset noisy")
    logger.info(f"Clean dir : {cfg.clean_dir}")
    logger.info(f"Noise dir : {cfg.noise_dir}")
    logger.info(f"Output dir : {cfg.output_dir}")
    logger.info(f"Sample rate : {cfg.sample_rate}")
    logger.info(f"Duration : {cfg.duration_sec}s")
    logger.info(f"Chunk samples : {cfg.chunk_samples}")
    logger.info(f"SNR range : {cfg.snr_min_db} à {cfg.snr_max_db} dB")
    logger.info(f"Batch size : {cfg.batch_size}")
    logger.info(f"Workers : {cfg.max_workers}")
    logger.info(f"Skip existing : {cfg.skip_existing}")
    logger.info(f"Seed : {cfg.seed}")
    logger.info(f"Deterministic : {cfg.deterministic}")

    clean_files = discover_audio_files(
        cfg.clean_dir,
        cfg.allowed_extensions,
    )

    noise_files = discover_audio_files(
        cfg.noise_dir,
        cfg.allowed_extensions,
    )

    if not clean_files:
        raise RuntimeError(f"Aucun fichier clean trouvé dans : {cfg.clean_dir}")

    if not noise_files:
        raise RuntimeError(f"Aucun fichier noise trouvé dans : {cfg.noise_dir}")

    clean_files = [str(path) for path in clean_files]
    noise_files = [str(path) for path in noise_files]

    logger.info(f"Fichiers clean trouvés : {len(clean_files)}")
    logger.info(f"Fichiers noise trouvés : {len(noise_files)}")

    split_counts = _split_counts(cfg)

    cfg_dict = cfg.to_dict()
    augment_cfg = augment_cfg or {"enabled": False}

    total_generated = 0
    total_skipped = 0
    total_errors = 0

    if cfg.max_workers <= 1:
        _init_worker(
            clean_files=clean_files,
            noise_files=noise_files,
            cfg_dict=cfg_dict,
            augment_cfg=augment_cfg,
        )

        for split, count in split_counts.items():
            logger.info(f"Génération split {split} : {count} samples")

            for batch_start in range(0, count, cfg.batch_size):
                batch_end = min(batch_start + cfg.batch_size, count)
                batch_indices = range(batch_start, batch_end)

                metadata_rows = []
                error_rows = []

                for idx in tqdm(
                    batch_indices,
                    desc=f"{split} [{batch_start}:{batch_end}]",
                ):
                    result = _generate_one_sample(split, idx)

                    metadata_rows.extend(result["metadata"])
                    error_rows.extend(result["errors"])

                    total_generated += result["generated"]
                    total_skipped += result["skipped"]
                    total_errors += len(result["errors"])

                append_rows(metadata_file, GEN_METADATA_FIELDS, metadata_rows)
                append_rows(errors_file, GEN_ERROR_FIELDS, error_rows)

                logger.info(
                    f"{split} batch {batch_start}-{batch_end} | "
                    f"generated={len(metadata_rows)} | "
                    f"errors={len(error_rows)}"
                )

    else:
        with ProcessPoolExecutor(
            max_workers=cfg.max_workers,
            initializer=_init_worker,
            initargs=(clean_files, noise_files, cfg_dict, augment_cfg),
        ) as executor:

            for split, count in split_counts.items():
                logger.info(f"Génération split {split} : {count} samples")

                for batch_start in range(0, count, cfg.batch_size):
                    batch_end = min(batch_start + cfg.batch_size, count)
                    batch_indices = list(range(batch_start, batch_end))

                    futures = [
                        executor.submit(_generate_one_sample, split, idx)
                        for idx in batch_indices
                    ]

                    metadata_rows = []
                    error_rows = []

                    for future in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc=f"{split} [{batch_start}:{batch_end}]",
                    ):
                        result = future.result()

                        metadata_rows.extend(result["metadata"])
                        error_rows.extend(result["errors"])

                        total_generated += result["generated"]
                        total_skipped += result["skipped"]
                        total_errors += len(result["errors"])

                    append_rows(metadata_file, GEN_METADATA_FIELDS, metadata_rows)
                    append_rows(errors_file, GEN_ERROR_FIELDS, error_rows)

                    logger.info(
                        f"{split} batch {batch_start}-{batch_end} | "
                        f"generated={len(metadata_rows)} | "
                        f"errors={len(error_rows)}"
                    )

    logger.info("Génération terminée")
    logger.info(f"Samples générés : {total_generated}")
    logger.info(f"Samples ignorés car déjà existants : {total_skipped}")
    logger.info(f"Erreurs : {total_errors}")
    logger.info(f"Metadata : {metadata_file}")
    logger.info(f"Errors : {errors_file}")
    logger.info(f"Logs : {log_file}")