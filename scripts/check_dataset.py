"""Validate generated dataset structure, audio files, metadata, and template mix.

This script performs a non-destructive health check on the active dataset
configuration and generated outputs. It is intentionally warning-oriented:
small smoke-test datasets can be structurally valid even when they are too small
for final training.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import csv
import math
import statistics
import sys

import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "dataset_config.yaml"

MAX_FILES_TO_CHECK_PER_FOLDER = 200
MIN_RMS_DB = -70.0
MAX_PEAK = 1.0
SPLITS = ("train", "val", "test")


def amp_to_db(value: float) -> float:
    """Convert a linear amplitude value to dB with numerical protection.

    Args:
        value: Linear amplitude value.

    Returns:
        Decibel value after clamping near-zero input.
    """
    return 20.0 * math.log10(max(float(value), 1e-12))


def rms_db(audio: np.ndarray) -> float:
    """Compute the RMS level of an audio array in decibels.

    Args:
        audio: Audio samples.

    Returns:
        RMS level expressed in decibels.
    """
    return amp_to_db(float(np.sqrt(np.mean(audio ** 2) + 1e-12)))


def peak(audio: np.ndarray) -> float:
    """Return the absolute peak amplitude of an audio array.

    Args:
        audio: Audio samples.

    Returns:
        Maximum absolute sample amplitude, or ``0.0`` for empty input.
    """
    if len(audio) == 0:
        return 0.0
    return float(np.max(np.abs(audio)))


def load_config() -> dict:
    """Load the active dataset YAML configuration.

    Returns:
        Parsed YAML dictionary.

    Raises:
        FileNotFoundError: If ``configs/dataset_config.yaml`` is missing.
    """
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_path(path_value: str) -> Path:
    """Resolve a YAML path relative to the project root when needed.

    Args:
        path_value: Absolute or project-relative path from config.

    Returns:
        Resolved ``Path``.
    """
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def discover_wavs(folder: Path) -> list[Path]:
    """Recursively list WAV files from a folder, returning an empty list if missing.

    Args:
        folder: Folder to scan.

    Returns:
        Sorted list of ``.wav`` files.
    """
    if not folder.exists():
        return []
    return sorted(folder.rglob("*.wav"))


def fmt_number(value: float | int | None, digits: int = 2) -> str:
    """Format optional numeric values for compact terminal tables.

    Args:
        value: Number to format, or ``None``.
        digits: Decimal precision for floating-point values.

    Returns:
        Human-readable string.
    """
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def describe(values: list[float]) -> dict[str, float | None]:
    """Return min, mean, and max for a numeric series.

    Args:
        values: Numeric values to summarize.

    Returns:
        Dictionary with ``min``, ``mean``, and ``max`` keys. Values are ``None``
        when the input list is empty.
    """
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": min(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def print_section(title: str) -> None:
    """Print a consistent section header for terminal reports.

    Args:
        title: Section title displayed between separators.

    Returns:
        None.
    """
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def check_exists(path: Path, label: str, warnings: list[str], required: bool = True) -> bool:
    """Report whether an expected file or folder exists.

    Args:
        path: File or folder expected by the pipeline.
        label: Human-readable label printed in the report.
        warnings: Mutable warning list populated when the path is missing.
        required: Whether missing path should be printed as error-level.

    Returns:
        ``True`` when the path exists, otherwise ``False``.
    """
    if path.exists():
        print(f"[OK] {label}: {path}")
        return True

    level = "ERROR" if required else "WARNING"
    print(f"[{level}] {label} not found: {path}")
    warnings.append(f"{label} not found: {path}")
    return False


def read_generation_context(config: dict) -> dict:
    """Extract generation paths, counts, and validation constants from YAML.

    Args:
        config: Parsed dataset YAML dictionary.

    Returns:
        Normalized context dictionary used by later checks.

    Raises:
        ValueError: If the ``generation`` section is missing.
    """
    generation = config.get("generation", {})
    if not generation:
        raise ValueError("Missing 'generation' section in YAML.")

    sample_rate = int(generation.get("sample_rate", 16000))
    duration_sec = float(generation.get("duration_sec", 3.0))

    return {
        "sample_rate": sample_rate,
        "duration_sec": duration_sec,
        "expected_samples": int(sample_rate * duration_sec),
        "clean_dir": resolve_path(generation.get("clean_dir", "data/processed/clean_chunks")),
        "noise_dir": resolve_path(generation.get("noise_dir", "data/processed/noise_chunks")),
        "output_dir": resolve_path(generation.get("output_dir", "data/processed/generated")),
        "metadata_dir": resolve_path(generation.get("metadata_dir", "data/metadata")),
        "snr_min": float(generation.get("snr_min_db", -5.0)),
        "snr_max": float(generation.get("snr_max_db", 20.0)),
        "expected_counts": {
            "train": int(generation.get("num_train_samples", 0)),
            "val": int(generation.get("num_val_samples", 0)),
            "test": int(generation.get("num_test_samples", 0)),
        },
        "apply_clean_augment": bool(generation.get("apply_clean_augment", False)),
        "apply_noise_augment": bool(generation.get("apply_noise_augment", False)),
        "apply_post_noisy_augment": bool(generation.get("apply_post_noisy_augment", False)),
        "post_noisy_augment_probability": float(generation.get("post_noisy_augment_probability", 0.0)),
    }


def check_python_architecture(warnings: list[str]) -> None:
    """Verify that the expected dataset pipeline files still exist.

    Args:
        warnings: Mutable list receiving missing-file warnings.

    Returns:
        None.
    """
    print_section("1. Project architecture")

    required_files = [
        "configs/dataset_config.yaml",
        "scripts/prepare_clean.py",
        "scripts/prepare_noise.py",
        "scripts/generate_noisy.py",
        "scripts/check_dataset.py",
        "scripts/run_full_pipeline.py",
        "src/dataset_builder/audio.py",
        "src/dataset_builder/chunking.py",
        "src/dataset_builder/config.py",
        "src/dataset_builder/mixing.py",
        "src/dataset_builder/augment.py",
        "src/dataset_builder/dataset_generation.py",
    ]

    for file in required_files:
        check_exists(PROJECT_ROOT / file, file, warnings)


def check_config(ctx: dict, warnings: list[str]) -> None:
    """Print and sanity-check active generation settings.

    Args:
        ctx: Normalized generation context from ``read_generation_context``.
        warnings: Mutable list receiving configuration warnings.

    Returns:
        None.
    """
    print_section("2. Active configuration")

    print(f"sample_rate                  : {ctx['sample_rate']}")
    print(f"duration_sec                 : {ctx['duration_sec']}")
    print(f"expected_samples             : {ctx['expected_samples']}")
    print(f"clean_dir                    : {ctx['clean_dir']}")
    print(f"noise_dir                    : {ctx['noise_dir']}")
    print(f"output_dir                   : {ctx['output_dir']}")
    print(f"metadata_dir                 : {ctx['metadata_dir']}")
    print(f"snr_min_db / snr_max_db       : {ctx['snr_min']} / {ctx['snr_max']}")
    print(f"expected train/val/test samples : {ctx['expected_counts']}")
    print(f"augment clean/noise/post     : {ctx['apply_clean_augment']} / {ctx['apply_noise_augment']} / {ctx['apply_post_noisy_augment']}")
    print(f"post-noisy probability       : {ctx['post_noisy_augment_probability']}")

    if ctx["snr_min"] > ctx["snr_max"]:
        warnings.append("snr_min_db is greater than snr_max_db.")
    if ctx["snr_min"] < -10:
        warnings.append("snr_min_db is very low: some samples will be extremely noisy.")
    if ctx["snr_max"] > 30:
        warnings.append("snr_max_db is very high: some noise tracks may be almost inaudible.")

    total_expected = sum(ctx["expected_counts"].values())
    if total_expected < 100:
        warnings.append("Generated dataset is very small: fine for testing, weak for training.")


def check_wav_file(path: Path, expected_sr: int, expected_samples: int) -> tuple[list[str], dict[str, float]]:
    """Validate one WAV file and return errors plus basic audio stats.

    Args:
        path: WAV file to inspect.
        expected_sr: Required sample rate in Hz.
        expected_samples: Required number of frames.

    Returns:
        Tuple ``(errors, stats)``. ``errors`` contains machine-readable failure
        labels; ``stats`` contains ``rms_db`` and ``peak`` values when readable.
    """
    errors: list[str] = []
    stats = {"rms_db": float("nan"), "peak": float("nan")}

    try:
        info = sf.info(path)
        if info.samplerate != expected_sr:
            errors.append(f"sample_rate={info.samplerate}")
        if info.channels != 1:
            errors.append(f"channels={info.channels}")
        if info.frames != expected_samples:
            errors.append(f"samples={info.frames}")

        audio, _ = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim != 1:
            errors.append(f"shape={audio.shape}")
        if not np.all(np.isfinite(audio)):
            errors.append("nan_or_inf")

        p = peak(audio)
        r = rms_db(audio)
        stats["peak"] = p
        stats["rms_db"] = r

        if p > MAX_PEAK:
            errors.append(f"clipping_peak={p:.4f}")
        if r < MIN_RMS_DB:
            errors.append(f"too_silent_rms={r:.2f}dB")
    except Exception as exc:
        errors.append(f"read_error={repr(exc)}")

    return errors, stats


def check_audio_folder(folder: Path, label: str, expected_sr: int, expected_samples: int, warnings: list[str]) -> dict:
    """Validate prepared chunks inside one folder.

    Args:
        folder: Prepared clean or noise chunk folder.
        label: Folder label used in report output.
        expected_sr: Required sample rate in Hz.
        expected_samples: Required number of samples per file.
        warnings: Mutable list receiving folder-level warnings.

    Returns:
        Summary dictionary containing discovered count, checked valid count,
        checked error count, and RMS values.
    """
    print_section(f"3. Prepared audio: {label}")

    wavs = discover_wavs(folder)
    print(f"Folder              : {folder}")
    print(f"WAV files found     : {len(wavs)}")

    if not wavs:
        warnings.append(f"No WAV files found in {folder}.")
        return {"count": 0, "valid": 0, "errors": 0, "rms_values": []}

    files_to_check = wavs[:MAX_FILES_TO_CHECK_PER_FOLDER]
    valid_count = 0
    error_count = 0
    rms_values: list[float] = []

    for path in tqdm(files_to_check, desc=f"Check {label}"):
        errors, stats = check_wav_file(path, expected_sr, expected_samples)
        if np.isfinite(stats["rms_db"]):
            rms_values.append(stats["rms_db"])

        if errors:
            error_count += 1
            print(f"[ERROR] {path.name} -> {errors}")
        else:
            valid_count += 1

    rms_stats = describe(rms_values)
    print(f"Valid checked files  : {valid_count}/{len(files_to_check)}")
    print(f"Checked errors       : {error_count}")
    print(
        "RMS dB min/mean/max  : "
        f"{fmt_number(rms_stats['min'])} / {fmt_number(rms_stats['mean'])} / {fmt_number(rms_stats['max'])}"
    )

    if len(wavs) > MAX_FILES_TO_CHECK_PER_FOLDER:
        print(f"[INFO] Check limited to {MAX_FILES_TO_CHECK_PER_FOLDER} files out of {len(wavs)}")

    if error_count:
        warnings.append(f"{label} contains {error_count} invalid file(s) in the checked sample.")
    if label == "clean_chunks" and len(wavs) < 50:
        warnings.append("Few clean chunks: the model may lack speech variety.")

    return {"count": len(wavs), "valid": valid_count, "errors": error_count, "rms_values": rms_values}


def check_generated_pairs(ctx: dict, warnings: list[str]) -> dict:
    """Validate noisy/clean split folders, file pairing, and sample-level audio.

    Args:
        ctx: Normalized generation context from ``read_generation_context``.
        warnings: Mutable list receiving split and pair warnings.

    Returns:
        Per-split summary dictionary with pair counts and checked errors.
    """

    print_section("4. Generated noisy/clean pairs")

    output_dir = ctx["output_dir"]
    expected_sr = ctx["sample_rate"]
    expected_samples = ctx["expected_samples"]
    summary = {}

    for split in SPLITS:
        noisy_dir = output_dir / split / "noisy"
        clean_dir = output_dir / split / "clean"

        noisy_files = discover_wavs(noisy_dir)
        clean_files = discover_wavs(clean_dir)
        noisy_names = {path.name for path in noisy_files}
        clean_names = {path.name for path in clean_files}
        common_names = sorted(noisy_names & clean_names)
        expected_count = ctx["expected_counts"][split]

        print(f"\n--- {split} ---")
        print(f"Noisy / clean / pairs   : {len(noisy_files)} / {len(clean_files)} / {len(common_names)}")
        print(f"Expected from config    : {expected_count}")

        if expected_count and len(common_names) != expected_count:
            warnings.append(f"{split}: found {len(common_names)} pairs for {expected_count} expected.")

        missing_clean = sorted(noisy_names - clean_names)
        missing_noisy = sorted(clean_names - noisy_names)
        if missing_clean:
            warnings.append(f"{split}: {len(missing_clean)} clean files missing.")
            print(f"[ERROR] Missing clean files: {missing_clean[:10]}")
        if missing_noisy:
            warnings.append(f"{split}: {len(missing_noisy)} noisy files missing.")
            print(f"[ERROR] Missing noisy files: {missing_noisy[:10]}")

        files_to_check = common_names[:MAX_FILES_TO_CHECK_PER_FOLDER]
        valid_pairs = 0
        pair_errors = 0
        noisy_rms_values: list[float] = []
        clean_rms_values: list[float] = []

        for name in tqdm(files_to_check, desc=f"Check pairs {split}"):
            noisy_path = noisy_dir / name
            clean_path = clean_dir / name
            noisy_errors, noisy_stats = check_wav_file(noisy_path, expected_sr, expected_samples)
            clean_errors, clean_stats = check_wav_file(clean_path, expected_sr, expected_samples)

            if np.isfinite(noisy_stats["rms_db"]):
                noisy_rms_values.append(noisy_stats["rms_db"])
            if np.isfinite(clean_stats["rms_db"]):
                clean_rms_values.append(clean_stats["rms_db"])

            if noisy_errors or clean_errors:
                pair_errors += 1
                print(f"[ERROR] Invalid pair: {name}")
                print(f"  noisy : {noisy_errors}")
                print(f"  clean : {clean_errors}")
                continue

            noisy, _ = sf.read(noisy_path, dtype="float32")
            clean, _ = sf.read(clean_path, dtype="float32")
            if np.allclose(noisy, clean, atol=1e-5):
                pair_errors += 1
                print(f"[WARNING] Noisy is almost identical to clean: {name}")
                continue

            valid_pairs += 1

        noisy_rms = describe(noisy_rms_values)
        clean_rms = describe(clean_rms_values)
        print(f"Valid checked pairs      : {valid_pairs}/{len(files_to_check)}")
        print(f"Pair errors              : {pair_errors}")
        print(f"RMS noisy mean           : {fmt_number(noisy_rms['mean'])} dB")
        print(f"RMS clean mean           : {fmt_number(clean_rms['mean'])} dB")

        if pair_errors:
            warnings.append(f"{split}: {pair_errors} invalid pair(s) in the checked sample.")

        summary[split] = {
            "pairs": len(common_names),
            "valid_checked": valid_pairs,
            "errors_checked": pair_errors,
        }

    return summary


def read_csv_rows(path: Path) -> list[dict]:
    """Read a metadata CSV file and return rows as dictionaries.

    Args:
        path: CSV file path.

    Returns:
        List of row dictionaries, or an empty list when the file is missing.
    """
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def split_augments(value: str) -> list[str]:
    """Split metadata augmentation fields while ignoring codec failure markers.

    Args:
        value: Pipe-separated augmentation metadata field.

    Returns:
        List of applied augmentation names excluding ``codec_failed`` markers.
    """
    return [item for item in value.split("|") if item and item != "codec_failed"]


def check_metadata(ctx: dict, warnings: list[str]) -> None:
    """Validate generation metadata, template distribution, and config snapshot.

    Args:
        ctx: Normalized generation context from ``read_generation_context``.
        warnings: Mutable list receiving metadata warnings.

    Returns:
        None.
    """
    print_section("5. Metadata and traceability")

    metadata_dir = ctx["metadata_dir"]
    output_dir = ctx["output_dir"]
    metadata_path = metadata_dir / "generated_metadata.csv"
    errors_path = metadata_dir / "generated_errors.csv"
    config_used_path = output_dir / "config_used.yaml"

    check_exists(config_used_path, "config_used.yaml", warnings, required=False)

    rows = read_csv_rows(metadata_path)
    print(f"Metadata              : {metadata_path}")
    print(f"Metadata rows          : {len(rows)}")

    if not rows:
        warnings.append("generated_metadata.csv is missing or empty.")
    else:
        split_counts = Counter(row.get("split", "") for row in rows)
        snr_values: list[float] = []
        clean_rms_values: list[float] = []
        noisy_rms_values: list[float] = []
        missing_files = 0
        clean_aug_counts: Counter[str] = Counter()
        noise_aug_counts: Counter[str] = Counter()
        post_aug_counts: Counter[str] = Counter()
        template_counts: Counter[str] = Counter()
        codec_failures = 0

        for row in rows:
            for file_key in ("noisy_file", "clean_file"):
                file_path = Path(row.get(file_key, ""))
                if file_path and not file_path.exists():
                    missing_files += 1

            try:
                snr_values.append(float(row.get("snr_db", "")))
            except ValueError:
                pass
            try:
                clean_rms_values.append(float(row.get("clean_rms_db", "")))
            except ValueError:
                pass
            try:
                noisy_rms_values.append(float(row.get("noisy_rms_db", "")))
            except ValueError:
                pass

            clean_aug_counts.update(split_augments(row.get("clean_augmentations", "")))
            noise_aug_counts.update(split_augments(row.get("noise_augmentations", "")))
            post_aug_counts.update(split_augments(row.get("post_noisy_augmentations", "")))
            template_name = row.get("template_name", "").strip()
            if template_name:
                template_counts[template_name] += 1
            if "codec_failed" in row.get("post_noisy_augmentations", ""):
                codec_failures += 1

        snr_stats = describe(snr_values)
        clean_rms_stats = describe(clean_rms_values)
        noisy_rms_stats = describe(noisy_rms_values)

        print(f"Split distribution      : {dict(split_counts)}")
        print(
            "SNR dB min/mean/max     : "
            f"{fmt_number(snr_stats['min'])} / {fmt_number(snr_stats['mean'])} / {fmt_number(snr_stats['max'])}"
        )
        print(
            "RMS clean mean          : "
            f"{fmt_number(clean_rms_stats['mean'])} dB"
        )
        print(
            "RMS noisy mean          : "
            f"{fmt_number(noisy_rms_stats['mean'])} dB"
        )
        print(f"Top aug clean           : {clean_aug_counts.most_common(6)}")
        print(f"Top aug noise           : {noise_aug_counts.most_common(6)}")
        print(f"Top aug post-noisy      : {post_aug_counts.most_common(6)}")
        print(f"Template distribution   : {dict(template_counts)}")

        if missing_files:
            warnings.append(f"{missing_files} file(s) referenced by metadata are missing.")
        if codec_failures:
            warnings.append(f"{codec_failures} codec augmentation(s) failed. Check FFmpeg.")
        if snr_values and (min(snr_values) < ctx["snr_min"] - 2 or max(snr_values) > ctx["snr_max"] + 2):
            warnings.append("Some SNR values are outside the expected range with a 2 dB tolerance.")

    error_rows = read_csv_rows(errors_path)
    print(f"Generation errors       : {len(error_rows)}")
    if error_rows:
        warnings.append(f"{len(error_rows)} generation error(s) recorded.")
        print("Error examples:")
        for row in error_rows[:5]:
            print(row)


def print_final_summary(warnings: list[str]) -> int:
    """Print the final health-check status and return a process exit code.

    Args:
        warnings: Collected warning messages from all check stages.

    Returns:
        Process exit code. Currently returns ``0`` even with warnings because
        warnings are actionable diagnostics rather than hard failures.
    """
    print_section("6. Result")

    if warnings:
        print(f"[WARNING] {len(warnings)} item(s) to review:")
        for warning in warnings:
            print(f"- {warning}")
        print("\nCHECKUP COMPLETED WITH WARNINGS")
        return 0

    print("[OK] Dataset is coherent for the performed checks.")
    print("CHECKUP COMPLETED")
    return 0


def main() -> None:
    """Run the complete dataset health check from the active YAML config.

    Returns:
        None. Prints a structured terminal report and exits non-zero only if a
        future hard-failure status is introduced.
    """
    print("SPEECH ENHANCEMENT DATASET CHECKUP")
    print(f"Project root: {PROJECT_ROOT}")

    warnings: list[str] = []
    config = load_config()
    ctx = read_generation_context(config)

    check_python_architecture(warnings)
    check_config(ctx, warnings)

    check_audio_folder(
        folder=ctx["clean_dir"],
        label="clean_chunks",
        expected_sr=ctx["sample_rate"],
        expected_samples=ctx["expected_samples"],
        warnings=warnings,
    )

    check_audio_folder(
        folder=ctx["noise_dir"],
        label="noise_chunks",
        expected_sr=ctx["sample_rate"],
        expected_samples=ctx["expected_samples"],
        warnings=warnings,
    )

    check_generated_pairs(ctx, warnings)
    check_metadata(ctx, warnings)

    exit_code = print_final_summary(warnings)
    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
