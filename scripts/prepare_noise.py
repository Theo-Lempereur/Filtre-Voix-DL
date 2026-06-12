"""Prepare raw noise files into validated fixed-length chunks.

This command reads the ``noise_preprocessing`` section from
``configs/dataset_config.yaml`` and converts raw background recordings into
reusable mono WAV chunks. Short noise can optionally be repeated because
background ambience is less sensitive to padding artifacts than speech.
"""

from pathlib import Path
import sys
import random as py_random
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.dataset_builder.config import load_noise_config, NoisePreprocessingConfig
from src.dataset_builder.audio import (
    load_audio,
    save_wav,
    rms_db,
    peak,
    is_too_silent,
    limit_peak,
)
from src.dataset_builder.utils import (
    setup_logger,
    discover_audio_files,
    short_hash,
    safe_stem,
)
from src.dataset_builder.metadata import init_csv, append_rows


NOISE_METADATA_FIELDS = [
    "chunk_id",
    "source_file",
    "output_file",
    "chunk_index",
    "start_sample",
    "start_sec",
    "duration_sec",
    "sample_rate",
    "num_samples",
    "rms_db",
    "peak",
]

NOISE_ERROR_FIELDS = [
    "source_file",
    "stage",
    "error",
]


def make_noise_chunks(
    audio: np.ndarray,
    chunk_samples: int,
    repeat_short_files: bool = True,
) -> list[tuple[np.ndarray, int]]:
    """Split noise into fixed chunks, optionally repeating short files.

    Args:
        audio: Mono noise samples.
        chunk_samples: Exact number of samples required per chunk.
        repeat_short_files: Whether a short noise file should be tiled to one
            full chunk instead of being skipped.

    Returns:
        List of ``(chunk, start_sample)`` tuples. Repeated short files use
        ``0`` as the source start offset.
    """

    chunks = []
    audio_len = len(audio)

    if audio_len == 0:
        return chunks

    if audio_len < chunk_samples:
        if not repeat_short_files:
            return chunks

        repeats = int(np.ceil(chunk_samples / audio_len))
        repeated = np.tile(audio, repeats)[:chunk_samples]
        return [(repeated.astype(np.float32), 0)]

    num_full_chunks = audio_len // chunk_samples

    for idx in range(num_full_chunks):
        start = idx * chunk_samples
        end = start + chunk_samples
        chunk = audio[start:end].astype(np.float32)
        chunks.append((chunk, start))

    return chunks


def process_one_noise_file(file_path: str, cfg_dict: dict) -> dict:
    """Load, split, validate, and save chunks for one noise file.

    Args:
        file_path: Source noise file path serialized as a string for
            multiprocessing compatibility.
        cfg_dict: Plain dictionary produced by ``NoisePreprocessingConfig.to_dict``.

    Returns:
        Dictionary with ``metadata`` rows, ``errors`` rows, and ``num_chunks``.

    Notes:
        Exceptions are captured into the returned ``errors`` list so one corrupt
        source file does not stop the whole preprocessing job.
    """

    cfg = NoisePreprocessingConfig(
        input_dir=Path(cfg_dict["input_dir"]),
        output_dir=Path(cfg_dict["output_dir"]),
        metadata_dir=Path(cfg_dict["metadata_dir"]),
        logs_dir=Path(cfg_dict["logs_dir"]),
        sample_rate=cfg_dict["sample_rate"],
        mono=cfg_dict["mono"],
        chunk_duration_sec=cfg_dict["chunk_duration_sec"],
        max_files=cfg_dict["max_files"],
        shuffle_files=cfg_dict["shuffle_files"],
        random_seed=cfg_dict["random_seed"],
        repeat_short_files=cfg_dict["repeat_short_files"],
        min_duration_sec=cfg_dict["min_duration_sec"],
        silence_threshold_db=cfg_dict["silence_threshold_db"],
        min_non_silent_ratio=cfg_dict["min_non_silent_ratio"],
        peak_limit=cfg_dict["peak_limit"],
        max_workers=cfg_dict["max_workers"],
        skip_existing=cfg_dict["skip_existing"],
        allowed_extensions=tuple(cfg_dict["allowed_extensions"]),
    )

    file_path = Path(file_path)

    metadata_rows = []
    error_rows = []

    try:
        audio, sr = load_audio(
            file_path,
            target_sr=cfg.sample_rate,
            mono=cfg.mono,
        )

        duration_sec = len(audio) / cfg.sample_rate

        if duration_sec < cfg.min_duration_sec:
            error_rows.append({
                "source_file": str(file_path),
                "stage": "duration_check",
                "error": f"noise_file_too_short_{duration_sec:.3f}s",
            })
            return {
                "metadata": metadata_rows,
                "errors": error_rows,
                "num_chunks": 0,
            }

        audio = limit_peak(audio, peak_limit=cfg.peak_limit)

        chunks = make_noise_chunks(
            audio=audio,
            chunk_samples=cfg.chunk_samples,
            repeat_short_files=cfg.repeat_short_files,
        )

        if not chunks:
            error_rows.append({
                "source_file": str(file_path),
                "stage": "chunking",
                "error": "no_noise_chunk_created",
            })
            return {
                "metadata": metadata_rows,
                "errors": error_rows,
                "num_chunks": 0,
            }

        file_hash = short_hash(str(file_path.resolve()))
        stem = safe_stem(file_path.name)

        valid_chunk_count = 0

        for chunk_index, (chunk, start_sample) in enumerate(chunks):
            if len(chunk) != cfg.chunk_samples:
                error_rows.append({
                    "source_file": str(file_path),
                    "stage": f"chunk_{chunk_index}",
                    "error": f"invalid_chunk_length_{len(chunk)}",
                })
                continue

            if not np.all(np.isfinite(chunk)):
                error_rows.append({
                    "source_file": str(file_path),
                    "stage": f"chunk_{chunk_index}",
                    "error": "nan_or_inf",
                })
                continue

            if is_too_silent(
                chunk,
                silence_threshold_db=cfg.silence_threshold_db,
                min_non_silent_ratio=cfg.min_non_silent_ratio,
            ):
                error_rows.append({
                    "source_file": str(file_path),
                    "stage": f"chunk_{chunk_index}",
                    "error": "noise_chunk_too_silent",
                })
                continue

            chunk = limit_peak(chunk, peak_limit=cfg.peak_limit)

            chunk_id = f"{stem}_{file_hash}_noise_{chunk_index:05d}"
            output_file = cfg.output_dir / f"{chunk_id}.wav"

            if cfg.skip_existing and output_file.exists():
                continue

            save_wav(output_file, chunk, cfg.sample_rate)

            metadata_rows.append({
                "chunk_id": chunk_id,
                "source_file": str(file_path),
                "output_file": str(output_file),
                "chunk_index": chunk_index,
                "start_sample": start_sample,
                "start_sec": round(start_sample / cfg.sample_rate, 6),
                "duration_sec": cfg.chunk_duration_sec,
                "sample_rate": cfg.sample_rate,
                "num_samples": len(chunk),
                "rms_db": round(rms_db(chunk), 4),
                "peak": round(peak(chunk), 6),
            })

            valid_chunk_count += 1

        return {
            "metadata": metadata_rows,
            "errors": error_rows,
            "num_chunks": valid_chunk_count,
        }

    except Exception as e:
        error_rows.append({
            "source_file": str(file_path),
            "stage": "exception",
            "error": repr(e),
        })

        return {
            "metadata": metadata_rows,
            "errors": error_rows,
            "num_chunks": 0,
        }


def main() -> None:
    """Run the noise preprocessing stage from dataset_config.yaml.

    Returns:
        None. The function writes WAV chunks, metadata CSV files, error CSV
        files, and timestamped logs as side effects.
    """

    config_path = PROJECT_ROOT / "configs" / "dataset_config.yaml"
    cfg = load_noise_config(config_path)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.metadata_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = cfg.logs_dir / f"prepare_noise_{timestamp}.log"
    metadata_file = cfg.metadata_dir / "noise_metadata.csv"
    errors_file = cfg.metadata_dir / "noise_errors.csv"

    logger = setup_logger(log_file)

    logger.info("Starting NOISE preprocessing")
    logger.info(f"Input dir : {cfg.input_dir}")
    logger.info(f"Output dir : {cfg.output_dir}")
    logger.info(f"Sample rate : {cfg.sample_rate}")
    logger.info(f"Chunk duration : {cfg.chunk_duration_sec}s")
    logger.info(f"Workers : {cfg.max_workers}")

    init_csv(metadata_file, NOISE_METADATA_FIELDS)
    init_csv(errors_file, NOISE_ERROR_FIELDS)

    files = discover_audio_files(cfg.input_dir, cfg.allowed_extensions)

    if cfg.shuffle_files:
        py_random.seed(cfg.random_seed)
        py_random.shuffle(files)

    if cfg.max_files is not None:
        files = files[:cfg.max_files]

    logger.info(f"Noise files selected: {len(files)}")

    if not files:
        logger.warning("No noise audio files found.")
        return

    total_chunks = 0
    total_errors = 0

    cfg_dict = cfg.to_dict()

    if cfg.max_workers <= 1:
        for file_path in tqdm(files, desc="Preprocessing NOISE"):
            result = process_one_noise_file(str(file_path), cfg_dict)

            append_rows(metadata_file, NOISE_METADATA_FIELDS, result["metadata"])
            append_rows(errors_file, NOISE_ERROR_FIELDS, result["errors"])

            total_chunks += result["num_chunks"]
            total_errors += len(result["errors"])

    else:
        with ProcessPoolExecutor(max_workers=cfg.max_workers) as executor:
            futures = {
                executor.submit(process_one_noise_file, str(file_path), cfg_dict): file_path
                for file_path in files
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing NOISE"):
                result = future.result()

                append_rows(metadata_file, NOISE_METADATA_FIELDS, result["metadata"])
                append_rows(errors_file, NOISE_ERROR_FIELDS, result["errors"])

                total_chunks += result["num_chunks"]
                total_errors += len(result["errors"])

    logger.info("NOISE preprocessing completed")
    logger.info(f"Generated noise chunks: {total_chunks}")
    logger.info(f"Errors / skipped files: {total_errors}")
    logger.info(f"Metadata : {metadata_file}")
    logger.info(f"Errors: {errors_file}")
    logger.info(f"Logs : {log_file}")


if __name__ == "__main__":
    main()
