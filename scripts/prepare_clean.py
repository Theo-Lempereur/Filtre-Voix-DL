"""Prepare clean speech files into validated fixed-length chunks.

This command reads the ``clean_preprocessing`` section from
``configs/dataset_config.yaml`` and converts raw speech recordings into reusable
mono WAV chunks. It writes successful chunk metadata and per-file/per-chunk
errors to CSV so the dataset can be audited before generation.
"""

from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import random 
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.dataset_builder.config import load_clean_config, CleanPreprocessingConfig
from src.dataset_builder.audio import (
    load_audio,
    normalize_rms,
    split_into_chunks,
    save_wav,
    rms_db,
    peak,
)
from src.dataset_builder.validator import validate_loaded_audio, validate_chunk
from src.dataset_builder.utils import (
    setup_logger,
    discover_audio_files,
    short_hash,
    safe_stem,
)
from src.dataset_builder.metadata import (
    init_csv,
    append_rows,
    CLEAN_METADATA_FIELDS,
    ERROR_METADATA_FIELDS,
)


def process_one_file(file_path: str, cfg_dict: dict) -> dict:
    """Load, normalize, split, validate, and save chunks for one clean file.

    Args:
        file_path: Source clean speech file path serialized as a string for
            multiprocessing compatibility.
        cfg_dict: Plain dictionary produced by ``CleanPreprocessingConfig.to_dict``.

    Returns:
        Dictionary with three keys:
        ``metadata`` contains rows for successfully written chunks, ``errors``
        contains structured error rows, and ``num_chunks`` counts valid chunks.

    Notes:
        Exceptions are captured into the returned ``errors`` list so one corrupt
        source file does not stop the whole preprocessing job.
    """

    cfg = CleanPreprocessingConfig(
        input_dir=Path(cfg_dict["input_dir"]),
        output_dir=Path(cfg_dict["output_dir"]),
        metadata_dir=Path(cfg_dict["metadata_dir"]),
        logs_dir=Path(cfg_dict["logs_dir"]),
        sample_rate=cfg_dict["sample_rate"],
        mono=cfg_dict["mono"],
        chunk_duration_sec=cfg_dict["chunk_duration_sec"],
        pad_short_files=cfg_dict["pad_short_files"],
        normalize_rms=cfg_dict["normalize_rms"],
        target_rms_db=cfg_dict["target_rms_db"],
        max_gain_db=cfg_dict["max_gain_db"],
        peak_limit=cfg_dict["peak_limit"],
        min_duration_sec=cfg_dict["min_duration_sec"],
        silence_threshold_db=cfg_dict["silence_threshold_db"],
        min_non_silent_ratio=cfg_dict["min_non_silent_ratio"],
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

        load_errors = validate_loaded_audio(
            audio,
            sample_rate=sr,
            expected_sr=cfg.sample_rate,
        )

        if load_errors:
            error_rows.append({
                "source_file": str(file_path),
                "stage": "load_validation",
                "error": "|".join(load_errors),
            })
            return {
                "metadata": metadata_rows,
                "errors": error_rows,
                "num_chunks": 0,
            }

        duration_sec = len(audio) / cfg.sample_rate

        if duration_sec < cfg.min_duration_sec and not cfg.pad_short_files:
            error_rows.append({
                "source_file": str(file_path),
                "stage": "duration_check",
                "error": f"file_too_short_{duration_sec:.3f}s",
            })
            return {
                "metadata": metadata_rows,
                "errors": error_rows,
                "num_chunks": 0,
            }

        if cfg.normalize_rms:
            audio = normalize_rms(
                audio,
                target_rms_db=cfg.target_rms_db,
                max_gain_db=cfg.max_gain_db,
                peak_limit=cfg.peak_limit,
            )

        chunks = split_into_chunks(
            audio,
            chunk_samples=cfg.chunk_samples,
            pad_short=cfg.pad_short_files,
        )

        if not chunks:
            error_rows.append({
                "source_file": str(file_path),
                "stage": "chunking",
                "error": "no_valid_chunk_created",
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
            chunk_errors = validate_chunk(
                chunk,
                expected_samples=cfg.chunk_samples,
                silence_threshold_db=cfg.silence_threshold_db,
                min_non_silent_ratio=cfg.min_non_silent_ratio,
                peak_limit=cfg.peak_limit,
            )

            if chunk_errors:
                error_rows.append({
                    "source_file": str(file_path),
                    "stage": f"chunk_{chunk_index}",
                    "error": "|".join(chunk_errors),
                })
                continue

            chunk_id = f"{stem}_{file_hash}_chunk_{chunk_index:05d}"
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
    """Run the clean speech preprocessing stage from dataset_config.yaml.

    Returns:
        None. The function writes WAV chunks, metadata CSV files, error CSV
        files, and timestamped logs as side effects.
    """

    config_path = PROJECT_ROOT / "configs" / "dataset_config.yaml"
    cfg = load_clean_config(config_path)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.metadata_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = cfg.logs_dir / f"prepare_clean_{timestamp}.log"
    metadata_file = cfg.metadata_dir / "clean_metadata.csv"
    errors_file = cfg.metadata_dir / "clean_errors.csv"

    logger = setup_logger(log_file)

    logger.info("Starting CLEAN preprocessing")
    logger.info(f"Input dir : {cfg.input_dir}")
    logger.info(f"Output dir : {cfg.output_dir}")
    logger.info(f"Sample rate : {cfg.sample_rate}")
    logger.info(f"Chunk duration : {cfg.chunk_duration_sec}s")
    logger.info(f"Workers : {cfg.max_workers}")

    init_csv(metadata_file, CLEAN_METADATA_FIELDS)
    init_csv(errors_file, ERROR_METADATA_FIELDS)

    files = discover_audio_files(cfg.input_dir, cfg.allowed_extensions)

    if cfg.shuffle_files:
        random.seed(cfg.random_seed)
        random.shuffle(files)

    if cfg.max_files is not None:
        files = files[:cfg.max_files]

    logger.info(f"Files selected: {len(files)}")

    if not files:
        logger.warning("No audio files found.")
        return

    total_chunks = 0
    total_errors = 0

    cfg_dict = cfg.to_dict()

    if cfg.max_workers <= 1:
        for file_path in tqdm(files, desc="Preprocessing CLEAN"):
            result = process_one_file(str(file_path), cfg_dict)

            append_rows(metadata_file, CLEAN_METADATA_FIELDS, result["metadata"])
            append_rows(errors_file, ERROR_METADATA_FIELDS, result["errors"])

            total_chunks += result["num_chunks"]
            total_errors += len(result["errors"])

    else:
        with ProcessPoolExecutor(max_workers=cfg.max_workers) as executor:
            futures = {
                executor.submit(process_one_file, str(file_path), cfg_dict): file_path
                for file_path in files
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing CLEAN"):
                result = future.result()

                append_rows(metadata_file, CLEAN_METADATA_FIELDS, result["metadata"])
                append_rows(errors_file, ERROR_METADATA_FIELDS, result["errors"])

                total_chunks += result["num_chunks"]
                total_errors += len(result["errors"])

    logger.info("Preprocessing completed")
    logger.info(f"Generated chunks: {total_chunks}")
    logger.info(f"Errors / skipped files: {total_errors}")
    logger.info(f"Metadata : {metadata_file}")
    logger.info(f"Errors: {errors_file}")
    logger.info(f"Logs : {log_file}")


if __name__ == "__main__":
    main()
