"""General-purpose helpers for dataset scripts: logging, discovery, and naming.

These utilities are intentionally small and dependency-light because they are
used by multiple command-line scripts and worker processes.
"""

from pathlib import Path
import logging
import hashlib
import re


def setup_logger(log_file: Path) -> logging.Logger:
    """Create a console-and-file logger for one dataset pipeline run.

    Args:
        log_file: Destination file for persistent logs.

    Returns:
        Logger configured with a stream handler and a file handler.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("clean_preprocessing")
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


def discover_audio_files(input_dir: Path, allowed_extensions: tuple[str, ...]) -> list[Path]:
    """Recursively list supported audio files from an input directory.

    Args:
        input_dir: Root folder to scan.
        allowed_extensions: Lowercase file extensions accepted by the pipeline.

    Returns:
        Sorted list of matching file paths.

    Raises:
        FileNotFoundError: If ``input_dir`` does not exist.
    """
    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Folder not found: {input_dir}")

    files = []

    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in allowed_extensions:
            files.append(path)

    return sorted(files)


def short_hash(text: str, length: int = 10) -> str:
    """Return a stable short SHA-1 prefix for filenames and identifiers.

    Args:
        text: Input text to hash.
        length: Number of hexadecimal characters to keep.

    Returns:
        Stable lowercase hash prefix.
    """
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def safe_stem(filename: str) -> str:
    """Convert a filename into a compact filesystem-safe lowercase stem.

    Args:
        filename: Original filename or path-like string.

    Returns:
        Lowercase stem containing only letters, digits, underscores, and dashes,
        truncated to keep generated filenames readable.
    """
    stem = Path(filename).stem.lower()
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", stem)
    stem = re.sub(r"_+", "_", stem)
    return stem[:60]
