from pathlib import Path
import logging
import hashlib
import re


def setup_logger(log_file: Path) -> logging.Logger:
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
    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Dossier introuvable : {input_dir}")

    files = []

    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in allowed_extensions:
            files.append(path)

    return sorted(files)


def short_hash(text: str, length: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def safe_stem(filename: str) -> str:
    stem = Path(filename).stem.lower()
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", stem)
    stem = re.sub(r"_+", "_", stem)
    return stem[:60]