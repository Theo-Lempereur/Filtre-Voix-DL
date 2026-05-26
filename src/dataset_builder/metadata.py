import csv
from pathlib import Path


CLEAN_METADATA_FIELDS = [
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


ERROR_METADATA_FIELDS = [
    "source_file",
    "stage",
    "error",
]


def init_csv(path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    if not rows:
        return

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        for row in rows:
            writer.writerow(row)