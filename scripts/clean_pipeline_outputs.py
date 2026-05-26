from pathlib import Path
import argparse
import shutil
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "dataset_config.yaml"


def load_yaml_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config introuvable : {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def get_pipeline_paths(config: dict) -> dict:
    clean_cfg = config.get("clean_preprocessing", {})
    noise_cfg = config.get("noise_preprocessing", {})
    gen_cfg = config.get("generation", {})

    return {
        "clean_chunks": resolve_path(
            clean_cfg.get("output_dir", "data/processed/clean_chunks")
        ),
        "noise_chunks": resolve_path(
            noise_cfg.get("output_dir", "data/processed/noise_chunks")
        ),
        "generated": resolve_path(
            gen_cfg.get("output_dir", "data/processed/generated")
        ),
        "metadata": resolve_path(
            gen_cfg.get("metadata_dir", "data/metadata")
        ),
        "clean_logs": resolve_path(
            clean_cfg.get("logs_dir", "data/logs/clean_preprocessing")
        ),
        "noise_logs": resolve_path(
            noise_cfg.get("logs_dir", "data/logs/noise_preprocessing")
        ),
        "generation_logs": resolve_path(
            gen_cfg.get("logs_dir", "data/logs/generation")
        ),
    }

def unique_paths(paths: list[Path]) -> list[Path]:
    seen = set()
    unique = []

    for path in paths:
        resolved = path.resolve()

        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)

    return unique

def remove_path(path: Path, dry_run: bool = False) -> None:
    if not path.exists():
        print(f"[SKIP] Introuvable : {path}")
        return

    if dry_run:
        print(f"[DRY-RUN] Supprimerait : {path}")
        return

    if path.is_dir():
        shutil.rmtree(path)
        print(f"[SUPPRIMÉ] Dossier : {path}")
    else:
        path.unlink()
        print(f"[SUPPRIMÉ] Fichier : {path}")


def recreate_folder(path: Path, dry_run: bool = False) -> None:
    if dry_run:
        print(f"[DRY-RUN] Recréerait : {path}")
        return

    path.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Dossier recréé : {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nettoie les sorties du pipeline dataset."
    )

    parser.add_argument("--all", action="store_true")
    parser.add_argument("--chunks", action="store_true")
    parser.add_argument("--clean-chunks", action="store_true")
    parser.add_argument("--noise-chunks", action="store_true")
    parser.add_argument("--generated", action="store_true")
    parser.add_argument("--metadata", action="store_true")
    parser.add_argument("--logs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    clean_chunks = args.all or args.chunks or args.clean_chunks
    noise_chunks = args.all or args.chunks or args.noise_chunks
    generated = args.all or args.generated
    metadata = args.all or args.metadata
    logs = args.all or args.logs

    if not any([clean_chunks, noise_chunks, generated, metadata, logs]):
        print("Aucune option choisie.")
        print("Exemple : python scripts/clean_pipeline_outputs.py --all")
        return

    config = load_yaml_config()
    paths = get_pipeline_paths(config)

    print("=" * 80)
    print("NETTOYAGE PIPELINE")
    print("=" * 80)
    print(f"Config utilisée : {CONFIG_PATH}")

    paths_to_remove = []

    if clean_chunks:
        paths_to_remove.append(paths["clean_chunks"])

    if noise_chunks:
        paths_to_remove.append(paths["noise_chunks"])

    if generated:
        paths_to_remove.append(paths["generated"])

    if metadata:
        paths_to_remove.append(paths["metadata"])

    if logs:
        paths_to_remove.extend([
            paths["clean_logs"],
            paths["noise_logs"],
            paths["generation_logs"],
        ])

    for path in unique_paths(paths_to_remove):
        remove_path(path, dry_run=args.dry_run)

    folders_to_recreate = []

    if clean_chunks:
        folders_to_recreate.append(paths["clean_chunks"])

    if noise_chunks:
        folders_to_recreate.append(paths["noise_chunks"])

    if generated:
        folders_to_recreate.append(paths["generated"])

    if metadata:
        folders_to_recreate.append(paths["metadata"])

    if logs:
        folders_to_recreate.extend([
            paths["clean_logs"],
            paths["noise_logs"],
            paths["generation_logs"],
        ])

    for folder in unique_paths(folders_to_recreate):
        recreate_folder(folder, dry_run=args.dry_run)

    print("=" * 80)
    print("NETTOYAGE TERMINÉ")
    print("=" * 80)


if __name__ == "__main__":
    main()