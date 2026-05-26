from pathlib import Path
import sys
import csv
import math

import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "dataset_config.yaml"

MAX_FILES_TO_CHECK_PER_FOLDER = 200
MIN_RMS_DB = -70.0
MAX_PEAK = 1.0


def amp_to_db(x: float) -> float:
    return 20.0 * math.log10(max(float(x), 1e-12))


def rms_db(audio: np.ndarray) -> float:
    return amp_to_db(float(np.sqrt(np.mean(audio ** 2) + 1e-12)))


def peak(audio: np.ndarray) -> float:
    if len(audio) == 0:
        return 0.0
    return float(np.max(np.abs(audio)))


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config introuvable : {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def check_exists(path: Path, label: str) -> bool:
    if path.exists():
        print(f"[OK] {label} : {path}")
        return True

    print(f"[ERREUR] {label} introuvable : {path}")
    return False


def discover_wavs(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(folder.rglob("*.wav"))


def check_python_architecture() -> None:
    print("\n=== 1. CHECK ARCHITECTURE PYTHON ===")

    required_files = [
        "configs/dataset_config.yaml",

        "scripts/prepare_clean.py",
        "scripts/generate_noisy.py",
        "scripts/check_dataset.py",

        "src/dataset_builder/audio.py",
        "src/dataset_builder/chunking.py",
        "src/dataset_builder/config.py",
        "src/dataset_builder/mixing.py",
        "src/dataset_builder/utils.py",
    ]

    optional_files = [
        "scripts/prepare_noise.py",
        "src/dataset_builder/augment.py",
        "src/dataset_builder/metadata.py",
        "src/dataset_builder/validator.py",
        "src/dataset_builder/dataset_generation.py",
    ]

    for file in required_files:
        check_exists(PROJECT_ROOT / file, file)

    print("\nFichiers optionnels :")
    for file in optional_files:
        if (PROJECT_ROOT / file).exists():
            print(f"[OK] {file}")
        else:
            print(f"[INFO] absent : {file}")


def check_config(config: dict) -> dict:
    print("\n=== 2. CHECK CONFIG YAML ===")

    generation = config.get("generation", {})

    if not generation:
        print("[ERREUR] Section 'generation' absente du YAML")
        return {}

    sample_rate = int(generation.get("sample_rate", 16000))
    duration_sec = float(generation.get("duration_sec", 3.0))
    expected_samples = int(sample_rate * duration_sec)

    clean_dir = resolve_path(generation.get("clean_dir", "data/processed/clean_chunks"))
    noise_dir = resolve_path(generation.get("noise_dir", "data/processed/noise_chunks"))
    output_dir = resolve_path(generation.get("output_dir", "data/processed/generated"))
    metadata_dir = resolve_path(generation.get("metadata_dir", "data/metadata"))

    print(f"sample_rate       : {sample_rate}")
    print(f"duration_sec      : {duration_sec}")
    print(f"expected_samples  : {expected_samples}")
    print(f"clean_dir         : {clean_dir}")
    print(f"noise_dir         : {noise_dir}")
    print(f"output_dir        : {output_dir}")
    print(f"metadata_dir      : {metadata_dir}")

    snr_min = float(generation.get("snr_min_db", -5.0))
    snr_max = float(generation.get("snr_max_db", 20.0))

    print(f"snr_min_db        : {snr_min}")
    print(f"snr_max_db        : {snr_max}")

    if snr_min < -10:
        print("[WARNING] snr_min_db très bas. Dataset potentiellement trop bruité.")

    if snr_max > 30:
        print("[WARNING] snr_max_db très haut. Bruit parfois presque inaudible.")

    apply_clean_augment = generation.get("apply_clean_augment", False)
    apply_noise_augment = generation.get("apply_noise_augment", False)
    apply_post_noisy_augment = generation.get("apply_post_noisy_augment", False)

    print(f"apply_clean_augment       : {apply_clean_augment}")
    print(f"apply_noise_augment       : {apply_noise_augment}")
    print(f"apply_post_noisy_augment  : {apply_post_noisy_augment}")

    if apply_post_noisy_augment is True:
        print("[ERREUR] apply_post_noisy_augment est activé.")
        print("Comme tu as retiré Discord/appels/codecs, mets-le à false.")

    else:
        print("[OK] Pas d'augmentation post-noisy activée.")

    return {
        "sample_rate": sample_rate,
        "duration_sec": duration_sec,
        "expected_samples": expected_samples,
        "clean_dir": clean_dir,
        "noise_dir": noise_dir,
        "output_dir": output_dir,
        "metadata_dir": metadata_dir,
        "snr_min": snr_min,
        "snr_max": snr_max,
    }


def check_wav_file(path: Path, expected_sr: int, expected_samples: int) -> list[str]:
    errors = []

    try:
        info = sf.info(path)

        if info.samplerate != expected_sr:
            errors.append(f"sample_rate={info.samplerate}")

        if info.channels != 1:
            errors.append(f"channels={info.channels}")

        if info.frames != expected_samples:
            errors.append(f"samples={info.frames}")

        audio, sr = sf.read(path, dtype="float32", always_2d=False)

        if audio.ndim != 1:
            errors.append(f"shape={audio.shape}")

        if not np.all(np.isfinite(audio)):
            errors.append("nan_or_inf")

        p = peak(audio)
        r = rms_db(audio)

        if p > MAX_PEAK:
            errors.append(f"clipping_peak={p:.4f}")

        if r < MIN_RMS_DB:
            errors.append(f"too_silent_rms={r:.2f}dB")

    except Exception as e:
        errors.append(f"read_error={repr(e)}")

    return errors


def check_audio_folder(folder: Path, label: str, expected_sr: int, expected_samples: int) -> None:
    print(f"\n=== 3. CHECK AUDIO : {label} ===")

    wavs = discover_wavs(folder)

    print(f"Dossier : {folder}")
    print(f"Fichiers WAV trouvés : {len(wavs)}")

    if not wavs:
        print(f"[ERREUR] Aucun WAV trouvé dans {folder}")
        return

    files_to_check = wavs[:MAX_FILES_TO_CHECK_PER_FOLDER]

    valid_count = 0
    error_count = 0

    for path in tqdm(files_to_check, desc=f"Check {label}"):
        errors = check_wav_file(path, expected_sr, expected_samples)

        if errors:
            error_count += 1
            print(f"[ERREUR] {path.name} -> {errors}")
        else:
            valid_count += 1

    print(f"[RESULTAT] {label} : {valid_count}/{len(files_to_check)} fichiers valides vérifiés")

    if len(wavs) > MAX_FILES_TO_CHECK_PER_FOLDER:
        print(f"[INFO] Check limité à {MAX_FILES_TO_CHECK_PER_FOLDER} fichiers sur {len(wavs)}")


def check_generated_pairs(output_dir: Path, expected_sr: int, expected_samples: int) -> None:
    print("\n=== 4. CHECK PAIRES NOISY / CLEAN ===")

    for split in ["train", "val", "test"]:
        noisy_dir = output_dir / split / "noisy"
        clean_dir = output_dir / split / "clean"

        print(f"\n--- Split {split} ---")

        if not noisy_dir.exists() and not clean_dir.exists():
            print(f"[INFO] Split {split} pas encore généré.")
            continue

        noisy_files = discover_wavs(noisy_dir)
        clean_files = discover_wavs(clean_dir)

        noisy_names = {p.name for p in noisy_files}
        clean_names = {p.name for p in clean_files}

        print(f"Noisy : {len(noisy_files)}")
        print(f"Clean : {len(clean_files)}")

        missing_clean = sorted(noisy_names - clean_names)
        missing_noisy = sorted(clean_names - noisy_names)

        if missing_clean:
            print(f"[ERREUR] Fichiers clean manquants : {len(missing_clean)}")
            print(missing_clean[:10])

        if missing_noisy:
            print(f"[ERREUR] Fichiers noisy manquants : {len(missing_noisy)}")
            print(missing_noisy[:10])

        common_names = sorted(noisy_names & clean_names)

        if not common_names:
            print("[WARNING] Aucune paire commune trouvée.")
            continue

        files_to_check = common_names[:MAX_FILES_TO_CHECK_PER_FOLDER]

        valid_pairs = 0
        pair_errors = 0

        for name in tqdm(files_to_check, desc=f"Check pairs {split}"):
            noisy_path = noisy_dir / name
            clean_path = clean_dir / name

            noisy_errors = check_wav_file(noisy_path, expected_sr, expected_samples)
            clean_errors = check_wav_file(clean_path, expected_sr, expected_samples)

            if noisy_errors or clean_errors:
                pair_errors += 1
                print(f"[ERREUR] Paire invalide : {name}")
                print(f"  noisy : {noisy_errors}")
                print(f"  clean : {clean_errors}")
                continue

            noisy, _ = sf.read(noisy_path, dtype="float32")
            clean, _ = sf.read(clean_path, dtype="float32")

            if len(noisy) != len(clean):
                pair_errors += 1
                print(f"[ERREUR] Longueurs différentes : {name}")
                continue

            if np.allclose(noisy, clean, atol=1e-5):
                pair_errors += 1
                print(f"[WARNING] Noisy presque identique au clean : {name}")
                continue

            valid_pairs += 1

        print(f"[RESULTAT] {split} : {valid_pairs}/{len(files_to_check)} paires valides vérifiées")
        print(f"Erreurs paires : {pair_errors}")


def check_metadata(metadata_dir: Path, snr_min: float, snr_max: float) -> None:
    print("\n=== 5. CHECK METADATA ===")

    metadata_path = metadata_dir / "generated_metadata.csv"
    errors_path = metadata_dir / "generated_errors.csv"

    if not metadata_path.exists():
        print(f"[WARNING] Metadata introuvable : {metadata_path}")
    else:
        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        print(f"Lignes metadata : {len(rows)}")

        if rows:
            required_columns = [
                "sample_id",
                "split",
                "noisy_file",
                "clean_file",
                "source_clean_file",
                "source_noise_file",
                "snr_db",
                "sample_rate",
                "duration_sec",
                "num_samples",
            ]

            for col in required_columns:
                if col not in rows[0]:
                    print(f"[ERREUR] Colonne manquante : {col}")

            missing_files = 0
            invalid_snr = 0
            post_aug_detected = 0

            for row in rows[:1000]:
                noisy_file = Path(row.get("noisy_file", ""))
                clean_file = Path(row.get("clean_file", ""))

                if noisy_file and not noisy_file.exists():
                    missing_files += 1

                if clean_file and not clean_file.exists():
                    missing_files += 1

                try:
                    snr = float(row.get("snr_db", "nan"))

                    if snr < snr_min - 2 or snr > snr_max + 2:
                        invalid_snr += 1

                except Exception:
                    invalid_snr += 1

                post_aug = row.get("post_noisy_augmentations", "")
                if post_aug.strip():
                    post_aug_detected += 1

            if missing_files:
                print(f"[ERREUR] Fichiers référencés manquants : {missing_files}")
            else:
                print("[OK] Les fichiers référencés existent.")

            if invalid_snr:
                print(f"[WARNING] SNR suspects : {invalid_snr}")
            else:
                print("[OK] SNR cohérents avec la config.")

            if post_aug_detected:
                print(f"[ERREUR] Augmentations post-noisy détectées : {post_aug_detected}")
                print("Tu as dit avoir retiré Discord/appels/codecs, donc cette colonne devrait être vide.")
            else:
                print("[OK] Aucune augmentation post-noisy détectée.")

    print("\n=== 6. CHECK ERREURS CSV ===")

    if not errors_path.exists():
        print("[OK] Aucun fichier generated_errors.csv trouvé.")
        return

    with open(errors_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Erreurs enregistrées : {len(rows)}")

    if rows:
        print("Exemples :")
        for row in rows[:5]:
            print(row)


def main() -> None:
    print("CHECKUP COMPLET DATASET SPEECH ENHANCEMENT")
    print(f"Racine projet : {PROJECT_ROOT}")

    config = load_config()

    check_python_architecture()

    cfg = check_config(config)

    if not cfg:
        print("\n[STOP] Config génération invalide.")
        return

    check_audio_folder(
        folder=cfg["clean_dir"],
        label="clean_chunks",
        expected_sr=cfg["sample_rate"],
        expected_samples=cfg["expected_samples"],
    )

    check_audio_folder(
        folder=cfg["noise_dir"],
        label="noise_chunks",
        expected_sr=cfg["sample_rate"],
        expected_samples=cfg["expected_samples"],
    )

    check_generated_pairs(
        output_dir=cfg["output_dir"],
        expected_sr=cfg["sample_rate"],
        expected_samples=cfg["expected_samples"],
    )

    check_metadata(
        metadata_dir=cfg["metadata_dir"],
        snr_min=cfg["snr_min"],
        snr_max=cfg["snr_max"],
    )

    print("\nCHECKUP TERMINÉ")


if __name__ == "__main__":
    main()