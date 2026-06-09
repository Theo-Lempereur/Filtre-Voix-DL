from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "dataset_config.yaml"


@dataclass(frozen=True)
class FieldSpec:
    section: str
    key: str
    label: str
    kind: str
    help_text: str = ""
    min_value: float | None = None
    max_value: float | None = None
    step: float = 1.0
    path_picker: bool = False


SCHEMA: dict[str, list[FieldSpec]] = {
    "Sources": [
        FieldSpec("clean_preprocessing", "input_dir", "Dossier voix clean", "str", "Banque d'audios de voix propres.", path_picker=True),
        FieldSpec("noise_preprocessing", "input_dir", "Dossier bruits", "str", "Banque de bruits de fond.", path_picker=True),
        FieldSpec("clean_preprocessing", "max_files", "Max fichiers clean", "int", "Nombre de fichiers clean utilisés. Vide = tous.", 0, 100000),
        FieldSpec("noise_preprocessing", "max_files", "Max fichiers bruit", "int", "Nombre de fichiers bruit utilisés. Vide = tous.", 0, 100000),
        FieldSpec("clean_preprocessing", "shuffle_files", "Mélanger clean", "bool"),
        FieldSpec("noise_preprocessing", "shuffle_files", "Mélanger bruits", "bool"),
        FieldSpec("clean_preprocessing", "random_seed", "Seed clean", "int"),
        FieldSpec("noise_preprocessing", "random_seed", "Seed bruits", "int"),
    ],
    "Préparation": [
        FieldSpec("clean_preprocessing", "sample_rate", "Sample rate", "int", "Fréquence cible en Hz.", 8000, 48000),
        FieldSpec("clean_preprocessing", "chunk_duration_sec", "Durée chunks clean", "float", "Durée des morceaux de voix en secondes.", 0.5, 30.0),
        FieldSpec("noise_preprocessing", "chunk_duration_sec", "Durée chunks bruit", "float", "Durée des morceaux de bruit en secondes.", 0.5, 30.0),
        FieldSpec("clean_preprocessing", "normalize_rms", "Normaliser RMS clean", "bool"),
        FieldSpec("clean_preprocessing", "target_rms_db", "RMS cible clean (dB)", "float", "Volume moyen visé pour les voix.", -60.0, -5.0),
        FieldSpec("clean_preprocessing", "pad_short_files", "Compléter voix courtes", "bool", "À éviter si tu veux des voix naturelles."),
        FieldSpec("noise_preprocessing", "repeat_short_files", "Répéter bruits courts", "bool"),
        FieldSpec("clean_preprocessing", "min_non_silent_ratio", "Ratio non-silence clean", "float", "Filtre les chunks presque silencieux.", 0.0, 1.0, 0.01),
        FieldSpec("noise_preprocessing", "min_non_silent_ratio", "Ratio non-silence bruit", "float", "Filtre les bruits presque silencieux.", 0.0, 1.0, 0.01),
    ],
    "Dataset": [
        FieldSpec("generation", "num_train_samples", "Samples train", "int", "Nombre de paires train noisy/clean.", 0, 1000000),
        FieldSpec("generation", "num_val_samples", "Samples validation", "int"),
        FieldSpec("generation", "num_test_samples", "Samples test", "int"),
        FieldSpec("generation", "snr_min_db", "SNR min (dB)", "float", "Plus bas = bruit plus fort.", -30.0, 40.0),
        FieldSpec("generation", "snr_max_db", "SNR max (dB)", "float", "Plus haut = bruit plus faible.", -30.0, 60.0),
        FieldSpec("generation", "batch_size", "Batch size génération", "int", "Taille des lots pendant la génération.", 1, 4096),
        FieldSpec("generation", "seed", "Seed génération", "int"),
        FieldSpec("generation", "deterministic", "Génération déterministe", "bool"),
        FieldSpec("generation", "skip_existing", "Ignorer fichiers existants", "bool"),
        FieldSpec("generation", "save_noise", "Sauver bruit ajouté", "bool", "Ajoute un dossier noise/ par split."),
    ],
    "Augmentations": [
        FieldSpec("generation", "apply_clean_augment", "Augmenter la voix clean", "bool", "Compression/EQ/téléphone sur la voix avant mix."),
        FieldSpec("generation", "apply_noise_augment", "Augmenter les bruits", "bool", "À activer seulement si tu veux transformer les bruits eux-mêmes."),
        FieldSpec("generation", "apply_post_noisy_augment", "Augmenter l'audio final", "bool", "Simule codec/micro après le mix noisy."),
        FieldSpec("generation", "post_noisy_augment_probability", "Probabilité post-mix", "prob", "Chance d'appliquer les effets au noisy final.", 0.0, 1.0, 0.01),
        FieldSpec("augmentations", "p_gain", "Gain", "prob"),
        FieldSpec("augmentations", "p_eq", "EQ", "prob"),
        FieldSpec("augmentations", "p_compression", "Compression voix", "prob"),
        FieldSpec("augmentations", "p_saturation", "Saturation micro", "prob"),
        FieldSpec("augmentations", "p_clipping", "Clipping", "prob"),
        FieldSpec("augmentations", "p_phone_filter", "Filtre téléphone", "prob"),
        FieldSpec("augmentations", "p_reverb", "Réverbération", "prob"),
        FieldSpec("augmentations", "p_codec", "Codec MP3/Opus", "prob", "Nécessite FFmpeg pour fonctionner."),
        FieldSpec("augmentations", "p_dropout", "Pertes réseau", "prob"),
        FieldSpec("augmentations", "p_quantization", "Quantification cheap", "prob"),
    ],
    "Compression": [
        FieldSpec("augmentations", "compressor_threshold_min_db", "Seuil min (dB)", "float", "Début de compression le plus sensible.", -60.0, 0.0),
        FieldSpec("augmentations", "compressor_threshold_max_db", "Seuil max (dB)", "float", "Début de compression le moins sensible.", -60.0, 0.0),
        FieldSpec("augmentations", "compressor_ratio_min", "Ratio min", "float", "Compression légère.", 1.0, 20.0),
        FieldSpec("augmentations", "compressor_ratio_max", "Ratio max", "float", "Compression forte.", 1.0, 20.0),
        FieldSpec("augmentations", "saturation_drive_min_db", "Saturation min (dB)", "float", "Drive minimum.", 0.0, 30.0),
        FieldSpec("augmentations", "saturation_drive_max_db", "Saturation max (dB)", "float", "Drive maximum.", 0.0, 30.0),
        FieldSpec("augmentations", "phone_highpass_min_hz", "Téléphone HP min", "float", "Coupe-bas minimum.", 20.0, 2000.0),
        FieldSpec("augmentations", "phone_lowpass_max_hz", "Téléphone LP max", "float", "Coupe-haut maximum.", 1000.0, 7900.0),
    ],
}


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError(
            "PyYAML est manquant. Installe les dépendances avec : pip install -r requirements.txt"
        )

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def yaml_scalar(value: Any, kind: str) -> str:
    if kind == "bool":
        return "true" if bool(value) else "false"
    if value is None or value == "":
        return "null"
    if kind in {"int", "float", "prob"}:
        return str(value)
    text = str(value).replace("\\", "/").replace('"', '\\"')
    return f'"{text}"'


def update_yaml_scalars(path: Path, updates: dict[tuple[str, str], tuple[Any, str]]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    section_pattern = re.compile(r"^([A-Za-z_][\w]*):\s*$")
    key_pattern = re.compile(r"^(\s{2})([A-Za-z_][\w]*):(\s*)(.*)$")

    current_section: str | None = None
    seen: set[tuple[str, str]] = set()
    new_lines: list[str] = []

    for line in lines:
        section_match = section_pattern.match(line)
        if section_match:
            current_section = section_match.group(1)
            new_lines.append(line)
            continue

        key_match = key_pattern.match(line)
        if key_match and current_section:
            indent, key, spacing, tail = key_match.groups()
            update_key = (current_section, key)

            if update_key in updates:
                value, kind = updates[update_key]
                comment = ""
                if " #" in tail:
                    comment = "  #" + tail.split(" #", 1)[1]
                new_lines.append(f"{indent}{key}:{spacing}{yaml_scalar(value, kind)}{comment}")
                seen.add(update_key)
                continue

        new_lines.append(line)

    missing = sorted(set(updates) - seen)
    if missing:
        names = ", ".join(f"{section}.{key}" for section, key in missing)
        raise KeyError(f"Clés introuvables dans le YAML : {names}")

    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
