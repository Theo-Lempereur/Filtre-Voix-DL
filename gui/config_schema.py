"""Schema, presets, and YAML update helpers for the dataset configuration GUI.

The GUI intentionally edits a small, curated surface instead of exposing the
entire YAML structure as raw text. This module defines that editable surface,
novice generation presets, and safe patch helpers that update scalar values or
the ``generation.template_mix`` block without reformatting the whole config.
"""

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
    """Describe one editable configuration field exposed by the expert UI.

    Attributes:
        section: Top-level YAML section containing the field.
        key: YAML key inside ``section``.
        label: Human-readable GUI label.
        kind: Field type used for widget choice and YAML serialization.
        help_text: Optional short description shown in the expert UI.
        min_value: Optional lower validation bound for numeric values.
        max_value: Optional upper validation bound for numeric values.
        step: Suggested numeric step for controls that support stepping.
        path_picker: Whether the field should expose a folder picker button.
    """
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
        FieldSpec("clean_preprocessing", "input_dir", "Clean voice folder", "str", "Folder containing clean speech recordings.", path_picker=True),
        FieldSpec("noise_preprocessing", "input_dir", "Noise folder", "str", "Folder containing background noise recordings.", path_picker=True),
        FieldSpec("clean_preprocessing", "max_files", "Max clean files", "int", "Number of clean files to use. Empty means all files.", 0, 100000),
        FieldSpec("noise_preprocessing", "max_files", "Max noise files", "int", "Number of noise files to use. Empty means all files.", 0, 100000),
    ],
    "Preparation": [
        FieldSpec("clean_preprocessing", "sample_rate", "Sample rate", "int", "Target frequency in Hz.", 8000, 48000),
        FieldSpec("clean_preprocessing", "chunk_duration_sec", "Chunk duration", "float", "Shared duration of generated audio chunks in seconds.", 0.5, 30.0),
        FieldSpec("clean_preprocessing", "normalize_rms", "Normalize clean RMS", "bool"),
        FieldSpec("clean_preprocessing", "target_rms_db", "Clean target RMS (dB)", "float", "Target average loudness for clean speech.", -60.0, -5.0),
        FieldSpec("clean_preprocessing", "pad_short_files", "Pad short speech files", "bool", "Keep this off for natural speech chunks."),
        FieldSpec("noise_preprocessing", "repeat_short_files", "Repeat short noise files", "bool"),
    ],
    "Dataset": [
        FieldSpec("generation", "num_train_samples", "Train samples", "int", "Number of noisy/clean training pairs.", 0, 1000000),
        FieldSpec("generation", "num_val_samples", "Validation samples", "int"),
        FieldSpec("generation", "num_test_samples", "Test samples", "int"),
        FieldSpec("generation", "snr_min_db", "Min SNR (dB)", "float", "Lower values create louder noise.", -30.0, 40.0),
        FieldSpec("generation", "snr_max_db", "Max SNR (dB)", "float", "Higher values create quieter noise.", -30.0, 60.0),
        FieldSpec("generation", "batch_size", "Generation batch size", "int", "Number of samples processed per batch.", 1, 4096),
        FieldSpec("generation", "seed", "Generation seed", "int"),
        FieldSpec("generation", "skip_existing", "Skip existing files", "bool"),
        FieldSpec("generation", "save_noise", "Save added noise", "bool", "Creates a noise/ folder inside each split."),
    ],
    "Augmentations": [
        FieldSpec("generation", "apply_clean_augment", "Augment clean speech", "bool", "Applies compression/EQ/phone effects before mixing."),
        FieldSpec("generation", "apply_noise_augment", "Augment noise", "bool", "Only enable when noise files themselves should be transformed."),
        FieldSpec("generation", "apply_post_noisy_augment", "Augment final noisy audio", "bool", "Simulates codec or microphone effects after mixing."),
        FieldSpec("generation", "post_noisy_augment_probability", "Post-mix probability", "prob", "Chance of applying effects to final noisy audio.", 0.0, 1.0, 0.01),
        FieldSpec("augmentations", "p_gain", "Gain", "prob"),
        FieldSpec("augmentations", "p_eq", "EQ", "prob"),
        FieldSpec("augmentations", "p_compression", "Voice compression", "prob"),
        FieldSpec("augmentations", "p_saturation", "Mic saturation", "prob"),
        FieldSpec("augmentations", "p_clipping", "Clipping", "prob"),
        FieldSpec("augmentations", "p_phone_filter", "Phone filter", "prob"),
        FieldSpec("augmentations", "p_reverb", "Reverb", "prob"),
        FieldSpec("augmentations", "p_codec", "MP3/Opus codec", "prob", "Requires FFmpeg."),
        FieldSpec("augmentations", "p_dropout", "Network dropouts", "prob"),
        FieldSpec("augmentations", "p_quantization", "Low-bit quantization", "prob"),
    ],
    "Compression": [
        FieldSpec("augmentations", "compressor_threshold_min_db", "Min threshold (dB)", "float", "Most sensitive compression threshold.", -60.0, 0.0),
        FieldSpec("augmentations", "compressor_threshold_max_db", "Max threshold (dB)", "float", "Least sensitive compression threshold.", -60.0, 0.0),
        FieldSpec("augmentations", "compressor_ratio_min", "Min ratio", "float", "Light compression ratio.", 1.0, 20.0),
        FieldSpec("augmentations", "compressor_ratio_max", "Max ratio", "float", "Strong compression ratio.", 1.0, 20.0),
        FieldSpec("augmentations", "saturation_drive_min_db", "Min saturation (dB)", "float", "Minimum saturation drive.", 0.0, 30.0),
        FieldSpec("augmentations", "saturation_drive_max_db", "Max saturation (dB)", "float", "Maximum saturation drive.", 0.0, 30.0),
        FieldSpec("augmentations", "phone_highpass_min_hz", "Phone HP min", "float", "Minimum high-pass cutoff.", 20.0, 2000.0),
        FieldSpec("augmentations", "phone_lowpass_max_hz", "Phone LP max", "float", "Maximum low-pass cutoff.", 1000.0, 7900.0),
    ],
}


PRESETS: dict[str, dict[tuple[str, str], Any]] = {
    "Classic": {
        ("generation", "snr_min_db"): -5.0,
        ("generation", "snr_max_db"): 20.0,
        ("generation", "apply_clean_augment"): False,
        ("generation", "apply_noise_augment"): False,
        ("generation", "apply_post_noisy_augment"): False,
        ("generation", "post_noisy_augment_probability"): 0.0,
        ("augmentations", "p_gain"): 0.0,
        ("augmentations", "p_eq"): 0.0,
        ("augmentations", "p_compression"): 0.0,
        ("augmentations", "p_saturation"): 0.0,
        ("augmentations", "p_clipping"): 0.0,
        ("augmentations", "p_phone_filter"): 0.0,
        ("augmentations", "p_reverb"): 0.0,
        ("augmentations", "p_codec"): 0.0,
        ("augmentations", "p_dropout"): 0.0,
        ("augmentations", "p_quantization"): 0.0,
    },
    "Microphone mode": {
        ("generation", "snr_min_db"): -3.0,
        ("generation", "snr_max_db"): 18.0,
        ("generation", "apply_clean_augment"): True,
        ("generation", "apply_noise_augment"): False,
        ("generation", "apply_post_noisy_augment"): True,
        ("generation", "post_noisy_augment_probability"): 0.35,
        ("augmentations", "p_gain"): 0.30,
        ("augmentations", "p_eq"): 0.45,
        ("augmentations", "p_compression"): 0.75,
        ("augmentations", "p_saturation"): 0.25,
        ("augmentations", "p_clipping"): 0.05,
        ("augmentations", "p_phone_filter"): 0.45,
        ("augmentations", "p_reverb"): 0.16,
        ("augmentations", "p_codec"): 0.22,
        ("augmentations", "p_dropout"): 0.08,
        ("augmentations", "p_quantization"): 0.12,
        ("augmentations", "compressor_threshold_min_db"): -34.0,
        ("augmentations", "compressor_threshold_max_db"): -16.0,
        ("augmentations", "compressor_ratio_min"): 2.5,
        ("augmentations", "compressor_ratio_max"): 6.0,
    },
    "Very noisy": {
        ("generation", "snr_min_db"): -10.0,
        ("generation", "snr_max_db"): 12.0,
        ("generation", "apply_clean_augment"): False,
        ("generation", "apply_noise_augment"): False,
        ("generation", "apply_post_noisy_augment"): False,
        ("generation", "post_noisy_augment_probability"): 0.0,
        ("augmentations", "p_gain"): 0.0,
        ("augmentations", "p_eq"): 0.0,
        ("augmentations", "p_compression"): 0.0,
        ("augmentations", "p_saturation"): 0.0,
        ("augmentations", "p_clipping"): 0.0,
        ("augmentations", "p_phone_filter"): 0.0,
        ("augmentations", "p_reverb"): 0.0,
        ("augmentations", "p_codec"): 0.0,
        ("augmentations", "p_dropout"): 0.0,
        ("augmentations", "p_quantization"): 0.0,
    },
}


def preset_to_sections(preset_name: str) -> dict[str, dict[str, Any]]:
    """Convert a named novice preset into section-key dictionaries.

    Args:
        preset_name: Key from ``PRESETS``.

    Returns:
        Nested mapping ``{section: {key: value}}`` suitable for template mix
        serialization.

    Raises:
        KeyError: If ``preset_name`` does not exist.
    """
    preset = PRESETS[preset_name]
    sections: dict[str, dict[str, Any]] = {}

    for (section, key), value in preset.items():
        sections.setdefault(section, {})[key] = value

    return sections


def build_template_mix(template_counts: dict[str, int]) -> list[dict[str, Any]]:
    """Build the generation.template_mix YAML value from novice preset counts.

    Args:
        template_counts: Mapping from preset name to requested sample count.

    Returns:
        List of template blocks. Presets with non-positive counts are omitted.
    """
    mix = []

    for preset_name, count in template_counts.items():
        if count <= 0:
            continue

        sections = preset_to_sections(preset_name)
        mix.append(
            {
                "name": preset_name,
                "count": int(count),
                "generation": sections.get("generation", {}),
                "augmentations": sections.get("augmentations", {}),
            }
        )

    return mix


def update_yaml_template_mix(path: Path, template_mix: list[dict[str, Any]]) -> None:
    """Replace the generation.template_mix block while preserving the rest of the file.

    Args:
        path: YAML file to patch.
        template_mix: New ``generation.template_mix`` value.

    Returns:
        None.

    Raises:
        RuntimeError: If PyYAML is not installed.
    """
    if yaml is None:
        raise RuntimeError("PyYAML is missing. Install dependencies with: pip install -r requirements.txt")

    lines = path.read_text(encoding="utf-8").splitlines()
    output: list[str] = []
    in_generation = False
    skipping_template_mix = False
    inserted = False

    block = yaml.safe_dump(
        {"template_mix": template_mix},
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    ).splitlines()
    block = ["  " + line for line in block]

    for line in lines:
        if re.match(r"^[A-Za-z_][\w]*:\s*$", line):
            if in_generation and not inserted:
                output.extend(block)
                inserted = True
            in_generation = line == "generation:"
            skipping_template_mix = False
            output.append(line)
            continue

        if in_generation and line.startswith("  template_mix:"):
            skipping_template_mix = True
            continue

        if skipping_template_mix:
            if line.startswith("  ") and not re.match(r"^  [A-Za-z_][\w]*:", line):
                continue
            if line.startswith("    ") or line.startswith("  -"):
                continue
            skipping_template_mix = False

        if in_generation and not inserted and line.startswith("  allowed_extensions:"):
            output.extend(block)
            inserted = True

        output.append(line)

    if in_generation and not inserted:
        output.extend(block)

    path.write_text("\n".join(output) + "\n", encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML with a clear dependency error when PyYAML is unavailable.

    Args:
        path: YAML file path.

    Returns:
        Parsed YAML dictionary, or an empty dictionary for an empty file.

    Raises:
        RuntimeError: If PyYAML is not installed.
    """
    if yaml is None:
        raise RuntimeError("PyYAML is missing. Install dependencies with: pip install -r requirements.txt")

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def yaml_scalar(value: Any, kind: str) -> str:
    """Serialize one scalar value while keeping YAML simple and readable.

    Args:
        value: Python value to serialize.
        kind: Field kind from ``FieldSpec``.

    Returns:
        YAML scalar string used by the lightweight patcher.
    """
    if kind == "bool":
        return "true" if bool(value) else "false"
    if value is None or value == "":
        return "null"
    if kind in {"int", "float", "prob"}:
        return str(value)
    text = str(value).replace("\\", "/").replace('"', '\\"')
    return f'"{text}"'


def update_yaml_scalars(path: Path, updates: dict[tuple[str, str], tuple[Any, str]]) -> None:
    """Patch top-level section scalar values without reformatting the whole YAML file.

    Args:
        path: YAML file to patch.
        updates: Mapping ``(section, key) -> (value, kind)``.

    Returns:
        None.

    Raises:
        KeyError: If any requested scalar key cannot be found in the YAML file.
    """
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
        raise KeyError(f"Keys not found in YAML: {names}")

    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
