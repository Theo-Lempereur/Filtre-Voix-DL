# Filtre-Voix-DL

Speech enhancement project focused on generating paired noisy/clean audio data for a denoising model.

The active dataset pipeline builds training examples from two separate audio banks:

- clean speech recordings
- background noise recordings such as street noise, rain, crowd, cars, cafe ambience, or white noise

The pipeline prepares both banks into fixed-length chunks, mixes clean speech with noise at controlled SNR values, optionally applies microphone/codec/compression effects, and writes a traceable dataset ready for training.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Open the visual dataset configuration interface:

```bash
python scripts/run_full_pipeline.py --gui
```

Run the complete dataset pipeline from scratch:

```bash
python scripts/run_full_pipeline.py --compile --reset-all
```

Regenerate only the final noisy/clean pairs after editing generation settings:

```bash
python scripts/run_full_pipeline.py --skip-clean --skip-noise --reset-generated
```

Run only the dataset health check:

```bash
python scripts/check_dataset.py
```

## Dataset GUI

The dataset GUI is launched through the main pipeline entry point:

```bash
python scripts/run_full_pipeline.py --gui
```

It has two separated workspaces:

- **Novice**: set only the number of examples per style. The total dataset size and train/validation/test split are computed automatically.
- **Expert**: edit source folders, chunk duration, split sizes, SNR range, augmentation probabilities, compression, codec, and technical generation settings.

The novice mode currently exposes three practical generation styles:

- `Classic`: clean speech mixed with background noise, without extra microphone effects.
- `Microphone mode`: adds voice compression, EQ, phone filtering, saturation, codec, and dropout probabilities.
- `Very noisy`: lowers the SNR range so the background noise becomes much stronger.

When the GUI saves novice settings, it writes `generation.template_mix` into `configs/dataset_config.yaml`. The generator then uses this mix to create the requested number of examples for each style.

Pipeline logs open in a separate GUI window when generation is started from the interface.

## Active Data Pipeline

```txt
raw clean speech
-> prepare_clean.py
-> data/processed/clean_chunks/

raw background noise
-> prepare_noise.py
-> data/processed/noise_chunks/

clean chunk + noise chunk + SNR + optional augmentations
-> generate_noisy.py
-> data/processed/generated/train|val|test/noisy
-> data/processed/generated/train|val|test/clean
```

Each generated run also copies the exact YAML configuration used at generation time:

```txt
data/processed/generated/config_used.yaml
```

This snapshot keeps datasets reproducible even after `configs/dataset_config.yaml` changes later.

## Important Files

```txt
configs/dataset_config.yaml          Active dataset configuration
gui/config_editor_modern.py          Visual configuration interface
gui/config_schema.py                 GUI schema and novice presets
scripts/run_full_pipeline.py         Main dataset pipeline entry point
scripts/prepare_clean.py             Clean speech preprocessing stage
scripts/prepare_noise.py             Noise preprocessing stage
scripts/generate_noisy.py            Final noisy/clean generation stage
scripts/check_dataset.py             Dataset health check
scripts/clean_pipeline_outputs.py    Cleanup helper
src/dataset_builder/                 Dataset pipeline implementation
```

## Expected Output

```txt
data/
├── processed/
│   ├── clean_chunks/
│   ├── noise_chunks/
│   └── generated/
│       ├── config_used.yaml
│       ├── train/
│       │   ├── noisy/
│       │   └── clean/
│       ├── val/
│       │   ├── noisy/
│       │   └── clean/
│       └── test/
│           ├── noisy/
│           └── clean/
├── metadata/
│   ├── clean_metadata.csv
│   ├── clean_errors.csv
│   ├── noise_metadata.csv
│   ├── noise_errors.csv
│   ├── generated_metadata.csv
│   └── generated_errors.csv
└── logs/
```

Generated pair filenames match between noisy and clean folders:

```txt
data/processed/generated/train/noisy/train_00000042.wav
data/processed/generated/train/clean/train_00000042.wav
```

## Dataset Health Check

`scripts/check_dataset.py` validates:

- required pipeline files
- active generation configuration
- prepared clean and noise chunks
- generated noisy/clean pairing
- sample rate, channel count, duration, RMS, and peak levels
- metadata consistency
- template distribution
- presence of `config_used.yaml`

The check reports warnings instead of failing for small experimental datasets, because a smoke-test dataset can be technically valid while still being too small for final model training.

## Configuration Notes

All editable dataset settings live in:

```txt
configs/dataset_config.yaml
```

Use forward slashes in YAML paths, even on Windows:

```yaml
clean_preprocessing:
  input_dir: "G:/My Drive/Filtre-Voix-DL/data/jules/clean"

noise_preprocessing:
  input_dir: "G:/My Drive/Filtre-Voix-DL/data/jules/noise"
```

FFmpeg is recommended when source files include `.mp3`, `.m4a`, `.aac`, or other compressed formats.

## Cleanup Commands

Remove all generated pipeline outputs:

```bash
python scripts/clean_pipeline_outputs.py --all
```

Preview cleanup without deleting files:

```bash
python scripts/clean_pipeline_outputs.py --all --dry-run
```

Remove only generated pairs, metadata, and logs while keeping prepared chunks:

```bash
python scripts/clean_pipeline_outputs.py --generated --metadata --logs
```
