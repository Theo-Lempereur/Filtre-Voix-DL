# Dataset Pipeline Guide

This folder contains local dataset inputs and generated outputs. Large audio files are not versioned in Git; they should stay on Google Drive, an external disk, or local folders referenced from `configs/dataset_config.yaml`.

The active pipeline creates paired audio for speech enhancement:

```txt
clean speech chunk + background noise chunk
-> controlled SNR mix
-> noisy input WAV + matching clean target WAV
```

## Setup

Install Python dependencies from the repository root:

```bash
pip install -r requirements.txt
```

FFmpeg is recommended for compressed source formats such as `.mp3`, `.m4a`, `.aac`, or `.ogg`.

## Visual Configuration

Open the dataset GUI:

```bash
python scripts/run_full_pipeline.py --gui
```

The GUI has two separate modes:

- **Novice**: choose how many examples to generate for `Classic`, `Microphone mode`, and `Very noisy`.
- **Expert**: edit paths, splits, SNR, chunk duration, augmentation probabilities, compression, codec, and generation settings.

Novice mode computes the total dataset size automatically from the three template quantities, then updates train/validation/test counts in the background.

When novice settings are saved, the GUI writes `generation.template_mix` into:

```txt
configs/dataset_config.yaml
```

The generator records the selected template name in `generated_metadata.csv`, so you can inspect how much data came from each style.

## Main Commands

Run the whole pipeline from scratch:

```bash
python scripts/run_full_pipeline.py --compile --reset-all
```

Run without cleanup:

```bash
python scripts/run_full_pipeline.py
```

Regenerate only final noisy/clean pairs:

```bash
python scripts/run_full_pipeline.py --skip-clean --skip-noise --reset-generated
```

Run only the dataset check:

```bash
python scripts/check_dataset.py
```

## Pipeline Stages

### 1. Clean Speech Preparation

Command:

```bash
python scripts/prepare_clean.py
```

Output:

```txt
data/processed/clean_chunks/
```

This stage:

- loads clean speech files
- converts audio to mono
- resamples to the configured sample rate
- optionally normalizes RMS
- splits files into fixed-length chunks
- filters silent or invalid chunks
- writes `clean_metadata.csv` and `clean_errors.csv`

### 2. Noise Preparation

Command:

```bash
python scripts/prepare_noise.py
```

Output:

```txt
data/processed/noise_chunks/
```

This stage:

- loads background noise files
- converts audio to mono
- resamples to the configured sample rate
- peak-limits the signal
- splits files into fixed-length chunks
- can repeat short noise files when configured
- writes `noise_metadata.csv` and `noise_errors.csv`

### 3. Noisy/Clean Generation

Command:

```bash
python scripts/generate_noisy.py
```

Output:

```txt
data/processed/generated/
├── config_used.yaml
├── train/
│   ├── noisy/
│   └── clean/
├── val/
│   ├── noisy/
│   └── clean/
└── test/
    ├── noisy/
    └── clean/
```

This stage:

- randomly selects clean and noise chunks
- crops aligned fixed-length arrays
- optionally augments clean speech or noise before mixing
- scales noise to the configured SNR range
- optionally augments the final noisy signal
- saves matching noisy/clean pairs
- writes `generated_metadata.csv` and `generated_errors.csv`
- copies the active configuration to `generated/config_used.yaml`

## Expected Audio Format

Prepared chunks and generated pairs should match the active config. The default is:

```txt
format      : WAV
channels    : mono
sample rate : 16000 Hz
duration    : 3.0 seconds
samples     : 48000
```

## Metadata

Metadata is written to:

```txt
data/metadata/
```

Important generated metadata fields include:

- `sample_id`
- `split`
- `noisy_file`
- `clean_file`
- `source_clean_file`
- `source_noise_file`
- `snr_db`
- `clean_rms_db`
- `noisy_rms_db`
- `clean_augmentations`
- `noise_augmentations`
- `post_noisy_augmentations`
- `template_name`

## Config Snapshot

Every generation run copies the current config to:

```txt
data/processed/generated/config_used.yaml
```

Use this file to know exactly how a dataset was generated, even if the main YAML config changes later.

## Dataset Check

Command:

```bash
python scripts/check_dataset.py
```

The check validates:

- project architecture
- active generation settings
- prepared clean and noise chunks
- noisy/clean pair counts
- missing files
- sample rate, duration, mono format, RMS, and peaks
- metadata row count and split distribution
- template distribution
- codec failures
- `config_used.yaml`

Warnings are actionable. They do not always mean the dataset is unusable; small smoke-test datasets will naturally produce warnings about low data volume.

## Cleanup

Remove all generated pipeline outputs:

```bash
python scripts/clean_pipeline_outputs.py --all
```

Preview cleanup first:

```bash
python scripts/clean_pipeline_outputs.py --all --dry-run
```

Remove only final generated pairs, metadata, and logs:

```bash
python scripts/clean_pipeline_outputs.py --generated --metadata --logs
```

Remove only prepared chunks:

```bash
python scripts/clean_pipeline_outputs.py --chunks
```

## Practical Recommendations

For a first local smoke test:

```yaml
clean_preprocessing:
  max_files: 10
  max_workers: 1

noise_preprocessing:
  max_files: 10
  max_workers: 1

generation:
  num_train_samples: 20
  num_val_samples: 5
  num_test_samples: 5
  batch_size: 8
  max_workers: 1
```

For a more useful training dataset:

- increase clean speech variety first
- keep noise sources diverse
- keep `max_workers: 1` until the pipeline is stable on your machine
- use `check_dataset.py` after each generated dataset
- keep `config_used.yaml` with any dataset you train on
