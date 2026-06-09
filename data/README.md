# Filtre-Voix-DL

Pipeline de génération de dataset pour du voice denoising.
à partir de deux banques séparées :

```txt
- une banque de voix clean
- une banque de bruits : métro, pluie, foule, voiture, café, etc.
```
  
Cela sert à préparer un dataset propre, structuré et vérifiable.

---

## Quick Start

### 1. Installer les dépendances

Depuis la racine du projet :

```bash
pip install -r requirements.txt
```

Ou sur Windows :

```bash
py -m pip install -r requirements.txt
```

FFmpeg est recommandé si tu utilises des fichiers `.mp3`, `.m4a`, `.aac`, etc.

---

### 2. Modifier la configuration

#### Option visuelle

Une interface locale permet de modifier les paramètres principaux sans éditer le YAML à la main :

```bash
python scripts/run_full_pipeline.py --gui
```

Sur Windows :

```bash
py scripts/run_full_pipeline.py --gui
```

Elle permet de régler :

```txt
- les dossiers d'entrée clean / noise
- le nombre de fichiers utilisés
- les durées et paramètres de preprocessing
- le nombre de samples train / val / test
- la plage de SNR
- les augmentations de voix : compression, EQ, téléphone, saturation, codec, etc.
```

Après sauvegarde, relance la génération du dataset final :

```bash
python scripts/run_full_pipeline.py --skip-clean --skip-noise --reset-generated
```

Tu peux aussi lancer cette génération directement depuis l'interface avec le bouton
`Générer dataset final`.

#### Option YAML

Tout se règle ici :

```txt
configs/dataset_config.yaml
```

Si les fichiers sont sur Google Drive, tu peux utiliser un chemin absolu :

```yaml
clean_preprocessing:
  input_dir: "G:/Mon Drive/Filtre-Voix-DL/data/jules/clean"

noise_preprocessing:
  input_dir: "G:/Mon Drive/Filtre-Voix-DL/data/jules/noise"
```

Utilise `/` au lieu de `\` dans le YAML.

---

### 3. Lancer tout le pipeline

Commande recommandée :

```bash
python scripts/run_full_pipeline.py --compile --reset-all
```

Sur Windows :

```bash
py scripts/run_full_pipeline.py --compile --reset-all
```

Cette commande fait tout :

```txt
1. vérifie la syntaxe Python
2. vide les anciennes sorties
3. prépare les voix clean
4. prépare les bruits
5. génère les paires noisy/clean
6. vérifie le dataset final
```

---

## Architecture

```txt
Filtre-Voix-DL/
│
├── configs/
│   └── dataset_config.yaml
│
├── data/
│   ├── README.md
│   │
│   ├── processed/
│   │   ├── clean_chunks/
│   │   ├── noise_chunks/
│   │   └── generated/
│   │       ├── train/
│   │       │   ├── noisy/
│   │       │   └── clean/
│   │       ├── val/
│   │       │   ├── noisy/
│   │       │   └── clean/
│   │       └── test/
│   │           ├── noisy/
│   │           └── clean/
│   │
│   ├── metadata/
│   │   ├── clean_metadata.csv
│   │   ├── clean_errors.csv
│   │   ├── noise_metadata.csv
│   │   ├── noise_errors.csv
│   │   ├── generated_metadata.csv
│   │   └── generated_errors.csv
│   │
│   └── logs/
│       ├── clean_preprocessing/
│       ├── noise_preprocessing/
│       └── generation/
│
├── scripts/
│   ├── prepare_clean.py
│   ├── prepare_noise.py
│   ├── generate_noisy.py
│   ├── check_dataset.py
│   ├── clean_pipeline_outputs.py
│   └── run_full_pipeline.py
│
└── src/
    └── dataset_builder/
        ├── audio.py
        ├── chunking.py
        ├── config.py
        ├── dataset_generation.py
        ├── mixing.py
        ├── augment.py
        ├── metadata.py
        ├── utils.py
        └── validator.py
```
---

## Dataset Structure

### Chunks préparés

Après preprocessing :

```txt
data/processed/
├── clean_chunks/
│   ├── voice_xxxxx_chunk_00000.wav
│   └── ...
└── noise_chunks/
    ├── metro_xxxxx_noise_00000.wav
    └── ...
```

Chaque chunk doit respecter :

```txt
format      : wav
canaux      : mono
sample rate : 16 kHz
durée       : 3 secondes
samples     : 48 000
```

---

### Dataset final

Après génération :

```txt
data/processed/generated/
├── train/
│   ├── noisy/
│   │   ├── train_00000000.wav
│   │   └── ...
│   └── clean/
│       ├── train_00000000.wav
│       └── ...
│
├── val/
│   ├── noisy/
│   └── clean/
│
└── test/
    ├── noisy/
    └── clean/
```

Chaque paire a le même nom :

```txt
train/noisy/train_00000042.wav
train/clean/train_00000042.wav
```

---

### `clean_preprocessing`

Prépare les voix clean.

Exemple :

```yaml
clean_preprocessing:
  input_dir: "data/clean"
  output_dir: "data/processed/clean_chunks"
  metadata_dir: "data/metadata"
  logs_dir: "data/logs/clean_preprocessing"

  sample_rate: 16000
  mono: true
  chunk_duration_sec: 3.0

  max_files: 50
  shuffle_files: true
  random_seed: 42

  pad_short_files: false

  normalize_rms: true
  target_rms_db: -25.0
  max_gain_db: 20.0
  peak_limit: 0.98

  min_duration_sec: 3.0
  silence_threshold_db: -50.0
  min_non_silent_ratio: 0.05

  max_workers: 1
  skip_existing: true
```
---

### `noise_preprocessing`

Prépare les bruits.

Exemple :

```yaml
noise_preprocessing:
  input_dir: "data/noise"
  output_dir: "data/processed/noise_chunks"
  metadata_dir: "data/metadata"
  logs_dir: "data/logs/noise_preprocessing"

  sample_rate: 16000
  mono: true
  chunk_duration_sec: 3.0

  max_files: 20
  shuffle_files: true
  random_seed: 42

  repeat_short_files: true

  min_duration_sec: 0.5
  silence_threshold_db: -60.0
  min_non_silent_ratio: 0.03

  peak_limit: 0.98

  max_workers: 1
  skip_existing: true
```

Pour les bruits, `repeat_short_files: true` est acceptable.  
Pour les voix clean, il vaut mieux éviter de compléter trop souvent avec du silence.

---

### `generation`

Génère les paires noisy/clean finales.

Exemple :

```yaml
generation:
  clean_dir: "data/processed/clean_chunks"
  noise_dir: "data/processed/noise_chunks"

  output_dir: "data/processed/generated"
  metadata_dir: "data/metadata"
  logs_dir: "data/logs/generation"

  sample_rate: 16000
  duration_sec: 3.0

  num_train_samples: 100
  num_val_samples: 20
  num_test_samples: 20

  snr_min_db: -5.0
  snr_max_db: 20.0

  min_clean_rms_db: -45.0
  min_noise_rms_db: -60.0

  peak_limit: 0.98
  avoid_clipping: true
  apply_gain_to_target: true

  batch_size: 16
  max_workers: 1

  seed: 42
  deterministic: true
  skip_existing: true

  save_noise: false

  apply_clean_augment: false
  apply_noise_augment: false
  apply_post_noisy_augment: false
```
Pour un premier test propre :

```yaml
generation:
  num_train_samples: 20
  num_val_samples: 5
  num_test_samples: 5
  batch_size: 8
  max_workers: 1

  apply_clean_augment: false
  apply_noise_augment: false
  apply_post_noisy_augment: false
```

---

## Pipeline

### 1. Préparer les voix clean

```bash
python scripts/prepare_clean.py
```

Fait :

```txt
raw clean audio
↓
wav
↓
mono
↓
16 kHz
↓
normalisation RMS
↓
chunks de 3 secondes
↓
data/processed/clean_chunks/
```

---

### 2. Préparer les bruits

```bash
python scripts/prepare_noise.py
```

Fait :

```txt
raw noise audio
↓
wav
↓
mono
↓
16 kHz
↓
chunks de 3 secondes
↓
data/processed/noise_chunks/
```

---

### 3. Générer le dataset final

```bash
python scripts/generate_noisy.py
```

Fait :

```txt
clean chunk
+
noise chunk
+
random SNR
↓
mix RMS
↓
noisy.wav + clean.wav
```

Le moteur interne utilisé est :

```txt
src/dataset_builder/dataset_generation.py
```

---

### 4. Vérifier le dataset

```bash
python scripts/check_dataset.py
```

Vérifie :

```txt
- architecture
- config
- clean chunks
- noise chunks
- paires noisy/clean
- durée 3 secondes
- 16 kHz
- mono
- metadata
- erreurs
```

---

## Appels de scripts utiles

### Tout lancer depuis zéro

```bash
python scripts/run_full_pipeline.py --compile --reset-all
```

---

### Tout lancer sans reset

```bash
python scripts/run_full_pipeline.py
```

---

### Refaire seulement le dataset final

Garde `clean_chunks/` et `noise_chunks/`, supprime seulement l’ancien dataset généré :

```bash
python scripts/run_full_pipeline.py --skip-clean --skip-noise --reset-generated
```

---

### Faire seulement le checkup

```bash
python scripts/check_dataset.py
```

---

### Nettoyer toutes les sorties

```bash
python scripts/clean_pipeline_outputs.py --all
```

Cela supprime :

```txt
data/processed/clean_chunks/
data/processed/noise_chunks/
data/processed/generated/
data/metadata/
data/logs/
```

---

### Voir ce qui serait supprimé sans supprimer

```bash
python scripts/clean_pipeline_outputs.py --all --dry-run
```

---

### Nettoyer seulement le dataset final

```bash
python scripts/clean_pipeline_outputs.py --generated --metadata --logs
```

---

### Nettoyer seulement les chunks

```bash
python scripts/clean_pipeline_outputs.py --chunks
```

---

## Exemple de test

```yaml
clean_preprocessing:
  max_files: 50
  max_workers: 1

noise_preprocessing:
  max_files: 20
  max_workers: 1

generation:
  num_train_samples: 100
  num_val_samples: 20
  num_test_samples: 20
  batch_size: 16
  max_workers: 1
```

Commande :

```bash
python scripts/run_full_pipeline.py --compile --reset-all
```

---

## Conseils importants

### Pour petit PC

Garder :

```yaml
max_workers: 1
```

Puis tester `2` seulement quand tout est stable.

---

### Pour éviter de mélanger ancien et nouveau dataset

Utiliser :

```bash
python scripts/run_full_pipeline.py --reset-generated --skip-clean --skip-noise
```

ou :

```bash
python scripts/run_full_pipeline.py --reset-all
```

---

### Pour un premier dataset propre

Désactiver les augmentations :

```yaml
generation:
  apply_clean_augment: false
  apply_noise_augment: false
  apply_post_noisy_augment: false
```

---

### Pour un vrai dataset

Il faut beaucoup plus de clean chunks.

Si le checkup affiche :

```txt
clean_chunks : 2 fichiers
```

le pipeline marche, mais le dataset manque de variété.

Augmenter :

```yaml
clean_preprocessing:
  max_files: 50
```

Puis relancer :

```bash
python scripts/prepare_clean.py
```

---
