"""Constantes partagées du projet : chemins Drive + paramètres audio.

Importer depuis n'importe quel notebook / module pour garantir que tout le monde
utilise les mêmes valeurs (sampling rate, STFT, etc.).
"""
import os

# --- Chemins Drive (Colab) ---
DRIVE_PROJECT = "/content/drive/MyDrive/Filtre-Voix-DL"

# Dataset principal : paires (voix bruitée, voix propre) fournies par le dataset public
DATA_CLEAN = os.path.join(DRIVE_PROJECT, "data/clean")   # version propre des enregistrements
DATA_NOISY = os.path.join(DRIVE_PROJECT, "data/noisy")   # version bruitée des mêmes enregistrements

# Sorties de l'entraînement et de l'inférence
CHECKPOINTS = os.path.join(DRIVE_PROJECT, "checkpoints")
OUTPUTS = os.path.join(DRIVE_PROJECT, "outputs")

# Optionnel — pour une augmentation synthétique éventuelle (extension)
DATA_NOISE_AUG = os.path.join(DRIVE_PROJECT, "data/noise_aug")

# Dossiers créés automatiquement par le notebook de setup
ALL_DIRS = [DATA_CLEAN, DATA_NOISY, CHECKPOINTS, OUTPUTS]

# --- Paramètres audio ---
SAMPLE_RATE = 16000          # Hz — standard pour la voix (vs 44.1k pour la musique)
CLIP_DURATION = 4.0          # secondes — durée fixe utilisée à l'entraînement
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)
MAX_CLIP_DURATION = 10.0     # secondes — durée max observée dans le dataset (pour info)

# Stratégie : crop centré ou aléatoire à CLIP_DURATION pour les clips plus longs,
# padding silence à droite pour les clips plus courts. À implémenter dans src/dataset.py.

# --- Paramètres STFT ---
N_FFT = 512                  # taille de la fenêtre FFT
HOP_LENGTH = 128             # décalage entre fenêtres (= 75% overlap)
WIN_LENGTH = 512             # longueur effective de la fenêtre
