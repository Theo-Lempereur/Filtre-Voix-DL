"""Constantes partagées du projet : chemins Drive + paramètres audio.

Importer depuis n'importe quel notebook / module pour garantir que tout le monde
utilise les mêmes valeurs (sampling rate, STFT, etc.).
"""
import os

# --- Chemins Drive (Colab) ---
DRIVE_PROJECT = "/content/drive/MyDrive/Filtre-Voix-DL"

DATA_CLEAN = os.path.join(DRIVE_PROJECT, "data/clean")
DATA_NOISE = os.path.join(DRIVE_PROJECT, "data/noise")
DATA_MIXED = os.path.join(DRIVE_PROJECT, "data/mixed")
CHECKPOINTS = os.path.join(DRIVE_PROJECT, "checkpoints")
OUTPUTS = os.path.join(DRIVE_PROJECT, "outputs")

ALL_DIRS = [DATA_CLEAN, DATA_NOISE, DATA_MIXED, CHECKPOINTS, OUTPUTS]

# --- Paramètres audio ---
SAMPLE_RATE = 16000      # Hz — standard pour la voix (vs 44.1k pour la musique)
CLIP_DURATION = 4.0      # secondes par clip d'entraînement
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)

# --- Paramètres STFT ---
N_FFT = 512              # taille de la fenêtre FFT
HOP_LENGTH = 128         # décalage entre fenêtres (= 75% overlap)
WIN_LENGTH = 512         # longueur effective de la fenêtre
