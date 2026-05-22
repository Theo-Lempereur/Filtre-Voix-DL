"""Constantes partagées du projet : chemins Drive + paramètres audio.

Importer depuis n'importe quel notebook / module pour garantir que tout le monde
utilise les mêmes valeurs (sampling rate, STFT, etc.).
"""
import os

# --- Chemins Drive (Colab) ---
DRIVE_PROJECT = "/content/drive/MyDrive/Filtre-Voix-DL"

# Dataset principal : paires (voix bruitée, voix propre) fournies par le dataset public
# Ces deux dossiers restent pour les notebooks d'exploration (00, 01, 02).
DATA_CLEAN = os.path.join(DRIVE_PROJECT, "data/clean")   # version propre des enregistrements
DATA_NOISY = os.path.join(DRIVE_PROJECT, "data/noisy")   # version bruitée des mêmes enregistrements

# Split train / val / test — utilisé pour l'entraînement et l'évaluation.
# La répartition est effectuée une seule fois (cf. scripts/split_data ou notebook 04).
DATA_TRAIN_CLEAN = os.path.join(DRIVE_PROJECT, "data/train/clean")
DATA_TRAIN_NOISY = os.path.join(DRIVE_PROJECT, "data/train/noisy")
DATA_VAL_CLEAN   = os.path.join(DRIVE_PROJECT, "data/val/clean")
DATA_VAL_NOISY   = os.path.join(DRIVE_PROJECT, "data/val/noisy")
DATA_TEST_CLEAN  = os.path.join(DRIVE_PROJECT, "data/test/clean")
DATA_TEST_NOISY  = os.path.join(DRIVE_PROJECT, "data/test/noisy")

# Sorties de l'entraînement et de l'inférence
CHECKPOINTS = os.path.join(DRIVE_PROJECT, "checkpoints")
OUTPUTS = os.path.join(DRIVE_PROJECT, "outputs")
LOGS = os.path.join(DRIVE_PROJECT, "logs")   # JSONL d'historique d'entraînement

# Optionnel — pour une augmentation synthétique éventuelle (extension)
DATA_NOISE_AUG = os.path.join(DRIVE_PROJECT, "data/noise_aug")

# Dossiers créés automatiquement par le notebook de setup
ALL_DIRS = [
    DATA_CLEAN, DATA_NOISY,
    DATA_TRAIN_CLEAN, DATA_TRAIN_NOISY,
    DATA_VAL_CLEAN, DATA_VAL_NOISY,
    DATA_TEST_CLEAN, DATA_TEST_NOISY,
    CHECKPOINTS, OUTPUTS, LOGS,
]

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


# --- Hyperparamètres d'entraînement (valeurs par défaut, surchargeables) ---
BATCH_SIZE     = 8           # tient sur un GPU T4 avec base_channels=32
NUM_WORKERS    = 2           # Colab : éviter > 2 (RAM partagée)
LR             = 1e-3        # Adam : 1e-3 est un bon point de départ
WEIGHT_DECAY   = 0.0         # pas de régularisation L2 explicite en v1
NUM_EPOCHS     = 50          # plafond ; early stopping coupera plus tôt si besoin
BASE_CHANNELS  = 32          # largeur du U-Net (≈7,85M params)
SEED           = 42          # graine globale (torch, numpy, random)
GRAD_CLIP_NORM = 1.0         # protège contre les NaN en début d'entraînement

# --- Split par défaut (si on génère un split automatiquement) ---
SPLIT_RATIOS = (0.8, 0.1, 0.1)   # train / val / test

# --- Politique de checkpoints ---
CKPT_EVERY_N_EPOCHS = 1      # 1 = on sauvegarde à chaque epoch
KEEP_LAST_N_CKPT    = 3      # rotation : on garde les N derniers epoch_*.pt
EARLY_STOP_PATIENCE = 7      # epochs sans amélioration de val avant arrêt
LR_PATIENCE         = 3      # epochs sans amélioration avant ReduceLROnPlateau
LR_FACTOR           = 0.1    # facteur de réduction du LR
