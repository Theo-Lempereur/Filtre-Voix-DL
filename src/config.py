"""Constantes partagées du projet : chemins Drive + paramètres audio.

Importer depuis n'importe quel notebook / module pour garantir que tout le monde
utilise les mêmes valeurs (sampling rate, STFT, etc.).
"""
import os
from pathlib import Path


# Chemin Colab par défaut — utilisé comme fallback si rien d'autre n'est trouvé,
# pour ne pas casser ``from src import config`` côté dev local sans Drive Desktop.
_DEFAULT_COLAB_ROOT = "/content/drive/MyDrive/Filtre-Voix-DL"


def _detect_project_root() -> str:
    """Détecte la racine du projet (dossier Drive partagé) selon l'environnement.

    Ordre de priorité :
      1. Variable d'environnement ``FILTRE_VOIX_DL_ROOT`` (override explicite).
      2. Colab : Drive monté sous ``/content/drive/MyDrive``.
      3. Local : chemins usuels de Google Drive for Desktop (Win/macOS/Linux).
      4. Fallback silencieux vers le chemin Colab (les notebooks d'exploration
         locale peuvent importer ``src.config`` sans Drive sync).

    L'import ne lève jamais : si rien n'est dispo, c'est ``ensure_project_root()``
    (appelé par les scripts d'entraînement) qui échouera proprement.
    """
    env = os.environ.get("FILTRE_VOIX_DL_ROOT")
    if env:
        return env

    if os.path.exists("/content/drive/MyDrive"):
        return _DEFAULT_COLAB_ROOT

    # Drive Desktop monte sous la lettre suivante libre (G: → Z:), et le nom du
    # sous-dossier dépend de la langue de Windows ("My Drive" en anglais,
    # "Mon Drive" en français). On balaie les combinaisons usuelles.
    win_letters = ["G:", "H:", "I:", "J:"]
    win_subdirs = ["My Drive", "Mon Drive"]
    candidates = [
        Path(f"{letter}/{subdir}/Filtre-Voix-DL")
        for letter in win_letters
        for subdir in win_subdirs
    ]
    candidates += [
        Path.home() / "Google Drive" / "My Drive" / "Filtre-Voix-DL",                                          # macOS / Linux
        Path.home() / "Library" / "CloudStorage" / "GoogleDrive-stonslemps@gmail.com" / "My Drive" / "Filtre-Voix-DL",  # macOS récent
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    return _DEFAULT_COLAB_ROOT


def ensure_project_root() -> str:
    """Vérifie que ``DRIVE_PROJECT`` pointe vers un dossier existant.

    Appelée par les scripts d'entraînement (``scripts/train_local.py``) pour
    échouer tôt avec un message clair si Drive Desktop n'est pas synchronisé.
    """
    if not os.path.isdir(DRIVE_PROJECT):
        raise RuntimeError(
            f"Racine du projet introuvable : {DRIVE_PROJECT!r}. "
            "Installer Google Drive for Desktop et synchroniser le dossier "
            "'Filtre-Voix-DL', ou définir la variable d'environnement "
            "FILTRE_VOIX_DL_ROOT vers le chemin local."
        )
    return DRIVE_PROJECT


# --- Racine du projet (Colab ou Drive Desktop local) ---
DRIVE_PROJECT = _detect_project_root()

# Racine du dépôt git (le fichier vit dans <repo>/src/config.py). Sert
# d'ancre pour les artefacts volumineux et re-générables (dataset) qu'on ne
# veut pas synchroniser sur Drive — contrairement aux checkpoints/logs qui
# sont partagés via DRIVE_PROJECT.
REPO_ROOT = str(Path(__file__).resolve().parents[1])

# Dataset principal : paires (voix bruitée, voix propre) fournies par le dataset public
# Ces deux dossiers restent pour les notebooks d'exploration (00, 01, 02).
DATA_CLEAN = os.path.join(DRIVE_PROJECT, "data/clean")   # version propre des enregistrements
DATA_NOISY = os.path.join(DRIVE_PROJECT, "data/noisy")   # version bruitée des mêmes enregistrements

# Split train / val / test — utilisé pour l'entraînement et l'évaluation.
# Pointe vers les paires bruité/propre générées par le pipeline de dataset
# (cf. scripts/run_full_pipeline.py + configs/dataset_config.yaml). Les
# fichiers portent le même nom dans noisy/ et clean/ (ex: train_00000042.wav),
# ce qui matche le mode "name" de PairedAudioDataset.list_pairs(). Vit dans
# le repo (relatif à REPO_ROOT) car ~5 GB et reproductible via le pipeline —
# inutile de saturer Drive avec ça.
_GENERATED_ROOT = os.path.join(REPO_ROOT, "data/processed/generated")
DATA_TRAIN_CLEAN = os.path.join(_GENERATED_ROOT, "train/clean")
DATA_TRAIN_NOISY = os.path.join(_GENERATED_ROOT, "train/noisy")
DATA_VAL_CLEAN   = os.path.join(_GENERATED_ROOT, "val/clean")
DATA_VAL_NOISY   = os.path.join(_GENERATED_ROOT, "val/noisy")
DATA_TEST_CLEAN  = os.path.join(_GENERATED_ROOT, "test/clean")
DATA_TEST_NOISY  = os.path.join(_GENERATED_ROOT, "test/noisy")

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
BATCH_SIZE     = 24          # mesuré : ~10 Go sur RTX 5070 12 Go en bf16 (base_channels=32).
                             # 32 déborde en mémoire partagée (lent). Plus de VRAM -> monter.
NUM_WORKERS    = 8           # local Ryzen 8c/16t ; sur Colab redescendre à 2 (RAM partagée)
LR             = 1e-3        # AdamW : 1e-3 est un bon point de départ
WEIGHT_DECAY   = 1e-4        # AdamW avec weight decay découplé (recommandé)
NUM_EPOCHS     = 50          # plafond ; early stopping coupera plus tôt si besoin
BASE_CHANNELS  = 32          # largeur du U-Net (≈7,85M params)
SEED           = 42          # graine globale (torch, numpy, random)
GRAD_CLIP_NORM = 1.0         # protège contre les NaN en début d'entraînement

# --- Performance / précision (CUDA) ---
# Ces options accélèrent l'entraînement sur GPU récent (RTX 30/40/50) sans
# changement perceptible de qualité. Auto-désactivées hors CUDA.
AMP            = True         # mixed precision bf16 : ~1.5-2x plus rapide, ~2x moins de VRAM
# torch.compile : gain réel sur Linux/RunPod, MAIS nécessite Triton (absent sous
# Windows -> on laisse désactivé par défaut). Activable via --compile : en cas
# d'absence de Triton, on retombe proprement en eager (cf. src.train).
COMPILE        = False
# Cache des waveforms décodées (float16, memmap) pour supprimer le coût
# librosa.load + STFT côté CPU. La STFT est alors faite sur GPU dans le forward.
USE_WAV_CACHE  = True
# Emplacement du cache. Par défaut dans le repo (régénérable, gitignoré). Sur un
# environnement distant (RunPod) où le repo est cloné à neuf mais où un Network
# Volume persistant est monté (ex: /workspace), pointer le cache DESSUS via la
# variable FILTRE_VOIX_DL_CACHE évite de ré-uploader ~18 Go à chaque session.
CACHE_DIR      = os.environ.get(
    "FILTRE_VOIX_DL_CACHE", os.path.join(REPO_ROOT, "data/processed/cache")
)

# --- Split par défaut (si on génère un split automatiquement) ---
SPLIT_RATIOS = (0.8, 0.1, 0.1)   # train / val / test

# --- Politique de checkpoints ---
CKPT_EVERY_N_EPOCHS = 1      # 1 = on sauvegarde à chaque epoch
KEEP_LAST_N_CKPT    = 3      # rotation : on garde les N derniers epoch_*.pt
EARLY_STOP_PATIENCE = 7      # epochs sans amélioration de val avant arrêt
LR_PATIENCE         = 3      # epochs sans amélioration avant ReduceLROnPlateau
LR_FACTOR           = 0.5    # facteur de réduction du LR (0.5 = baisses douces ;
                             # 0.1 décroissait trop vite -> LR mort vers ~1e-7 et plateau précoce)

# --- Sauvegarde et log INTRA-epoch (sécurité contre coupures Colab) ---
# Sauvegarde `last.pt` toutes les N secondes pendant une epoch, sans attendre
# la fin de l'epoch. À la reprise, on repart au début de l'epoch en cours
# avec les poids les plus récents — au pire on perd ces N secondes de calcul.
INTRA_EPOCH_SAVE_SECONDS = 15 * 60   # 15 minutes
# Log `train_loss_step` à wandb tous les N batches. Permet de voir la courbe
# d'apprentissage dès les premières minutes, sans attendre la fin d'epoch.
INTRA_EPOCH_LOG_EVERY    = 50        # batches


# ---------------------------------------------------------------------------
# Recette d'entraînement (figée pour la baseline "p2")
# ---------------------------------------------------------------------------
# Choix de modélisation pour le débruitage. Ce sont les paramètres qu'on a
# validés expérimentalement comme baseline qui supprime réellement du bruit
# (pas juste l'atténuer) sur le dataset 20 k. Les valeurs ci-dessous sont
# soit figées dans le code (input_repr=log1p, GroupNorm, AdamW, masque sigmoid,
# loss combo), soit modifiables via la GUI / CLI quand l'expérimentation peut
# être utile (poids des composantes loss, warmup, weight decay).

# Compression appliquée à la magnitude AVANT le U-Net. log(1+mag) rend la
# distribution plus gaussienne et concentre le réseau sur les zones perceptives.
# La cible reste en magnitude linéaire pour conserver la sémantique
# `sigmoid × noisy_mag` du masque.
INPUT_REPR = "log1p"

# Poids des composantes de la loss combinée. La loss totale est :
#   L = w_mse · MSE(pred_mag, clean_mag)
#     + w_l1c · L1(pred_mag^0.3, clean_mag^0.3)
#     + w_mr  · MR-STFT(pred_wav, clean_wav)
# Une composante à poids 0 est skippée (économie de compute). Modifiable
# depuis la GUI (panneau "Avancé").
LOSS_WEIGHTS = {
    "mse":     0.0,
    "l1_comp": 1.0,
    "mr_stft": 1.0,
}

# Tailles de FFT pour la MR-STFT loss (les hop_length sont déduits = n_fft // 4).
MR_STFT_FFT_SIZES = (256, 512, 1024)

# Nombre de groupes pour la GroupNorm (figée comme normalisation par défaut :
# indépendante du batch_size, plus stable que BN à petit batch).
NORM_GROUPS = 8

# Warmup linéaire du LR sur N epochs (0 = désactivé). Utile pour stabiliser
# les premières epochs avec AdamW + GroupNorm. Modifiable depuis la GUI.
LR_WARMUP_EPOCHS = 2


# ---------------------------------------------------------------------------
# Mode de sortie du modèle : masque magnitude (p2) vs complex spectral mapping
# ---------------------------------------------------------------------------
# "mask"    : recette p2 historique. Le réseau prédit un masque réel ∈ [0,1]
#             appliqué à la magnitude du bruité ; reconstruction avec la phase
#             du bruité. Plafond de SI-SDR dur (~+12 dB SI-SDRi mesuré) car la
#             phase n'est jamais corrigée.
# "complex" : complex spectral mapping. Le réseau prédit directement le réel et
#             l'imaginaire du spectre propre (2 canaux) → il RÉCUPÈRE la phase,
#             ce qui permet de dépasser le plafond magnitude. Entrée = Re/Im du
#             bruité compressés par |·|^c ; sortie = Re/Im propre compressés ;
#             décompression puis ISTFT pour la waveform.
OUTPUT_MODE = "mask"

# Compression puissance des spectres complexes (mode "complex" uniquement).
# S_comp = |S|^c · e^{jθ}. c<1 rééquilibre le budget de gradient vers les bins
# faible énergie (même esprit que log1p en magnitude). 0.3 est un choix usuel
# (Tan & Wang, complex spectral mapping).
CSM_COMPRESS = 0.3

# Poids de la loss en mode "complex" :
#   L = w_ri  · [L1(Re) + L1(Im)]      (réel/imaginaire compressés)
#     + w_mag · L1(|·|)                (magnitude compressée — stabilise)
#     + w_mr  · MR-STFT(pred_wav, clean_wav)
CSM_LOSS_WEIGHTS = {
    "ri":      1.0,
    "mag":     1.0,
    "mr_stft": 1.0,
}
