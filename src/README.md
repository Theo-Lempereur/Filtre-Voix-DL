# src/ — cœur Python du projet

Modules réutilisables importés par les scripts (`scripts/`), l'API (`serve/`) et
les notebooks. Aucun import circulaire : `model.py` ne dépend que de PyTorch.

## Modules

| Fichier | Rôle |
|---|---|
| `config.py` | Constantes partagées (audio, STFT, hyperparamètres) + détection de la racine projet (Drive). |
| `model.py` | Architecture **U-Net** 2D (encodeur/décodeur 4 niveaux, `GroupNorm`), têtes `mask` et `complex`. |
| `denoiser.py` | **Boîte d'inférence** réutilisable `Wallace` : durée libre (overlap-add 50 %), batch, détection `mask`/`complex`, contrat d'entrée 16 kHz mono. Source unique du forward d'inférence. |
| `dataset.py` | `Dataset` PyTorch pour les paires `noisy/clean`. |
| `audio.py` | Helpers STFT/ISTFT, magnitude/phase, `fix_length`, chargement audio. |
| `metrics.py` | Pertes (MSE, L1 compressée, MR-STFT), SI-SDR / SI-SDRi, `OverfitMonitor`. |
| `train.py` | Boucle d'entraînement (checkpoints, logs, reprise, modes `mask`/`complex`). |
| `checkpoint.py` | Sauvegarde/chargement des checkpoints, `peek_checkpoint_config`, sélection `best`/`latest`. |
| `scheduler.py` | Planification du learning rate. |
| `lock.py` | Lock d'entraînement partagé via Drive (évite deux runs simultanés). |
| `stop_signal.py` | Arrêt propre d'un entraînement à distance (fichier signal). |
| `logging_utils.py` | Logging console + fichier (stdout dupliqué sur Drive). |
| `dataset_builder/` | Préparation des sources et génération des paires `noisy/clean`. |

## Utilisation

Depuis la racine du dépôt (venv activé) :

```python
from src.denoiser import Wallace

box = Wallace("checkpoints/rp_csm_final/best_si_sdri.pt")   # chemin souple : fichier, dossier ou nom de run
clean = box.denoise(noisy_wav, sr=44100)                     # numpy → numpy, durée et sr quelconques
box.denoise_file("bruite.wav", "propre.wav")                 # fichier → fichier
```

> Pour utiliser le modèle **sans le dépôt** (service externe), voir le paquet
> autonome [`wallace`](../export/wallace/README.md).
