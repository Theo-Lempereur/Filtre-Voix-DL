# 🎙️ Wallace

**Un débruiteur de voix par deep learning — qui s'installe en une ligne, traite un audio de n'importe quelle durée, et restaure la voix au lieu de simplement l'atténuer.**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Modèle](https://img.shields.io/badge/mod%C3%A8le-production-2ea44f)
![SI--SDRi](https://img.shields.io/badge/SI--SDRi-~%2B11%20dB-blue)
![Installation](https://img.shields.io/badge/pip-depuis%20n'importe%20o%C3%B9-orange)

Wallace transforme une voix noyée dans le bruit de fond en une voix claire.
Au cœur : un **U-Net** entraîné sur spectrogrammes en **complex spectral mapping**,
qui prédit le spectre complexe propre (partie réelle + imaginaire) pour **récupérer
la phase** — là où un masque de magnitude classique plafonne, lui va plus loin.

```txt
voix + bruit de fond  ──►  U-Net (complex spectral mapping)  ──►  voix restaurée
```

---

## ✨ En bref

| | |
|---|---|
| 🎯 **Qualité** | ~**+11 dB de SI-SDRi** — restaure la voix, ne se contente pas de soustraire du bruit |
| ⏱️ **Durée libre** | Tout son, de la seconde à l'heure : fenêtrage 4 s + **overlap-add 50 %**, coutures inaudibles |
| 📦 **Portable** | Paquet **pip installable depuis n'importe où**, sans cloner le repo — modèle TorchScript embarqué |
| ⚡ **Rapide sur GPU** | ~**150× temps réel** sur GPU ; ~0,1× temps réel sur un CPU multi-cœurs |
| 🔌 **Prêt à servir** | API **FastAPI** clé en main + guide d'intégration web (JS / React / nginx / systemd) |
| 🛠️ **Chaîne complète** | Génération de dataset, entraînement, checkpoints, reprise, export — tout est ici |

---

## 🚀 Démarrage rapide — utiliser le modèle

Le modèle est distribué comme **paquet Python autonome**. Pas besoin de cloner le
dépôt, pas besoin du code d'entraînement : une URL suffit.

```bash
pip install https://github.com/Theo-Lempereur/Filtre-Voix-DL/releases/download/wallace-v1.0.0/wallace-1.0.0-py3-none-any.whl
```

```python
from wallace import Wallace

box = Wallace()                                  # modèle TorchScript embarqué, prêt à l'emploi
box.denoise_file("bruite.wav", "propre.wav")     # fichier → fichier, n'importe quelle durée

# ... ou en mémoire, pour un service :
wav_bytes = box.denoise_bytes(open("bruite.wav", "rb").read())   # bytes → bytes (WAV 16 kHz)

# ... ou sur GPU pour le temps réel :
box = Wallace(device="cuda")
```

Le paquet ne dépend que de `torch`, `numpy`, `soundfile`, `librosa` — **aucune
dépendance au code d'entraînement**. Contrat d'entrée : le modèle travaille en
**mono 16 kHz** ; `denoise(wav, sr=...)` met en mono et rééchantillonne pour vous.

> Détails d'intégration, (ré)génération et publication du paquet :
> [export/wallace/README.md](export/wallace/README.md).

---

## 📊 Performance

| Métrique | Valeur | Note |
|---|---|---|
| **SI-SDRi** | ~**+11 dB** | amélioration mesurée vs signal bruité (split test) |
| **Latence GPU** | ~**150× temps réel** | 30 s d'audio traité en ~0,2 s |
| **Latence CPU** | ~**0,1× temps réel** | 6 s d'audio en ~0,8 s sur un CPU 8 cœurs |
| **Format** | mono 16 kHz | sortie WAV, quel que soit le format d'entrée |

> Le modèle est *compute-bound* : une seule fenêtre de 4 s sature déjà un CPU
> multi-cœurs. **Pour du débit ou du temps réel, déployez sur GPU** — c'est la
> configuration pour laquelle il est conçu.

---

## 🧠 Comment ça marche

Le pipeline d'inférence est encapsulé dans une seule « boîte » réutilisable
([src/denoiser.py](src/denoiser.py), et sa copie autonome dans le paquet) :

1. **STFT** du signal bruité (`n_fft=512`, `hop=128`, 16 kHz).
2. **Power-compression** des parties réelle/imaginaire (`|S|^(c-1)`, `c=0.3`).
3. **U-Net** (encodeur/décodeur 4 niveaux, `GroupNorm`) → spectre complexe propre estimé.
4. **Décompression** + **ISTFT** → forme d'onde, **avec la phase reconstruite**.
5. Pour les sons > 4 s : **fenêtrage glissant 4 s + overlap-add pondéré (Hann, 50 %)**
   → reconstruction COLA-exacte, sans artefact de couture. Les fenêtres sont
   traitées **par batch** (mémoire bornée).

Le mode historique `mask` (`sigmoid × magnitude`, phase bruitée conservée) reste
supporté pour rétrocompatibilité, mais le **mode de production est `complex`** :
en apprenant la phase, il dépasse le plafond intrinsèque du masque de magnitude.

> Documentation technique complète (données, STFT, pertes, métriques, entraînement) :
> [MEMOIRE_TECHNIQUE.md](MEMOIRE_TECHNIQUE.md).

---

## 🔧 Construire, entraîner, réentraîner

Le dépôt couvre **toute la chaîne**, pas seulement l'inférence.

### Installation (développement)

Python 3.11 attendu. Voir [SETUP.md](SETUP.md) pour le guide complet (venv, Drive, GPU).

```bash
pip install -r requirements.txt            # base (CPU)
pip install -r requirements-serve.txt      # pour servir l'API
pip install -r requirements-gpu.txt        # surcouche CUDA 12.8 (entraînement GPU)
```

FFmpeg est recommandé si les sources contiennent du `.mp3`, `.m4a`, `.aac`, `.ogg`.

### 1. Générer un dataset

Paires `noisy/clean` alignées, avec SNR contrôlé et augmentations. Interface visuelle :

```bash
python scripts/run_full_pipeline.py --gui          # configuration (novice / expert)
python scripts/run_full_pipeline.py --compile --reset-all   # tout régénérer
python scripts/check_dataset.py                    # vérifier le dataset
```

Détail des étapes et de la config : [data/README.md](data/README.md). La config
active est `configs/dataset_config.yaml` ; chaque génération archive son
`config_used.yaml` pour une traçabilité totale.

### 2. Entraîner

```bash
python scripts/build_wav_cache.py --splits train val        # cache waveform (gros datasets)
python scripts/train_local.py --run-id mon_run --epochs 100 # entraînement
python scripts/train_local.py --run-id mon_run --resume last # reprise après coupure
```

Checkpoints produits : `best_si_sdri.pt`, `best.pt`, `last.pt`. Logs, checkpoints
et outputs vont sur la racine Drive du projet (survivent aux changements de machine).
Pour les gros runs sur GPU loué, voir [RUNPOD.md](RUNPOD.md).

### 3. Exporter un nouveau modèle

```bash
python scripts/export_model.py --ckpt mon_run       # → model.ts + metadata.json
```

Génère le paquet `wallace` autonome (TorchScript + métadonnées). Procédure
complète (build du wheel, publication en GitHub Release) :
[export/wallace/README.md](export/wallace/README.md).

---

## 🔌 Servir le modèle (API)

Une API **FastAPI** prête à l'emploi expose le débruiteur en HTTP.

```bash
python -m uvicorn serve.server:app --host 127.0.0.1 --port 8000 --workers 1
```

```txt
GET  /health    état du serveur + checkpoint chargé
GET  /models    checkpoints détectés
POST /denoise   fichier audio (durée quelconque) → WAV débruité 16 kHz mono
```

Intégration web complète (JavaScript, React, nginx, systemd, CORS) :
[serve/INTEGRATION.md](serve/INTEGRATION.md).

---

## 📁 Structure du projet

```txt
src/
├── denoiser.py              Boîte d'inférence réutilisable (durée libre, batch, mask/complex)
├── model.py                 Architecture U-Net
├── train.py                 Boucle d'entraînement
├── dataset.py               Dataset PyTorch (paires noisy/clean)
├── metrics.py               Pertes, SI-SDR, suivi d'overfitting
├── config.py                Chemins, audio, hyperparamètres
└── dataset_builder/         Préparation et génération du dataset
configs/dataset_config.yaml  Configuration dataset active
scripts/
├── run_full_pipeline.py     Pipeline dataset (GUI + CLI)
├── train_local.py           Entraînement local (lock partagé, reprise)
├── export_model.py          Génère le paquet wallace depuis un checkpoint
├── denoise_file.py          Débruite un fichier en CLI
└── listen_test.py           Génère noisy/pred/clean pour l'écoute + SI-SDR
serve/                       API d'inférence FastAPI + guide d'intégration
export/wallace/       Paquet autonome téléchargeable (TorchScript)
gui/                         Interfaces de configuration et d'entraînement
MEMOIRE_TECHNIQUE.md         Documentation technique détaillée
SETUP.md                     Guide d'installation et workflow d'équipe
```

---

## 🗺️ Limites & pistes

- **Latence CPU** : le modèle privilégie la **qualité** ; pour du temps réel CPU
  (type DeepFilterNet), il faudrait un modèle plus léger (features ERB,
  convolutions séparables) — piste de réentraînement, hors périmètre actuel.
- **API mono-worker** : suffisant pour un usage modéré ; pour de la charge,
  plusieurs workers ou une file d'attente.
- **MP3 / OGG en entrée** : nécessitent FFmpeg sur la machine (WAV/FLAC sans).

---

## 📚 Index de la documentation

| Document | Pour qui / quoi |
|---|---|
| [README.md](README.md) | Vue d'ensemble (ce fichier) |
| [export/wallace/README.md](export/wallace/README.md) | **Utiliser le paquet** dans un service |
| [serve/INTEGRATION.md](serve/INTEGRATION.md) | **Intégrer l'API** à un site web |
| [data/README.md](data/README.md) | Pipeline de génération du dataset |
| [SETUP.md](SETUP.md) | Installation & workflow de développement |
| [RUNPOD.md](RUNPOD.md) | Entraînement sur GPU loué (RunPod) |
| [MEMOIRE_TECHNIQUE.md](MEMOIRE_TECHNIQUE.md) | Référence technique approfondie |

---

## ⚖️ Licence & contexte

Projet académique de débruitage vocal supervisé, développé en équipe. Les gros
fichiers (audio, caches, checkpoints, datasets générés) restent **hors Git**
(Google Drive). `config_used.yaml` doit accompagner tout dataset d'entraînement.
</content>
