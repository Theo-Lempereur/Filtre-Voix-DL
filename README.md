# Filtre-Voix-DL

Filtre-Voix-DL est un projet de denoising vocal supervise. Il genere des paires
audio `bruite/propre`, entraine un U-Net sur spectrogrammes, puis expose le
modele sous forme d'API pour filtrer une voix bruitée.

Pour le detail technique complet du modele, des donnees, de Google Drive et de
l'inference, voir [MEMOIRE_TECHNIQUE.md](MEMOIRE_TECHNIQUE.md).

## Objectif

Le modele apprend a transformer une voix bruitée en voix plus propre :

```txt
voix propre + bruit de fond controle
-> exemple bruite
-> U-Net de debruitage
-> estimation de la voix propre
```

Le projet couvre toute la chaine :

- preparation des audios propres et des bruits ;
- generation de paires alignées `noisy/clean` avec SNR controle ;
- entrainement PyTorch avec checkpoints, logs et reprise ;
- cache waveform pour accelerer l'entrainement ;
- API FastAPI pour tester ou integrer le modele.

## Installation

Python 3.11 est attendu.

```bash
pip install -r requirements.txt
```

Pour servir le modele via HTTP :

```bash
pip install -r requirements-serve.txt
```

FFmpeg est recommande si les sources audio contiennent du `.mp3`, `.m4a`,
`.aac`, `.ogg`, ou si les augmentations codec sont activees.

## Donnees

Les donnees brutes volumineuses restent hors Git, typiquement dans Google Drive.
La configuration active est :

```txt
configs/dataset_config.yaml
```

Les chemins peuvent utiliser `${DRIVE_PROJECT}` :

```yaml
clean_preprocessing:
  input_dir: "${DRIVE_PROJECT}/data/jules/wav"

noise_preprocessing:
  input_dir: "${DRIVE_PROJECT}/data/jules/noise/free-sound"
```

La racine Drive est detectee automatiquement par `src/config.py`. Elle peut etre
forcee avec :

```bash
set FILTRE_VOIX_DL_ROOT=G:/Mon Drive/Filtre-Voix-DL
```

## Pipeline Dataset

Ouvrir l'interface visuelle de configuration :

```bash
python scripts/run_full_pipeline.py --gui
```

Regenerer tout le dataset :

```bash
python scripts/run_full_pipeline.py --compile --reset-all
```

Regenerer seulement les paires finales apres modification des reglages :

```bash
python scripts/run_full_pipeline.py --skip-clean --skip-noise --reset-generated
```

Verifier le dataset :

```bash
python scripts/check_dataset.py
```

Sortie attendue :

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

Les fichiers `noisy` et `clean` portent le meme nom pour permettre un appairage
direct pendant l'entrainement.

## Entrainement

Construire le cache waveform, recommande pour les gros datasets :

```bash
python scripts/build_wav_cache.py --splits train val
```

Lancer un entrainement :

```bash
python scripts/train_local.py --run-id p2_baseline --epochs 50
```

Reprendre apres interruption :

```bash
python scripts/train_local.py --run-id p2_baseline --resume last
```

Checkpoints principaux :

```txt
checkpoints/<run_id>/last.pt
checkpoints/<run_id>/best.pt
checkpoints/<run_id>/best_si_sdri.pt
```

Par defaut, les checkpoints et les logs sont ecrits dans la racine Drive du
projet afin de survivre aux changements de machine.

## Modele

La baseline actuelle est un U-Net 2D applique aux spectrogrammes :

- entree : magnitude du signal bruite compressee par `log1p` ;
- architecture : encodeur/decodeur a 4 niveaux avec skip connections ;
- normalisation : `GroupNorm`, stable meme avec petits batchs ;
- sortie : masque reel `sigmoid` applique a la magnitude bruitée ;
- reconstruction : ISTFT avec la phase du signal bruite.

Le mode `complex` (complex spectral mapping) prédit le spectre complexe propre
(Re/Im) pour **récupérer la phase**, ce qui dépasse le plafond du masque
magnitude. C'est le mode du modèle de production `rp_csm_final`, et tout le chemin
d'inférence (`src/denoiser.py`, `serve/`, paquet `voice_denoiser`) le gère.

## API

Configurer un checkpoint dans `.env` :

```ini
CKPT_PATH=G:\Mon Drive\Filtre-Voix-DL\checkpoints\p2_baseline\best.pt
DEVICE=cpu
CORS_ORIGINS=*
```

Demarrer le serveur :

```bash
python -m uvicorn serve.server:app --host 127.0.0.1 --port 8000 --workers 1
```

Endpoints utiles :

```txt
GET  /health    etat du serveur et checkpoint charge
GET  /models    checkpoints detectes
POST /denoise   fichier audio -> WAV debruite 16 kHz mono
```

L'audio d'entrée peut être de **durée quelconque** : il est découpé en fenêtres
de 4 s avec recouvrement 50 % (overlap-add), géré par `src/denoiser.py`.

## Modèle téléchargeable (paquet `voice_denoiser`)

Le modèle est aussi distribué comme **paquet Python autonome**, installable
**depuis n'importe où, sans cloner le repo**, via la GitHub Release :

```bash
pip install https://github.com/Theo-Lempereur/Filtre-Voix-DL/releases/download/voice-denoiser-v1.0.0/voice_denoiser-1.0.0-py3-none-any.whl
```

```python
from voice_denoiser import VoiceDenoiser
box = VoiceDenoiser()                          # modèle TorchScript embarqué
box.denoise_file("bruite.wav", "propre.wav")   # son de durée quelconque
```

Le paquet embarque le modèle (TorchScript) + le pipeline ; il ne dépend que de
`torch`, `numpy`, `soundfile`, `librosa` (aucune dépendance au repo). Pour
l'intégrer à un service et pour (ré)générer/publier le paquet, voir
[export/voice_denoiser/README.md](export/voice_denoiser/README.md) et
`scripts/export_model.py`.

## Fichiers Principaux

```txt
src/model.py                    Architecture U-Net
src/train.py                    Boucle d'entrainement
src/dataset.py                  Dataset PyTorch paire noisy/clean
src/metrics.py                  Loss, SI-SDR, suivi d'overfitting
src/config.py                   Chemins Drive, audio, hyperparametres
src/denoiser.py                 Boite d'inference reutilisable (long-audio)
src/dataset_builder/            Preparation et generation du dataset
configs/dataset_config.yaml     Configuration dataset active
gui/                            Interfaces de configuration et training
serve/                          API d'inference FastAPI
export/voice_denoiser/          Paquet autonome telechargeable (TorchScript)
scripts/export_model.py         Genere le paquet voice_denoiser depuis un ckpt
data/README.md                  Guide court du pipeline dataset
MEMOIRE_TECHNIQUE.md            Documentation technique detaillee
```

## Notes

- Les notebooks ne sont pas necessaires pour lancer le pipeline principal.
- Les gros fichiers audio, caches, checkpoints et datasets generes ne doivent
  pas etre versionnes dans Git.
- `config_used.yaml` doit etre conserve avec tout dataset utilise pour entrainer
  un modele, car il documente exactement sa generation.
