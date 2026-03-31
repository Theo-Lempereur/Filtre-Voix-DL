# 🎙️ Projet de fin d'année — Outil de débruitage audio par Deep Learning

## Contexte

Projet de fin d'année réalisé en groupe de **7 étudiants en informatique**, sur une durée de **2 mois**. L'équipe n'a pas de connaissances préalables en Machine Learning ou Deep Learning. L'objectif pédagogique est double : livrer un outil fonctionnel et acquérir des bases solides en DL en préparation d'une **spécialisation IA l'année suivante**.

---

## Objectif du projet

Construire un outil de **débruitage audio** capable d'isoler une voix humaine dans un enregistrement contenant des bruits de fond connus et prédéfinis.

> L'outil prend en entrée un fichier audio bruité, et retourne un fichier audio avec la voix isolée. **Pas de temps réel** — traitement par fichier.

### Bruits ciblés (scope volontairement restreint)
- Bruit blanc
- Soufflerie / ventilateur
- Bruit de souris (clics)

Ce scope restreint est une décision intentionnelle : il rend le problème faisable en 2 mois tout en permettant d'apprendre vraiment.

---

## Approche technique

### Pipeline général

```
Audio bruité
     ↓
STFT (librosa) → Spectrogramme bruité (image 2D)
     ↓
U-Net (modèle PyTorch entraîné)
     ↓
Spectrogramme propre
     ↓
ISTFT → Audio propre ✅
```

### Architecture choisie : U-Net sur spectrogramme

Le son brut est converti en **spectrogramme** via une transformée de Fourier à court terme (STFT). Ce spectrogramme est une image 2D (temps × fréquences) sur laquelle un **réseau de neurones convolutif de type U-Net** est entraîné à supprimer le bruit.

Le U-Net est une architecture encoder/decoder avec des **skip connections**, initialement conçue pour la segmentation d'images médicales et parfaitement adaptée à ce type de problème.

### Pourquoi U-Net et pas autre chose ?

| Architecture | Verdict |
|---|---|
| **U-Net + spectrogramme** ✅ | Adapté, bien documenté, apprenable en 2 mois |
| RNN / LSTM | Plus complexe, moins pédagogique pour débuter |
| Transformer | Trop complexe, hors scope |

---

## Dataset

Le dataset sera **généré synthétiquement** par l'équipe :

1. Enregistrements de voix propres (membres du groupe + datasets publics libres)
2. Bruits ciblés enregistrés ou générés programmatiquement (bruit blanc via numpy)
3. Mélange voix + bruit à différents niveaux de volume → milliers d'exemples

Avantage : aucune dépendance à un dataset externe rare ou payant.

---

## Stack technique

### Langage
Tout en **Python** (standard du DL).

### Packages principaux

| Package | Usage |
|---|---|
| `torch` + `torchaudio` | Framework Deep Learning (PyTorch) |
| `librosa` | Traitement audio, STFT/ISTFT, spectrogrammes |
| `soundfile` / `scipy` | Lecture/écriture fichiers audio |
| `numpy` | Manipulation des données, génération bruit blanc |
| `matplotlib` | Visualisation des spectrogrammes |
| `tensorboard` ou `wandb` | Suivi de l'entraînement en temps réel |

### Environnement
- **VS Code** + extension **Google Colab** (exécution GPU distante)
- **Google Drive** pour le stockage des données et checkpoints

### Interface utilisateur (optionnel)
- `gradio` ou `streamlit` — interface web simple en Python pur
- Ou front HTML/JS appelant le modèle via une API Python

---

## Infrastructure d'entraînement

### Temps d'entraînement estimé (scope restreint)
| Matériel | Durée estimée |
|---|---|
| CPU seul | Plusieurs jours — inutilisable |
| Google Colab (GPU T4 gratuit) | 4–10 heures |
| Azure VM GPU (T4) | 2–5 heures |

### Environnement de développement

Le développement se fait en local sur **VS Code** avec l'**extension Google Colab**, qui permet d'exécuter les notebooks directement sur les GPUs de Colab depuis VS Code. Cela combine le confort de l'IDE local (git, extensions, autocomplétion) avec la puissance de calcul cloud.

| Phase | Outil |
|---|---|
| Édition du code | **VS Code** (local) + extension Colab |
| Exécution GPU (principal) | **Google Colab** (via l'extension VS Code) |
| Exécution GPU (secours) | **Azure Education (100$)** — si Colab coupe pendant les longs entraînements |
| Suivi des runs | **Weights & Biases (wandb)** — gratuit étudiants |

### ⚠️ Règle absolue Azure
> Toujours **stopper la VM immédiatement** après l'entraînement.
> Une VM oubliée = crédits qui fondent inutilement (~1$/h).

### Stockage des données

Le dataset et les checkpoints sont stockés sur **Google Drive**, monté dans Colab :

```python
from google.colab import drive
drive.mount('/content/drive')
# Les données sont dans /content/drive/MyDrive/Filtre-Voix-DL/data/
```

> Le repo GitHub contient le code uniquement. Les fichiers audio et modèles entraînés restent sur Google Drive (trop volumineux pour git).

### Checkpoints
L'entraînement peut être interrompu et repris grâce aux checkpoints PyTorch :

```python
# Sauvegarder
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, "checkpoint_epoch_10.pth")

# Recharger
model.load_state_dict(torch.load("checkpoint_epoch_10.pth"))
```

---

## Ce que le projet n'est PAS

- ❌ Un wrapper autour de Demucs / Spleeter / DeepFilterNet
- ❌ Un outil temps réel
- ❌ Un modèle généraliste capable de supprimer n'importe quel bruit

---

## Objectifs pédagogiques

À la fin du projet, l'équipe aura appris :
- Les bases du traitement du signal audio (STFT, spectrogrammes)
- L'architecture CNN / U-Net
- L'entraînement d'un réseau de neurones from scratch avec PyTorch
- La génération et gestion d'un dataset
- L'utilisation d'infrastructure cloud GPU (Google Colab)
- Le suivi et l'évaluation d'un modèle DL

---

*Document généré en début de projet — à mettre à jour au fil de l'avancement.*