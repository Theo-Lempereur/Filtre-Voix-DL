# 🎙️ Projet de fin d'année — Outil de débruitage audio par Deep Learning

## Contexte

Projet de fin d'année réalisé par une équipe de **3 étudiants en informatique**, sur une durée de **5 semaines**. L'équipe n'a pas de connaissances préalables en Machine Learning ou Deep Learning. L'objectif pédagogique est double : livrer un outil fonctionnel et acquérir des bases solides en DL en préparation d'une **spécialisation IA l'année suivante**.

---

## Objectif du projet

Construire un outil de **débruitage audio** capable d'isoler une voix humaine dans un enregistrement contenant des bruits de fond ambiants courants.

> L'outil prend en entrée un fichier audio bruité, et retourne un fichier audio avec la voix isolée. **Pas de temps réel** — traitement par fichier.

### Bruits ciblés
- Bruit blanc / souffle
- Café / restaurant
- Brouhaha (foule, voix multiples en arrière-plan)
- Bruit de rue / circulation

Ce périmètre correspond aux bruits présents dans le dataset paire utilisé pour l'entraînement. D'autres types de bruits (ventilateur, clics, bruits domestiques spécifiques) pourront être envisagés en extension si le temps le permet — via augmentation synthétique du dataset.

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

L'équipe utilise un **dataset public de paires `(audio bruité, audio propre)`** déjà annoté. Chaque exemple consiste en :
- Un enregistrement de voix humaine bruitée (~4 secondes en moyenne, jusqu'à ~10 secondes)
- La version propre correspondante de la même voix
- Le bruit de fond couvre les catégories listées ci-dessus (blanc, café, brouhaha, rue)

**Avantages de cette approche** :
- Pas besoin de générer le mélange synthétiquement → pipeline plus simple, gain de temps
- Distribution de bruit réaliste (enregistrements en conditions ambiantes)
- Dataset déjà aligné et validé

> Le dataset est stocké sur le Drive partagé (`data/clean/` et `data/noisy/` ou structure équivalente fournie par le dataset). Le repo GitHub ne contient pas les fichiers audio.

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
- **Google Drive partagé** pour le stockage des données et checkpoints
- **Azure Education (100$)** en secours si Colab coupe pendant les longs entraînements

> Voir **[SETUP.md](SETUP.md)** pour le guide d'installation complet.

### Interface utilisateur (optionnel)
- `gradio` ou `streamlit` — interface web simple en Python pur
- Ou front HTML/JS appelant le modèle via une API Python

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
- L'exploration et l'exploitation d'un dataset audio annoté
- L'utilisation d'infrastructure cloud GPU (Google Colab)
- Le suivi et l'évaluation d'un modèle DL

---

*Document généré en début de projet — à mettre à jour au fil de l'avancement.*