# Guide d'installation

Tout ce qu'il faut faire **avant de commencer à coder**. A suivre dans l'ordre.

---

## 1. Cloner le repo

```bash
git clone https://github.com/Theo-Lempereur/Filtre-Voix-DL.git
```

---

## 2. Installer VS Code + extensions

1. Installer [VS Code](https://code.visualstudio.com/)
2. Installer les extensions suivantes :
   - **Google Colab** — permet d'exécuter les notebooks sur les GPUs de Colab depuis VS Code
   - **Jupyter** — support des fichiers `.ipynb`
   - **Python** — support Python (autocomplétion, linting)

---

## 3. Configurer Google Drive

Le dataset et les checkpoints sont stockés sur un **dossier Drive partagé** avec toute l'équipe. Le repo GitHub ne contient que le code.

### Accéder au Drive

1. Ouvrir ce lien : **[Dossier Drive partagé](https://drive.google.com/drive/folders/1z2M8g6Bhqr235VfNU5aKKxSQm8EQAQ94?usp=sharing)**
2. Le dossier apparaît dans **"Partagés avec moi"**
3. Clic droit sur le dossier `Filtre-Voix-DL` → **Organiser** → **Ajouter un raccourci** → **Mon Drive**
4. Vérifier que le dossier est visible dans **Mon Drive**

> **Important** : sans le raccourci dans Mon Drive, Colab ne pourra pas accéder au dossier.

### Structure du Drive

```
Filtre-Voix-DL/
├── data/
│   ├── clean/       # Voix propres (sans bruit)
│   ├── noise/       # Bruits isolés (blanc, ventilateur, clics)
│   └── mixed/       # Mélanges voix + bruit (entrées du modèle)
├── checkpoints/     # Modèles sauvegardés pendant l'entraînement
└── outputs/         # Audio débruité en sortie
```

---

## 4. Lancer une session de travail

A chaque session (chaque fois que tu ouvres le projet) :

1. Ouvrir le dossier `Filtre-Voix-DL` dans VS Code
2. Ouvrir `main.ipynb`
3. Se connecter à un runtime Colab via l'extension (bouton en haut à droite du notebook)
4. Exécuter les cellules de setup (pip install + mount Drive + création dossiers)

> Les packages (`librosa`, `wandb`, etc.) et le montage Drive sont à refaire à chaque session — l'environnement Colab est éphémère.

---

## 5. Structure du repo

```
Filtre-Voix-DL/
├── main.ipynb           # Notebook principal (setup + point d'entrée)
├── README.md            # Contexte et objectifs du projet
├── SETUP.md             # Ce fichier
├── .gitignore           # Exclut les .wav, .pth, etc.
├── src/                 # Modules Python réutilisables (model, dataset, utils)
└── notebooks/           # Notebooks par étape du pipeline
```

---

## Rappels

- **Ne jamais commit** de fichiers audio (`.wav`) ou de modèles (`.pth`) sur GitHub — ils vont sur Drive
- Les **checkpoints** permettent de reprendre un entraînement interrompu (par Colab ou volontairement)
- Si vous utilisez **Azure** comme GPU de secours : toujours stopper la VM immédiatement après l'entraînement (~1$/h)
