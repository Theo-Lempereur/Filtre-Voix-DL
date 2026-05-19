# Guide d'installation

Tout ce qu'il faut faire **avant de commencer à coder**. À suivre dans l'ordre.

---

## 1. Cloner le repo

```bash
git clone https://github.com/Theo-Lempereur/Filtre-Voix-DL.git
cd Filtre-Voix-DL
```

---

## 2. Installer Python 3.11

Même si le code s'exécute sur Google Colab, il faut un Python local pour :
- l'extension Jupyter de VS Code (autocomplétion, linting)
- développer/tester les modules `src/` sans GPU
- exécuter les outils dev (`nbstripout`, etc.)

> On vise **Python 3.11** pour rester aligné avec Colab. Vérifiez la version exacte de votre runtime Colab avec une cellule `!python --version` et adaptez si besoin.

### Windows

Télécharger depuis [python.org/downloads](https://www.python.org/downloads/release/python-3119/) (cocher "Add to PATH" pendant l'install).

### macOS / Linux

```bash
# macOS
brew install python@3.11

# Ubuntu
sudo apt install python3.11 python3.11-venv
```

Vérifier :
```bash
python --version   # doit afficher Python 3.11.x
```

---

## 3. Créer le venv et installer les dépendances

Un environnement virtuel isole les libs du projet du Python système. **Indispensable** pour que tout le monde ait les mêmes versions.

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> Si PowerShell bloque l'activation : `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Le venv (`.venv/`) est dans `.gitignore`, il ne sera pas committé.

### Sélectionner le venv dans VS Code

`Ctrl+Shift+P` → **Python: Select Interpreter** → choisir `.venv/Scripts/python.exe` (Windows) ou `.venv/bin/python` (macOS/Linux).

---

## 4. Installer nbstripout (filtre git pour les notebooks)

Évite que les sorties d'exécution, `execution_count`, et autres métadonnées volatiles polluent les diffs git.

```bash
# Depuis le venv activé (nbstripout est dans requirements.txt)
nbstripout --install --attributes .gitattributes
```

> **À refaire après chaque `git clone`** (le filtre se configure dans `.git/config`, non versionné).

Vérifier que ça marche : ouvrir un notebook, exécuter une cellule, faire `git diff main.ipynb` → ne doit montrer que les changements de code, pas les outputs.

---

## 5. Choisir sa branche dans le notebook Colab

Le repo est public : pas d'auth nécessaire pour le clone côté Colab. Chaque membre travaille sur **sa propre branche** pour ne pas se marcher dessus.

### Convention de nommage des branches

```
feature/<prenom>-<sujet>
# ex: feature/theo-dataset, feature/lea-model
```

### Configurer la cellule clone

Dans [main.ipynb](main.ipynb), modifier la variable `BRANCH` en haut de la cellule clone :

```python
BRANCH = 'feature/theo-dataset'   # ou 'main', etc.
```

Le notebook va `clone` (première fois) ou `fetch + checkout + pull` (sessions suivantes) cette branche. Il affiche la branche active + le dernier commit pour vérification.

> **Important** : pour tester tes modifs locales sur Colab, il faut `commit + push` d'abord. La cellule `%autoreload` recharge automatiquement les modules si tu refais `!git pull` en cours de session.

---

## 6. Configurer Google Drive

Le dataset et les checkpoints sont stockés sur un **dossier Drive partagé** avec toute l'équipe. Le repo GitHub ne contient que le code.

### Accéder au Drive

1. Ouvrir : **[Dossier Drive partagé](https://drive.google.com/drive/folders/1z2M8g6Bhqr235VfNU5aKKxSQm8EQAQ94?usp=sharing)**
2. Le dossier apparaît dans **"Partagés avec moi"**
3. Clic droit sur `Filtre-Voix-DL` → **Organiser** → **Ajouter un raccourci** → **Mon Drive**
4. Vérifier qu'il est bien dans **Mon Drive**

> Sans le raccourci dans Mon Drive, Colab ne pourra pas accéder au dossier.

### Structure du Drive

```
Filtre-Voix-DL/
├── data/
│   ├── clean/       # Voix propres
│   ├── noise/       # Bruits isolés
│   └── mixed/       # Mélanges voix + bruit
├── checkpoints/     # Modèles sauvegardés
└── outputs/         # Audio débruité
```

---

## 7. Installer VS Code + extensions

1. Installer [VS Code](https://code.visualstudio.com/)
2. Extensions :
   - **Google Colab** — exécution sur GPU Colab depuis VS Code
   - **Jupyter** — support `.ipynb`
   - **Python** — autocomplétion, linting

---

## 8. Lancer une session de travail

À chaque session :

1. Ouvrir le dossier `Filtre-Voix-DL` dans VS Code
2. Activer le venv (VS Code le fait souvent automatiquement)
3. Ouvrir `main.ipynb`
4. Se connecter à un runtime Colab (bouton en haut à droite)
5. Exécuter les cellules de setup dans l'ordre :
   - `%pip install` (versions épinglées)
   - Mount Drive
   - Clone/pull du repo dans Colab
   - Import de `src.config` et création des dossiers
   - Vérification

> L'environnement Colab est éphémère : pip install et mount Drive sont à refaire à chaque session.

---

## 9. Structure du repo

```
Filtre-Voix-DL/
├── main.ipynb           # Notebook principal (setup + point d'entrée)
├── README.md            # Contexte et objectifs
├── SETUP.md             # Ce fichier
├── requirements.txt     # Dépendances Python (versions épinglées)
├── .python-version      # Python 3.11
├── .gitattributes       # Filtre nbstripout
├── .gitignore           # Exclut .wav, .pth, .venv, etc.
├── src/
│   ├── __init__.py
│   └── config.py        # Constantes partagées (chemins, params audio)
└── notebooks/           # Notebooks par étape du pipeline
```

---

## Rappels

- **Ne jamais commit** de fichiers audio (`.wav`) ou de modèles (`.pth`) sur GitHub — ils vont sur Drive
- **Toujours travailler dans le venv** activé (sinon les versions divergent)
- Les **checkpoints** permettent de reprendre un entraînement interrompu
- Si vous utilisez **Azure** comme GPU de secours : toujours stopper la VM après l'entraînement (~1$/h)
