# Guide d'installation et workflow

À lire **en entier avant de commencer à coder**. Ce document est ton point d'entrée unique : installation, workflow git, lancement d'une session, dépannage.

> ⏱️ Prévoir ~30 min pour le setup initial.

---

## Sommaire

1. [Cloner le repo](#1-cloner-le-repo)
2. [Installer Python 3.11](#2-installer-python-311)
3. [Créer le venv et installer les dépendances](#3-créer-le-venv-et-installer-les-dépendances)
4. [Installer nbstripout (filtre git pour les notebooks)](#4-installer-nbstripout-filtre-git-pour-les-notebooks)
5. [Configurer Google Drive](#5-configurer-google-drive)
6. [Installer VS Code + extensions](#6-installer-vs-code--extensions)
7. [Workflow git de l'équipe](#7-workflow-git-de-léquipe)
8. [Lancer une session de travail](#8-lancer-une-session-de-travail)
9. [Workflow après une modification de code](#9-workflow-après-une-modification-de-code)
10. [Dépannage (FAQ)](#10-dépannage-faq)
11. [Structure du repo](#11-structure-du-repo)
12. [Règles d'or](#12-règles-dor)

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

> On vise **Python 3.11** pour rester aligné avec Colab.

### Windows

1. Télécharger Python 3.11 depuis [python.org/downloads/release/python-3119](https://www.python.org/downloads/release/python-3119/)
2. **Cocher "Add to PATH"** pendant l'installation
3. Vérifier dans PowerShell :
   ```powershell
   py -0
   ```
   Doit lister `-V:3.11` parmi les versions installées.

> Si tu as déjà Python (ex: 3.12), garde-le, 3.11 sera installé à côté. Le launcher `py -3.11` permet de choisir explicitement.

### macOS

```bash
brew install python@3.11
python3.11 --version   # vérification
```

### Linux (Ubuntu/Debian)

```bash
sudo apt install python3.11 python3.11-venv
python3.11 --version
```

---

## 3. Créer le venv et installer les dépendances

Un environnement virtuel isole les libs du projet du Python système. **Indispensable** pour que tout le monde ait les mêmes versions.

### Windows (PowerShell)

Depuis le dossier du repo :

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Si PowerShell bloque l'activation avec une erreur d'exécution de scripts :
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> Le venv (`.venv/`) est dans `.gitignore`, il ne sera pas committé.

### Vérification

Après installation :
```powershell
python --version   # Python 3.11.x
pip list | findstr torch   # doit afficher torch 2.10.0 (CPU)
```

> Côté local, c'est la version **CPU** de torch qui est installée. Aucun GPU local n'est utilisé — l'entraînement se fait sur Colab.

### Sélectionner le venv dans VS Code

`Ctrl+Shift+P` → **Python: Select Interpreter** → choisir `.\.venv\Scripts\python.exe` (Windows) ou `.venv/bin/python` (macOS/Linux).

> Si l'interpréteur n'apparaît pas, clique **Enter interpreter path** → **Find** → navigue vers le binaire.

---

## 4. Installer nbstripout (filtre git pour les notebooks)

Évite que les sorties d'exécution, `execution_count`, et autres métadonnées volatiles polluent les diffs git. Sans ça, chaque ré-exécution d'une cellule génère un diff inutile.

```bash
# Depuis le venv activé
nbstripout --install --attributes .gitattributes
git config --local filter.nbstripout.extrakeys "metadata.kernelspec metadata.language_info"
git config --local core.autocrlf false
```

> **À refaire après chaque `git clone`** (ces trois réglages se font dans `.git/config`, qui n'est pas versionné).

- La 2ᵉ commande supprime les métadonnées de version kernel (Python 3.11 local vs 3.12 Colab)
- La 3ᵉ commande désactive la conversion automatique des fins de ligne de Windows — sans ça, git génère de faux diffs à chaque exécution de notebook

### Vérifier que ça marche

Ouvre un notebook, exécute une cellule sur Colab, puis :
```bash
git diff main.ipynb
```
Ne doit afficher **que les changements de code/markdown**, ni les outputs, ni les `execution_count`, ni les versions Python.

---

## 5. Configurer Google Drive

Le dataset, les checkpoints et les outputs sont stockés sur un **dossier Drive partagé** avec toute l'équipe. Le repo GitHub ne contient que le code.

### Accéder au Drive

1. Ouvrir : **[Dossier Drive partagé](https://drive.google.com/drive/folders/1z2M8g6Bhqr235VfNU5aKKxSQm8EQAQ94?usp=sharing)**
2. Le dossier apparaît dans **"Partagés avec moi"**
3. Clic droit sur `Filtre-Voix-DL` → **Organiser** → **Ajouter un raccourci** → **Mon Drive**
4. Vérifier qu'il est bien dans **Mon Drive**

> ⚠️ Sans le raccourci dans Mon Drive, Colab ne peut pas accéder au dossier (il cherche `/content/drive/MyDrive/Filtre-Voix-DL`).

### Structure du Drive

```
Filtre-Voix-DL/
├── data/
│   ├── clean/       # Voix propres (du dataset paire)
│   └── noisy/       # Voix bruitées correspondantes (du dataset paire)
├── checkpoints/     # Modèles sauvegardés pendant l'entraînement
└── outputs/         # Audio débruité produit par le modèle
```

Les dossiers `checkpoints/` et `outputs/` sont créés automatiquement par la cellule de setup du notebook si absents. Les dossiers `data/clean` et `data/noisy` doivent contenir le **dataset paire** téléchargé en amont par l'équipe (cf. [README.md](README.md) section Dataset).

---

## 6. Installer VS Code + extensions

1. Installer [VS Code](https://code.visualstudio.com/)
2. Extensions à installer :
   - **Google Colab** — exécution sur GPU Colab depuis VS Code
   - **Jupyter** — support `.ipynb`
   - **Python** — autocomplétion, linting

---

## 7. Workflow git de l'équipe

À **7 personnes sur un même repo**, la discipline git est essentielle. Le principe : personne ne push sur `main` directement, tout passe par des **Pull Requests**.

### Convention de nommage des branches

```
feature/<prenom>-<sujet>
```

Exemples :
- `feature/theo-dataset`
- `feature/lea-model`
- `feature/marc-training-loop`

### Créer ta branche au début d'une nouvelle tâche

Depuis ton PC, dans le repo :

```bash
git checkout main
git pull origin main                    # repartir de la version à jour
git checkout -b feature/<prenom>-<sujet>
git push -u origin feature/<prenom>-<sujet>
```

Le `-u` indique à git que cette branche locale suit la branche distante du même nom (utile pour les `git push` suivants).

### Travailler sur ta branche

Pendant ta tâche :
```bash
# modifier des fichiers ...
git add <fichiers>
git commit -m "Message clair décrivant ce que tu fais"
git push                                # va sur ta branche, pas sur main
```

Bonnes pratiques :
- **Commits petits et fréquents** (1 fonctionnalité = 1 commit) plutôt qu'un gros commit en fin de journée
- Messages explicites : `Ajoute classe AudioDataset` plutôt que `update`
- Si tu modifies du code de quelqu'un d'autre, vérifie sur quelle branche tu es (`git branch` affiche la branche active)

### Synchroniser ta branche avec `main` régulièrement

Pendant que tu bosses sur ta branche, les autres mergent leurs PRs dans `main`. Pour récupérer leurs changements :

```bash
git checkout main
git pull origin main
git checkout feature/<ta-branche>
git merge main                          # ou git rebase main si tu préfères
```

> À faire **au moins une fois par jour** si plusieurs personnes mergent en parallèle. Évite les conflits massifs en fin de projet.

### Ouvrir une Pull Request quand ta tâche est finie

1. Push ta dernière version : `git push`
2. Va sur [GitHub](https://github.com/Theo-Lempereur/Filtre-Voix-DL) → onglet **Pull requests** → **New pull request**
3. Base : `main` ← Compare : `feature/<ta-branche>`
4. Rédige un titre clair + description (qu'est-ce que ça apporte, comment tester)
5. Demande à **au moins un autre membre** de relire avant de merger
6. Une fois mergée, supprime ta branche (GitHub propose le bouton) et localement :
   ```bash
   git checkout main
   git pull
   git branch -d feature/<ta-branche>
   ```

### Règle d'or

> **Personne ne push directement sur `main`.** Toute modification de `main` passe par une PR avec relecture.

---

## 8. Lancer une session de travail

À chaque fois que tu ouvres le projet :

### Préparer ton environnement local

1. Ouvre le dossier `Filtre-Voix-DL` dans VS Code
2. Active le venv (VS Code le fait automatiquement si l'interpréteur est sélectionné — vérifie que `(.venv)` apparaît dans le terminal)
3. Mets ta branche à jour si tu reviens après un moment :
   ```bash
   git checkout feature/<ta-branche>
   git pull
   ```

### Ouvrir et configurer le notebook

4. Ouvre `main.ipynb`
5. **Connecte-toi à un runtime Colab** (bouton **Colab** ou kernel selector en haut à droite → choisir un runtime Colab GPU T4)
6. **Modifie la variable `BRANCH`** dans la cellule clone pour pointer sur **ta** branche :
   ```python
   BRANCH = 'feature/<ton-prenom>-<sujet>'
   ```

### Exécuter les cellules de setup dans l'ordre

| Cellule | Rôle | Temps |
|---|---|---|
| `%pip install librosa==... soundfile==... wandb==...` | Force les versions (libs déjà préinstallées sur Colab) | ~20s |
| `from google.colab import drive; drive.mount(...)` | Monte ton Google Drive (autorise l'accès au popup) | ~10s |
| Cellule **clone** (`BRANCH = ...`) | Clone ou pull ta branche dans `/content/Filtre-Voix-DL` | ~5s |
| `importlib.reload(config)` | Force le re-import si la cellule a déjà été exécutée | <1s |
| Cellule **autoreload** | Tente d'activer l'auto-reload — affiche un message "non chargé" sur Colab Python 3.12 (bug connu, sans impact) | <1s |
| `from src import config` + création dossiers | Importe les constantes partagées + crée les dossiers Drive si absents | ~2s |
| Cellule vérification | Affiche l'arborescence Drive | <1s |

À la fin, tu dois voir :
```
Sample rate : 16000 Hz | clip : 4.0s | n_fft=512, hop=128
Drive prêt !
Filtre-Voix-DL/
  data/
    ...
```

> L'environnement Colab est éphémère : pip install et mount Drive sont à refaire à **chaque session**. Le runtime se déconnecte après ~90 min d'inactivité.

---

## 8 bis. Entraînement local long (hors Colab)

Colab reste idéal pour **tester rapidement** une modif (dataset, model, mini-train de 2 epochs). Mais pour les **sessions longues** (≥ 1h), les coupures Colab (~90 min d'inactivité, déconnexion GPU) deviennent un problème. On lance alors l'entraînement **en local**, sur le GPU de la personne disponible.

> Le code est le même qu'en Colab : c'est juste un autre point d'entrée (`scripts/train_local.py`). Pas de fork, pas de duplication.

### Pré-requis

1. **Google Drive for Desktop** installé et synchronisant le dossier `Filtre-Voix-DL` localement.
   - Windows : le dossier apparaît sous `G:\My Drive\Filtre-Voix-DL` (ou `H:\` selon ton install).
   - macOS : `~/Library/CloudStorage/GoogleDrive-<email>/My Drive/Filtre-Voix-DL`.
   - Vérifie que `data/train/`, `data/val/`, `checkpoints/` y sont bien présents (sync terminée).
   - Si ton chemin Drive est inhabituel, définis la variable d'environnement `FILTRE_VOIX_DL_ROOT` vers le chemin local.

2. **Accélération matérielle** (sinon l'entraînement tournera sur CPU, ~50× plus lent — inadapté pour du long).

   La sélection de device est automatique : `train.py` prend **CUDA → MPS → CPU** dans cet ordre. Selon ta plateforme :

   | Plateforme | Action | Device utilisé |
   |---|---|---|
   | **Windows / Linux + GPU NVIDIA** (RTX 20/30/40/50) | `pip uninstall -y torch torchaudio` puis `pip install -r requirements-gpu.txt` (CUDA 12.8) | `cuda` |
   | **Mac Apple Silicon** (M1/M2/M3/M4) | Rien à faire — MPS (Metal) est inclus dans torch stock | `mps` |
   | **Mac Intel** | Rien à faire — pas d'accélération possible sur Mac Intel | `cpu` (déconseillé pour long) |
   | **Windows / Linux sans GPU NVIDIA** | Rien à faire | `cpu` (déconseillé pour long) |

   Vérifier ce qui est dispo :
   ```powershell
   # Windows / Linux NVIDIA
   python -c "import torch; print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
   # Mac Silicon
   python -c "import torch; print('mps:', torch.backends.mps.is_available())"
   ```

   > Pour les RTX 50 (Blackwell), CUDA 12.8 est **obligatoire** — c'est ce que pointe `requirements-gpu.txt`. Pour des GPU plus anciens, cu128 reste compatible.

   > Si tu n'as pas de GPU local, coordonne avec un équipier qui en a un : le lock partagé sur Drive est justement fait pour ça.

### Lancer une session

Depuis la racine du repo, venv activé :

```powershell
# Lancement nominal (run de nuit)
python scripts/train_local.py --run-id unet_big --epochs 100 --batch-size 16

# Surcharge fine via fichier JSON
python scripts/train_local.py --run-id unet_v2 --config-file configs/v2.json

# Reprise après crash / coupure
python scripts/train_local.py --run-id unet_big --resume last
```

Tous les checkpoints, logs et outputs vont sur **Drive** (donc partagés avec l'équipe en temps réel). Le stdout est dupliqué dans `<DRIVE>/logs/<run-id>.log` : n'importe quel membre peut suivre la session depuis son poste.

### Le lock d'entraînement partagé

Pour éviter que deux personnes lancent un entraînement en même temps (collision Drive, confusion d'équipe), le script pose un fichier de lock sur Drive : **`<DRIVE>/.training_lock.json`**.

- Il vit **hors git** (Drive uniquement) → il traverse les branches, les machines et les comptes.
- Tant qu'il existe et qu'il a un heartbeat récent (< 5 min), tout autre lancement est refusé avec un message identifiant le détenteur (utilisateur, machine, branche, run_id, PID).
- Le heartbeat est mis à jour toutes les 60 s pendant l'entraînement.
- Au-delà de 5 min sans heartbeat (= process mort, crash, coupure de courant), le lock est considéré comme stale et le prochain `train_local.py` le reprend automatiquement avec un warning.

**Voir qui entraîne en ce moment** (depuis n'importe quel poste) :

```powershell
# Windows
type "G:\My Drive\Filtre-Voix-DL\.training_lock.json"
# macOS / Linux
cat ~/Google\ Drive/My\ Drive/Filtre-Voix-DL/.training_lock.json
```

**Forcer un démarrage** si tu es certain que l'autre run est mort (et que ton équipier ne répond pas sur Discord) :

```powershell
python scripts/train_local.py --run-id ... --force
```

> ⚠️ Avant `--force`, demande sur le canal de l'équipe. Deux entraînements simultanés sur Drive saturent la sync et peuvent corrompre les checkpoints.

**Sémantique de `--force`** — c'est juste un override du *check* de lock, **pas un kill switch**. Concrètement :

- `--force` écrase le fichier lock avec ton payload et démarre ton run.
- Si l'autre run est réellement mort (crash, fenêtre fermée) → tout va bien, tu prends la suite.
- Si l'autre run est encore vivant → au prochain heartbeat (max 60 s plus tard), il détecte que le lock ne lui appartient plus et **s'arrête tout seul** avec `LockStolenError`. Pendant ces ≤ 60 s, deux trainings tournent en parallèle.
- `--force` ne tue jamais un autre process directement — c'est impossible de manière fiable cross-machine via Drive.

Donc : utilise `--force` uniquement si tu es **sûr** que l'autre est mort. Sinon préviens d'abord pour qu'il arrête proprement (Ctrl+C).

### Quand utiliser quoi

| Tu veux… | Outil | Pourquoi |
|---|---|---|
| Tester une modif `src/*.py` en 5 min | **Colab** + `notebooks/04_training_session.ipynb` avec `max_train_samples=20` | Pas besoin de GPU local, setup en 2 cellules |
| Explorer/visualiser les données | **Colab** ou local CPU + notebooks 00-02 | Pas besoin d'entraîner |
| Entraînement complet (≥ 1h) | **Local** + `scripts/train_local.py` | Pas de coupure 90 min, GPU stable |
| Entraînement de nuit | **Local** + `scripts/train_local.py` en `Start-Process` détaché | Survit à la fermeture de VS Code |

---

## 9. Workflow après une modification de code

C'est **le cycle le plus important à maîtriser** pour bosser efficacement.

### Le problème

Quand tu modifies un fichier `src/*.py` sur ton PC, Colab ne le voit **pas automatiquement**. Le code modifié existe à 4 niveaux indépendants :

```
[1] Ton PC (fichier .py modifié)
     ↓ git push
[2] GitHub
     ↓ git pull (sur Colab)
[3] Colab — disque (/content/Filtre-Voix-DL/src/...)
     ↓ reload / restart kernel
[4] Colab — mémoire Python (modules déjà importés)
```

### Le cycle complet

**Sur ton PC :**
```bash
# 1. Modifier src/dataset.py (ou autre)
git add src/dataset.py
git commit -m "Ajoute lecture des fichiers audio"
git push                                # part sur GitHub (ta branche)
```

**Dans Colab :**
1. **Ré-exécute la cellule clone** → fait `git pull`, ton disque Colab est à jour
2. **Ré-exécute la cellule reload** (`importlib.reload(config)`) → ou utilise `importlib.reload(<module>)` pour d'autres modules
3. **Continue avec tes cellules de travail**

### Alternative simple : Restart kernel

Si la situation devient confuse (plusieurs modules à recharger, état bizarre), le plus sûr est :

**Runtime → Restart session** dans Colab, puis **Run All** depuis le haut du notebook.

Ça repart d'un état propre en ~20 secondes. Coût marginal en DL où une époque d'entraînement prend des minutes.

### ⚠️ Le piège classique

Si tu modifies un fichier **localement** sans faire `git push`, puis tu ré-exécutes la cellule clone sur Colab → tu rechargeras l'**ancienne** version (celle qui est sur GitHub). Toujours `push` avant de retester sur Colab.

---

## 10. Dépannage (FAQ)

### `Autoreload non chargé (No module named 'imp')`

**Bug Colab connu** : Python 3.12 a supprimé le module `imp`, mais l'IPython installé sur Colab l'utilise encore dans `autoreload`.

→ **Sans impact**, la cellule continue. Utiliser `importlib.reload(<module>)` ou **Restart kernel** quand tu modifies un fichier dans `src/`.

### `Drive already mounted at /content/drive`

Pas une erreur, juste une information. Tu peux passer à la cellule suivante.

### `ModuleNotFoundError: No module named 'src'`

La cellule clone n'a pas tourné (ou pas tourné avec succès). Vérifie qu'elle a affiché `Branche : ...` et `HEAD : ...`.

### `userdata.SecretNotFoundError` ou `TimeoutException` sur secrets

Tu as une vieille version du notebook qui demande un token GitHub. Pull la dernière version de ta branche — le repo est public, plus besoin de token.

### `pip install` plante avec un conflit de versions

Ton venv est probablement pollué. Recrée-le :
```powershell
deactivate
Remove-Item -Recurse -Force .venv
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### `nbstripout` ne strip pas les outputs

Tu l'as installé dans le venv mais pas configuré pour ce clone. Re-fais les trois commandes de la section 4 :
```bash
nbstripout --install --attributes .gitattributes
git config --local filter.nbstripout.extrakeys "metadata.kernelspec metadata.language_info"
git config --local core.autocrlf false
```

### Git me montre des notebooks "modifiés" alors que je n'ai rien changé

Cause probable : `core.autocrlf=true` dans ta config git globale Windows. Git convertit les fins de ligne au checkout, ce qui crée de faux diffs.

Fix en deux commandes :
```bash
git config --local core.autocrlf false
git restore .
```
`git status` doit ensuite afficher `nothing to commit`. Si tu viens de cloner le repo, fais plutôt les 3 commandes de la section 4 d'un coup.

### Git me montre des diffs de version Python (`3.11` → `3.12.13`) dans les notebooks

Colab écrit sa version Python dans les métadonnées du notebook à chaque run. C'est du bruit pur, à ne pas committer.

Fix : la deuxième commande de la section 4 règle ça. Si tu ne l'as pas encore faite :
```bash
git config --local filter.nbstripout.extrakeys "metadata.kernelspec metadata.language_info"
```
Puis annule les changements en attente : `git checkout -- *.ipynb`

### Le runtime Colab est lent / je n'ai pas de GPU

Vérifie : **Runtime → Change runtime type → T4 GPU**. Le runtime gratuit est limité, parfois indisponible aux heures de pointe.

### J'ai un conflit de merge entre `main` et ma branche

Demande de l'aide à quelqu'un de l'équipe qui maîtrise git. Surtout **ne pas faire `git reset --hard`** sans comprendre — tu peux perdre du travail.

### J'ai pushé un `.wav` ou un `.pth` par erreur

```bash
git rm --cached <fichier>
git commit -m "Retire <fichier> du tracking"
git push
```
Note que le fichier reste dans l'historique git. Si c'était volumineux ou sensible, demande de l'aide.

### Le secret token GitHub est dans mon historique de commits

Va sur [github.com/settings/tokens](https://github.com/settings/tokens) et **révoque-le immédiatement**. Puis génère-en un nouveau si besoin.

---

## 11. Structure du repo

```
Filtre-Voix-DL/
├── main.ipynb              # Notebook principal (setup + point d'entrée Colab)
├── README.md               # Contexte et objectifs du projet
├── SETUP.md                # Ce fichier
├── requirements.txt        # Dépendances Python (CPU, alignées Colab)
├── requirements-gpu.txt    # Surcouche CUDA pour entraînement local GPU
├── .python-version         # Python 3.11
├── .gitattributes          # Filtre nbstripout
├── .gitignore              # Exclut .wav, .pth, .venv, etc.
├── src/                    # Modules Python réutilisables
│   ├── __init__.py
│   ├── config.py           # Constantes partagées + détection racine projet
│   ├── lock.py             # Lock d'entraînement partagé via Drive
│   ├── train.py            # Boucle d'entraînement
│   └── ...                 # model, dataset, checkpoint, metrics, audio, logging_utils
├── scripts/
│   ├── split_data.py       # Split train/val/test du dataset
│   └── train_local.py      # CLI entraînement local (hors Colab)
└── notebooks/              # Notebooks par étape du pipeline
```

À mesure que le projet avance, `src/` accueillera :
- `dataset.py` — PyTorch Dataset pour les paires (audio bruité, audio propre)
- `model.py` — Architecture U-Net
- `train.py` — Boucle d'entraînement
- `audio.py` — Helpers STFT/ISTFT
- `evaluate.py` — Métriques (SNR, PESQ, etc.)

---

## 12. Règles d'or

1. **Toujours bosser dans un venv activé** localement — sinon les versions divergent entre membres.
2. **Toujours bosser sur ta branche**, jamais directement sur `main`.
3. **`git pull` ta branche avant de commencer**, `git push` souvent.
4. **Avant de tester une modif sur Colab : commit + push.** Sinon Colab récupère l'ancienne version.
5. **Ne jamais commit** de fichiers audio (`.wav`), de modèles (`.pth`), de tokens/clés API.
6. **Ne jamais push sur `main`** directement — toute modif passe par une PR.
7. **Pull Request review** : faire relire ton code par au moins une personne avant de merger.
8. **En cas de doute git** (conflit, état bizarre, perte potentielle) : demande de l'aide avant de faire `--hard` quoi que ce soit.
9. **Si vous utilisez Azure** (GPU de secours) : toujours stopper la VM après l'entraînement (~1$/h).

---

*Document maintenu par l'équipe — si tu trouves une étape qui manque ou qui ne marche plus, ouvre une PR pour l'améliorer.*
