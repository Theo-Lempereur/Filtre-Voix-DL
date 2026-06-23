# wallace — ajouter le débruiteur à un service

Boîte **autonome** : le modèle de débruitage de voix empaqueté avec ses outils,
importable **sans cloner le repo d'entraînement**. Une entrée (audio), une sortie
(audio débruité), durée **quelconque**.

---

## 1. Contenu du paquet

```
wallace/
├── __init__.py        # expose Wallace
├── denoiser.py        # la boîte (forward + fenêtrage 4 s + overlap-add 50 %)
├── model.ts           # le modèle TorchScript (architecture + poids embarqués)
└── metadata.json      # contrat d'entrée + provenance (run, epoch, SI-SDRi)
```

> `model.ts` et `metadata.json` sont **générés** par `scripts/export_model.py`
> (repo d'entraînement). Si tu n'as que la source du paquet, voir §6.

## 2. Installation

**Depuis n'importe où (recommandé)** — directement depuis la GitHub Release, sans
cloner le repo, juste une URL :

```bash
pip install https://github.com/Theo-Lempereur/Filtre-Voix-DL/releases/download/wallace-v1.0.0/wallace-1.0.0-py3-none-any.whl
```

**Depuis une copie locale du paquet** (si tu as le dossier avec `model.ts` dedans) :

```bash
pip install ./wallace
```

Dans les deux cas, les dépendances (`torch`, `numpy`, `soundfile`, `librosa`) sont
tirées automatiquement. **Aucune dépendance au repo d'entraînement.** Python ≥ 3.9,
CPU suffisant.

## 3. Contrat d'entrée

Le modèle a été entraîné en **mono 16 kHz**. `denoise(wav, sr=...)` met en mono et
rééchantillonne automatiquement si tu fournis `sr`. Respecter ce contrat est
important : un son fourni à un autre sample rate **sans** préciser `sr` sera mal
débruité.

## 4. Usage minimal

```python
from wallace import Wallace

box = Wallace()                       # trouve model.ts / metadata.json dans le paquet
                                      # Wallace(device="cuda") pour forcer le GPU

# Fichier -> fichier (n'importe quelle durée)
box.denoise_file("bruite.wav", "propre.wav")

# Tableau numpy -> tableau numpy
import numpy as np
clean = box.denoise(noisy_wav, sr=44100)   # resample + mono gérés

# Bytes -> bytes (WAV mono 16 kHz PCM 16) — pratique pour un service HTTP
wav_bytes = box.denoise_bytes(raw_audio_bytes)
```

## 5. Intégration dans un service (ex. notre `serve/` FastAPI)

Le modèle se charge **une fois** au démarrage, puis chaque requête appelle
`denoise_bytes`. Exemple complet d'un service minimal :

```python
# server.py
from fastapi import FastAPI, UploadFile, Response, HTTPException
from wallace import Wallace

app = FastAPI()
box: Wallace | None = None

@app.on_event("startup")
def _load():
    global box
    box = Wallace()                            # ou Wallace(device="cuda")

@app.get("/health")
def health():
    return {"ready": box is not None, "model": box.meta if box else None}

@app.post("/denoise")
async def denoise(file: UploadFile):
    if box is None:
        raise HTTPException(503, "model not ready")
    try:
        out = box.denoise_bytes(await file.read())
    except ValueError as exc:                  # audio illisible / vide / NaN
        raise HTTPException(400, str(exc))
    return Response(out, media_type="audio/wav")
```

Lancer : `uvicorn server:app`. Tester : `curl -F file=@bruite.wav http://localhost:8000/denoise --output propre.wav`.

> Dans **notre** repo, `serve/server.py` suit déjà ce schéma (chargement au
> lifespan + `denoise_bytes`). Pour le faire pointer sur le paquet exporté au lieu
> du chemin repo, il suffit de remplacer `serve.inference.load_model(...)` par
> `Wallace()` et `denoise_bytes(audio_bytes, model, ...)` par
> `model.denoise_bytes(audio_bytes)` — les signatures sont volontairement proches.

## 6. (Ré)générer et publier le modèle

Depuis le **repo d'entraînement** :

```bash
# 1. (re)générer model.ts + metadata.json depuis un checkpoint
python scripts/export_model.py --ckpt rp_csm_final

# 2. construire le wheel autonome (modèle embarqué)
python -m pip wheel --no-deps -w export/wallace/dist ./export/wallace

# 3. publier en GitHub Release -> installable par URL depuis n'importe où
gh release create wallace-v1.0.1 \
    export/wallace/dist/wallace-1.0.1-py3-none-any.whl \
    --title "wallace v1.0.1" --notes "nouveau modèle ..."
```

Bumpe la version dans `pyproject.toml` à chaque nouveau modèle. Pour juste tester
en local sans release : `pip install ./export/wallace`.

## 7. Provenance

Tout est traçable dans `metadata.json` (et `box.meta`) : `run_id`, `epoch`,
`val_si_sdri`, paramètres STFT, `csm_compress`, version de torch, date d'export.

---

### Notes

- **Mode** : ce paquet ne gère que le **complex spectral mapping** (le modèle de
  prod) ; il refuse un modèle d'un autre mode.
- **Vitesse** : les fenêtres de 4 s sont traitées **par batch** (`Wallace(
  batch_size=...)`, défaut 8, mémoire bornée) et le module TorchScript est passé
  dans `optimize_for_inference` au chargement. Le modèle est *compute-bound* : sur
  un CPU multi-cœurs il tourne déjà à ~0,1× temps réel et le batch n'y change rien
  (une fenêtre sature déjà les threads) ; **le vrai levier est le GPU** (~150×
  temps réel). Pour de la latence type temps-réel CPU, il faudrait un modèle plus
  léger (réentraînement) — hors scope du paquet.
- **Fidélité** : le forward du paquet est une copie exacte du chemin validé du
  repo (`src/denoiser.py`, mode complex) — sorties identiques à < 1e-4 près.
