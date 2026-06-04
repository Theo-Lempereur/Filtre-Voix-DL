# Entraînement headless sur RunPod (RTX 5090)

Guide pour lancer un gros run sur un GPU loué (RunPod), sans GUI, à distance.
Le code est le même qu'en local — seul le point d'entrée change
(`scripts/train_local.py`), et les chemins pointent vers un **Network Volume**
persistant au lieu de Google Drive.

> Pré-requis local : avoir déjà **construit le cache float16** (train + val) sur
> ta machine. C'est lui qu'on uploade — le pod ne décode aucun WAV.
> ```powershell
> python scripts/build_wav_cache.py --splits train val
> ```
> Cache produit dans `data/processed/cache/` :
> `train_noisy.npy`, `train_clean.npy`, `train_names.json` (+ idem `val_*`).

---

## 1. Stockage : un Network Volume (obligatoire)

| Type RunPod | Persistant ? | Verdict |
|---|---|---|
| Container disk | ❌ effacé au stop du pod | non |
| **Network Volume** (monté sur `/workspace`) | ✅ survit stop/restart | ✅ **à prendre** |

- **Taille** : **50 Go** (cache ×4 ≈ 18 Go + venv ~5 Go + checkpoints + marge).
- **Région** : choisis-en une où des **RTX 5090** sont disponibles — le volume
  est verrouillé sur sa région, le pod doit être dans la même.
- Coût : ~0,07 $/Go/mois (≈ 3,50 $/mois pour 50 Go), facturé **même pod éteint**.
  Tu peux le supprimer entre deux campagnes pour ne plus payer.

Arborescence cible sur le volume :
```
/workspace/
├── Filtre-Voix-DL/   # le repo cloné
├── venv/             # environnement Python (persistant)
├── cache/            # cache float16 uploadé  → FILTRE_VOIX_DL_CACHE
└── runs/             # checkpoints/ logs/ outputs/ + lock → FILTRE_VOIX_DL_ROOT
```

---

## 2. Créer le pod

1. **Template** : « RunPod PyTorch » récent avec **CUDA 12.8** (Blackwell/RTX 50).
   À défaut, n'importe quel template Ubuntu + CUDA 12.8 — le bootstrap réinstalle
   torch dans son propre venv de toute façon.
2. **GPU** : RTX 5090 (32 Go).
3. **Network Volume** : attache celui créé à l'étape 1, monté sur `/workspace`.
4. Démarre, puis connecte-toi en SSH (bouton « Connect » → SSH, ou le terminal web).

---

## 3. Bootstrap (une fois par pod neuf)

```bash
cd /workspace
git clone https://github.com/Theo-Lempereur/Filtre-Voix-DL.git   # si pas déjà là
cd Filtre-Voix-DL
git checkout <ta-branche>          # la branche contenant les optims + ces scripts

bash scripts/runpod_bootstrap.sh
```

Le script crée `runs/` et `cache/`, monte un venv sur le volume, installe
`requirements-train.txt` (torch cu128, sans GUI), et vérifie que CUDA voit le GPU.
Idempotent : relançable sans risque.

---

## 4. Uploader le cache float16 (~18 Go)

Le cache doit atterrir dans **`/workspace/cache/`**. Deux méthodes :

### Option A — `runpodctl` (peer-to-peer, simple)
Sur ta machine **locale** (Windows, depuis la racine du repo) :
```powershell
# Regroupe le cache en une archive (évite 6 transferts séparés)
tar -cf cache.tar -C data/processed/cache .
runpodctl send cache.tar
```
`runpodctl send` affiche un code. Sur le **pod** :
```bash
cd /workspace/cache
runpodctl receive <code-affiché>
tar -xf cache.tar && rm cache.tar
```

### Option B — `scp`/`rsync` over SSH
RunPod fournit une commande SSH (hôte + port). Depuis ta machine :
```bash
# rsync reprend là où il s'est arrêté en cas de coupure (recommandé pour 18 Go)
rsync -avP -e "ssh -p <PORT>" data/processed/cache/ root@<HOST>:/workspace/cache/
```

Vérifie sur le pod :
```bash
ls -lh /workspace/cache    # doit lister train_*.npy / val_*.npy + *_names.json
```

---

## 5. wandb (suivi live)

Tu as choisi wandb. Sur le pod, avant de lancer :
```bash
export WANDB_API_KEY="ta_clé"      # depuis wandb.ai/authorize
```
(ou ajoute la ligne dans `scripts/runpod_env.sh`, déconseillé si tu pushes le repo).
Sans clé, le logger retombe automatiquement sur le JSONL local — aucun crash.

---

## 6. Lancer un entraînement détaché

```bash
cd /workspace/Filtre-Voix-DL
export WANDB_API_KEY="ta_clé"

bash scripts/runpod_train.sh \
    --run-id rp_5090_bc48_bs64 \
    --epochs 80 \
    --batch-size 64 \
    --base-channels 48 \
    --lr 2e-3 \
    --compile
```

- Le run tourne dans une **session tmux** → survit à une coupure SSH.
- `--compile` active `torch.compile` (Triton dispo sous Linux → vrai gain, ~20-40 %).
- Le wrapper vérifie d'abord que le cache est présent et nomme la session
  `train-<run-id>`.

**Suivre / piloter :**
```bash
tail -f /workspace/runs/logs/rp_5090_bc48_bs64.log   # log live
tmux attach -t train-rp_5090_bc48_bs64               # console (détacher : Ctrl-b puis d)
nvidia-smi -l 2                                       # occupation GPU/VRAM
```

**Arrêt propre** (sauvegarde `last.pt` avant de sortir) :
```bash
mkdir -p /workspace/runs/control && touch /workspace/runs/control/rp_5090_bc48_bs64.stop
```

**Reprise** après stop/crash :
```bash
bash scripts/runpod_train.sh --run-id rp_5090_bc48_bs64 --resume last
```

---

## 7. Paramètres conseillés sur 5090 (32 Go)

Tu as ~2,6× plus de VRAM que la 5070. Points de départ (monter par paliers en
surveillant `nvidia-smi`) :

| Paramètre | 5070 local | 5090 départ | Plafond à viser |
|---|---|---|---|
| `--batch-size` | 24 | **64** | 96–128 |
| `--base-channels` | 32 | **48** | 64 |
| `--compile` | off (Windows) | **on** | — |
| `--lr` | 1,5e-3 | **2e-3** (≈ 1e-3·√4) | ajuster |

> Le scheduler utilise désormais `LR_FACTOR=0.5` (baisses douces) — pas besoin de
> le repasser, c'est le défaut. Un nouveau `--run-id` repart de zéro avec cette valeur.
>
> Astuce VRAM : sur 32 Go tu peux réactiver `cudnn.benchmark=True` pour un petit
> bonus (il est forcé à `False` dans `src/train.py` pour protéger la 5070 12 Go).
> Optionnel — laisse tel quel pour un premier run.

---

## 8. Récupérer les checkpoints / logs

Tout est dans `/workspace/runs/` (persistant). Pour rapatrier le meilleur modèle
sur ta machine locale :
```bash
# depuis le pod
runpodctl send /workspace/runs/checkpoints/rp_5090_bc48_bs64/best_si_sdri.pt
# ou rsync inverse depuis ta machine :
rsync -avP -e "ssh -p <PORT>" root@<HOST>:/workspace/runs/checkpoints/ ./checkpoints_rp/
```

Le `history.jsonl` (`/workspace/runs/logs/<run-id>/`) se lit avec le notebook 05
pour tracer les courbes au retour.

> ⚠️ **Avant de supprimer le pod** : assure-toi d'avoir rapatrié les checkpoints,
> ou garde le Network Volume (les données y survivent ; seul le pod est éphémère).

---

## Récap des variables d'environnement

Définies par `scripts/runpod_env.sh` (sourcé par les deux autres scripts) :

| Variable | Rôle | Défaut |
|---|---|---|
| `WORKSPACE` | racine du Network Volume | `/workspace` |
| `FILTRE_VOIX_DL_ROOT` | checkpoints/logs/outputs/lock | `$WORKSPACE/runs` |
| `FILTRE_VOIX_DL_CACHE` | cache memmap float16 | `$WORKSPACE/cache` |
| `WANDB_API_KEY` | suivi wandb (optionnel) | — |
| `PYTHONIOENCODING` | UTF-8 (caractères ★ des logs) | `utf-8` |
