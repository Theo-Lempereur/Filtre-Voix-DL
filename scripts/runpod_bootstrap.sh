#!/usr/bin/env bash
# Bootstrap d'un pod RunPod pour l'entraînement headless du U-Net de débruitage.
#
# À lancer UNE FOIS sur un pod fraîchement démarré, depuis la racine du repo
# cloné sur le Network Volume :
#
#     cd /workspace/Filtre-Voix-DL
#     bash scripts/runpod_bootstrap.sh
#
# Idempotent : relançable sans casser un venv/cache existant.
#
# Étapes :
#   1. Crée l'arborescence persistante sur le Network Volume.
#   2. Crée un venv Python (sur le volume → survit aux restarts du pod).
#   3. Installe requirements-train.txt (torch cu128 + audio, sans GUI).
#   4. Vérifie que CUDA voit bien le GPU.
set -euo pipefail

# Racine du repo = parent du dossier scripts/ (résolu depuis ce fichier).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Charge les chemins (WORKSPACE, FILTRE_VOIX_DL_ROOT, FILTRE_VOIX_DL_CACHE).
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/runpod_env.sh"

echo
echo "=== 1. Arborescence persistante ==="
mkdir -p "$FILTRE_VOIX_DL_ROOT" "$FILTRE_VOIX_DL_CACHE"
echo "  runs  : $FILTRE_VOIX_DL_ROOT"
echo "  cache : $FILTRE_VOIX_DL_CACHE"

echo
echo "=== 2. venv Python ($WORKSPACE/venv) ==="
if [ ! -f "$WORKSPACE/venv/bin/activate" ]; then
    python3 -m venv "$WORKSPACE/venv"
    echo "  venv créé."
else
    echo "  venv déjà présent — réutilisé."
fi
# shellcheck disable=SC1091
source "$WORKSPACE/venv/bin/activate"
python -m pip install --upgrade pip >/dev/null

echo
echo "=== 3. Dépendances (requirements-train.txt) ==="
pip install -r "$REPO_ROOT/requirements-train.txt"

echo
echo "=== 4. Vérification GPU / CUDA ==="
python - <<'PY'
import torch
ok = torch.cuda.is_available()
print(f"  torch          : {torch.__version__}")
print(f"  cuda.available : {ok}")
if ok:
    print(f"  device         : {torch.cuda.get_device_name(0)}")
    print(f"  capability     : {torch.cuda.get_device_capability(0)}")
else:
    raise SystemExit("[bootstrap] ERREUR : CUDA indisponible — vérifie le template du pod.")
PY

echo
echo "=== Bootstrap terminé ==="
echo "Étapes suivantes :"
echo "  1. Uploader le cache float16 dans : $FILTRE_VOIX_DL_CACHE"
echo "     (train_noisy.npy / train_clean.npy / train_names.json + idem val)"
echo "  2. Lancer un entraînement détaché :"
echo "       bash scripts/runpod_train.sh --run-id mon_run --epochs 80 --batch-size 48 --base-channels 48 --compile"
