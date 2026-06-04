# shellcheck shell=bash
# Variables d'environnement pour l'entraînement headless sur RunPod.
#
# À SOURCER (pas exécuter) avant toute commande d'entraînement :
#     source scripts/runpod_env.sh
#
# Hypothèse : un Network Volume persistant est monté sur $WORKSPACE (/workspace
# par défaut sur RunPod). On y range le venv, le cache, et les sorties (runs)
# pour qu'ils survivent à un stop/restart du pod.

# Racine persistante (Network Volume). Surcharger si monté ailleurs.
export WORKSPACE="${WORKSPACE:-/workspace}"

# Racine projet : checkpoints/, logs/, outputs/ et le lock atterrissent ici.
# Remplace le dossier Google Drive utilisé en local (cf. src/config.py).
export FILTRE_VOIX_DL_ROOT="${FILTRE_VOIX_DL_ROOT:-$WORKSPACE/runs}"

# Cache memmap float16 (uploadé depuis la machine locale). Hors du repo cloné
# pour ne pas le ré-uploader à chaque session.
export FILTRE_VOIX_DL_CACHE="${FILTRE_VOIX_DL_CACHE:-$WORKSPACE/cache}"

# UTF-8 forcé : les résumés d'epoch contiennent des caractères non-ASCII (★).
export PYTHONIOENCODING="utf-8"

# Clé wandb : à renseigner avant de sourcer, ou ici en dur (déconseillé).
#   export WANDB_API_KEY="xxxxxxxx"
# Si absente, le logger retombe proprement sur le JSONL local.

# Active le venv persistant s'il existe.
if [ -f "$WORKSPACE/venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$WORKSPACE/venv/bin/activate"
fi

echo "[runpod_env] WORKSPACE=$WORKSPACE"
echo "[runpod_env] FILTRE_VOIX_DL_ROOT=$FILTRE_VOIX_DL_ROOT"
echo "[runpod_env] FILTRE_VOIX_DL_CACHE=$FILTRE_VOIX_DL_CACHE"
echo "[runpod_env] wandb : $( [ -n "${WANDB_API_KEY:-}" ] && echo "clé présente" || echo "absente (JSONL only)" )"
