#!/usr/bin/env bash
# Lance un entraînement headless DÉTACHÉ sur RunPod, à l'épreuve des coupures SSH.
#
# Usage (tous les flags sont passés tels quels à scripts/train_local.py) :
#     bash scripts/runpod_train.sh --run-id mon_run --epochs 80 \
#          --batch-size 48 --base-channels 48 --compile
#
# Reprise :
#     bash scripts/runpod_train.sh --run-id mon_run --resume last
#
# Le run tourne dans une session tmux nommée d'après --run-id : il survit à la
# fermeture du terminal / déconnexion SSH. On peut s'y rattacher ou suivre le log.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/runpod_env.sh"

if [ ! -f "$WORKSPACE/venv/bin/activate" ]; then
    echo "[runpod_train] venv introuvable — lance d'abord scripts/runpod_bootstrap.sh" >&2
    exit 1
fi

# --- Récupère la valeur de --run-id pour nommer la session tmux + le log ---
RUN_ID=""
prev=""
for arg in "$@"; do
    if [ "$prev" = "--run-id" ]; then RUN_ID="$arg"; break; fi
    case "$arg" in --run-id=*) RUN_ID="${arg#--run-id=}"; break;; esac
    prev="$arg"
done
if [ -z "$RUN_ID" ]; then
    echo "[runpod_train] --run-id est obligatoire." >&2
    exit 1
fi

# --- Sanity check : le cache train/val est-il bien présent ? ---
missing=0
for f in train_noisy.npy train_clean.npy train_names.json \
         val_noisy.npy val_clean.npy val_names.json; do
    if [ ! -f "$FILTRE_VOIX_DL_CACHE/$f" ]; then
        echo "[runpod_train] cache manquant : $FILTRE_VOIX_DL_CACHE/$f" >&2
        missing=1
    fi
done
if [ "$missing" = 1 ]; then
    echo "[runpod_train] Uploade le cache float16 avant de lancer (cf. RUNPOD.md)." >&2
    exit 1
fi

LOG_PATH="$FILTRE_VOIX_DL_ROOT/logs/$RUN_ID.log"
SESSION="train-$RUN_ID"

# Construit la commande à lancer dans tmux (quoting sûr des arguments).
CMD="source '$REPO_ROOT/scripts/runpod_env.sh' >/dev/null; python scripts/train_local.py"
for arg in "$@"; do CMD+=" $(printf '%q' "$arg")"; done

if command -v tmux >/dev/null 2>&1; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "[runpod_train] une session tmux '$SESSION' existe déjà." >&2
        echo "  Rattache-toi : tmux attach -t $SESSION" >&2
        exit 1
    fi
    tmux new-session -d -s "$SESSION" "$CMD"
    echo "[runpod_train] lancé dans tmux (session '$SESSION')."
    echo "  Suivre le log : tail -f '$LOG_PATH'"
    echo "  Rattacher     : tmux attach -t $SESSION   (détacher : Ctrl-b puis d)"
    echo "  Arrêter propre: mkdir -p '$FILTRE_VOIX_DL_ROOT/control' && touch '$FILTRE_VOIX_DL_ROOT/control/$RUN_ID.stop'"
else
    # Pas de tmux : fallback nohup (détaché du terminal).
    mkdir -p "$(dirname "$LOG_PATH")"
    nohup bash -c "$CMD" >/dev/null 2>&1 &
    echo "[runpod_train] tmux absent — lancé via nohup (PID $!)."
    echo "  Suivre le log : tail -f '$LOG_PATH'"
fi
