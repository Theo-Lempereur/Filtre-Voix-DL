"""Gestion des checkpoints d'entraînement.

Indispensable sur Colab où la session peut couper à tout moment.
On adopte trois fichiers par run :

    checkpoints/<run_id>/
        last.pt           ← état après la dernière epoch terminée
        best.pt           ← état correspondant à la meilleure val_loss
        epoch_NNN.pt      ← historique récent (rotation des N derniers)

Conventions :
    - Toutes les écritures passent par un fichier `.tmp` renommé en final via
      `os.replace`. Si la session coupe au milieu d'un `torch.save`, on conserve
      l'ancien fichier intact au lieu d'une moitié de fichier corrompu.
    - Un checkpoint stocke : poids modèle, état optimizer, état scheduler,
      `epoch` (dernier numéro d'epoch *terminé*), `best_val_loss`,
      `run_id`, `config` et `history`.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Sauvegarde
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str | os.PathLike,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    epoch: int,
    best_val_loss: float,
    run_id: str,
    config: dict,
    history: dict,
    best_val_si_sdri: float = float("-inf"),
) -> None:
    """Sauvegarde atomique d'un checkpoint complet.

    `epoch` est le numéro de la dernière epoch terminée (1-indexed côté
    notebook : si epoch=3 alors les epochs 1, 2 et 3 ont été jouées).

    `best_val_si_sdri` est le meilleur SI-SDRi observé sur la val. Par défaut
    `-inf` pour la rétrocompatibilité (anciens appels qui ne passent pas le
    paramètre se comportent comme avant côté sauvegarde).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state":        model.state_dict(),
        "optimizer_state":    optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state":    scheduler.state_dict() if scheduler is not None else None,
        "epoch":              int(epoch),
        "best_val_loss":      float(best_val_loss),
        "best_val_si_sdri":   float(best_val_si_sdri),
        "run_id":             run_id,
        "config":             dict(config),
        "history":            dict(history),
    }

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)   # atomique sur le même système de fichiers


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def load_checkpoint(
    path: str | os.PathLike,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> dict:
    """Restaure un checkpoint et renvoie le dict complet.

    Le modèle est restauré in-place. Si `optimizer` / `scheduler` sont fournis,
    leur état est également restauré (utile pour reprendre l'entraînement à
    l'identique).

    Retourne : {epoch, best_val_loss, run_id, config, history}.
    """
    path = Path(path)
    # weights_only=False car on stocke l'état d'un optimizer/scheduler complet,
    # pas seulement des tenseurs. Le fichier provient toujours de notre code.
    payload = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(payload["model_state"], strict=strict)
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        try:
            scheduler.load_state_dict(payload["scheduler_state"])
        except (KeyError, TypeError, ValueError) as e:
            # Format de scheduler différent (ex : ancien ReduceLROnPlateau
            # vers nouveau DualMetricPlateauScheduler). On garde le scheduler
            # fraîchement initialisé plutôt que de planter.
            print(f"[checkpoint] scheduler_state incompatible, ignoré ({e}).")

    return {
        "epoch":            int(payload.get("epoch", 0)),
        "best_val_loss":    float(payload.get("best_val_loss", float("inf"))),
        "best_val_si_sdri": float(payload.get("best_val_si_sdri", float("-inf"))),
        "run_id":           payload.get("run_id", ""),
        "config":           dict(payload.get("config", {})),
        "history":          dict(payload.get("history", {})),
    }


def peek_checkpoint_config(path: str | os.PathLike) -> dict:
    """Lit uniquement le dict ``config`` d'un checkpoint, sans restaurer le modèle.

    Utile à la reprise : on récupère les hyperparamètres d'origine pour les
    réutiliser, AVANT de construire le modèle / l'optimizer (qui dépendent
    de ces hyperparamètres).
    """
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return dict(payload.get("config", {}))


# ---------------------------------------------------------------------------
# Recherche et rotation
# ---------------------------------------------------------------------------

_EPOCH_RE = re.compile(r"^epoch_(\d+)\.pt$")


def find_latest_checkpoint(ckpt_dir: str | os.PathLike) -> str | None:
    """Renvoie le chemin du checkpoint le plus avancé, ou None.

    Préférence : `last.pt` (mis à jour à chaque epoch) si présent ; sinon le
    plus grand `epoch_NNN.pt`. `best.pt` n'est pas utilisé pour reprendre
    l'entraînement (la meilleure val_loss n'est pas forcément la dernière).
    """
    d = Path(ckpt_dir)
    if not d.exists():
        return None

    last = d / "last.pt"
    if last.is_file():
        return str(last)

    epochs = []
    for p in d.iterdir():
        m = _EPOCH_RE.match(p.name)
        if m and p.is_file():
            epochs.append((int(m.group(1)), p))
    if not epochs:
        return None
    epochs.sort(key=lambda x: x[0])
    return str(epochs[-1][1])


def find_best_checkpoint(ckpt_dir: str | os.PathLike) -> str | None:
    """Renvoie `<ckpt_dir>/best.pt` s'il existe."""
    p = Path(ckpt_dir) / "best.pt"
    return str(p) if p.is_file() else None


def rotate_checkpoints(ckpt_dir: str | os.PathLike, keep_last_n: int) -> list[str]:
    """Garde uniquement les `keep_last_n` `epoch_*.pt` les plus récents.

    Ne touche jamais à `last.pt` ni `best.pt`. Retourne la liste des fichiers
    supprimés (utile pour les logs).
    """
    d = Path(ckpt_dir)
    if not d.exists() or keep_last_n < 0:
        return []

    epochs: list[tuple[int, Path]] = []
    for p in d.iterdir():
        m = _EPOCH_RE.match(p.name)
        if m and p.is_file():
            epochs.append((int(m.group(1)), p))
    if len(epochs) <= keep_last_n:
        return []

    epochs.sort(key=lambda x: x[0])
    to_delete = epochs[:-keep_last_n] if keep_last_n > 0 else epochs
    removed: list[str] = []
    for _, p in to_delete:
        try:
            p.unlink()
            removed.append(str(p))
        except OSError:
            pass
    return removed
