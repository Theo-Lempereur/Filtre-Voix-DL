"""Signal d'arrêt propre d'un entraînement via un fichier flag.

La GUI (ou n'importe quel outil externe) appelle ``request_stop(run_id)`` pour
demander au training en cours de s'arrêter proprement à la fin du batch courant.
La boucle d'entraînement consulte ``is_stop_requested(run_id)`` à chaque batch
et, si vrai, sauvegarde ``last.pt`` puis sort.

Pourquoi un fichier plutôt qu'un signal POSIX ? Sur Windows il est difficile
d'envoyer CTRL_C_EVENT à un sous-process Python lancé via QProcess sans risquer
de couper en pleine écriture de checkpoint. Un flag fichier est portable et
totalement déterministe.
"""
from __future__ import annotations

import os
from pathlib import Path

from . import config as cfg


def _control_dir() -> Path:
    p = Path(cfg.DRIVE_PROJECT) / "control"
    p.mkdir(parents=True, exist_ok=True)
    return p


def stop_flag_path(run_id: str) -> Path:
    return _control_dir() / f"{run_id}.stop"


def request_stop(run_id: str) -> None:
    """Crée le flag stop. Idempotent."""
    path = stop_flag_path(run_id)
    path.touch(exist_ok=True)


def is_stop_requested(run_id: str) -> bool:
    return stop_flag_path(run_id).exists()


def clear_stop_flag(run_id: str) -> None:
    """Supprime le flag s'il existe. Idempotent."""
    try:
        stop_flag_path(run_id).unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass
