"""Lock d'entraînement partagé via Drive.

Empêche deux sessions d'entraînement de tourner en même temps dans l'équipe,
même sur des branches git différentes ou sur des machines différentes — le
lock vit dans le dossier Drive synchronisé chez tout le monde via Drive
for Desktop.

Mécanisme :
    - Un fichier JSON ``<DRIVE_PROJECT>/.training_lock.json`` décrit le run actif.
    - Le process en cours met à jour ``heartbeat_at`` régulièrement (via
      ``lock.heartbeat()`` appelé depuis la boucle d'entraînement).
    - Un lock dont le heartbeat date de plus de ``STALE_AFTER_SECONDS`` est
      considéré comme mort (crash, kill -9, perte de courant) et peut être
      écrasé sans ``--force``.
    - L'écriture du fichier est atomique (``.tmp`` + ``os.replace``), pattern
      déjà utilisé dans ``src.checkpoint``.

Limite acceptée : Drive Desktop a un délai de sync (typiquement quelques
secondes pour un petit JSON). Pour une équipe de 7 où les sessions durent des
heures, la fenêtre de course (deux ``acquire`` à la seconde près) est
négligeable.
"""
from __future__ import annotations

import atexit
import getpass
import json
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import config as cfg


LOCK_FILENAME = ".training_lock.json"
STALE_AFTER_SECONDS = 5 * 60   # 5 min sans heartbeat = lock mort


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _git_branch() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or "unknown"
    except Exception:
        return "unknown"


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or "unknown"
    except Exception:
        return "unknown"


def _lock_path() -> Path:
    return Path(cfg.DRIVE_PROJECT) / LOCK_FILENAME


def _read_lock() -> Optional[dict]:
    path = _lock_path()
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _write_lock_atomic(payload: dict) -> None:
    path = _lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _is_stale(lock: dict) -> bool:
    try:
        hb = _parse_iso(lock["heartbeat_at"])
    except (KeyError, ValueError):
        return True
    age = (datetime.now(timezone.utc) - hb).total_seconds()
    return age > STALE_AFTER_SECONDS


def _format_existing(lock: dict) -> str:
    return (
        f"  utilisateur : {lock.get('user', '?')}@{lock.get('hostname', '?')}\n"
        f"  branche     : {lock.get('branch', '?')} ({lock.get('commit', '?')})\n"
        f"  run_id      : {lock.get('run_id', '?')}\n"
        f"  pid         : {lock.get('pid', '?')}\n"
        f"  démarré le  : {lock.get('started_at', '?')}\n"
        f"  heartbeat   : {lock.get('heartbeat_at', '?')}"
    )


class LockBusyError(RuntimeError):
    """Levée quand un lock vivant existe déjà et que ``force=False``."""


class LockStolenError(RuntimeError):
    """Levée quand notre lock a été repris par quelqu'un d'autre (--force).

    Détectée au prochain heartbeat. Doit faire échouer l'entraînement courant
    pour qu'il ne reste pas deux runs concurrents sur Drive.
    """


@dataclass
class TrainingLock:
    """Context manager pour acquérir le lock d'entraînement partagé.

    Usage::

        with TrainingLock(run_id="unet_big", force=False) as lock:
            for epoch in range(...):
                for batch in ...:
                    train_step(...)
                    lock.heartbeat()  # à appeler périodiquement

    Le lock est automatiquement relâché en sortie du ``with`` (succès, exception,
    ou interruption). En cas de crash brutal (kill -9, panne), le fichier reste
    sur Drive mais sera considéré comme stale après ``STALE_AFTER_SECONDS``.
    """

    run_id: str
    force: bool = False

    def __post_init__(self) -> None:
        self._acquired: bool = False
        self._payload: dict = {}

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "TrainingLock":
        cfg.ensure_project_root()

        existing = _read_lock()
        if existing is not None and not _is_stale(existing):
            if not self.force:
                raise LockBusyError(
                    "Un entraînement est déjà en cours :\n"
                    + _format_existing(existing)
                    + "\n\nUtiliser --force si tu es certain qu'il est mort "
                    "(le lock se libère seul après "
                    f"{STALE_AFTER_SECONDS // 60} min sans heartbeat)."
                )
            print(
                "[lock] --force : écrasement d'un lock encore vivant détenu par "
                f"{existing.get('user', '?')}@{existing.get('hostname', '?')}",
                file=sys.stderr,
            )
        elif existing is not None and _is_stale(existing):
            print(
                "[lock] reprise d'un lock stale (heartbeat trop ancien) — "
                f"ancien détenteur : {existing.get('user', '?')}@"
                f"{existing.get('hostname', '?')}",
                file=sys.stderr,
            )

        now = _now_iso()
        self._payload = {
            "user": getpass.getuser(),
            "hostname": socket.gethostname(),
            "branch": _git_branch(),
            "commit": _git_commit(),
            "run_id": self.run_id,
            "pid": os.getpid(),
            "started_at": now,
            "heartbeat_at": now,
        }
        _write_lock_atomic(self._payload)
        self._acquired = True
        atexit.register(self._release_safe)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._release_safe()

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def heartbeat(self) -> None:
        """Met à jour ``heartbeat_at`` après avoir vérifié qu'on est toujours
        le détenteur du lock.

        Si quelqu'un d'autre l'a pris entre-temps (via ``--force``), lève
        ``LockStolenError`` : cette exception doit faire échouer l'entraînement
        courant pour éviter d'avoir deux runs concurrents sur Drive.
        """
        if not self._acquired:
            return

        current = _read_lock()
        if current is not None and (
            current.get("pid") != self._payload.get("pid")
            or current.get("hostname") != self._payload.get("hostname")
        ):
            # Quelqu'un nous a pris le lock — on abandonne sans réécrire.
            self._acquired = False
            raise LockStolenError(
                "Le lock d'entraînement a été repris par "
                f"{current.get('user', '?')}@{current.get('hostname', '?')} "
                f"(run_id={current.get('run_id', '?')}). "
                "Arrêt de la session courante."
            )

        self._payload["heartbeat_at"] = _now_iso()
        try:
            _write_lock_atomic(self._payload)
        except OSError as e:
            # Drive sync bloqué, disque plein, etc. — on ne casse pas
            # l'entraînement pour ça, mais on prévient.
            print(f"[lock] heartbeat échoué : {e}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Interne
    # ------------------------------------------------------------------

    def _release_safe(self) -> None:
        if not self._acquired:
            return
        self._acquired = False
        path = _lock_path()
        try:
            current = _read_lock()
            # Ne supprime que si on est encore le détenteur (évite d'effacer
            # un lock pris entre-temps par quelqu'un d'autre via --force).
            if current is not None and current.get("pid") == self._payload.get("pid") \
                    and current.get("hostname") == self._payload.get("hostname"):
                path.unlink(missing_ok=True)
        except OSError as e:
            print(f"[lock] release échoué : {e}", file=sys.stderr)
