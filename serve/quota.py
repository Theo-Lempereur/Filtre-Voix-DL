"""Quota journalier par IP, stocké en SQLite.

Utilisé uniquement pour le mode « clé partagée » (NVIDIA gratuit). Quand un
utilisateur fournit sa propre clé, aucun quota n'est appliqué de notre côté
(c'est NVIDIA qui applique le sien).

Choix : SQLite plutôt que Redis/in-memory.
    - un seul fichier, aucune dépendance à lancer ;
    - survit aux redémarrages du serveur (contrairement à l'in-memory) ;
    - parfait pour un serveur mono-worker (cf. INTEGRATION.md).

La « remise à zéro quotidienne » est implicite : la clé primaire est
``(day, ip)``. Un nouveau jour => une nouvelle ligne, le compteur repart de 0.
"""
from __future__ import annotations

import os
import sqlite3
import threading
from datetime import date, datetime, timezone
from pathlib import Path

# Chemin du fichier SQLite (configurable). Par défaut à la racine du repo.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DB_PATH = Path(os.environ.get("QUOTA_DB_PATH", _REPO_ROOT / "quota.sqlite3"))

# Limite gratuite par IP et par jour (mode clé partagée). Configurable.
def free_daily_limit() -> int:
    try:
        return max(0, int(os.environ.get("FREE_DAILY_LIMIT", "5")))
    except ValueError:
        return 5


_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _today() -> str:
    """Jour courant en UTC (évite les sauts liés au fuseau du serveur)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        _conn.execute(
            "CREATE TABLE IF NOT EXISTS usage ("
            "  day   TEXT NOT NULL,"
            "  ip    TEXT NOT NULL,"
            "  count INTEGER NOT NULL DEFAULT 0,"
            "  PRIMARY KEY (day, ip)"
            ")"
        )
        _conn.commit()
    return _conn


def get_count(ip: str) -> int:
    """Nombre d'utilisations déjà consommées par cette IP aujourd'hui."""
    with _lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT count FROM usage WHERE day = ? AND ip = ?",
            (_today(), ip),
        ).fetchone()
        return int(row[0]) if row else 0


def remaining(ip: str) -> int:
    """Utilisations restantes aujourd'hui pour cette IP (mode gratuit)."""
    return max(0, free_daily_limit() - get_count(ip))


def has_quota(ip: str) -> bool:
    return get_count(ip) < free_daily_limit()


def increment(ip: str) -> int:
    """Incrémente le compteur du jour pour cette IP, renvoie le nouveau total.

    Fait aussi un petit ménage des lignes des jours précédents pour ne pas
    laisser grossir la base indéfiniment.
    """
    with _lock:
        conn = _get_conn()
        day = _today()
        conn.execute(
            "INSERT INTO usage (day, ip, count) VALUES (?, ?, 1) "
            "ON CONFLICT(day, ip) DO UPDATE SET count = count + 1",
            (day, ip),
        )
        # Ménage : on ne garde que le jour courant.
        conn.execute("DELETE FROM usage WHERE day <> ?", (day,))
        conn.commit()
        row = conn.execute(
            "SELECT count FROM usage WHERE day = ? AND ip = ?", (day, ip)
        ).fetchone()
        return int(row[0]) if row else 1
