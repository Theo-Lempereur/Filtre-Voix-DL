"""Logger d'entraînement : wandb (en ligne) + JSONL (local).

Pourquoi deux destinations :
- **wandb** : dashboard live, comparaison de runs, partage avec l'équipe.
- **JSONL** : indépendant du réseau et du compte wandb. Reste lisible après
  coupure de session, sert de source unique au notebook 05 d'analyse,
  archivable dans `outputs/` ou Drive pour le rapport final.

Le module est tolérant : si wandb n'est pas connecté ou que `use_wandb=False`,
il continue à fonctionner avec uniquement le JSONL.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from . import config as cfg


class Logger:
    """Logger minimaliste : wandb + JSONL local.

    Usage type :

        logger = Logger(run_id="20260522-base32", config_dict=config)
        logger.log(epoch=1, metrics={"train_loss": 0.13, "val_loss": 0.15})
        logger.log_audio("noisy_sample", wav, step=1, sample_rate=16000)
        logger.finish()
    """

    def __init__(
        self,
        run_id: str,
        config_dict: dict,
        use_wandb: bool = True,
        log_dir: str | os.PathLike = cfg.LOGS,
        project: str = "filtre-voix-dl",
    ):
        self.run_id = run_id
        self.log_dir = Path(log_dir) / run_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.log_dir / "history.jsonl"

        # On ouvre en mode append pour préserver l'historique en cas de reprise.
        self._fp = open(self.jsonl_path, "a", encoding="utf-8", buffering=1)

        # Écrit un en-tête de session pour que le notebook 05 puisse retrouver
        # la config exacte de chaque tronçon de run.
        self._write({
            "type": "session_start",
            "run_id": run_id,
            "timestamp": time.time(),
            "config": _to_jsonable(config_dict),
        })

        self.use_wandb = False
        self._wandb = None
        if use_wandb:
            try:
                import wandb  # type: ignore
                wandb.init(project=project, name=run_id,
                           config=_to_jsonable(config_dict),
                           id=run_id, resume="allow")
                self._wandb = wandb
                self.use_wandb = True
            except Exception as e:  # pas de connexion, pas de login, etc.
                print(f"[Logger] wandb désactivé ({type(e).__name__}: {e}). "
                      f"Logs locaux uniquement.")

    # ---- API publique ----

    def log(self, epoch: int, metrics: dict[str, Any]) -> None:
        """Écrit une ligne de métriques pour cette `epoch`."""
        record = {
            "type": "metrics",
            "epoch": int(epoch),
            "timestamp": time.time(),
            **_to_jsonable(metrics),
        }
        self._write(record)
        if self.use_wandb and self._wandb is not None:
            try:
                self._wandb.log({**metrics, "epoch": epoch}, step=epoch)
            except Exception:
                pass  # on n'interrompt jamais l'entraînement pour un log

    def log_audio(self, name: str, waveform, step: int,
                  sample_rate: int = cfg.SAMPLE_RATE) -> None:
        """Envoie un échantillon audio à wandb (no-op si wandb indisponible).

        `waveform` peut être un numpy array 1D ou un tenseur torch 1D.
        """
        if not self.use_wandb or self._wandb is None:
            return
        try:
            import numpy as np
            if hasattr(waveform, "detach"):
                waveform = waveform.detach().cpu().numpy()
            waveform = np.asarray(waveform).astype("float32").reshape(-1)
            self._wandb.log(
                {name: self._wandb.Audio(waveform, sample_rate=sample_rate)},
                step=step,
            )
        except Exception:
            pass

    def finish(self) -> None:
        """Ferme proprement le JSONL et wandb."""
        try:
            self._write({"type": "session_end", "timestamp": time.time()})
            self._fp.close()
        except Exception:
            pass
        if self.use_wandb and self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass

    # ---- contexte ----

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish()

    # ---- privé ----

    def _write(self, record: dict) -> None:
        self._fp.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Lecture du JSONL (utilisée par le notebook 05)
# ---------------------------------------------------------------------------

def read_history(run_dir: str | os.PathLike) -> list[dict]:
    """Charge toutes les lignes JSONL d'un run et les renvoie comme liste."""
    path = Path(run_dir) / "history.jsonl"
    if not path.is_file():
        return []
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # ligne tronquée (très rare avec line-buffering=1) → on ignore
                continue
    return records


def metrics_records(records: list[dict]) -> list[dict]:
    """Filtre les enregistrements de type "metrics" depuis `read_history`."""
    return [r for r in records if r.get("type") == "metrics"]


# ---------------------------------------------------------------------------
# Helpers privés
# ---------------------------------------------------------------------------

def _to_jsonable(obj: Any) -> Any:
    """Convertit récursivement un objet en valeurs sérialisables JSON."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # tenseur scalaire, numpy scalar, etc.
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)
