"""CLI pour lancer un entraînement long en local (hors Colab).

Usage minimal :
    python scripts/train_local.py --run-id unet_big --epochs 100

Reprise après crash :
    python scripts/train_local.py --run-id unet_big --resume last

Le script :
  1. Vérifie que la racine projet Drive est accessible (``ensure_project_root``).
  2. Construit le dict de config en surchargant les défauts de ``src.config``.
  3. Acquiert le lock partagé (refus strict si un autre membre entraîne déjà,
     ``--force`` pour override explicite).
  4. Loggue stdout/stderr à la fois dans le terminal et dans
     ``<DRIVE_PROJECT>/logs/<run_id>.log`` pour pouvoir consulter le log
     depuis n'importe quel poste de l'équipe.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Permet ``python scripts/train_local.py`` depuis la racine du repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as cfg
from src.lock import LockBusyError
from src.train import train


class _Tee:
    """Duplique chaque écriture vers stdout et un fichier (mode append, line-buffered)."""

    def __init__(self, original, file_path: Path):
        self._original = original
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = file_path.open("a", encoding="utf-8", buffering=1)

    def write(self, data: str) -> int:
        self._original.write(data)
        try:
            self._file.write(data)
        except OSError:
            pass
        return len(data)

    def flush(self) -> None:
        self._original.flush()
        try:
            self._file.flush()
        except OSError:
            pass

    def isatty(self) -> bool:
        return getattr(self._original, "isatty", lambda: False)()

    def fileno(self) -> int:
        return self._original.fileno()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Entraînement local du U-Net de débruitage de voix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-id", required=True,
                   help="Identifiant unique du run (sert de sous-dossier checkpoints/<run-id>/).")
    p.add_argument("--resume", default=None,
                   help="'last' / 'best' / 'best_si_sdri' (cherche dans "
                        "checkpoints/<run-id>/), ou chemin absolu vers un .pt.")
    p.add_argument("--config-file", default=None,
                   help="Fichier JSON optionnel d'overrides de config (mergé après les flags).")

    # Hyperparamètres usuels — None signifie 'utiliser le défaut de src.config'.
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--base-channels", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)

    # Sous-échantillonnage (pour smoke tests).
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)

    # Logging / lock.
    p.add_argument("--no-wandb", action="store_true",
                   help="Désactive wandb (utile en local sans clé API).")
    p.add_argument("--force", action="store_true",
                   help="Force l'acquisition du lock même si un autre run semble vivant.")
    return p


def _resolve_resume(arg: str | None, run_id: str) -> str | None:
    if not arg:
        return None
    if arg in ("last", "best", "best_si_sdri"):
        return str(Path(cfg.CHECKPOINTS) / run_id / f"{arg}.pt")
    return arg


def _build_config(args: argparse.Namespace) -> dict:
    config: dict = {"run_id": args.run_id}

    if args.config_file:
        with open(args.config_file, "r", encoding="utf-8") as f:
            config.update(json.load(f))

    # Les flags CLI ont la priorité sur le fichier de config.
    if args.epochs is not None:        config["num_epochs"]    = args.epochs
    if args.batch_size is not None:    config["batch_size"]    = args.batch_size
    if args.lr is not None:            config["lr"]            = args.lr
    if args.num_workers is not None:   config["num_workers"]   = args.num_workers
    if args.base_channels is not None: config["base_channels"] = args.base_channels
    if args.seed is not None:          config["seed"]          = args.seed
    if args.no_wandb:                  config["use_wandb"]     = False

    return config


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # 1. Vérifie l'accès à la racine Drive (échec clair si Drive Desktop manque).
    cfg.ensure_project_root()

    # 2. Branche le tee stdout/stderr vers logs/<run_id>.log sur Drive.
    log_path = Path(cfg.LOGS) / f"{args.run_id}.log"
    sys.stdout = _Tee(sys.stdout, log_path)
    sys.stderr = _Tee(sys.stderr, log_path)
    print(f"[train_local] log : {log_path}")
    print(f"[train_local] racine projet : {cfg.DRIVE_PROJECT}")

    # 3. Lance l'entraînement (le lock est posé à l'intérieur de train()).
    config = _build_config(args)
    resume_from = _resolve_resume(args.resume, args.run_id)

    try:
        train(
            config=config,
            resume_from=resume_from,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            force_lock=args.force,
        )
    except LockBusyError as e:
        print(f"[train_local] {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
