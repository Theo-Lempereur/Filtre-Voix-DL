"""Génère des WAV (noisy, pred, clean) pour écouter à l'oreille le rendu du modèle.

Tire N paires aléatoires du split test, fait passer le noisy dans le modèle
chargé depuis le checkpoint choisi, et écrit les 3 fichiers côte à côte dans
``<DRIVE_PROJECT>/outputs/listen/<run>/<idx>_<basename>/``.

Usage :
    python scripts/listen_test.py
    python scripts/listen_test.py --run big_newds --ckpt best.pt --n 10 --seed 1
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import audio, config as cfg
from src.denoiser import Denoiser


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default=None,
                   help="Nom du dossier de run sous CHECKPOINTS/. Si omis, menu interactif.")
    p.add_argument("--ckpt", default=None,
                   help="Nom du fichier checkpoint (best_si_sdri.pt, best.pt, last.pt...). "
                        "Si omis, menu interactif.")
    p.add_argument("--n", type=int, default=5, help="Nombre d'échantillons à générer.")
    p.add_argument("--seed", type=int, default=0, help="Graine pour le tirage aléatoire.")
    p.add_argument("--device", default=None,
                   help="Force le device (cuda / cpu). Auto si non précisé.")
    return p.parse_args()


def _list_runs() -> list[str]:
    root = Path(cfg.CHECKPOINTS)
    if not root.is_dir():
        return []
    runs = []
    for d in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir() or d.name.startswith("."):
            continue
        # Garde uniquement les runs qui ont au moins un .pt
        if any(d.glob("*.pt")):
            runs.append(d.name)
    return runs


def _list_ckpts(run_dir: Path) -> list[str]:
    """Trie : best_si_sdri > best > last > epoch_NNN (du plus récent)."""
    priority = {"best_si_sdri.pt": 0, "best.pt": 1, "last.pt": 2}
    files = list(run_dir.glob("*.pt"))

    def key(p: Path):
        return (priority.get(p.name, 3), -p.stat().st_mtime, p.name)

    return [p.name for p in sorted(files, key=key)]


def _prompt_choice(label: str, options: list[str], default_idx: int = 0) -> str:
    if not options:
        raise RuntimeError(f"Aucun {label} disponible.")
    print(f"\n=== Choix du {label} ===")
    for i, opt in enumerate(options):
        marker = " (défaut)" if i == default_idx else ""
        print(f"  [{i}] {opt}{marker}")
    raw = input(f"Numéro [{default_idx}] : ").strip()
    if not raw:
        return options[default_idx]
    try:
        idx = int(raw)
        if 0 <= idx < len(options):
            return options[idx]
    except ValueError:
        pass
    print(f"  -> invalide, on prend [{default_idx}] {options[default_idx]}")
    return options[default_idx]


def resolve_run_and_ckpt(run_arg: str | None, ckpt_arg: str | None) -> tuple[str, str]:
    """Renvoie (run, ckpt) en demandant à l'utilisateur si nécessaire."""
    run = run_arg
    if run is None:
        runs = _list_runs()
        if not runs:
            raise RuntimeError(f"Aucun run trouvé sous {cfg.CHECKPOINTS}.")
        run = _prompt_choice("run", runs, default_idx=0)

    run_dir = Path(cfg.CHECKPOINTS) / run
    if not run_dir.is_dir():
        raise RuntimeError(f"Run introuvable : {run_dir}")

    ckpt = ckpt_arg
    if ckpt is None:
        ckpts = _list_ckpts(run_dir)
        if not ckpts:
            raise RuntimeError(f"Aucun .pt dans {run_dir}.")
        ckpt = _prompt_choice("checkpoint", ckpts, default_idx=0)

    return run, ckpt


def pick_pairs(n: int, seed: int) -> list[tuple[str, str, str]]:
    """Renvoie une liste de (basename, noisy_path, clean_path)."""
    noisy_dir = Path(cfg.DATA_TEST_NOISY)
    clean_dir = Path(cfg.DATA_TEST_CLEAN)
    noisy_files = {p.name for p in noisy_dir.glob("*.wav")}
    clean_files = {p.name for p in clean_dir.glob("*.wav")}
    common = sorted(noisy_files & clean_files)
    if not common:
        raise RuntimeError(
            f"Aucune paire trouvée entre {noisy_dir} et {clean_dir}. "
            "Le split test a-t-il été généré (scripts/run_full_pipeline.py) ?"
        )
    n = min(n, len(common))
    rng = random.Random(seed)
    picked = rng.sample(common, n)
    return [(name, str(noisy_dir / name), str(clean_dir / name)) for name in picked]


def si_sdr_db(est: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> float:
    """SI-SDR (scale-invariant) en dB entre une estimation et une référence.

    Accompagne l'écoute d'un repère chiffré par extrait. ⚠️ Intrusif : biaisé par
    la qualité imparfaite de la référence « clean » — à lire à titre indicatif,
    l'oreille reste le juge n°1.
    """
    est = est.astype(np.float64); ref = ref.astype(np.float64)
    n = min(len(est), len(ref))
    est, ref = est[:n], ref[:n]
    est = est - est.mean(); ref = ref - ref.mean()
    alpha = float(np.dot(est, ref) / (np.dot(ref, ref) + eps))
    target = alpha * ref
    noise = est - target
    return float(10.0 * np.log10((np.dot(target, target) + eps)
                                 / (np.dot(noise, noise) + eps)))


def main() -> int:
    # Force UTF-8 sur stdout/stderr : les résumés contiennent des caractères
    # non-ASCII (Δ, ★) qui plantent une console Windows cp1252 à l'affichage.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args()

    run, ckpt = resolve_run_and_ckpt(args.run, args.ckpt)
    ckpt_path = Path(cfg.CHECKPOINTS) / run / ckpt
    if not ckpt_path.is_file():
        print(f"[err] checkpoint introuvable : {ckpt_path}", file=sys.stderr)
        return 1

    # La boîte gère le chargement (détection mask/complex), le contrat d'entrée
    # et le forward. Sur un clip de 4 s, `denoise` = une fenêtre (non-régression).
    denoiser = Denoiser(ckpt_path, device=args.device)
    info = denoiser.info
    mode_str = denoiser.output_mode + (
        f" (c={denoiser.c_comp})" if denoiser.output_mode == "complex" else "")
    print(f"[ckpt] {ckpt_path.name} | epoch={info['epoch']} "
          f"| val_loss={info['best_val_loss']:.4f} "
          f"| val_si_sdri={info['best_val_si_sdri']:.3f} "
          f"| mode={mode_str} | base_channels={denoiser.base_channels} "
          f"| device={denoiser.device}")

    pairs = pick_pairs(args.n, args.seed)
    ckpt_stem = Path(ckpt).stem
    out_root = Path(cfg.OUTPUTS) / "listen" / run / ckpt_stem
    out_root.mkdir(parents=True, exist_ok=True)

    sdri_sum = 0.0
    for idx, (name, noisy_path, clean_path) in enumerate(pairs):
        stem = Path(name).stem
        out_dir = out_root / f"{idx:02d}_{stem}"
        out_dir.mkdir(parents=True, exist_ok=True)

        noisy_wav = audio.fix_length(audio.load_audio(noisy_path), mode="center")
        clean_wav = audio.fix_length(audio.load_audio(clean_path), mode="center")
        pred_wav = denoiser.denoise(noisy_wav)

        sf.write(out_dir / "noisy.wav", noisy_wav, cfg.SAMPLE_RATE)
        sf.write(out_dir / "pred.wav", pred_wav, cfg.SAMPLE_RATE)
        sf.write(out_dir / "clean.wav", clean_wav, cfg.SAMPLE_RATE)

        # Repère chiffré par extrait, à confronter à l'écoute.
        sdr_pred = si_sdr_db(pred_wav, clean_wav)
        sdr_noisy = si_sdr_db(noisy_wav, clean_wav)
        sdri_sum += sdr_pred - sdr_noisy
        print(f"  [{idx:02d}] {name}  SI-SDR pred {sdr_pred:+.2f} dB "
              f"(noisy {sdr_noisy:+.2f}, Δ {sdr_pred - sdr_noisy:+.2f} dB) -> {out_dir}")

    if pairs:
        print(f"\n[moyenne] SI-SDRi sur {len(pairs)} extraits : "
              f"{sdri_sum / len(pairs):+.2f} dB  (indicatif — l'oreille prime)")
    print(f"OK -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
