"""Débruite un fichier audio de durée **quelconque** (> 4 s géré).

S'appuie sur la boîte ``src.denoiser.Wallace`` : chargement automatique du mode
(mask / complex) depuis le checkpoint, contrat d'entrée mono 16 kHz, et
fenêtrage 4 s + overlap-add 50 % pour les sons longs.

Usage :
    python scripts/denoise_file.py --ckpt rp_csm_final --in bruite.wav --out propre.wav
    python scripts/denoise_file.py --ckpt "G:/.../checkpoints/rp_csm_final/best_si_sdri.pt" \
        --in long.wav --out long_clean.wav --overlap 0.5

``--ckpt`` accepte : un fichier .pt, un dossier de run (prend best.pt), ou un
**nom de run** (cherché sous ``config.CHECKPOINTS``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import audio, config as cfg
from src.denoiser import Wallace


def _resolve_ckpt_arg(ckpt: str) -> str:
    """Résout ``--ckpt`` : chemin tel quel s'il existe, sinon nom de run sous CHECKPOINTS."""
    p = Path(ckpt)
    if p.exists():
        return ckpt
    candidate = Path(cfg.CHECKPOINTS) / ckpt
    if candidate.exists():
        return str(candidate)
    # Laisse Wallace tenter sa propre résolution (ex: nom sans extension) et
    # lever une erreur claire si vraiment introuvable.
    return ckpt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True,
                   help="Fichier .pt, dossier de run, ou nom de run (sous CHECKPOINTS).")
    p.add_argument("--in", dest="in_path", required=True, help="Audio bruité d'entrée.")
    p.add_argument("--out", dest="out_path", required=True, help="WAV débruité de sortie.")
    p.add_argument("--device", default=None, help="cuda / cpu (auto si omis).")
    p.add_argument("--overlap", type=float, default=0.5,
                   help="Recouvrement entre fenêtres de 4 s (0.5 = 50 %%, défaut).")
    return p.parse_args()


def main() -> int:
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args()
    ckpt = _resolve_ckpt_arg(args.ckpt)

    box = Wallace(ckpt, device=args.device, overlap=args.overlap)
    info = box.info
    mode_str = box.output_mode + (
        f" (c={box.c_comp})" if box.output_mode == "complex" else "")
    print(f"[ckpt] {Path(box.ckpt_path).name} | epoch={info['epoch']} "
          f"| val_si_sdri={info['best_val_si_sdri']:.3f} | mode={mode_str} "
          f"| device={box.device}")

    wav = audio.load_audio(args.in_path)        # mono 16 kHz float32 (resample inclus)
    pred = box.denoise(wav)                      # durée quelconque -> même durée

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), pred, cfg.SAMPLE_RATE)

    sr = cfg.SAMPLE_RATE
    print(f"[in]  {len(wav)} samples ({len(wav) / sr:.2f} s)")
    print(f"[out] {len(pred)} samples ({len(pred) / sr:.2f} s) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
