"""Mesure les bornes 'oracle' de SI-SDR sur le val set — diagnostic de plafond.

Objectif : savoir où se situe le plafond théorique de la formulation actuelle
(masque de magnitude borné [0,1] + phase du bruité) AVANT d'investir un run.

On compare, en SI-SDR waveform (la vraie métrique perceptive), plusieurs
reconstructions oracle (qui utilisent la cible propre, donc inatteignables par
un vrai modèle — ce sont des PLAFONDS) :

  baseline_noisy   : le signal bruité tel quel (référence, = 0 dB d'amélioration)
  irm_noisyphase   : masque idéal borné [0,1]  + phase bruité   <- plafond de TON modèle actuel
  oraclemag_noisyphase : magnitude propre exacte + phase bruité <- plafond si on lève la borne [0,1]
  oraclemag_cleanphase : magnitude propre + phase propre        <- sanity (≈ reconstruction parfaite)

Lecture :
  - Si `irm_noisyphase` ≈ ton SI-SDRi actuel → tu es au plafond magnitude :
    plus de capacité/données ne sert à rien, il FAUT passer phase-aware (cIRM /
    complex mapping / time-domain).
  - L'écart `oraclemag_cleanphase` − `oraclemag_noisyphase` = le 'prix' de la
    phase : combien de dB on laisse sur la table en gardant la phase bruité.

Usage :
    python scripts/oracle_ceiling.py --n 512
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as cfg


def _stft(wav: torch.Tensor):
    win = torch.hann_window(cfg.WIN_LENGTH, device=wav.device, dtype=wav.dtype)
    spec = torch.stft(wav, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH,
                      win_length=cfg.WIN_LENGTH, window=win, center=True,
                      pad_mode="constant", return_complex=True)
    return spec


def _istft(spec: torch.Tensor, length: int) -> torch.Tensor:
    win = torch.hann_window(cfg.WIN_LENGTH, device=spec.device, dtype=torch.float32)
    return torch.istft(spec, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH,
                       win_length=cfg.WIN_LENGTH, window=win, center=True,
                       length=length, return_complex=False)


def _si_sdr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    dot = (pred * target).sum(1, keepdim=True)
    tgt_e = (target * target).sum(1, keepdim=True) + eps
    s_t = (dot / tgt_e) * target
    e_n = pred - s_t
    num = (s_t * s_t).sum(1) + eps
    den = (e_n * e_n).sum(1) + eps
    return (10.0 * torch.log10(num / den))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=512, help="Nb d'exemples val à évaluer.")
    p.add_argument("--cache-dir", default=cfg.CACHE_DIR)
    p.add_argument("--split", default="val")
    p.add_argument("--batch", type=int, default=64)
    args = p.parse_args(argv)

    cdir = Path(args.cache_dir)
    noisy = np.load(cdir / f"{args.split}_noisy.npy", mmap_mode="r")
    clean = np.load(cdir / f"{args.split}_clean.npy", mmap_mode="r")
    n = min(args.n, noisy.shape[0])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[oracle] {n} exemples '{args.split}' | device={device} | clip={cfg.CLIP_SAMPLES}")

    acc = {k: [] for k in
           ("baseline_noisy", "irm_noisyphase", "oraclemag_noisyphase", "oraclemag_cleanphase")}

    for start in range(0, n, args.batch):
        end = min(start + args.batch, n)
        nw = torch.from_numpy(np.ascontiguousarray(noisy[start:end], dtype=np.float32)).to(device)
        cw = torch.from_numpy(np.ascontiguousarray(clean[start:end], dtype=np.float32)).to(device)
        L = nw.shape[-1]

        ns = _stft(nw); cs = _stft(cw)
        n_mag = ns.abs(); n_ph = torch.angle(ns)
        c_mag = cs.abs(); c_ph = torch.angle(cs)

        # 1) baseline : bruité brut
        acc["baseline_noisy"].append(_si_sdr(nw, cw))
        # 2) masque idéal borné [0,1] + phase bruité  (== min(clean_mag, noisy_mag))
        irm_mag = torch.minimum(c_mag, n_mag)
        rec = _istft(torch.polar(irm_mag, n_ph), L)
        acc["irm_noisyphase"].append(_si_sdr(rec, cw))
        # 3) magnitude propre exacte (non bornée) + phase bruité
        rec = _istft(torch.polar(c_mag, n_ph), L)
        acc["oraclemag_noisyphase"].append(_si_sdr(rec, cw))
        # 4) magnitude propre + phase propre (sanity ≈ parfait)
        rec = _istft(torch.polar(c_mag, c_ph), L)
        acc["oraclemag_cleanphase"].append(_si_sdr(rec, cw))

    res = {k: float(torch.cat(v).mean().item()) for k, v in acc.items()}
    base = res["baseline_noisy"]
    print("\n=== SI-SDR waveform (dB) — moyennes ===")
    for k in ("baseline_noisy", "irm_noisyphase", "oraclemag_noisyphase", "oraclemag_cleanphase"):
        sdr = res[k]
        print(f"  {k:24s} : {sdr:+7.2f} dB   (SI-SDRi {sdr - base:+6.2f})")
    print("\n=== Lecture ===")
    print(f"  Plafond de ta formulation actuelle (masque [0,1] + phase bruité) : "
          f"SI-SDRi {res['irm_noisyphase'] - base:+.2f} dB")
    print(f"  Gain potentiel en levant la borne du masque : "
          f"{res['oraclemag_noisyphase'] - res['irm_noisyphase']:+.2f} dB")
    print(f"  Prix de la phase (ce qu'on perd en gardant la phase bruité) : "
          f"{res['oraclemag_cleanphase'] - res['oraclemag_noisyphase']:+.2f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
