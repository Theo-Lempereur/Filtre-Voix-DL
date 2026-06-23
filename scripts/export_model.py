"""Exporte un checkpoint **complex** vers le paquet autonome ``wallace``.

Sérialise le U-Net en **TorchScript** (architecture + poids embarqués → le service
consommateur n'a plus besoin de `src/model.py`) et écrit les métadonnées (contrat
d'entrée + provenance). Les deux artefacts atterrissent dans le dossier du paquet
``export/wallace/wallace/`` pour être embarqués au ``pip install``.

Usage :
    python scripts/export_model.py --ckpt rp_csm_final
    python scripts/export_model.py --ckpt "G:/.../best_si_sdri.pt" --out <dossier>

``--ckpt`` accepte : un fichier .pt, un dossier de run, ou un **nom de run**
(cherché sous ``config.CHECKPOINTS``).
"""
from __future__ import annotations

import argparse
import datetime
import io
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as cfg
from src.denoiser import Wallace

_DEFAULT_OUT = (Path(__file__).resolve().parents[1]
                / "export" / "wallace" / "wallace")


def _resolve_ckpt_arg(ckpt: str) -> str:
    p = Path(ckpt)
    if p.exists():
        return ckpt
    candidate = Path(cfg.CHECKPOINTS) / ckpt
    if candidate.exists():
        return str(candidate)
    return ckpt  # laisse Wallace tenter sa résolution / lever une erreur claire


def main() -> int:
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True,
                    help="Fichier .pt, dossier de run, ou nom de run (sous CHECKPOINTS).")
    ap.add_argument("--out", default=str(_DEFAULT_OUT),
                    help="Dossier de sortie (défaut : le paquet wallace).")
    args = ap.parse_args()

    ckpt = _resolve_ckpt_arg(args.ckpt)
    box = Wallace(ckpt, device="cpu")          # charge + valide le modèle
    if box.output_mode != "complex":
        print(f"[export] ERREUR : output_mode={box.output_mode!r} — "
              f"l'export ne gère que le mode 'complex'.", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Modèle -> TorchScript (architecture + poids embarqués).
    # On sauve via un buffer puis write_bytes : le save C++ de torch ne résout pas
    # les chemins non-ASCII sous Windows (ex. dossier « Théo »), Python si.
    scripted = torch.jit.script(box.model)
    ts_path = out / "model.ts"
    _buf = io.BytesIO()
    torch.jit.save(scripted, _buf)
    ts_path.write_bytes(_buf.getvalue())

    # 2) Métadonnées : contrat d'entrée + provenance.
    info = box.info
    meta = {
        "name": "wallace",
        "run_id": info.get("run_id") or Path(box.ckpt_path).parent.name,
        "epoch": info.get("epoch"),
        "val_si_sdri": info.get("best_val_si_sdri"),
        "output_mode": "complex",
        "sample_rate": cfg.SAMPLE_RATE,
        "n_fft": cfg.N_FFT,
        "hop_length": cfg.HOP_LENGTH,
        "win_length": cfg.WIN_LENGTH,
        "clip_samples": cfg.CLIP_SAMPLES,
        "csm_compress": box.c_comp,
        "base_channels": box.base_channels,
        "overlap_default": 0.5,
        "source_ckpt": str(box.ckpt_path),
        "torch_version": torch.__version__,
        "exported_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    meta_path = out / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    size_mo = ts_path.stat().st_size / 1e6
    print(f"[export] model.ts      -> {ts_path} ({size_mo:.1f} Mo)")
    print(f"[export] metadata.json -> {meta_path}")
    print(f"[export] run={meta['run_id']} epoch={meta['epoch']} "
          f"si_sdri={meta['val_si_sdri']:.3f} base_channels={meta['base_channels']}")
    print(f"[export] Paquet prêt. Installe-le dans un service :")
    print(f"           pip install ./export/wallace")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
