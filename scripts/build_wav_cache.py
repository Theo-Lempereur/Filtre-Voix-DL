"""Pré-calcule un cache memmap des waveforms décodées (float16) par split.

Objectif : supprimer le coût `librosa.load` + STFT côté CPU au DataLoader.
Une fois ce cache construit, `PairedAudioDataset(cache_dir=..., cache_split=...)`
ne fait plus que slicer un `.npy` mmap ; la STFT est calculée sur GPU dans le
forward (cf. `src.audio.stft_torch`).

Pour chaque split, on écrit dans `--cache-dir` :
    <split>_noisy.npy   (N, CLIP_SAMPLES) float16
    <split>_clean.npy   (N, CLIP_SAMPLES) float16
    <split>_names.json  liste ordonnée des noms (aligne le dataset au memmap)

Usage :
    python scripts/build_wav_cache.py                 # train + val
    python scripts/build_wav_cache.py --splits train val test
    python scripts/build_wav_cache.py --force         # reconstruit même si présent

Idempotent : un split déjà présent est sauté sauf `--force`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Permet `python scripts/build_wav_cache.py` depuis la racine du repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import audio as A
from src import config as cfg
from src.dataset import list_pairs

try:  # barre de progression optionnelle
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(it, **kwargs):
        return it


# split -> (noisy_dir, clean_dir)
_SPLIT_DIRS = {
    "train": (cfg.DATA_TRAIN_NOISY, cfg.DATA_TRAIN_CLEAN),
    "val":   (cfg.DATA_VAL_NOISY,   cfg.DATA_VAL_CLEAN),
    "test":  (cfg.DATA_TEST_NOISY,  cfg.DATA_TEST_CLEAN),
}


def _build_split(split: str, cache_dir: Path, force: bool) -> None:
    noisy_dir, clean_dir = _SPLIT_DIRS[split]
    names_path = cache_dir / f"{split}_names.json"
    noisy_npy  = cache_dir / f"{split}_noisy.npy"
    clean_npy  = cache_dir / f"{split}_clean.npy"

    if not force and names_path.exists() and noisy_npy.exists() and clean_npy.exists():
        print(f"[cache] split '{split}' déjà présent — sauté (--force pour reconstruire).")
        return

    pairs = list_pairs(noisy_dir, clean_dir)
    if not pairs:
        print(f"[cache] split '{split}' : aucune paire dans\n"
              f"        {noisy_dir}\n        {clean_dir}\n        -> sauté.")
        return

    n = len(pairs)
    clip = cfg.CLIP_SAMPLES
    print(f"[cache] split '{split}' : {n} paires -> {clip} samples (float16).")

    cache_dir.mkdir(parents=True, exist_ok=True)
    # open_memmap streame sur disque (pas de buffer RAM de N×clip).
    noisy_arr = np.lib.format.open_memmap(
        noisy_npy, mode="w+", dtype=np.float16, shape=(n, clip))
    clean_arr = np.lib.format.open_memmap(
        clean_npy, mode="w+", dtype=np.float16, shape=(n, clip))

    names: list[str] = []
    for i, (name, noisy_path, clean_path) in enumerate(tqdm(pairs, desc=split)):
        noisy = A.fix_length(A.load_audio(noisy_path), clip, mode="start")
        clean = A.fix_length(A.load_audio(clean_path), clip, mode="start")
        noisy_arr[i] = noisy.astype(np.float16)
        clean_arr[i] = clean.astype(np.float16)
        names.append(name)

    noisy_arr.flush()
    clean_arr.flush()
    del noisy_arr, clean_arr  # ferme les memmap proprement (Windows)
    names_path.write_text(json.dumps(names), encoding="utf-8")
    print(f"[cache] split '{split}' écrit dans {cache_dir}.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", nargs="+", default=["train", "val"],
                   choices=list(_SPLIT_DIRS), help="Splits à (re)construire.")
    p.add_argument("--cache-dir", default=cfg.CACHE_DIR,
                   help="Dossier de sortie du cache.")
    p.add_argument("--force", action="store_true",
                   help="Reconstruit même si le cache existe déjà.")
    args = p.parse_args(argv)

    cache_dir = Path(args.cache_dir)
    print(f"[cache] cache-dir : {cache_dir}")
    for split in args.splits:
        _build_split(split, cache_dir, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
