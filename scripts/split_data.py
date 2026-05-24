"""Génère le split train/val/test à partir du dataset brut.

Le dataset d'origine est organisé ainsi :

    data/clean/test_xxx.wav        <- voix propre
    data/noisy/train_xxx.wav       <- meme voix avec bruit

Les noms `test_*` (côté clean) et `train_*` (côté noisy) sont historiques et
sans rapport avec un quelconque split — c'est juste la convention du dataset
public. On les renomme à la copie pour éviter toute confusion :

    data/train/clean/pair_xxx.wav   (copie de data/clean/test_xxx.wav)
    data/train/noisy/pair_xxx.wav   (copie de data/noisy/train_xxx.wav)
    data/val/...
    data/test/...

Le numéro `xxx` est conservé : il sert d'identifiant de paire et permet de
retrouver le fichier d'origine dans `data/clean,noisy/` à tout moment.

Un manifest CSV (`data/split_manifest.csv`) est écrit en sortie ; il liste
chaque paire avec son split et son nom d'origine pour une traçabilité totale.

Usage :
    python scripts/split_data.py
    python scripts/split_data.py --dry-run         # juste affiche le plan
    python scripts/split_data.py --seed 123        # autre tirage
    python scripts/split_data.py --ratios 0.7 0.15 0.15
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import sys
from pathlib import Path

# Permet d'exécuter le script depuis la racine du repo sans installer le package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as cfg


PAIR_NUM_RE = re.compile(r"(\d+)")
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}


# ---------------------------------------------------------------------------
# Découverte des fichiers
# ---------------------------------------------------------------------------

def _extract_number(filename: str) -> int | None:
    """Renvoie le premier groupe de chiffres trouvé dans le nom, ou None."""
    m = PAIR_NUM_RE.search(filename)
    return int(m.group(1)) if m else None


def discover_pairs(clean_dir: Path, noisy_dir: Path) -> list[tuple[int, Path, Path]]:
    """Liste les paires (numero, clean_path, noisy_path) appariées par numéro.

    Robuste aux trous : si seul un côté possède le numéro xxx, la paire est
    ignorée (avec un avertissement résumé).
    """
    def _index(folder: Path) -> dict[int, Path]:
        out: dict[int, Path] = {}
        if not folder.is_dir():
            return out
        for p in folder.iterdir():
            if not p.is_file() or p.suffix.lower() not in AUDIO_EXTS:
                continue
            n = _extract_number(p.name)
            if n is None:
                continue
            if n in out:
                # Collision : deux fichiers portent le même numéro. On garde
                # le plus ancien et on prévient.
                print(f"  ! collision numéro {n} dans {folder.name} : "
                      f"{p.name} ignoré (déjà {out[n].name})")
                continue
            out[n] = p
        return out

    clean_idx = _index(clean_dir)
    noisy_idx = _index(noisy_dir)

    common = sorted(set(clean_idx) & set(noisy_idx))
    only_clean = sorted(set(clean_idx) - set(noisy_idx))
    only_noisy = sorted(set(noisy_idx) - set(clean_idx))

    if only_clean:
        print(f"  ! {len(only_clean)} numéro(s) présents uniquement côté clean, "
              f"ignorés (ex: {only_clean[:3]})")
    if only_noisy:
        print(f"  ! {len(only_noisy)} numéro(s) présents uniquement côté noisy, "
              f"ignorés (ex: {only_noisy[:3]})")

    return [(n, clean_idx[n], noisy_idx[n]) for n in common]


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def split_pairs(pairs: list, ratios: tuple[float, float, float],
                seed: int) -> dict[str, list]:
    """Shuffle déterministe puis découpe en train/val/test."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Les ratios doivent sommer à 1, reçus : {ratios}")

    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])
    # le test prend le reste pour ne perdre aucune paire au calcul flottant
    return {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train:n_train + n_val],
        "test":  shuffled[n_train + n_val:],
    }


# ---------------------------------------------------------------------------
# Copie et manifest
# ---------------------------------------------------------------------------

def apply_split(splits: dict[str, list], *, dry_run: bool) -> list[dict]:
    """Copie les fichiers vers data/<split>/{clean,noisy}/ et renvoie le manifest.

    Renomme chaque paire `pair_<num>.wav` en conservant `num` du nom d'origine.
    """
    targets = {
        "train": (Path(cfg.DATA_TRAIN_CLEAN), Path(cfg.DATA_TRAIN_NOISY)),
        "val":   (Path(cfg.DATA_VAL_CLEAN),   Path(cfg.DATA_VAL_NOISY)),
        "test":  (Path(cfg.DATA_TEST_CLEAN),  Path(cfg.DATA_TEST_NOISY)),
    }

    manifest: list[dict] = []
    for split, items in splits.items():
        clean_dst_dir, noisy_dst_dir = targets[split]
        if not dry_run:
            clean_dst_dir.mkdir(parents=True, exist_ok=True)
            noisy_dst_dir.mkdir(parents=True, exist_ok=True)

        for num, clean_src, noisy_src in items:
            new_name = f"pair_{num:04d}{clean_src.suffix.lower()}"
            clean_dst = clean_dst_dir / new_name
            noisy_dst = noisy_dst_dir / f"pair_{num:04d}{noisy_src.suffix.lower()}"

            if not dry_run:
                # `copy2` préserve les métadonnées et écrase si déjà présent
                # (utile pour ré-exécuter le script avec une autre seed).
                shutil.copy2(clean_src, clean_dst)
                shutil.copy2(noisy_src, noisy_dst)

            manifest.append({
                "pair_id": num,
                "split":   split,
                "clean_original": clean_src.name,
                "noisy_original": noisy_src.name,
                "clean_new":      str(clean_dst.relative_to(Path(cfg.DRIVE_PROJECT))) if Path(cfg.DRIVE_PROJECT) in clean_dst.parents else str(clean_dst),
                "noisy_new":      str(noisy_dst.relative_to(Path(cfg.DRIVE_PROJECT))) if Path(cfg.DRIVE_PROJECT) in noisy_dst.parents else str(noisy_dst),
            })
    return manifest


def write_manifest(manifest: list[dict], path: Path) -> None:
    """Écrit le manifest CSV. Trie par pair_id pour une lecture humaine facile."""
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_sorted = sorted(manifest, key=lambda r: r["pair_id"])
    fieldnames = ["pair_id", "split", "clean_original", "noisy_original",
                  "clean_new", "noisy_new"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_sorted)


# ---------------------------------------------------------------------------
# Entrée
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clean-dir", default=cfg.DATA_CLEAN,
                        help="dossier source des fichiers propres")
    parser.add_argument("--noisy-dir", default=cfg.DATA_NOISY,
                        help="dossier source des fichiers bruités")
    parser.add_argument("--seed", type=int, default=cfg.SEED,
                        help="seed pour le shuffle déterministe")
    parser.add_argument("--ratios", type=float, nargs=3,
                        default=list(cfg.SPLIT_RATIOS), metavar=("TRAIN", "VAL", "TEST"),
                        help="ratios de split (doivent sommer à 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="affiche le plan mais ne copie rien")
    args = parser.parse_args()

    clean_dir = Path(args.clean_dir)
    noisy_dir = Path(args.noisy_dir)

    print(f"[split] sources : {clean_dir}  |  {noisy_dir}")
    pairs = discover_pairs(clean_dir, noisy_dir)
    print(f"[split] {len(pairs)} paires appariées par numéro.")
    if not pairs:
        print("[split] rien à faire.")
        return

    splits = split_pairs(pairs, tuple(args.ratios), seed=args.seed)
    print(f"[split] répartition : "
          f"train={len(splits['train'])}  "
          f"val={len(splits['val'])}  "
          f"test={len(splits['test'])}  "
          f"(seed={args.seed}, ratios={args.ratios})")

    if args.dry_run:
        print("[split] --dry-run : aucun fichier copié.")
        return

    manifest = apply_split(splits, dry_run=False)
    manifest_path = Path(cfg.DRIVE_PROJECT) / "data" / "split_manifest.csv"
    write_manifest(manifest, manifest_path)
    print(f"[split] manifest écrit : {manifest_path}")
    print("[split] terminé.")


if __name__ == "__main__":
    main()
