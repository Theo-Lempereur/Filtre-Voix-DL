"""Dataset PyTorch pour les paires (audio bruité, audio propre).

Convention : un même nom de fichier doit exister dans `DATA_NOISY` et `DATA_CLEAN`.
Le dataset liste l'intersection des deux dossiers, charge l'audio, ajuste la durée
à `CLIP_SAMPLES`, puis renvoie soit les waveforms, soit les spectrogrammes selon
la valeur de `return_spectrogram`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from . import config
from . import audio as A


AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg")


def _list_audio_files(folder: str) -> dict[str, str]:
    """Renvoie {nom_de_fichier: chemin_complet} pour les fichiers audio du dossier."""
    folder = Path(folder)
    if not folder.exists():
        return {}
    out: dict[str, str] = {}
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            out[p.name] = str(p)
    return out


def list_pairs(noisy_dir: str = config.DATA_NOISY,
               clean_dir: str = config.DATA_CLEAN,
               pair_by: str = "auto") -> list[tuple[str, str, str]]:
    """Liste les paires de fichiers (noisy, clean).

    Parameters
    ----------
    pair_by : "name"  → apparie par nom de fichier identique (ex: p001.wav / p001.wav)
              "index" → apparie par ordre trié (ex: train_001.wav ↔ test_001.wav)
              "auto"  → essaie "name" d'abord, bascule sur "index" si 0 paires trouvées

    Returns
    -------
    list de tuples (label, noisy_path, clean_path), triée.
    """
    noisy = _list_audio_files(noisy_dir)
    clean = _list_audio_files(clean_dir)

    if pair_by in ("name", "auto"):
        common = sorted(set(noisy.keys()) & set(clean.keys()))
        if common:
            return [(name, noisy[name], clean[name]) for name in common]
        if pair_by == "name":
            return []

    # pair_by == "index" (ou fallback auto)
    noisy_sorted = sorted(noisy.values())
    clean_sorted = sorted(clean.values())
    n = min(len(noisy_sorted), len(clean_sorted))
    return [(f"pair_{i:04d}", noisy_sorted[i], clean_sorted[i]) for i in range(n)]


class PairedAudioDataset(Dataset):
    """Dataset PyTorch listant des paires (bruité, propre).

    Parameters
    ----------
    noisy_dir, clean_dir : dossiers contenant des fichiers audio du même nom.
    sample_rate         : SR cible (resampling au chargement si besoin).
    clip_samples        : nombre d'échantillons fixes à la sortie.
    crop_mode           : "center" / "random" / "start" (cf. audio.fix_length).
    return_spectrogram  : si True renvoie magnitude + phase ; sinon waveforms.
    return_waveform     : si True ET return_spectrogram=True, retourne aussi
                          `noisy_wav` et `clean_wav` (utilisé par les losses
                          MR-STFT / SI-SDR temporel / cIRM).
    files               : optionnel, liste de noms de fichiers à utiliser (sinon tout).
    """

    def __init__(self,
                 noisy_dir: str = config.DATA_NOISY,
                 clean_dir: str = config.DATA_CLEAN,
                 sample_rate: int = config.SAMPLE_RATE,
                 clip_samples: int = config.CLIP_SAMPLES,
                 crop_mode: str = "random",
                 return_spectrogram: bool = True,
                 return_waveform: bool = False,
                 pair_by: str = "auto",
                 files: Iterable[str] | None = None,
                 cache_dir: str | None = None,
                 cache_split: str | None = None,
                 waveform_only: bool = False):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.sample_rate = sample_rate
        self.clip_samples = clip_samples
        self.crop_mode = crop_mode
        self.return_spectrogram = return_spectrogram
        self.return_waveform = return_waveform
        # Mode waveform-only : __getitem__ ne renvoie que les waveforms
        # (noisy_wav / clean_wav) ; la STFT est faite plus tard sur GPU. C'est le
        # chemin du pipeline d'entraînement optimisé. Le cache l'implique.
        self.waveform_only = bool(waveform_only or cache_dir is not None)
        self._use_cache = False
        # Les memmaps sont ouverts PARESSEUSEMENT dans __getitem__ (donc dans
        # chaque worker). On ne garde que les chemins en attribut : un objet
        # np.memmap ne se pickle pas vers les workers (spawn Windows -> crash).
        self._cache_noisy_path: str | None = None
        self._cache_clean_path: str | None = None
        self._cache_noisy = None
        self._cache_clean = None

        # --- Chemin cache memmap (float16) : pas de listing de dossier ni de
        # décodage WAV à chaque accès — on slice un .npy mmap. ---
        if cache_dir is not None:
            if cache_split is None:
                raise ValueError("cache_split est requis quand cache_dir est fourni.")
            cdir = Path(cache_dir)
            names_path = cdir / f"{cache_split}_names.json"
            noisy_npy  = cdir / f"{cache_split}_noisy.npy"
            clean_npy  = cdir / f"{cache_split}_clean.npy"
            if not (names_path.exists() and noisy_npy.exists() and clean_npy.exists()):
                raise FileNotFoundError(
                    f"Cache introuvable pour le split '{cache_split}' dans {cdir}. "
                    "Lance d'abord `python scripts/build_wav_cache.py`."
                )
            names = json.loads(names_path.read_text(encoding="utf-8"))
            # Lecture du header (shape) sans conserver le memmap ouvert.
            n_noisy = int(np.load(noisy_npy, mmap_mode="r").shape[0])
            n_clean = int(np.load(clean_npy, mmap_mode="r").shape[0])
            if not (len(names) == n_noisy == n_clean):
                raise ValueError(
                    f"Cache '{cache_split}' incohérent : {len(names)} noms vs "
                    f"{n_noisy}/{n_clean} lignes."
                )
            self._use_cache = True
            self._cache_noisy_path = str(noisy_npy)
            self._cache_clean_path = str(clean_npy)
            self.pairs = [(n, "", "") for n in names]
            return

        pairs = list_pairs(noisy_dir, clean_dir, pair_by=pair_by)
        if files is not None:
            wanted = set(files)
            pairs = [p for p in pairs if p[0] in wanted]
        if not pairs:
            raise FileNotFoundError(
                f"Aucune paire trouvée entre {noisy_dir} et {clean_dir}. "
                "Vérifie que les fichiers ont le même nom dans les deux dossiers."
            )
        self.pairs = pairs

    # ---- API PyTorch ----

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        name, noisy_path, clean_path = self.pairs[idx]

        if self._use_cache:
            if self._cache_noisy is None:
                # Ouverture paresseuse, une fois par worker (cf. __init__).
                self._cache_noisy = np.load(self._cache_noisy_path, mmap_mode="r")
                self._cache_clean = np.load(self._cache_clean_path, mmap_mode="r")
            # Slice memmap float16 -> copie float32 contiguë.
            noisy = np.ascontiguousarray(self._cache_noisy[idx], dtype=np.float32)
            clean = np.ascontiguousarray(self._cache_clean[idx], dtype=np.float32)
        else:
            noisy = A.load_audio(noisy_path, sr=self.sample_rate)
            clean = A.load_audio(clean_path, sr=self.sample_rate)

        # Pour garder l'alignement bruité/propre, on choisit un offset commun
        # quand on est en mode random.
        if self.crop_mode == "random":
            length = min(noisy.shape[-1], clean.shape[-1])
            noisy = noisy[..., :length]
            clean = clean[..., :length]
            if length > self.clip_samples:
                start = np.random.randint(0, length - self.clip_samples + 1)
                noisy = noisy[..., start:start + self.clip_samples]
                clean = clean[..., start:start + self.clip_samples]
            else:
                noisy = A.fix_length(noisy, self.clip_samples, mode="start")
                clean = A.fix_length(clean, self.clip_samples, mode="start")
        else:
            noisy = A.fix_length(noisy, self.clip_samples, mode=self.crop_mode)
            clean = A.fix_length(clean, self.clip_samples, mode=self.crop_mode)

        if self.waveform_only:
            # Pipeline optimisé : on ne renvoie que les waveforms ; la STFT
            # (noisy_mag/phase, clean_mag) est calculée sur GPU dans le forward.
            return {
                "name": name,
                "noisy_wav": torch.from_numpy(noisy).float(),
                "clean_wav": torch.from_numpy(clean).float(),
            }

        if self.return_spectrogram:
            noisy_spec = A.stft(noisy)
            clean_spec = A.stft(clean)
            # Pour le U-Net on cible la magnitude ; on conserve la phase noisy
            # pour pouvoir reconstruire à l'inférence.
            noisy_mag, noisy_phase = A.magnitude_phase(noisy_spec)
            clean_mag = np.abs(clean_spec)

            out = {
                "name": name,
                "noisy_mag": torch.from_numpy(noisy_mag).float().unsqueeze(0),
                "clean_mag": torch.from_numpy(clean_mag).float().unsqueeze(0),
                "noisy_phase": torch.from_numpy(noisy_phase).float(),
            }
            if self.return_waveform:
                # Utilisé par les losses domaine temporel (MR-STFT, SI-SDR wav)
                # et par cIRM (qui a besoin du clean_wav pour la loss SI-SDR).
                out["noisy_wav"] = torch.from_numpy(noisy).float()
                out["clean_wav"] = torch.from_numpy(clean).float()
            return out

        return {
            "name": name,
            "noisy": torch.from_numpy(noisy).float(),
            "clean": torch.from_numpy(clean).float(),
        }
