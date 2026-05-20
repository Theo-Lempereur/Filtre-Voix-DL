"""Dataset PyTorch pour les paires (audio bruité, audio propre).

Convention : un même nom de fichier doit exister dans `DATA_NOISY` et `DATA_CLEAN`.
Le dataset liste l'intersection des deux dossiers, charge l'audio, ajuste la durée
à `CLIP_SAMPLES`, puis renvoie soit les waveforms, soit les spectrogrammes selon
la valeur de `return_spectrogram`.
"""
from __future__ import annotations

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
               clean_dir: str = config.DATA_CLEAN) -> list[tuple[str, str, str]]:
    """Liste les paires de fichiers présentes dans les deux dossiers.

    Returns
    -------
    list de tuples (filename, noisy_path, clean_path), triée par filename.
    """
    noisy = _list_audio_files(noisy_dir)
    clean = _list_audio_files(clean_dir)
    common = sorted(set(noisy.keys()) & set(clean.keys()))
    return [(name, noisy[name], clean[name]) for name in common]


class PairedAudioDataset(Dataset):
    """Dataset PyTorch listant des paires (bruité, propre).

    Parameters
    ----------
    noisy_dir, clean_dir : dossiers contenant des fichiers audio du même nom.
    sample_rate         : SR cible (resampling au chargement si besoin).
    clip_samples        : nombre d'échantillons fixes à la sortie.
    crop_mode           : "center" / "random" / "start" (cf. audio.fix_length).
    return_spectrogram  : si True renvoie magnitude + phase ; sinon waveforms.
    files               : optionnel, liste de noms de fichiers à utiliser (sinon tout).
    """

    def __init__(self,
                 noisy_dir: str = config.DATA_NOISY,
                 clean_dir: str = config.DATA_CLEAN,
                 sample_rate: int = config.SAMPLE_RATE,
                 clip_samples: int = config.CLIP_SAMPLES,
                 crop_mode: str = "random",
                 return_spectrogram: bool = True,
                 files: Iterable[str] | None = None):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.sample_rate = sample_rate
        self.clip_samples = clip_samples
        self.crop_mode = crop_mode
        self.return_spectrogram = return_spectrogram

        pairs = list_pairs(noisy_dir, clean_dir)
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

        if self.return_spectrogram:
            noisy_spec = A.stft(noisy)
            clean_spec = A.stft(clean)
            # Pour le U-Net on cible la magnitude ; on conserve la phase noisy
            # pour pouvoir reconstruire à l'inférence.
            noisy_mag, noisy_phase = A.magnitude_phase(noisy_spec)
            clean_mag = np.abs(clean_spec)

            return {
                "name": name,
                "noisy_mag": torch.from_numpy(noisy_mag).float().unsqueeze(0),
                "clean_mag": torch.from_numpy(clean_mag).float().unsqueeze(0),
                "noisy_phase": torch.from_numpy(noisy_phase).float(),
            }

        return {
            "name": name,
            "noisy": torch.from_numpy(noisy).float(),
            "clean": torch.from_numpy(clean).float(),
        }
