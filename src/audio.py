"""Helpers audio : chargement, crop/pad, STFT/ISTFT, visualisation.

Tous les helpers s'appuient sur les constantes de `src.config` (SAMPLE_RATE,
CLIP_SAMPLES, N_FFT, HOP_LENGTH, WIN_LENGTH) pour rester cohérents partout.
"""
from __future__ import annotations

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

from . import config


# ---------------------------------------------------------------------------
# Chargement / mise en forme du signal temporel
# ---------------------------------------------------------------------------

def load_audio(path: str, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """Charge un fichier audio en mono float32, rééchantillonné à `sr`."""
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav.astype(np.float32)


def fix_length(wav: np.ndarray, n_samples: int = config.CLIP_SAMPLES,
               mode: str = "center") -> np.ndarray:
    """Crop ou pad le signal pour atteindre exactement `n_samples`.

    - mode="center" : crop centré, pad silence symétrique
    - mode="random" : crop à offset aléatoire (utile pour l'entraînement)
    - mode="start"  : on garde le début, pad silence à droite
    """
    length = wav.shape[-1]
    if length == n_samples:
        return wav
    if length > n_samples:
        if mode == "random":
            start = np.random.randint(0, length - n_samples + 1)
        elif mode == "center":
            start = (length - n_samples) // 2
        else:
            start = 0
        return wav[..., start:start + n_samples]
    # length < n_samples -> padding silence
    pad = n_samples - length
    if mode == "center":
        left = pad // 2
        right = pad - left
    else:
        left, right = 0, pad
    return np.pad(wav, (left, right), mode="constant")


# ---------------------------------------------------------------------------
# STFT / ISTFT
# ---------------------------------------------------------------------------

def stft(wav: np.ndarray,
         n_fft: int = config.N_FFT,
         hop_length: int = config.HOP_LENGTH,
         win_length: int = config.WIN_LENGTH) -> np.ndarray:
    """STFT complexe (shape : [freq, time])."""
    return librosa.stft(wav, n_fft=n_fft, hop_length=hop_length,
                        win_length=win_length)


def istft(spec: np.ndarray,
          hop_length: int = config.HOP_LENGTH,
          win_length: int = config.WIN_LENGTH,
          length: int | None = None) -> np.ndarray:
    """Inverse de `stft` : spectro complexe -> signal temporel."""
    return librosa.istft(spec, hop_length=hop_length, win_length=win_length,
                         length=length)


def magnitude_phase(spec_complex: np.ndarray):
    """Sépare un spectre complexe en (magnitude, phase)."""
    return np.abs(spec_complex), np.angle(spec_complex)


def reconstruct(magnitude: np.ndarray, phase: np.ndarray,
                hop_length: int = config.HOP_LENGTH,
                win_length: int = config.WIN_LENGTH,
                length: int | None = None) -> np.ndarray:
    """Reconstruit un signal à partir d'une magnitude (débruité) et d'une phase."""
    spec = magnitude * np.exp(1j * phase)
    return istft(spec, hop_length=hop_length, win_length=win_length, length=length)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_waveform(wav: np.ndarray, sr: int = config.SAMPLE_RATE,
                  title: str = "Waveform", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 2.5))
    t = np.arange(wav.shape[-1]) / sr
    ax.plot(t, wav, linewidth=0.6)
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.set_xlim(0, t[-1] if t.size else 0)
    ax.grid(alpha=0.3)
    return ax


def plot_spectrogram(wav_or_spec, sr: int = config.SAMPLE_RATE,
                     n_fft: int = config.N_FFT,
                     hop_length: int = config.HOP_LENGTH,
                     title: str = "Spectrogramme (dB)",
                     ax=None, colorbar: bool = True):
    """Affiche un spectrogramme en dB.

    Accepte soit un signal temporel 1D, soit un spectre complexe 2D.
    """
    if wav_or_spec.ndim == 1:
        spec = stft(wav_or_spec, n_fft=n_fft, hop_length=hop_length)
    else:
        spec = wav_or_spec
    mag_db = librosa.amplitude_to_db(np.abs(spec), ref=np.max)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mag_db, sr=sr, hop_length=hop_length,
                                   x_axis="time", y_axis="hz", ax=ax)
    ax.set_title(title)
    if colorbar:
        plt.colorbar(img, ax=ax, format="%+2.0f dB")
    return ax


def plot_pair(noisy_wav: np.ndarray, clean_wav: np.ndarray,
              sr: int = config.SAMPLE_RATE):
    """Affiche côte à côte la paire (bruité, propre) : waveform + spectro."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 6))
    plot_waveform(noisy_wav, sr=sr, title="Noisy — waveform", ax=axes[0, 0])
    plot_waveform(clean_wav, sr=sr, title="Clean — waveform", ax=axes[0, 1])
    plot_spectrogram(noisy_wav, sr=sr, title="Noisy — spectro", ax=axes[1, 0])
    plot_spectrogram(clean_wav, sr=sr, title="Clean — spectro", ax=axes[1, 1])
    fig.tight_layout()
    return fig
