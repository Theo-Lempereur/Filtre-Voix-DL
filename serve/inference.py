"""Inférence en mémoire pour servir le modèle de débruitage via HTTP.

Ce module est l'équivalent « sans système de fichiers » de
``scripts/listen_test.py`` : il charge un checkpoint en reconstruisant
l'architecture exacte du U-Net, puis débruite un flux audio passé sous forme
de ``bytes`` et renvoie un WAV (``bytes``) prêt à être streamé au client.

Aucune écriture disque n'est faite : tout passe par ``io.BytesIO``.

Pipeline (identique à ``listen_test.denoise`` mais généralisé) :
    bytes audio
      → librosa.load (decode, resample 16 kHz mono)
      → fix_length(CLIP_SAMPLES, center)
      → STFT → magnitude / phase
      → compression input_repr (log1p / pow03 / linear)
      → U-Net → masque (magnitude ou cIRM)
      → reconstruction ISTFT
      → soundfile.write (PCM_16 WAV)
      → bytes
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import torch

# Garantit que `src/` est importable quelle que soit la façon dont uvicorn est
# lancé (depuis la racine du repo, via systemd avec un autre CWD, etc.).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import audio, config as cfg  # noqa: E402
from src.checkpoint import (  # noqa: E402
    find_best_checkpoint,
    find_latest_checkpoint,
    load_checkpoint,
    peek_checkpoint_config,
)
from src.model import UNet  # noqa: E402


# Taille minimale exploitable : au moins une fenêtre STFT complète.
_MIN_SAMPLES = cfg.N_FFT


def resolve_ckpt_path(path: str) -> str:
    """Résout un CKPT_PATH souple vers un fichier .pt concret.

    Accepte trois formes :
      - un fichier .pt           -> renvoyé tel quel
      - un nom sans extension    -> ``best`` devient ``best.pt`` s'il existe
      - un dossier de run        -> prend ``best.pt``, sinon le dernier
                                    checkpoint (``last.pt`` / ``epoch_NNN.pt``)

    Raises
    ------
    FileNotFoundError
        Si aucun checkpoint exploitable n'est trouvé.
    """
    p = Path(path)

    # 1) Fichier existant tel quel.
    if p.is_file():
        return str(p)

    # 2) Dossier de run -> best.pt puis dernier checkpoint.
    if p.is_dir():
        best = find_best_checkpoint(p)
        if best:
            return best
        latest = find_latest_checkpoint(p)
        if latest:
            return latest
        raise FileNotFoundError(
            f"Aucun checkpoint .pt trouvé dans le dossier de run : {p}"
        )

    # 3) Nom sans extension (ex: ".../best" -> ".../best.pt").
    if p.suffix != ".pt":
        with_ext = p.with_suffix(".pt")
        if with_ext.is_file():
            return str(with_ext)

    raise FileNotFoundError(
        f"Checkpoint introuvable : {path!r}. Attendu : un fichier .pt "
        f"(ex: .../best.pt) ou un dossier de run contenant best.pt."
    )


def load_model(ckpt_path: str, device: str) -> tuple[UNet, str, str]:
    """Charge un checkpoint et renvoie ``(model, input_repr, resolved_path)``.

    ``ckpt_path`` peut être un fichier .pt, un nom sans extension, ou un dossier
    de run (cf. ``resolve_ckpt_path``). Reconstruit le ``UNet`` à l'identique de
    l'entraînement en lisant la config stockée dans le checkpoint
    (cf. ``scripts/listen_test.py``).

    Le modèle est renvoyé en mode ``eval()`` et placé sur ``device``.

    Raises
    ------
    FileNotFoundError
        Si aucun checkpoint exploitable n'est trouvé.
    RuntimeError
        Si l'architecture du checkpoint ne correspond pas (load_state_dict strict).
    """
    ckpt_path = resolve_ckpt_path(ckpt_path)
    ckpt_cfg = peek_checkpoint_config(ckpt_path)

    # La signature réelle de UNet est (base_channels, norm_groups). On lit ces
    # deux hyperparamètres depuis la config du checkpoint avec des fallbacks
    # robustes (les anciens checkpoints ne stockent que base_channels).
    base_channels = int(ckpt_cfg.get("base_channels") or 32)
    norm_groups = int(ckpt_cfg.get("norm_groups") or 8)

    model = UNet(base_channels=base_channels, norm_groups=norm_groups).to(device)

    load_checkpoint(ckpt_path, model, device=device)
    model.eval()

    # `input_repr` n'est pas stocké dans les checkpoints existants. Le pipeline
    # d'inférence de référence (scripts/listen_test.py) applique `log1p` en dur,
    # donc c'est le fallback correct — surtout PAS "linear" qui donnerait du bruit.
    input_repr = str(ckpt_cfg.get("input_repr") or "log1p")
    return model, input_repr, ckpt_path


@torch.no_grad()
def _denoise_wav(
    model: UNet,
    noisy_wav: np.ndarray,
    device: str,
    input_repr: str,
) -> np.ndarray:
    """Débruite un waveform (CLIP_SAMPLES samples) -> waveform prédit.

    Reprise fidèle de ``scripts/listen_test.denoise`` : gère la compression de
    l'entrée (``input_repr``) ainsi que le mode magnitude classique et le mode
    cIRM (masque complexe + ISTFT avec phase corrigée).
    """
    spec = audio.stft(noisy_wav)
    mag, phase = audio.magnitude_phase(spec)
    mag_t = torch.from_numpy(mag).float().unsqueeze(0).unsqueeze(0).to(device)  # (1,1,F,T)

    if input_repr == "log1p":
        model_in = torch.log1p(mag_t)
    elif input_repr == "pow03":
        model_in = torch.clamp(mag_t, min=0.0).pow(0.3)
    else:  # "linear" ou inconnu
        model_in = mag_t

    out = model(model_in, mag_t)

    if getattr(model, "use_cirm", False):
        # `out` est un tenseur complexe (B, 1, F, T) : masque complexe appliqué
        # au spectre complexe noisy, puis ISTFT avec phase corrigée par le réseau.
        phase_t = torch.from_numpy(phase).float().unsqueeze(0).unsqueeze(0).to(device)
        noisy_complex = torch.complex(mag_t * torch.cos(phase_t),
                                      mag_t * torch.sin(phase_t))
        pred_complex = (out * noisy_complex).squeeze(0).squeeze(0)
        window = torch.hann_window(cfg.N_FFT, device=device)
        pred = torch.istft(
            pred_complex,
            n_fft=cfg.N_FFT,
            hop_length=cfg.HOP_LENGTH,
            win_length=cfg.N_FFT,
            window=window,
            length=cfg.CLIP_SAMPLES,
        ).cpu().numpy()
        return pred.astype(np.float32)

    # Mode magnitude classique : masque réel × phase noisy.
    pred_mag = out.squeeze(0).squeeze(0).cpu().numpy()
    return audio.reconstruct(pred_mag, phase, length=cfg.CLIP_SAMPLES)


def denoise_bytes(
    audio_bytes: bytes,
    model: UNet,
    device: str,
    input_repr: str,
) -> bytes:
    """Pipeline complet de débruitage en mémoire : ``bytes`` -> ``bytes``.

    Accepte tout format décodable par librosa (WAV/FLAC nativement, MP3/OGG si
    ffmpeg est installé). Renvoie toujours un WAV mono 16 kHz PCM 16 bits.

    Raises
    ------
    ValueError
        Si l'audio ne peut pas être décodé, est vide / trop court, ou si la
        sortie du modèle contient des NaN/Inf.
    """
    # 1) Decode -> waveform mono 16 kHz
    try:
        wav, _ = librosa.load(io.BytesIO(audio_bytes), sr=cfg.SAMPLE_RATE, mono=True)
    except Exception as exc:  # noqa: BLE001 - on remappe toute erreur de decode
        raise ValueError(f"Cannot decode audio: {exc}") from exc

    wav = np.asarray(wav, dtype=np.float32)

    # 2) Validation entrée
    if wav.size == 0:
        raise ValueError("Audio is empty after decoding.")
    if wav.size < _MIN_SAMPLES:
        raise ValueError(
            f"Audio too short: {wav.size} samples (minimum ~{_MIN_SAMPLES})."
        )

    # 3) Durée fixe (crop/pad centré à 4 s)
    wav = audio.fix_length(wav, cfg.CLIP_SAMPLES, mode="center")

    # 4) Inférence
    pred_wav = _denoise_wav(model, wav, device, input_repr)

    # 5) Validation sortie
    if not np.isfinite(pred_wav).all():
        raise ValueError("Model produced NaN or Inf in output waveform.")

    # 6) Encode WAV PCM_16 en mémoire
    out_buf = io.BytesIO()
    sf.write(out_buf, pred_wav, cfg.SAMPLE_RATE, format="WAV", subtype="PCM_16")
    out_buf.seek(0)
    return out_buf.read()
