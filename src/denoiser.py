"""`Wallace` : la « boîte » d'inférence réutilisable du débruiteur de voix.

Objet unique et autonome qu'on construit avec un checkpoint et à qui on donne une
forme d'onde (de durée **quelconque**) pour récupérer la forme d'onde débruitée :

    from src.denoiser import Wallace

    box = Wallace("checkpoints/rp_csm_final/best_si_sdri.pt")
    clean = box.denoise(noisy_wav)            # numpy float32, n'importe quelle durée
    box.denoise_file("in.wav", "out.wav")     # confort fichier -> fichier
    wav_bytes = box.denoise_bytes(raw_bytes)  # pour le service HTTP

Elle gère **en interne**, et est la **source unique** du forward d'inférence :
  - le chargement du modèle avec **détection du mode** (`mask` ↔ `complex`) lu
    depuis la config du checkpoint (cf. `src/train.py` / `scripts/listen_test.py`) ;
  - le **contrat d'entrée** (mono, 16 kHz) — protège du décalage train/prod ;
  - le **traitement des sons > 4 s** par fenêtrage 4 s + overlap-add pondéré (WOLA)
    avec recouvrement 50 %, fenêtre de Hann de synthèse → coutures inaudibles.

⚠️ `_denoise_window` doit rester synchronisé avec `forward_fn_complex` de
`src/train.py` (mêmes formules de compression/décompression, `c=CSM_COMPRESS`).
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import torch

from . import audio, config as cfg
from .checkpoint import (
    find_best_checkpoint,
    find_latest_checkpoint,
    load_checkpoint,
    peek_checkpoint_config,
)
from .model import UNet


# ---------------------------------------------------------------------------
# Résolution souple d'un chemin de checkpoint
# ---------------------------------------------------------------------------

def resolve_ckpt_path(path: str | os.PathLike) -> str:
    """Résout un chemin souple vers un fichier ``.pt`` concret.

    Accepte : un fichier ``.pt`` (renvoyé tel quel), un **dossier de run**
    (``best.pt`` sinon le dernier checkpoint), ou un **nom sans extension**
    (``.../best`` -> ``.../best.pt``).

    Raises
    ------
    FileNotFoundError
        Si aucun checkpoint exploitable n'est trouvé.
    """
    p = Path(path)
    if p.is_file():
        return str(p)
    if p.is_dir():
        best = find_best_checkpoint(p)
        if best:
            return best
        latest = find_latest_checkpoint(p)
        if latest:
            return latest
        raise FileNotFoundError(f"Aucun checkpoint .pt trouvé dans le dossier : {p}")
    if p.suffix != ".pt":
        with_ext = p.with_suffix(".pt")
        if with_ext.is_file():
            return str(with_ext)
    raise FileNotFoundError(
        f"Checkpoint introuvable : {path!r}. Attendu : un fichier .pt "
        f"(ex: .../best_si_sdri.pt) ou un dossier de run contenant best.pt."
    )


# ---------------------------------------------------------------------------
# La boîte
# ---------------------------------------------------------------------------

class Wallace:
    """Débruiteur de voix prêt à l'emploi, durée d'entrée quelconque.

    Paramètres
    ----------
    ckpt_path  : chemin souple vers le checkpoint (cf. ``resolve_ckpt_path``).
    device     : "cuda" / "cpu" / torch.device. Auto si None.
    overlap    : recouvrement entre fenêtres de 4 s (0.5 = 50 %, défaut).
    batch_size : nb de fenêtres de 4 s traitées en un seul forward (mémoire
                 bornée : un son long est découpé en paquets de cette taille).
    optimize   : tente ``torch.jit.script`` + ``optimize_for_inference`` au
                 chargement (fusion + MKLDNN). **Désactivé par défaut** : mesuré
                 sans gain runtime sur ce modèle (compute-bound, une fenêtre sature
                 déjà les threads) alors que le scripting coûte au démarrage. Repli
                 silencieux sur l'eager si le scripting échoue.
    """

    def __init__(self, ckpt_path: str | os.PathLike, device=None,
                 overlap: float = 0.5, batch_size: int = 8,
                 optimize: bool = False):
        if not (0.0 <= overlap < 1.0):
            raise ValueError(f"overlap doit être dans [0, 1) — reçu {overlap}.")
        self.device = (torch.device(device) if device is not None
                       else torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        resolved = resolve_ckpt_path(ckpt_path)
        ckpt_cfg = peek_checkpoint_config(resolved)

        base_channels = int(ckpt_cfg.get("base_channels") or cfg.BASE_CHANNELS)
        norm_groups = int(ckpt_cfg.get("norm_groups") or cfg.NORM_GROUPS)
        # Détecte le mode d'entraînement pour reconstruire le modèle à l'identique.
        # Anciens checkpoints sans `output_mode` -> "mask" (rétrocompatible).
        self.output_mode = str(ckpt_cfg.get("output_mode") or "mask")
        self.c_comp = float(ckpt_cfg.get("csm_compress")
                            or getattr(cfg, "CSM_COMPRESS", 0.3))
        # `input_repr` (mode masque) : fallback "log1p" comme le pipeline de réf.
        self.input_repr = str(ckpt_cfg.get("input_repr") or "log1p")

        if self.output_mode == "complex":
            model = UNet(base_channels=base_channels, norm_groups=norm_groups,
                         in_channels=2, out_channels=2, head="linear")
        else:
            model = UNet(base_channels=base_channels, norm_groups=norm_groups)
        model = model.to(self.device)
        self.info = load_checkpoint(resolved, model, device=self.device)
        model.eval()

        # Optimisation inférence : script + fusion + MKLDNN. Mathématiquement
        # équivalent (pas de fold conv+GroupNorm, juste layout/fusions sûres) ;
        # on retombe sur l'eager si le scripting bute sur ce checkpoint.
        self.optimized = False
        if optimize:
            try:
                scripted = torch.jit.script(model)
                model = torch.jit.optimize_for_inference(scripted)
                self.optimized = True
            except Exception:
                model.eval()  # repli : eager reste correct

        self.model = model
        self.ckpt_path = resolved
        self.base_channels = base_channels
        self.norm_groups = norm_groups
        self.overlap = overlap
        self.batch_size = max(1, int(batch_size))
        # Métadonnées du contrat d'entrée (utiles pour un packaging futur).
        self.sample_rate = cfg.SAMPLE_RATE
        self.clip_samples = cfg.CLIP_SAMPLES

    # ------------------------------------------------------------------ #
    # Forward sur UNE fenêtre de 4 s — SOURCE UNIQUE du forward d'inférence
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _denoise_window(self, wav_4s: np.ndarray) -> np.ndarray:
        """Débruite un waveform de `CLIP_SAMPLES` échantillons -> même longueur.

        fp32 pur (pas d'autocast) pour le rendu le plus fidèle des poids appris.
        """
        if self.output_mode == "complex":
            return self._forward_complex(wav_4s)
        return self._forward_mask(wav_4s)

    def _forward_mask(self, noisy_wav: np.ndarray) -> np.ndarray:
        """Recette masque : `sigmoid × noisy_mag`, ISTFT avec la phase noisy."""
        spec = audio.stft(noisy_wav)
        mag, phase = audio.magnitude_phase(spec)
        mag_t = torch.from_numpy(mag).float().unsqueeze(0).unsqueeze(0).to(self.device)

        if self.input_repr == "log1p":
            model_in = torch.log1p(mag_t)
        elif self.input_repr == "pow03":
            model_in = torch.clamp(mag_t, min=0.0).pow(0.3)
        else:  # "linear" ou inconnu
            model_in = mag_t

        pred_mag_t = self.model(model_in, mag_t)
        pred_mag = pred_mag_t.squeeze(0).squeeze(0).cpu().numpy()
        return audio.reconstruct(pred_mag, phase, length=cfg.CLIP_SAMPLES)

    def _forward_complex(self, noisy_wav: np.ndarray) -> np.ndarray:
        """Complex spectral mapping sur UNE fenêtre (wrapper batch B=1)."""
        return self._forward_complex_batch(noisy_wav[None, :])[0]

    @torch.no_grad()
    def _forward_complex_batch(self, wav_batch: np.ndarray) -> np.ndarray:
        """Complex spectral mapping vectorisé sur un batch de fenêtres.

        Entrée `(B, CLIP_SAMPLES)` -> sortie `(B, CLIP_SAMPLES)`. STFT/forward/
        ISTFT portent nativement la dimension batch : le résultat par fenêtre est
        identique (à epsilon flottant près) au passage séquentiel. Reproduit
        `forward_fn_complex` de `src/train.py` à l'identique (`c=CSM_COMPRESS`).
        """
        eps = 1e-8
        c = self.c_comp
        wav = torch.from_numpy(
            np.ascontiguousarray(wav_batch, dtype=np.float32)).to(self.device)  # (B, T)
        win = torch.hann_window(cfg.WIN_LENGTH, device=self.device, dtype=torch.float32)

        ns = torch.stft(wav, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH,
                        win_length=cfg.WIN_LENGTH, window=win, center=True,
                        pad_mode="constant", return_complex=True)              # (B, F, T)
        n_mag = ns.abs()
        n_scale = (n_mag + eps).pow(c - 1.0)
        model_input = torch.stack([ns.real * n_scale, ns.imag * n_scale], dim=1)  # (B,2,F,T)

        pred_ri = self.model(model_input).float()
        pred_re, pred_im = pred_ri[:, 0], pred_ri[:, 1]                        # (B,F,T)

        pred_cmag = torch.sqrt(pred_re ** 2 + pred_im ** 2 + eps)
        up = pred_cmag.pow((1.0 - c) / c)
        pred_spec = torch.complex(pred_re * up, pred_im * up)

        pred_wav = torch.istft(pred_spec, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH,
                               win_length=cfg.WIN_LENGTH, window=win, center=True,
                               length=cfg.CLIP_SAMPLES, return_complex=False)  # (B, T)
        return pred_wav.cpu().numpy()

    @torch.no_grad()
    def _denoise_windows(self, windows: np.ndarray) -> np.ndarray:
        """Débruite un batch de fenêtres `(B, CLIP_SAMPLES)` -> `(B, CLIP_SAMPLES)`.

        Mode complex : vectorisé en un seul forward. Mode masque (moins critique,
        STFT librosa par fenêtre) : boucle.
        """
        if self.output_mode == "complex":
            return self._forward_complex_batch(windows)
        return np.stack([self._forward_mask(w) for w in windows])

    # ------------------------------------------------------------------ #
    # API principale : durée quelconque
    # ------------------------------------------------------------------ #
    def denoise(self, wav: np.ndarray, sr: int | None = None) -> np.ndarray:
        """Débruite un waveform de durée **quelconque** -> waveform 16 kHz mono.

        Applique le contrat d'entrée (mono + resample 16 kHz si `sr` fourni), puis
        traite par fenêtres de 4 s avec overlap-add 50 % au-delà de 4 s.
        """
        wav = np.asarray(wav, dtype=np.float32)
        # Contrat d'entrée : mono.
        if wav.ndim > 1:
            ch_axis = 0 if wav.shape[0] < wav.shape[-1] else wav.ndim - 1
            wav = wav.mean(axis=ch_axis).astype(np.float32)
        # Contrat d'entrée : 16 kHz.
        if sr is not None and sr != cfg.SAMPLE_RATE:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=cfg.SAMPLE_RATE)
            wav = np.asarray(wav, dtype=np.float32)

        n = wav.shape[-1]
        clip = cfg.CLIP_SAMPLES

        # Cas court : une seule fenêtre (= comportement historique, non-régression).
        if n <= clip:
            win4 = audio.fix_length(wav, clip, mode="center")
            out = self._denoise_window(win4)
            if n < clip:
                left = (clip - n) // 2          # fix_length(center) pad symétrique
                out = out[left:left + n]
            return np.asarray(out, dtype=np.float32)

        # Cas long : WOLA (weighted overlap-add).
        return self._overlap_add(wav)

    def _overlap_add(self, wav: np.ndarray) -> np.ndarray:
        """Fenêtrage 4 s + recombinaison overlap-add pondérée (Hann)."""
        clip = cfg.CLIP_SAMPLES
        stride = max(1, int(round(clip * (1.0 - self.overlap))))  # 50% -> 32000
        pad = clip // 2                                           # bords couverts à plein poids
        n = wav.shape[-1]

        wp = np.pad(wav, (pad, pad), mode="reflect")
        L = wp.shape[-1]
        synth = np.hanning(clip).astype(np.float64)               # fenêtre de synthèse

        out = np.zeros(L, dtype=np.float64)
        wsum = np.zeros(L, dtype=np.float64)

        starts = list(range(0, L - clip + 1, stride))
        if not starts or starts[-1] != L - clip:
            starts.append(L - clip)                               # garantit la couverture de la fin

        # Toutes les fenêtres, traitées par paquets de `batch_size` (mémoire bornée).
        windows = np.stack([wp[s:s + clip] for s in starts])      # (Nw, clip)
        preds = np.empty((len(starts), clip), dtype=np.float32)
        for i in range(0, len(starts), self.batch_size):
            preds[i:i + self.batch_size] = self._denoise_windows(
                windows[i:i + self.batch_size])

        for k, s in enumerate(starts):
            out[s:s + clip] += preds[k] * synth
            wsum[s:s + clip] += synth

        result = out / np.maximum(wsum, 1e-8)
        result = result[pad:pad + n]                              # retire le padding miroir
        return np.asarray(result, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Confort : fichier / bytes
    # ------------------------------------------------------------------ #
    def denoise_file(self, in_path: str | os.PathLike,
                     out_path: str | os.PathLike) -> None:
        """Débruite un fichier audio (durée quelconque) et écrit un WAV 16 kHz."""
        wav = audio.load_audio(str(in_path))     # mono 16 kHz float32 (resample inclus)
        pred = self.denoise(wav)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), pred, cfg.SAMPLE_RATE)

    def denoise_bytes(self, audio_bytes: bytes) -> bytes:
        """Pipeline en mémoire ``bytes`` -> ``bytes`` (WAV mono 16 kHz PCM 16).

        Accepte tout format décodable par librosa (WAV/FLAC natif, MP3/OGG si
        ffmpeg présent).

        Raises
        ------
        ValueError
            Si l'audio ne peut être décodé, est vide / trop court, ou si la sortie
            contient des NaN/Inf.
        """
        try:
            wav, _ = librosa.load(io.BytesIO(audio_bytes), sr=cfg.SAMPLE_RATE, mono=True)
        except Exception as exc:  # noqa: BLE001 - on remappe toute erreur de decode
            raise ValueError(f"Cannot decode audio: {exc}") from exc

        wav = np.asarray(wav, dtype=np.float32)
        if wav.size == 0:
            raise ValueError("Audio is empty after decoding.")
        if wav.size < cfg.N_FFT:
            raise ValueError(
                f"Audio too short: {wav.size} samples (minimum ~{cfg.N_FFT})."
            )

        pred = self.denoise(wav)
        if not np.isfinite(pred).all():
            raise ValueError("Model produced NaN or Inf in output waveform.")

        out_buf = io.BytesIO()
        sf.write(out_buf, pred, cfg.SAMPLE_RATE, format="WAV", subtype="PCM_16")
        out_buf.seek(0)
        return out_buf.read()
