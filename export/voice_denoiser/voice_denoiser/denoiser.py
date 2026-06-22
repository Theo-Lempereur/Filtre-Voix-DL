"""VoiceDenoiser — débruiteur de voix **autonome** (paquet ``voice_denoiser``).

Boîte importable **sans le repo d'entraînement** : charge un modèle TorchScript
(``model.ts`` — architecture + poids embarqués) et ses métadonnées
(``metadata.json``), puis débruite un audio de **durée quelconque** (fenêtrage
4 s + overlap-add 50 %).

    from voice_denoiser import VoiceDenoiser

    box = VoiceDenoiser()                      # artefacts trouvés dans le paquet
    clean = box.denoise(noisy_wav, sr=16000)   # numpy float32 -> numpy float32
    box.denoise_file("in.wav", "out.wav")      # fichier -> fichier
    wav_bytes = box.denoise_bytes(raw_bytes)   # bytes -> bytes (WAV 16 kHz)

Contrat d'entrée : le modèle a été entraîné en **mono 16 kHz**. ``denoise`` met
en mono et rééchantillonne automatiquement si on lui passe ``sr``.

⚠️ Mode **complex spectral mapping uniquement**. Le forward et l'overlap-add sont
une **copie volontaire** de ``src/denoiser.py`` (mode complex) du repo
d'entraînement — ils doivent rester synchronisés en cas d'évolution.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import torch

_PKG_DIR = Path(__file__).resolve().parent


class VoiceDenoiser:
    """Débruiteur prêt à l'emploi, durée d'entrée quelconque (mode complex).

    Paramètres
    ----------
    bundle_dir : dossier contenant ``model.ts`` + ``metadata.json``. Défaut : le
        dossier du paquet lui-même (les artefacts sont livrés avec le paquet).
    device     : "cuda" / "cpu" / torch.device. Auto si None.
    overlap    : recouvrement entre fenêtres de 4 s (défaut : valeur du metadata).
    batch_size : nb de fenêtres de 4 s traitées en un seul forward (mémoire
        bornée : un son long est découpé en paquets de cette taille).
    """

    def __init__(self, bundle_dir=None, device=None, overlap=None, batch_size=8):
        bundle = Path(bundle_dir) if bundle_dir is not None else _PKG_DIR
        ts_path = bundle / "model.ts"
        meta_path = bundle / "metadata.json"
        if not ts_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(
                f"Artefacts manquants dans {bundle} (model.ts / metadata.json). "
                f"Génère-les avec scripts/export_model.py du repo d'entraînement."
            )

        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        if self.meta.get("output_mode") != "complex":
            raise ValueError(
                "voice_denoiser ne supporte que output_mode='complex' "
                f"(metadata: {self.meta.get('output_mode')!r})."
            )

        self.device = (torch.device(device) if device is not None
                       else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # Lecture via buffer : le load C++ de torch ne résout pas les chemins
        # non-ASCII sous Windows ; Python si. Sans effet sur un chemin classique.
        loaded = torch.jit.load(io.BytesIO(ts_path.read_bytes()),
                                map_location=self.device)
        loaded.eval()
        # Optimisation inférence (fusion + MKLDNN sur CPU). Le .ts livré reste le
        # module « pur », portable ; l'optimisation est appliquée ici, sur la
        # machine cible. Repli silencieux si la version de torch ne la gère pas.
        try:
            self.model = torch.jit.optimize_for_inference(loaded)
        except Exception:
            self.model = loaded

        m = self.meta
        self.sample_rate = int(m["sample_rate"])
        self.n_fft = int(m["n_fft"])
        self.hop_length = int(m["hop_length"])
        self.win_length = int(m["win_length"])
        self.clip_samples = int(m["clip_samples"])
        self.c_comp = float(m["csm_compress"])
        self.overlap = float(overlap if overlap is not None
                             else m.get("overlap_default", 0.5))
        if not (0.0 <= self.overlap < 1.0):
            raise ValueError(f"overlap doit être dans [0, 1) — reçu {self.overlap}.")
        self.batch_size = max(1, int(batch_size))

    # ------------------------------------------------------------------ #
    # Forward complex sur UNE fenêtre de `clip_samples` échantillons.
    # Copie fidèle de src/denoiser._forward_complex (fp32).
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _denoise_window(self, wav_4s: np.ndarray) -> np.ndarray:
        """Forward complex sur UNE fenêtre (wrapper batch B=1)."""
        return self._denoise_windows(wav_4s[None, :])[0]

    @torch.no_grad()
    def _denoise_windows(self, wav_batch: np.ndarray) -> np.ndarray:
        """Forward complex vectorisé : `(B, clip_samples)` -> `(B, clip_samples)`.

        STFT/forward/ISTFT portent nativement la dimension batch ; le résultat par
        fenêtre est identique (à epsilon près) au passage séquentiel.
        """
        eps = 1e-8
        c = self.c_comp
        wav = torch.from_numpy(
            np.ascontiguousarray(wav_batch, dtype=np.float32)).to(self.device)  # (B, T)
        win = torch.hann_window(self.win_length, device=self.device, dtype=torch.float32)

        ns = torch.stft(wav, n_fft=self.n_fft, hop_length=self.hop_length,
                        win_length=self.win_length, window=win, center=True,
                        pad_mode="constant", return_complex=True)              # (B, F, T)
        n_mag = ns.abs()
        n_scale = (n_mag + eps).pow(c - 1.0)
        model_input = torch.stack([ns.real * n_scale, ns.imag * n_scale], dim=1)

        pred_ri = self.model(model_input).float()
        pred_re, pred_im = pred_ri[:, 0], pred_ri[:, 1]                        # (B,F,T)

        pred_cmag = torch.sqrt(pred_re ** 2 + pred_im ** 2 + eps)
        up = pred_cmag.pow((1.0 - c) / c)
        pred_spec = torch.complex(pred_re * up, pred_im * up)

        pred_wav = torch.istft(pred_spec, n_fft=self.n_fft, hop_length=self.hop_length,
                               win_length=self.win_length, window=win, center=True,
                               length=self.clip_samples, return_complex=False)  # (B, T)
        return pred_wav.cpu().numpy()

    # ------------------------------------------------------------------ #
    # API principale : durée quelconque
    # ------------------------------------------------------------------ #
    def denoise(self, wav: np.ndarray, sr: int | None = None) -> np.ndarray:
        """Débruite un waveform de durée quelconque -> waveform 16 kHz mono."""
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:                                   # -> mono
            ch_axis = 0 if wav.shape[0] < wav.shape[-1] else wav.ndim - 1
            wav = wav.mean(axis=ch_axis).astype(np.float32)
        if sr is not None and sr != self.sample_rate:      # -> 16 kHz
            import librosa
            wav = np.asarray(librosa.resample(wav, orig_sr=sr,
                                              target_sr=self.sample_rate), dtype=np.float32)

        n = wav.shape[-1]
        clip = self.clip_samples
        if n <= clip:                                      # une seule fenêtre
            win4 = _fix_center(wav, clip)
            out = self._denoise_window(win4)
            if n < clip:
                left = (clip - n) // 2
                out = out[left:left + n]
            return np.asarray(out, dtype=np.float32)
        return self._overlap_add(wav)                      # WOLA

    def _overlap_add(self, wav: np.ndarray) -> np.ndarray:
        clip = self.clip_samples
        stride = max(1, int(round(clip * (1.0 - self.overlap))))
        pad = clip // 2
        n = wav.shape[-1]

        wp = np.pad(wav, (pad, pad), mode="reflect")
        L = wp.shape[-1]
        synth = np.hanning(clip).astype(np.float64)

        out = np.zeros(L, dtype=np.float64)
        wsum = np.zeros(L, dtype=np.float64)

        starts = list(range(0, L - clip + 1, stride))
        if not starts or starts[-1] != L - clip:
            starts.append(L - clip)

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
        return np.asarray(result[pad:pad + n], dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Confort : fichier / bytes
    # ------------------------------------------------------------------ #
    def denoise_file(self, in_path, out_path) -> None:
        """Débruite un fichier audio (durée quelconque) -> WAV 16 kHz."""
        import soundfile as sf
        wav, sr = sf.read(str(in_path), dtype="float32", always_2d=False)
        pred = self.denoise(wav, sr=sr)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), pred, self.sample_rate)

    def denoise_bytes(self, audio_bytes: bytes) -> bytes:
        """Pipeline en mémoire ``bytes`` -> ``bytes`` (WAV mono 16 kHz PCM 16)."""
        import soundfile as sf
        try:
            wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        except Exception as exc_sf:
            # Formats compressés (MP3/OGG) : repli librosa (nécessite ffmpeg).
            try:
                import librosa
                wav, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=False)
            except Exception:
                raise ValueError(f"Cannot decode audio: {exc_sf}") from exc_sf

        wav = np.asarray(wav, dtype=np.float32)
        if wav.size == 0:
            raise ValueError("Audio is empty after decoding.")

        pred = self.denoise(wav, sr=sr)
        if not np.isfinite(pred).all():
            raise ValueError("Model produced NaN or Inf in output waveform.")

        out_buf = io.BytesIO()
        sf.write(out_buf, pred, self.sample_rate, format="WAV", subtype="PCM_16")
        out_buf.seek(0)
        return out_buf.read()


def _fix_center(wav: np.ndarray, n_samples: int) -> np.ndarray:
    """Crop centré / pad silence symétrique à `n_samples` (mirroir de audio.fix_length center)."""
    length = wav.shape[-1]
    if length == n_samples:
        return wav
    if length > n_samples:
        start = (length - n_samples) // 2
        return wav[..., start:start + n_samples]
    pad = n_samples - length
    left = pad // 2
    right = pad - left
    return np.pad(wav, (left, right), mode="constant")
