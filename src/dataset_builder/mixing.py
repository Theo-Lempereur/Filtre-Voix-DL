from dataclasses import dataclass
import math
import random

import numpy as np


EPS = 1e-12


@dataclass
class MixConfig:
    sample_rate: int = 16000
    duration_sec: float = 3.0

    snr_min_db: float = -5.0
    snr_max_db: float = 20.0

    # RMS / silence
    min_clean_rms_db: float = -45.0
    min_noise_rms_db: float = -60.0

    # Sécurité audio
    peak_limit: float = 0.98
    avoid_clipping: bool = True

    # Si True, applique le même gain à noisy, clean et noise
    # quand le mix risque de clipper.
    apply_gain_to_target: bool = True


@dataclass
class MixResult:
    noisy: np.ndarray
    clean_target: np.ndarray
    noise_scaled: np.ndarray

    snr_db: float
    clean_rms_db: float
    noise_rms_db_before: float
    noise_rms_db_after: float
    noisy_rms_db: float

    peak_clean: float
    peak_noise: float
    peak_noisy: float
    global_gain_db: float


def db_to_amp(db: float) -> float:
    return float(10 ** (db / 20.0))


def amp_to_db(amp: float) -> float:
    amp = max(float(amp), EPS)
    return float(20.0 * math.log10(amp))


def ensure_float32(audio: np.ndarray, name: str = "audio") -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim != 1:
        raise ValueError(f"{name} must be mono 1D audio. Got shape: {audio.shape}")

    if audio.size == 0:
        raise ValueError(f"{name} is empty.")

    if not np.all(np.isfinite(audio)):
        raise ValueError(f"{name} contains NaN or Inf.")

    return audio


def rms(audio: np.ndarray) -> float:
    audio = ensure_float32(audio)
    return float(np.sqrt(np.mean(audio ** 2) + EPS))


def rms_db(audio: np.ndarray) -> float:
    return amp_to_db(rms(audio))


def peak(audio: np.ndarray) -> float:
    audio = ensure_float32(audio)
    return float(np.max(np.abs(audio)))


def validate_same_length(clean: np.ndarray, noise: np.ndarray) -> None:
    if len(clean) != len(noise):
        raise ValueError(
            f"Length mismatch: clean={len(clean)} samples, noise={len(noise)} samples."
        )


def random_snr(
    snr_min_db: float = -5.0,
    snr_max_db: float = 20.0,
    rng: random.Random | None = None,
) -> float:
    if rng is None:
        rng = random

    return float(rng.uniform(snr_min_db, snr_max_db))


def scale_noise_to_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """
    Scale le bruit pour atteindre un SNR donné.

    Formule :
    SNR dB = 20 * log10(clean_rms / noise_rms)

    Donc :
    target_noise_rms = clean_rms / 10^(SNR/20)
    """
    clean = ensure_float32(clean, "clean")
    noise = ensure_float32(noise, "noise")

    validate_same_length(clean, noise)

    clean_rms = rms(clean)
    noise_rms = rms(noise)

    if clean_rms <= EPS:
        raise ValueError("Clean RMS is too low or silent.")

    if noise_rms <= EPS:
        raise ValueError("Noise RMS is too low or silent.")

    target_noise_rms = clean_rms / db_to_amp(snr_db)
    gain = target_noise_rms / (noise_rms + EPS)

    noise_scaled = noise * gain
    return noise_scaled.astype(np.float32)


def compute_real_snr_db(clean: np.ndarray, noise_scaled: np.ndarray) -> float:
    clean_rms = rms(clean)
    noise_rms = rms(noise_scaled)

    return float(20.0 * math.log10((clean_rms + EPS) / (noise_rms + EPS)))


def apply_common_gain_if_needed(
    clean: np.ndarray,
    noise_scaled: np.ndarray,
    noisy: np.ndarray,
    peak_limit: float = 0.98,
    apply_gain_to_target: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Évite le clipping.

    Très important :
    Si on baisse le volume du noisy, il faut aussi baisser le clean target
    avec le même gain, sinon le target ne correspond plus parfaitement.
    """
    noisy_peak = peak(noisy)

    if noisy_peak <= peak_limit:
        return clean, noise_scaled, noisy, 0.0

    gain = peak_limit / (noisy_peak + EPS)
    gain_db = amp_to_db(gain)

    noisy = noisy * gain
    noise_scaled = noise_scaled * gain

    if apply_gain_to_target:
        clean = clean * gain

    return (
        clean.astype(np.float32),
        noise_scaled.astype(np.float32),
        noisy.astype(np.float32),
        gain_db,
    )


def mix_clean_noise(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    cfg: MixConfig | None = None,
) -> MixResult:
    """
    Mix principal pour speech enhancement.

    Entrée :
    - clean : voix clean mono float32
    - noise : bruit mono float32
    - snr_db : SNR voulu

    Sortie :
    - noisy : clean + bruit
    - clean_target : clean correspondant au noisy
    - noise_scaled : bruit réellement ajouté
    """
    if cfg is None:
        cfg = MixConfig()

    clean = ensure_float32(clean, "clean")
    noise = ensure_float32(noise, "noise")

    validate_same_length(clean, noise)

    expected_samples = int(cfg.sample_rate * cfg.duration_sec)

    if len(clean) != expected_samples:
        raise ValueError(
            f"Invalid clean length: {len(clean)} samples, expected {expected_samples}."
        )

    if len(noise) != expected_samples:
        raise ValueError(
            f"Invalid noise length: {len(noise)} samples, expected {expected_samples}."
        )

    clean_rms_db_value = rms_db(clean)
    noise_rms_db_before = rms_db(noise)

    if clean_rms_db_value < cfg.min_clean_rms_db:
        raise ValueError(
            f"Clean too silent: {clean_rms_db_value:.2f} dB RMS."
        )

    if noise_rms_db_before < cfg.min_noise_rms_db:
        raise ValueError(
            f"Noise too silent: {noise_rms_db_before:.2f} dB RMS."
        )

    noise_scaled = scale_noise_to_snr(
        clean=clean,
        noise=noise,
        snr_db=snr_db,
    )

    noisy = clean + noise_scaled

    global_gain_db = 0.0

    if cfg.avoid_clipping:
        clean, noise_scaled, noisy, global_gain_db = apply_common_gain_if_needed(
            clean=clean,
            noise_scaled=noise_scaled,
            noisy=noisy,
            peak_limit=cfg.peak_limit,
            apply_gain_to_target=cfg.apply_gain_to_target,
        )

    noisy = noisy.astype(np.float32)
    clean_target = clean.astype(np.float32)
    noise_scaled = noise_scaled.astype(np.float32)

    if not np.all(np.isfinite(noisy)):
        raise ValueError("Noisy contains NaN or Inf after mix.")

    if not np.all(np.isfinite(clean_target)):
        raise ValueError("Clean target contains NaN or Inf after mix.")

    real_snr = compute_real_snr_db(clean_target, noise_scaled)

    return MixResult(
        noisy=noisy,
        clean_target=clean_target,
        noise_scaled=noise_scaled,

        snr_db=real_snr,
        clean_rms_db=rms_db(clean_target),
        noise_rms_db_before=noise_rms_db_before,
        noise_rms_db_after=rms_db(noise_scaled),
        noisy_rms_db=rms_db(noisy),

        peak_clean=peak(clean_target),
        peak_noise=peak(noise_scaled),
        peak_noisy=peak(noisy),
        global_gain_db=global_gain_db,
    )


def mix_with_random_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    cfg: MixConfig | None = None,
    rng: random.Random | None = None,
) -> MixResult:
    if cfg is None:
        cfg = MixConfig()

    snr = random_snr(
        snr_min_db=cfg.snr_min_db,
        snr_max_db=cfg.snr_max_db,
        rng=rng,
    )

    return mix_clean_noise(
        clean=clean,
        noise=noise,
        snr_db=snr,
        cfg=cfg,
    )