"""SNR-controlled clean/noise mixing for paired speech enhancement samples.

This module creates the final supervised pair used by the denoising model:
``noisy = clean + scaled_noise``. It enforces duration, sample-rate-derived
length, silence thresholds, clipping protection, and diagnostic metadata.

All functions expect mono ``float32`` arrays shaped as ``(samples,)`` once the
preprocessing stage has prepared clean and noise chunks.
"""

from dataclasses import dataclass
import math
import random

import numpy as np


EPS = 1e-12


@dataclass
class MixConfig:
    """Configuration controlling SNR mixing, silence filtering, and clipping safety.

    Attributes:
        sample_rate: Expected audio sample rate in Hz.
        duration_sec: Expected duration of each generated sample.
        snr_min_db: Lower bound used when drawing random SNR values.
        snr_max_db: Upper bound used when drawing random SNR values.
        min_clean_rms_db: Minimum accepted clean speech RMS level.
        min_noise_rms_db: Minimum accepted raw noise RMS level.
        peak_limit: Maximum absolute amplitude allowed after mixing.
        avoid_clipping: Whether to attenuate mixtures that exceed ``peak_limit``.
        apply_gain_to_target: Whether the same anti-clipping gain should also be
            applied to the clean target, keeping target and noisy aligned.
    """

    sample_rate: int = 16000
    duration_sec: float = 3.0

    snr_min_db: float = -5.0
    snr_max_db: float = 20.0

    # RMS / silence
    min_clean_rms_db: float = -45.0
    min_noise_rms_db: float = -60.0

    # Audio safety
    peak_limit: float = 0.98
    avoid_clipping: bool = True

    # If True, apply the same gain to noisy, clean, and noise when the mix may clip.
    apply_gain_to_target: bool = True


@dataclass
class MixResult:
    """Mixed audio arrays and diagnostic metrics for one generated sample.

    Attributes:
        noisy: Final model input, containing clean speech plus scaled noise.
        clean_target: Final model target aligned with ``noisy``.
        noise_scaled: Exact noise signal added to the clean speech.
        snr_db: Actual measured SNR after scaling and optional gain.
        clean_rms_db: RMS level of the clean target.
        noise_rms_db_before: RMS level of the raw selected noise chunk.
        noise_rms_db_after: RMS level of the scaled noise.
        noisy_rms_db: RMS level of the final noisy signal.
        peak_clean: Absolute peak of the clean target.
        peak_noise: Absolute peak of the scaled noise.
        peak_noisy: Absolute peak of the noisy signal.
        global_gain_db: Anti-clipping gain applied to the mixture, in dB.
    """

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
    """Convert a decibel gain value to a linear amplitude multiplier.

    Args:
        db: Gain value in decibels.

    Returns:
        Linear amplitude multiplier.
    """
    return float(10 ** (db / 20.0))


def amp_to_db(amp: float) -> float:
    """Convert a linear amplitude value to decibels with epsilon protection.

    Args:
        amp: Linear amplitude value.

    Returns:
        Decibel value after clamping near-zero input.
    """
    amp = max(float(amp), EPS)
    return float(20.0 * math.log10(amp))


def ensure_float32(audio: np.ndarray, name: str = "audio") -> np.ndarray:
    """Validate mono finite audio and return it as a float32 NumPy array.

    Args:
        audio: Input audio-like array.
        name: Human-readable signal name used in error messages.

    Returns:
        Mono finite ``float32`` array.

    Raises:
        ValueError: If audio is empty, non-mono, or contains NaN/Inf values.
    """
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim != 1:
        raise ValueError(f"{name} must be mono 1D audio. Got shape: {audio.shape}")

    if audio.size == 0:
        raise ValueError(f"{name} is empty.")

    if not np.all(np.isfinite(audio)):
        raise ValueError(f"{name} contains NaN or Inf.")

    return audio


def rms(audio: np.ndarray) -> float:
    """Compute the linear RMS level of a mono audio signal.

    Args:
        audio: Mono audio samples.

    Returns:
        Linear RMS value.
    """
    audio = ensure_float32(audio)
    return float(np.sqrt(np.mean(audio ** 2) + EPS))


def rms_db(audio: np.ndarray) -> float:
    """Compute RMS level in decibels.

    Args:
        audio: Mono audio samples.

    Returns:
        RMS level expressed in decibels.
    """
    return amp_to_db(rms(audio))


def peak(audio: np.ndarray) -> float:
    """Return the absolute peak amplitude after validating the signal.

    Args:
        audio: Mono audio samples.

    Returns:
        Maximum absolute sample amplitude.
    """
    audio = ensure_float32(audio)
    return float(np.max(np.abs(audio)))


def validate_same_length(clean: np.ndarray, noise: np.ndarray) -> None:
    """Raise an error when clean and noise arrays cannot be mixed sample-aligned.

    Args:
        clean: Clean speech samples.
        noise: Noise samples.

    Returns:
        None.

    Raises:
        ValueError: If both arrays do not have exactly the same length.
    """
    if len(clean) != len(noise):
        raise ValueError(
            f"Length mismatch: clean={len(clean)} samples, noise={len(noise)} samples."
        )


def random_snr(
    snr_min_db: float = -5.0,
    snr_max_db: float = 20.0,
    rng: random.Random | None = None,
) -> float:
    """Draw one SNR value from the configured range.

    Args:
        snr_min_db: Lower SNR bound.
        snr_max_db: Upper SNR bound.
        rng: Optional random generator for deterministic runs.

    Returns:
        Random SNR value in decibels.
    """
    if rng is None:
        rng = random

    return float(rng.uniform(snr_min_db, snr_max_db))


def scale_noise_to_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """
    Scale noise to reach a target SNR.

    Formula:
    SNR dB = 20 * log10(clean_rms / noise_rms)

    Therefore:
    target_noise_rms = clean_rms / 10^(SNR/20)

    Args:
        clean: Mono clean speech samples.
        noise: Mono noise samples with the same length as ``clean``.
        snr_db: Target signal-to-noise ratio in decibels.

    Returns:
        Noise signal scaled so that ``clean`` and the result approximately match
        the requested SNR.

    Raises:
        ValueError: If inputs are invalid, not aligned, or too silent to scale.
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
    """Compute the actual SNR after scaling and optional global gain.

    Args:
        clean: Clean target signal.
        noise_scaled: Noise signal added to the clean target.

    Returns:
        Measured SNR in decibels.
    """
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
    Avoid clipping.

    If noisy is attenuated, the clean target must receive the same gain so the target
    still matches the mixture.

    Args:
        clean: Clean target signal.
        noise_scaled: Already SNR-scaled noise signal.
        noisy: Sum of clean and scaled noise.
        peak_limit: Maximum accepted absolute amplitude.
        apply_gain_to_target: Whether to apply the same attenuation to ``clean``.

    Returns:
        Tuple ``(clean, noise_scaled, noisy, gain_db)`` after optional attenuation.
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
    Main speech enhancement mix function.

    Args:
        clean: Mono ``float32`` clean speech chunk.
        noise: Mono ``float32`` noise chunk with the same length as ``clean``.
        snr_db: Target SNR used to scale the noise.
        cfg: Optional mixing configuration. Defaults to ``MixConfig()``.

    Returns:
        ``MixResult`` containing output arrays and diagnostic levels.

    Raises:
        ValueError: If lengths, duration, silence thresholds, or final numeric
            checks fail.
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
    """Mix clean and noise audio after drawing a random target SNR.

    Args:
        clean: Mono clean speech chunk.
        noise: Mono noise chunk.
        cfg: Optional mixing configuration.
        rng: Optional random generator for deterministic SNR draws.

    Returns:
        ``MixResult`` from ``mix_clean_noise``.
    """
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
