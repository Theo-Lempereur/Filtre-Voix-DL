"""Validation helpers for loaded audio and prepared fixed-length chunks.

The preprocessing scripts use these functions to collect structured error codes
instead of raising immediately. That makes it possible to continue processing a
batch while still writing actionable diagnostics to CSV.
"""

import numpy as np

from .audio import peak, rms_db, is_too_silent


def validate_loaded_audio(audio: np.ndarray, sample_rate: int, expected_sr: int) -> list[str]:
    """Return validation error codes for freshly loaded mono audio.

    Args:
        audio: Loaded audio array after decoding and optional mono conversion.
        sample_rate: Actual sample rate returned by the loader.
        expected_sr: Required sample rate from configuration.

    Returns:
        List of machine-readable error codes. An empty list means the audio
        passed validation.
    """
    errors = []

    if audio is None:
        errors.append("audio_none")
        return errors

    if len(audio) == 0:
        errors.append("audio_empty")

    if sample_rate != expected_sr:
        errors.append(f"invalid_sample_rate_{sample_rate}")

    if audio.ndim != 1:
        errors.append(f"audio_not_mono_shape_{audio.shape}")

    if not np.all(np.isfinite(audio)):
        errors.append("audio_contains_nan_or_inf")

    return errors


def validate_chunk(
    chunk: np.ndarray,
    expected_samples: int,
    silence_threshold_db: float,
    min_non_silent_ratio: float,
    peak_limit: float,
) -> list[str]:
    """Return validation error codes for one prepared dataset chunk.

    Args:
        chunk: Mono fixed-length audio candidate.
        expected_samples: Exact number of samples required.
        silence_threshold_db: Silence threshold used by activity detection.
        min_non_silent_ratio: Minimum active-sample ratio required.
        peak_limit: Configured peak limit used to detect overly loud chunks.

    Returns:
        List of machine-readable error codes. An empty list means the chunk is
        valid for dataset generation.
    """
    errors = []

    if len(chunk) != expected_samples:
        errors.append(f"invalid_chunk_length_{len(chunk)}")

    if not np.all(np.isfinite(chunk)):
        errors.append("chunk_contains_nan_or_inf")

    if is_too_silent(
        chunk,
        silence_threshold_db=silence_threshold_db,
        min_non_silent_ratio=min_non_silent_ratio,
    ):
        errors.append("chunk_too_silent")

    if peak(chunk) > 1.0:
        errors.append("chunk_clipping")

    if peak(chunk) > peak_limit + 0.05:
        errors.append("chunk_peak_too_high")

    if rms_db(chunk) < -80:
        errors.append("chunk_rms_too_low")

    return errors
