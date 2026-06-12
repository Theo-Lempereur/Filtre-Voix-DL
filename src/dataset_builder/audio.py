"""Core audio loading, conversion, normalization, chunking, and WAV export helpers.

This module is used by the clean and noise preprocessing scripts. It keeps the
low-level audio rules in one place: decoding, mono conversion, resampling,
loudness normalization, silence checks, fixed-duration chunking, and WAV export.

All public helpers operate on NumPy arrays and return ``float32`` audio whenever
they transform sample data. The pipeline assumes mono waveforms shaped as
``(samples,)`` after loading.
"""

from pathlib import Path
import subprocess
import shutil
import math

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


EPS = 1e-12


def db_to_amp(db: float) -> float:
    """Convert a decibel gain value to a linear amplitude multiplier.

    Args:
        db: Gain value in decibels. Positive values amplify the signal, negative
            values attenuate it.

    Returns:
        Linear amplitude multiplier equivalent to ``db``.
    """
    return float(10 ** (db / 20))


def amp_to_db(amp: float) -> float:
    """Convert a linear amplitude value to decibels with epsilon protection.

    Args:
        amp: Linear amplitude value. Values less than ``EPS`` are clamped to
            avoid ``log10(0)``.

    Returns:
        Amplitude expressed in decibels.
    """
    return float(20 * math.log10(max(amp, EPS)))


def rms_db(audio: np.ndarray) -> float:
    """Compute the RMS level of an audio array in decibels.

    Args:
        audio: Audio samples as a NumPy array. The function is intended for mono
            arrays but only requires numeric sample data.

    Returns:
        Root-mean-square level expressed in dBFS-like units.
    """
    rms = float(np.sqrt(np.mean(audio ** 2) + EPS))
    return amp_to_db(rms)


def peak(audio: np.ndarray) -> float:
    """Return the absolute peak amplitude of an audio array.

    Args:
        audio: Audio samples as a NumPy array.

    Returns:
        Maximum absolute sample value, or ``0.0`` for an empty array.
    """
    return float(np.max(np.abs(audio))) if len(audio) else 0.0


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert a mono or multichannel array to a mono signal.

    Args:
        audio: Either a mono array shaped ``(samples,)`` or a channel-last array
            shaped ``(samples, channels)``.

    Returns:
        Mono audio array. Existing mono input is returned unchanged; multichannel
        input is averaged across channels.

    Raises:
        ValueError: If ``audio`` has an unsupported number of dimensions.
    """
    if audio.ndim == 1:
        return audio

    if audio.ndim == 2:
        return np.mean(audio, axis=1)

    raise ValueError(f"Invalid audio dimensions: {audio.shape}")


def resample_audio(audio: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio with polyphase filtering and return float32 samples.

    Args:
        audio: Mono audio samples.
        original_sr: Current sample rate in Hz.
        target_sr: Desired sample rate in Hz.

    Returns:
        Audio resampled to ``target_sr`` as ``float32``. If the sample rate is
        already correct, only the dtype conversion is applied.
    """
    if original_sr == target_sr:
        return audio.astype(np.float32)

    gcd = math.gcd(original_sr, target_sr)
    up = target_sr // gcd
    down = original_sr // gcd

    audio = resample_poly(audio, up, down)
    return audio.astype(np.float32)


def load_with_soundfile(path: Path) -> tuple[np.ndarray, int]:
    """Read an audio file through soundfile without format conversion.

    Args:
        path: Audio file path supported by libsndfile.

    Returns:
        Tuple ``(audio, sample_rate)`` where ``audio`` is read as ``float32``.

    Raises:
        RuntimeError: Propagated from soundfile when the file cannot be decoded.
    """
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    return audio, sr


def load_with_ffmpeg(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    """Decode an audio file through FFmpeg into mono float32 PCM.

    This fallback is used when soundfile cannot read a compressed format. FFmpeg
    performs mono conversion and resampling in the subprocess.

    Args:
        path: Audio file path to decode.
        target_sr: Desired output sample rate in Hz.

    Returns:
        Tuple ``(audio, target_sr)`` with mono ``float32`` samples.

    Raises:
        RuntimeError: If FFmpeg is not installed or if FFmpeg fails to decode the
            input file.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "Unable to read this file with soundfile, and FFmpeg is not installed or not available in PATH."
        )

    command = [
        "ffmpeg",
        "-v", "error",
        "-i", str(path),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(target_sr),
        "pipe:1",
    ]

    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if process.returncode != 0:
        raise RuntimeError(process.stderr.decode("utf-8", errors="ignore"))

    audio = np.frombuffer(process.stdout, dtype=np.float32)
    return audio, target_sr


def load_audio(path: str | Path, target_sr: int, mono: bool = True) -> tuple[np.ndarray, int]:
    """Load, validate, optionally mono-convert, and resample an audio file.

    Args:
        path: Source audio file path.
        target_sr: Desired sample rate in Hz for downstream pipeline stages.
        mono: Whether to convert multichannel input to mono.

    Returns:
        Tuple ``(audio, target_sr)`` where ``audio`` is finite ``float32`` data.

    Raises:
        ValueError: If the decoded audio is empty or contains non-finite values.
        RuntimeError: If both soundfile and FFmpeg decoding fail.
    """
    path = Path(path)

    try:
        audio, sr = load_with_soundfile(path)
    except Exception:
        audio, sr = load_with_ffmpeg(path, target_sr)

    if audio.size == 0:
        raise ValueError("Audio file is empty.")

    if mono:
        audio = to_mono(audio)

    audio = np.asarray(audio, dtype=np.float32)

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio file contains NaN or infinite values.")

    audio = resample_audio(audio, sr, target_sr)

    if not np.all(np.isfinite(audio)):
        raise ValueError("Invalid audio after resampling.")

    return audio.astype(np.float32), target_sr


def limit_peak(audio: np.ndarray, peak_limit: float = 0.98) -> np.ndarray:
    """Scale audio down only when its absolute peak exceeds the configured limit.

    Args:
        audio: Audio samples to protect from clipping.
        peak_limit: Maximum allowed absolute amplitude.

    Returns:
        ``float32`` audio. The waveform is unchanged when already below the
        limit; otherwise it is scaled uniformly.
    """
    current_peak = peak(audio)

    if current_peak <= peak_limit:
        return audio.astype(np.float32)

    audio = audio / (current_peak + EPS) * peak_limit
    return audio.astype(np.float32)


def normalize_rms(
    audio: np.ndarray,
    target_rms_db: float = -25.0,
    max_gain_db: float = 20.0,
    peak_limit: float = 0.98,
) -> np.ndarray:
    """Normalize average loudness while respecting gain and peak constraints.

    Args:
        audio: Mono audio samples.
        target_rms_db: Desired RMS level after normalization.
        max_gain_db: Maximum positive gain allowed, preventing very quiet clips
            from being amplified excessively.
        peak_limit: Maximum absolute amplitude after normalization.

    Returns:
        Normalized ``float32`` audio. Near-silent input below ``-90`` dB RMS is
        returned unchanged to avoid boosting silence.
    """
    current_rms_db = rms_db(audio)

    if current_rms_db < -90:
        return audio.astype(np.float32)

    gain_db = target_rms_db - current_rms_db
    gain_db = min(gain_db, max_gain_db)

    gain = db_to_amp(gain_db)
    audio = audio * gain

    audio = limit_peak(audio, peak_limit=peak_limit)
    return audio.astype(np.float32)


def is_too_silent(
    audio: np.ndarray,
    silence_threshold_db: float = -50.0,
    min_non_silent_ratio: float = 0.05,
) -> bool:
    """Return whether an audio segment is too quiet or mostly silent.

    Args:
        audio: Mono audio samples.
        silence_threshold_db: Amplitude threshold used to classify samples as
            active or silent.
        min_non_silent_ratio: Minimum fraction of samples that must exceed the
            threshold.

    Returns:
        ``True`` when the segment should be rejected by preprocessing.
    """
    threshold = db_to_amp(silence_threshold_db)
    non_silent_ratio = float(np.mean(np.abs(audio) > threshold))

    if non_silent_ratio < min_non_silent_ratio:
        return True

    if rms_db(audio) < silence_threshold_db:
        return True

    return False


def split_into_chunks(
    audio: np.ndarray,
    chunk_samples: int,
    pad_short: bool = False,
) -> list[tuple[np.ndarray, int]]:
    """Split a mono signal into fixed-size chunks with optional zero padding.

    Args:
        audio: Mono source audio samples.
        chunk_samples: Exact number of samples required per output chunk.
        pad_short: Whether to keep a short file by zero-padding it to one chunk.

    Returns:
        List of ``(chunk, start_sample)`` tuples. ``start_sample`` is the offset
        in the original source file before any padding.
    """
    chunks = []

    audio_len = len(audio)

    if audio_len < chunk_samples:
        if not pad_short:
            return []

        padded = np.zeros(chunk_samples, dtype=np.float32)
        padded[:audio_len] = audio
        return [(padded, 0)]

    num_full_chunks = audio_len // chunk_samples

    for idx in range(num_full_chunks):
        start = idx * chunk_samples
        end = start + chunk_samples
        chunk = audio[start:end].astype(np.float32)
        chunks.append((chunk, start))

    return chunks


def save_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write a peak-limited mono WAV file as PCM 16-bit.

    Args:
        path: Destination path. Parent directories are created automatically.
        audio: Mono audio samples to save.
        sample_rate: Output sample rate in Hz.

    Returns:
        None.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    audio = limit_peak(audio, peak_limit=0.999)
    sf.write(path, audio, sample_rate, subtype="PCM_16")
