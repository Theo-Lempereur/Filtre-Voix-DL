"""Audio augmentation primitives used to simulate microphones, codecs, and calls.

The generator uses these effects to make synthetic noisy examples closer to
real-world capture conditions: phone bandwidth, compression, cheap microphone
saturation, codec losses, short dropouts, quantization, gain shifts, and light
reverb.

Every transform accepts mono ``float32`` audio shaped as ``(samples,)`` and
returns ``float32`` audio with the same length unless explicitly documented
otherwise. Randomized transforms receive a Python ``random.Random`` instance so
dataset generation can remain deterministic when configured.
"""

from dataclasses import dataclass, field
from pathlib import Path
import math
import random
import shutil
import subprocess
import tempfile

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt


EPS = 1e-12


@dataclass
class AugmentConfig:
    """Configuration for probabilistic audio degradations used during generation.

    Attributes:
        sample_rate: Expected sample rate in Hz.
        duration_sec: Expected duration of each augmented clip.
        enabled: Master switch for the augmentation pipeline.
        p_*: Independent application probability for each effect family.
        *_min / *_max: Random sampling ranges used by the corresponding effect.
        codec_choices: FFmpeg codecs allowed for roundtrip simulation.
        peak_limit: Maximum absolute amplitude after each safety limiting step.
    """

    sample_rate: int = 16000
    duration_sec: float = 3.0

    enabled: bool = True

    # Application probabilities
    p_gain: float = 0.35
    p_eq: float = 0.25
    p_compression: float = 0.20
    p_saturation: float = 0.12
    p_clipping: float = 0.04
    p_phone_filter: float = 0.08
    p_reverb: float = 0.12
    p_codec: float = 0.06
    p_dropout: float = 0.03
    p_quantization: float = 0.05

    # Ranges gain
    gain_min_db: float = -6.0
    gain_max_db: float = 6.0

    # EQ
    eq_gain_min_db: float = -6.0
    eq_gain_max_db: float = 6.0

    # Phone / VoIP band limits
    phone_highpass_min_hz: float = 250.0
    phone_highpass_max_hz: float = 350.0
    phone_lowpass_min_hz: float = 3000.0
    phone_lowpass_max_hz: float = 3800.0

    # Compression
    compressor_threshold_min_db: float = -30.0
    compressor_threshold_max_db: float = -12.0
    compressor_ratio_min: float = 1.5
    compressor_ratio_max: float = 4.0

    # Saturation
    saturation_drive_min_db: float = 2.0
    saturation_drive_max_db: float = 10.0

    # Clipping
    clipping_threshold_min: float = 0.55
    clipping_threshold_max: float = 0.95

    # Lightweight reverb
    reverb_wet_min: float = 0.03
    reverb_wet_max: float = 0.18
    reverb_decay_min_ms: float = 80.0
    reverb_decay_max_ms: float = 500.0
    reverb_reflections_min: int = 3
    reverb_reflections_max: int = 10

    # Codec
    codec_choices: tuple[str, ...] = ("opus", "mp3")
    opus_bitrate_min_kbps: int = 8
    opus_bitrate_max_kbps: int = 32
    mp3_bitrate_min_kbps: int = 32
    mp3_bitrate_max_kbps: int = 96

    # Dropout
    dropout_count_min: int = 1
    dropout_count_max: int = 3
    dropout_ms_min: float = 10.0
    dropout_ms_max: float = 80.0

    # Quantization
    quantization_bits_min: int = 8
    quantization_bits_max: int = 12

    # Audio safety
    peak_limit: float = 0.98


@dataclass
class AugmentResult:
    """Augmented audio plus a compact trace of applied effects and parameters.

    Attributes:
        audio: Final augmented mono signal.
        applied: Ordered list of augmentation names applied to the signal.
        metadata: Effect parameters sampled during augmentation, suitable for
            storage in generation metadata.
    """

    audio: np.ndarray
    applied: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


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
    return float(20.0 * math.log10(max(float(amp), EPS)))


def ensure_float32(audio: np.ndarray, name: str = "audio") -> np.ndarray:
    """Validate mono finite audio and return it as a float32 NumPy array.

    Args:
        audio: Input audio-like array.
        name: Signal name used in validation error messages.

    Returns:
        Mono finite ``float32`` array.

    Raises:
        ValueError: If audio is empty, non-mono, or contains NaN/Inf values.
    """
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim != 1:
        raise ValueError(f"{name} must be mono 1D audio. Got shape {audio.shape}")

    if audio.size == 0:
        raise ValueError(f"{name} is empty.")

    if not np.all(np.isfinite(audio)):
        raise ValueError(f"{name} contains NaN or Inf.")

    return audio


def peak(audio: np.ndarray) -> float:
    """Return the absolute peak amplitude after validating the signal.

    Args:
        audio: Mono audio samples.

    Returns:
        Maximum absolute sample amplitude.
    """
    audio = ensure_float32(audio)
    return float(np.max(np.abs(audio)))


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


def limit_peak(audio: np.ndarray, peak_limit: float = 0.98) -> np.ndarray:
    """Scale audio down only when its absolute peak exceeds the configured limit.

    Args:
        audio: Mono audio samples.
        peak_limit: Maximum allowed absolute amplitude.

    Returns:
        Peak-limited ``float32`` audio.
    """
    audio = ensure_float32(audio)

    current_peak = peak(audio)

    if current_peak <= peak_limit:
        return audio.astype(np.float32)

    return (audio / (current_peak + EPS) * peak_limit).astype(np.float32)


def chance(rng: random.Random, probability: float) -> bool:
    """Return whether a random draw falls below a probability threshold.

    Args:
        rng: Python random generator.
        probability: Probability between 0 and 1.

    Returns:
        ``True`` when the sampled value activates an effect.
    """
    return rng.random() < probability


def rand_uniform(rng: random.Random, min_value: float, max_value: float) -> float:
    """Draw a floating-point value from an inclusive configuration range.

    Args:
        rng: Python random generator.
        min_value: Lower bound.
        max_value: Upper bound.

    Returns:
        Sampled float value.
    """
    return float(rng.uniform(min_value, max_value))


def apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    """Apply a constant gain in decibels to the whole signal.

    Args:
        audio: Mono audio samples.
        gain_db: Gain amount in decibels.

    Returns:
        Gain-adjusted ``float32`` audio.
    """
    audio = ensure_float32(audio)
    return (audio * db_to_amp(gain_db)).astype(np.float32)


def butter_filter(
    audio: np.ndarray,
    sample_rate: int,
    cutoff,
    filter_type: str,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth SOS filter and return float32 audio.

    Args:
        audio: Mono audio samples.
        sample_rate: Sample rate in Hz.
        cutoff: Cutoff frequency or frequency pair accepted by scipy.
        filter_type: Butterworth filter type, such as ``"lowpass"``.
        order: Filter order.

    Returns:
        Filtered ``float32`` audio.
    """
    audio = ensure_float32(audio)

    sos = butter(
        N=order,
        Wn=cutoff,
        btype=filter_type,
        fs=sample_rate,
        output="sos",
    )

    filtered = sosfilt(sos, audio)
    return filtered.astype(np.float32)


def lowpass(audio: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    """Apply a low-pass filter with cutoff clamped to the valid Nyquist range.

    Args:
        audio: Mono audio samples.
        sample_rate: Sample rate in Hz.
        cutoff_hz: Desired cutoff frequency.

    Returns:
        Low-pass filtered audio.
    """
    nyquist = sample_rate / 2.0
    cutoff_hz = min(float(cutoff_hz), nyquist - 100.0)
    cutoff_hz = max(cutoff_hz, 100.0)

    return butter_filter(
        audio,
        sample_rate=sample_rate,
        cutoff=cutoff_hz,
        filter_type="lowpass",
        order=4,
    )


def highpass(audio: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    """Apply a high-pass filter with a safe minimum cutoff.

    Args:
        audio: Mono audio samples.
        sample_rate: Sample rate in Hz.
        cutoff_hz: Desired cutoff frequency.

    Returns:
        High-pass filtered audio.
    """
    cutoff_hz = max(float(cutoff_hz), 20.0)

    return butter_filter(
        audio,
        sample_rate=sample_rate,
        cutoff=cutoff_hz,
        filter_type="highpass",
        order=4,
    )


def bandpass(audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> np.ndarray:
    """Apply a band-pass filter after validating and clamping the frequency range.

    Args:
        audio: Mono audio samples.
        sample_rate: Sample rate in Hz.
        low_hz: Lower cutoff frequency.
        high_hz: Upper cutoff frequency.

    Returns:
        Band-pass filtered audio.

    Raises:
        ValueError: If the low cutoff is greater than or equal to the high cutoff.
    """
    nyquist = sample_rate / 2.0

    low_hz = max(float(low_hz), 20.0)
    high_hz = min(float(high_hz), nyquist - 100.0)

    if low_hz >= high_hz:
        raise ValueError(f"Invalid bandpass range: {low_hz} - {high_hz}")

    return butter_filter(
        audio,
        sample_rate=sample_rate,
        cutoff=[low_hz, high_hz],
        filter_type="bandpass",
        order=4,
    )


def phone_filter(
    audio: np.ndarray,
    sample_rate: int,
    highpass_hz: float = 300.0,
    lowpass_hz: float = 3400.0,
) -> np.ndarray:
    """
    Simulate a classic phone or voice-call band limit.

    The filter removes low-end rumble and high-end detail.

    Args:
        audio: Mono audio samples.
        sample_rate: Sample rate in Hz.
        highpass_hz: Lower cutoff of the simulated phone band.
        lowpass_hz: Upper cutoff of the simulated phone band.

    Returns:
        Band-limited and peak-protected audio.
    """
    audio = bandpass(
        audio,
        sample_rate=sample_rate,
        low_hz=highpass_hz,
        high_hz=lowpass_hz,
    )

    return limit_peak(audio, peak_limit=0.98)


def three_band_eq(
    audio: np.ndarray,
    sample_rate: int,
    low_gain_db: float,
    mid_gain_db: float,
    high_gain_db: float,
) -> np.ndarray:
    """
    Fast three-band EQ.

    - low : < 250 Hz
    - mid : 250 Hz to 4000 Hz
    - high : > 4000 Hz

    Args:
        audio: Mono audio samples.
        sample_rate: Sample rate in Hz.
        low_gain_db: Gain applied to the low band.
        mid_gain_db: Gain applied to the mid band.
        high_gain_db: Gain applied to the high band.

    Returns:
        Equalized and peak-protected audio.
    """
    audio = ensure_float32(audio)

    low = butter_filter(
        audio,
        sample_rate=sample_rate,
        cutoff=250.0,
        filter_type="lowpass",
        order=2,
    )

    mid = butter_filter(
        audio,
        sample_rate=sample_rate,
        cutoff=[250.0, 4000.0],
        filter_type="bandpass",
        order=2,
    )

    high = butter_filter(
        audio,
        sample_rate=sample_rate,
        cutoff=4000.0,
        filter_type="highpass",
        order=2,
    )

    out = (
        low * db_to_amp(low_gain_db)
        + mid * db_to_amp(mid_gain_db)
        + high * db_to_amp(high_gain_db)
    )

    return limit_peak(out.astype(np.float32), peak_limit=0.98)


def soft_saturation(audio: np.ndarray, drive_db: float = 6.0) -> np.ndarray:
    """
    Apply soft saturation similar to a cheap mic or preamp.

    Args:
        audio: Mono audio samples.
        drive_db: Input drive before the tanh waveshaper.

    Returns:
        Saturated and peak-protected audio.
    """
    audio = ensure_float32(audio)

    drive = db_to_amp(drive_db)
    driven = audio * drive

    saturated = np.tanh(driven) / np.tanh(drive)
    return limit_peak(saturated.astype(np.float32), peak_limit=0.98)


def hard_clipping(audio: np.ndarray, threshold: float = 0.8) -> np.ndarray:
    """
    Apply hard clipping.

    Use this with a low probability.

    Args:
        audio: Mono audio samples.
        threshold: Absolute clipping threshold before rescaling.

    Returns:
        Clipped ``float32`` audio.
    """
    audio = ensure_float32(audio)

    threshold = float(np.clip(threshold, 0.1, 1.0))
    clipped = np.clip(audio, -threshold, threshold)
    clipped = clipped / threshold * min(threshold, 0.98)

    return clipped.astype(np.float32)


def static_compressor(
    audio: np.ndarray,
    threshold_db: float = -18.0,
    ratio: float = 3.0,
    makeup_gain_db: float = 0.0,
) -> np.ndarray:
    """
    Fast static compressor.

    This is not a full studio compressor, but it is useful to simulate:
    - Discord
    - phone calls
    - micro gaming
    - webcam

    Args:
        audio: Mono audio samples.
        threshold_db: Level above which gain reduction is applied.
        ratio: Compression ratio used above the threshold.
        makeup_gain_db: Optional gain added after compression.

    Returns:
        Compressed and peak-protected audio.
    """
    audio = ensure_float32(audio)

    abs_audio = np.abs(audio) + EPS
    level_db = 20.0 * np.log10(abs_audio)

    over_db = level_db - threshold_db

    gain_reduction_db = np.where(
        over_db > 0.0,
        threshold_db + over_db / ratio - level_db,
        0.0,
    )

    gain_db = gain_reduction_db + makeup_gain_db
    gain = np.power(10.0, gain_db / 20.0)

    compressed = audio * gain
    return limit_peak(compressed.astype(np.float32), peak_limit=0.98)


def light_reverb(
    audio: np.ndarray,
    sample_rate: int,
    rng: random.Random,
    wet: float = 0.08,
    decay_ms: float = 250.0,
    num_reflections: int = 6,
) -> np.ndarray:
    """
    Lightweight CPU-friendly reverb.

    This avoids expensive convolution so the pipeline stays usable on small machines.

    Args:
        audio: Mono audio samples.
        sample_rate: Sample rate in Hz.
        rng: Python random generator used for reflection delays and gains.
        wet: Wet signal amount mixed back into the dry signal.
        decay_ms: Approximate decay window in milliseconds.
        num_reflections: Number of synthetic reflections to add.

    Returns:
        Reverberated and peak-protected audio.
    """
    audio = ensure_float32(audio)

    wet = float(np.clip(wet, 0.0, 0.5))
    decay_ms = max(float(decay_ms), 20.0)

    reverbed = audio.copy()
    total_len = len(audio)

    for _ in range(num_reflections):
        delay_ms = rng.uniform(15.0, min(decay_ms, 140.0))
        delay_samples = int(sample_rate * delay_ms / 1000.0)

        if delay_samples <= 0 or delay_samples >= total_len:
            continue

        decay_factor = math.exp(-delay_ms / decay_ms)
        reflection_gain = rng.uniform(0.15, 0.65) * decay_factor

        reverbed[delay_samples:] += audio[:-delay_samples] * reflection_gain

    out = (1.0 - wet) * audio + wet * reverbed
    return limit_peak(out.astype(np.float32), peak_limit=0.98)


def quantize_audio(audio: np.ndarray, bits: int = 10) -> np.ndarray:
    """
    Simulate low-bit quantization.

    Useful for webcam, low-quality microphone, or cheap capture artifacts.

    Args:
        audio: Mono audio samples.
        bits: Number of quantization bits.

    Returns:
        Quantized ``float32`` audio with the same length as input.
    """
    audio = ensure_float32(audio)

    bits = int(np.clip(bits, 4, 16))
    levels = 2 ** bits

    clipped = np.clip(audio, -1.0, 1.0)
    quantized = np.round((clipped + 1.0) * (levels - 1) / 2.0)
    restored = (quantized * 2.0 / (levels - 1)) - 1.0

    return restored.astype(np.float32)


def random_dropouts(
    audio: np.ndarray,
    sample_rate: int,
    rng: random.Random,
    min_count: int = 1,
    max_count: int = 3,
    min_ms: float = 10.0,
    max_ms: float = 80.0,
) -> np.ndarray:
    """
    Simulate short VoIP-like network dropouts.

    Use sparingly.

    Args:
        audio: Mono audio samples.
        sample_rate: Sample rate in Hz.
        rng: Python random generator used for dropout positions and lengths.
        min_count: Minimum number of dropout regions.
        max_count: Maximum number of dropout regions.
        min_ms: Minimum dropout duration in milliseconds.
        max_ms: Maximum dropout duration in milliseconds.

    Returns:
        Audio with short faded silent gaps.
    """
    audio = ensure_float32(audio)
    out = audio.copy()

    count = rng.randint(min_count, max_count)

    for _ in range(count):
        dropout_ms = rng.uniform(min_ms, max_ms)
        dropout_samples = int(sample_rate * dropout_ms / 1000.0)

        if dropout_samples <= 0 or dropout_samples >= len(out):
            continue

        start = rng.randint(0, len(out) - dropout_samples)
        end = start + dropout_samples

        fade_len = min(int(sample_rate * 0.005), dropout_samples // 2)

        if fade_len > 0:
            fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)

            out[start:start + fade_len] *= fade_out
            out[end - fade_len:end] *= fade_in
            out[start + fade_len:end - fade_len] = 0.0
        else:
            out[start:end] = 0.0

    return out.astype(np.float32)


def ffmpeg_codec_simulation(
    audio: np.ndarray,
    sample_rate: int,
    codec: str = "opus",
    bitrate_kbps: int = 24,
) -> np.ndarray:
    """
    Simulate a real codec roundtrip through FFmpeg.

    Requires FFmpeg. If FFmpeg is not available, this function raises an error.

    Args:
        audio: Mono audio samples.
        sample_rate: Sample rate in Hz.
        codec: Codec name. Supported values are ``"opus"`` and ``"mp3"``.
        bitrate_kbps: Target bitrate for the encoded temporary file.

    Returns:
        Decoded audio after the lossy codec roundtrip, trimmed or padded to the
        original length.

    Raises:
        RuntimeError: If FFmpeg is missing, encoding fails, decoding fails, or
            the decoded sample rate is invalid.
        ValueError: If ``codec`` is unsupported.
    """
    audio = ensure_float32(audio)

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg not found. Codec augmentation skipped.")

    codec = codec.lower().strip()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        input_wav = tmp_dir / "input.wav"
        output_file = tmp_dir / f"encoded.{codec}"
        decoded_wav = tmp_dir / "decoded.wav"

        sf.write(input_wav, audio, sample_rate, subtype="PCM_16")

        if codec == "opus":
            encode_cmd = [
                "ffmpeg",
                "-y",
                "-v", "error",
                "-i", str(input_wav),
                "-c:a", "libopus",
                "-b:a", f"{bitrate_kbps}k",
                "-ar", str(sample_rate),
                "-ac", "1",
                str(output_file),
            ]

        elif codec == "mp3":
            encode_cmd = [
                "ffmpeg",
                "-y",
                "-v", "error",
                "-i", str(input_wav),
                "-c:a", "libmp3lame",
                "-b:a", f"{bitrate_kbps}k",
                "-ar", str(sample_rate),
                "-ac", "1",
                str(output_file),
            ]

        else:
            raise ValueError(f"Unsupported codec: {codec}")

        decode_cmd = [
            "ffmpeg",
            "-y",
            "-v", "error",
            "-i", str(output_file),
            "-ar", str(sample_rate),
            "-ac", "1",
            str(decoded_wav),
        ]

        encode_process = subprocess.run(
            encode_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        if encode_process.returncode != 0:
            raise RuntimeError(
                encode_process.stderr.decode("utf-8", errors="ignore")
            )

        decode_process = subprocess.run(
            decode_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        if decode_process.returncode != 0:
            raise RuntimeError(
                decode_process.stderr.decode("utf-8", errors="ignore")
            )

        decoded, sr = sf.read(decoded_wav, dtype="float32", always_2d=False)

        if sr != sample_rate:
            raise RuntimeError(f"Codec output sample rate invalid: {sr}")

        decoded = np.asarray(decoded, dtype=np.float32)

        if decoded.ndim == 2:
            decoded = np.mean(decoded, axis=1)

        if len(decoded) < len(audio):
            padded = np.zeros_like(audio)
            padded[:len(decoded)] = decoded
            decoded = padded

        if len(decoded) > len(audio):
            decoded = decoded[:len(audio)]

        return limit_peak(decoded.astype(np.float32), peak_limit=0.98)


def apply_random_augmentations(
    audio: np.ndarray,
    cfg: AugmentConfig,
    rng: random.Random | None = None,
    allow_codec: bool = True,
    name: str = "audio",
) -> AugmentResult:
    """
    Modular augmentation pipeline.

    Args:
        audio: Mono ``float32`` audio. Its length must match
            ``cfg.sample_rate * cfg.duration_sec``.
        cfg: Augmentation configuration controlling probabilities and ranges.
        rng: Optional Python random generator for deterministic augmentation.
        allow_codec: Whether FFmpeg codec simulation is allowed for this call.
        name: Signal name used in validation error messages.

    Returns:
        ``AugmentResult`` containing final audio, applied effect names, and
        metadata describing sampled parameters.

    Raises:
        ValueError: If input audio is invalid or has the wrong duration.
    """
    if rng is None:
        rng = random.Random()

    audio = ensure_float32(audio, name=name)

    expected_samples = int(cfg.sample_rate * cfg.duration_sec)

    if len(audio) != expected_samples:
        raise ValueError(
            f"{name} invalid length: {len(audio)} samples, expected {expected_samples}."
        )

    if not cfg.enabled:
        return AugmentResult(audio=audio.astype(np.float32))

    applied = []
    metadata = {}

    out = audio.copy()

    # Light gain
    if chance(rng, cfg.p_gain):
        gain_db = rand_uniform(rng, cfg.gain_min_db, cfg.gain_max_db)
        out = apply_gain(out, gain_db)
        out = limit_peak(out, cfg.peak_limit)

        applied.append("gain")
        metadata["gain_db"] = round(gain_db, 4)

    # Three-band EQ
    if chance(rng, cfg.p_eq):
        low_gain_db = rand_uniform(rng, cfg.eq_gain_min_db, cfg.eq_gain_max_db)
        mid_gain_db = rand_uniform(rng, cfg.eq_gain_min_db, cfg.eq_gain_max_db)
        high_gain_db = rand_uniform(rng, cfg.eq_gain_min_db, cfg.eq_gain_max_db)

        out = three_band_eq(
            out,
            sample_rate=cfg.sample_rate,
            low_gain_db=low_gain_db,
            mid_gain_db=mid_gain_db,
            high_gain_db=high_gain_db,
        )

        applied.append("eq")
        metadata["eq_low_gain_db"] = round(low_gain_db, 4)
        metadata["eq_mid_gain_db"] = round(mid_gain_db, 4)
        metadata["eq_high_gain_db"] = round(high_gain_db, 4)

    # Phone / VoIP band limiting
    if chance(rng, cfg.p_phone_filter):
        hp = rand_uniform(
            rng,
            cfg.phone_highpass_min_hz,
            cfg.phone_highpass_max_hz,
        )

        lp = rand_uniform(
            rng,
            cfg.phone_lowpass_min_hz,
            cfg.phone_lowpass_max_hz,
        )

        out = phone_filter(
            out,
            sample_rate=cfg.sample_rate,
            highpass_hz=hp,
            lowpass_hz=lp,
        )

        applied.append("phone_filter")
        metadata["phone_highpass_hz"] = round(hp, 2)
        metadata["phone_lowpass_hz"] = round(lp, 2)

    # Compression
    if chance(rng, cfg.p_compression):
        threshold_db = rand_uniform(
            rng,
            cfg.compressor_threshold_min_db,
            cfg.compressor_threshold_max_db,
        )

        ratio = rand_uniform(
            rng,
            cfg.compressor_ratio_min,
            cfg.compressor_ratio_max,
        )

        out = static_compressor(
            out,
            threshold_db=threshold_db,
            ratio=ratio,
            makeup_gain_db=0.0,
        )

        applied.append("compression")
        metadata["compressor_threshold_db"] = round(threshold_db, 4)
        metadata["compressor_ratio"] = round(ratio, 4)

    # Soft saturation
    if chance(rng, cfg.p_saturation):
        drive_db = rand_uniform(
            rng,
            cfg.saturation_drive_min_db,
            cfg.saturation_drive_max_db,
        )

        out = soft_saturation(out, drive_db=drive_db)

        applied.append("saturation")
        metadata["saturation_drive_db"] = round(drive_db, 4)

    # Hard clipping
    if chance(rng, cfg.p_clipping):
        threshold = rand_uniform(
            rng,
            cfg.clipping_threshold_min,
            cfg.clipping_threshold_max,
        )

        out = hard_clipping(out, threshold=threshold)

        applied.append("clipping")
        metadata["clipping_threshold"] = round(threshold, 4)

    # Lightweight reverb
    if chance(rng, cfg.p_reverb):
        wet = rand_uniform(rng, cfg.reverb_wet_min, cfg.reverb_wet_max)
        decay_ms = rand_uniform(
            rng,
            cfg.reverb_decay_min_ms,
            cfg.reverb_decay_max_ms,
        )

        reflections = rng.randint(
            cfg.reverb_reflections_min,
            cfg.reverb_reflections_max,
        )

        out = light_reverb(
            out,
            sample_rate=cfg.sample_rate,
            rng=rng,
            wet=wet,
            decay_ms=decay_ms,
            num_reflections=reflections,
        )

        applied.append("reverb")
        metadata["reverb_wet"] = round(wet, 4)
        metadata["reverb_decay_ms"] = round(decay_ms, 4)
        metadata["reverb_reflections"] = reflections

    # Low-bit quantization
    if chance(rng, cfg.p_quantization):
        bits = rng.randint(cfg.quantization_bits_min, cfg.quantization_bits_max)

        out = quantize_audio(out, bits=bits)

        applied.append("quantization")
        metadata["quantization_bits"] = bits

    # Short VoIP-like dropouts
    if chance(rng, cfg.p_dropout):
        out = random_dropouts(
            out,
            sample_rate=cfg.sample_rate,
            rng=rng,
            min_count=cfg.dropout_count_min,
            max_count=cfg.dropout_count_max,
            min_ms=cfg.dropout_ms_min,
            max_ms=cfg.dropout_ms_max,
        )

        applied.append("dropout")
        metadata["dropout"] = True

    # Real codec roundtrip through FFmpeg
    if allow_codec and chance(rng, cfg.p_codec):
        codec = rng.choice(cfg.codec_choices)

        if codec == "opus":
            bitrate = rng.randint(
                cfg.opus_bitrate_min_kbps,
                cfg.opus_bitrate_max_kbps,
            )
        elif codec == "mp3":
            bitrate = rng.randint(
                cfg.mp3_bitrate_min_kbps,
                cfg.mp3_bitrate_max_kbps,
            )
        else:
            bitrate = 32

        try:
            out = ffmpeg_codec_simulation(
                out,
                sample_rate=cfg.sample_rate,
                codec=codec,
                bitrate_kbps=bitrate,
            )

            applied.append("codec")
            metadata["codec"] = codec
            metadata["codec_bitrate_kbps"] = bitrate

        except Exception as e:
            applied.append("codec_failed")
            metadata["codec_error"] = repr(e)

    out = limit_peak(out, cfg.peak_limit)

    metadata["rms_db_after_aug"] = round(rms_db(out), 4)
    metadata["peak_after_aug"] = round(peak(out), 6)

    return AugmentResult(
        audio=out.astype(np.float32),
        applied=applied,
        metadata=metadata,
    )
