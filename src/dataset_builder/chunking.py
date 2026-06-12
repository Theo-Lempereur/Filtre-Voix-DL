"""Cropping and chunking utilities for prepared audio and generated pairs.

The dataset pipeline uses two complementary strategies:

* deterministic fixed chunks during preprocessing, so raw files can be converted
  into reusable clean/noise assets;
* random crops during generation, so each noisy/clean pair can sample different
  regions from prepared assets while preserving alignment.

All crop offsets are expressed in samples, not seconds, to keep metadata exact.
"""

from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np
import soundfile as sf


@dataclass
class CropResult:
    """Audio crop payload with source offsets expressed in samples.

    Attributes:
        audio: Cropped mono audio as a ``float32`` NumPy array.
        start_sample: Inclusive start offset in the source audio.
        end_sample: Exclusive end offset in the source audio.
        num_samples: Number of samples in ``audio``.
    """

    audio: np.ndarray
    start_sample: int
    end_sample: int
    num_samples: int


def duration_to_samples(duration_sec: float, sample_rate: int) -> int:
    """Convert a duration in seconds to a rounded sample count.

    Args:
        duration_sec: Duration expressed in seconds.
        sample_rate: Sample rate in Hz.

    Returns:
        Number of samples corresponding to the duration.
    """
    return int(round(duration_sec * sample_rate))


def make_rng(seed: int | None = None) -> np.random.Generator:
    """Create a NumPy random generator for deterministic crop selection.

    Args:
        seed: Optional seed. Passing the same seed reproduces crop positions.

    Returns:
        NumPy random generator instance.
    """
    return np.random.default_rng(seed)


def stable_seed_from_text(text: str, base_seed: int = 42) -> int:
    """
    Create a stable seed from text.

    Useful for multiprocessing because each file or sample can receive a unique seed.

    Args:
        text: Stable text identifier, such as a path or sample id.
        base_seed: Project-level seed offset.

    Returns:
        Unsigned 32-bit seed derived from the text and base seed.
    """
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
    value = int(h[:8], 16)
    return (value + base_seed) % (2**32)


def ensure_float32(audio: np.ndarray) -> np.ndarray:
    """Validate that an audio array is finite and return it as float32.

    Args:
        audio: Audio-like array.

    Returns:
        ``float32`` NumPy array.

    Raises:
        ValueError: If the array contains NaN or infinite values.
    """
    audio = np.asarray(audio, dtype=np.float32)

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains NaN or Inf.")

    return audio


def to_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert stereo or multichannel audio to mono.

    Expected shapes:
    - mono : (samples,)
    - stereo : (samples, channels)

    Args:
        audio: Mono or channel-last multichannel audio.

    Returns:
        Mono audio array.

    Raises:
        ValueError: If ``audio`` has an unsupported shape.
    """
    if audio.ndim == 1:
        return audio

    if audio.ndim == 2:
        return np.mean(audio, axis=1)

    raise ValueError(f"Invalid audio shape: {audio.shape}")


def pad_audio(
    audio: np.ndarray,
    target_samples: int,
    mode: str = "zero",
) -> np.ndarray:
    """
    Pad audio that is shorter than target_samples.

    mode:
    - zero: add silence
    - repeat: tile the input audio

    Args:
        audio: Mono audio samples.
        target_samples: Required output length.
        mode: Padding strategy, either ``"zero"`` or ``"repeat"``.

    Returns:
        Audio exactly ``target_samples`` long.

    Raises:
        ValueError: If repeat padding is requested for empty audio or if the mode
            is unknown.
    """
    audio = ensure_float32(audio)

    if len(audio) >= target_samples:
        return audio[:target_samples]

    if mode == "zero":
        output = np.zeros(target_samples, dtype=np.float32)
        output[:len(audio)] = audio
        return output

    if mode == "repeat":
        if len(audio) == 0:
            raise ValueError("Cannot repeat empty audio.")

        repeats = int(np.ceil(target_samples / len(audio)))
        output = np.tile(audio, repeats)[:target_samples]
        return output.astype(np.float32)

    raise ValueError(f"Unknown padding mode: {mode}")


def random_crop_array(
    audio: np.ndarray,
    crop_samples: int,
    rng: np.random.Generator,
    pad_if_short: bool = False,
    pad_mode: str = "zero",
) -> CropResult:
    """
    Take a random crop from an audio array.

    Cases:
    - audio longer than crop_samples: random crop
    - audio exactly crop_samples: return unchanged
    - audio shorter than crop_samples:
        - pad_if_short=False: raise an error
        - pad_if_short=True: pad the signal

    Args:
        audio: Mono source audio.
        crop_samples: Exact number of samples to return.
        rng: NumPy generator used to draw the crop start.
        pad_if_short: Whether shorter audio should be padded instead of rejected.
        pad_mode: Padding mode passed to ``pad_audio``.

    Returns:
        ``CropResult`` containing the crop and source offsets.

    Raises:
        ValueError: If audio is non-mono, too short without padding, or invalid.
    """
    audio = ensure_float32(audio)

    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio, got shape {audio.shape}")

    audio_len = len(audio)

    if audio_len == crop_samples:
        return CropResult(
            audio=audio.astype(np.float32),
            start_sample=0,
            end_sample=crop_samples,
            num_samples=crop_samples,
        )

    if audio_len < crop_samples:
        if not pad_if_short:
            raise ValueError(
                f"Audio too short: {audio_len} samples, expected {crop_samples}."
            )

        padded = pad_audio(audio, crop_samples, mode=pad_mode)

        return CropResult(
            audio=padded,
            start_sample=0,
            end_sample=crop_samples,
            num_samples=crop_samples,
        )

    max_start = audio_len - crop_samples
    start = int(rng.integers(0, max_start + 1))
    end = start + crop_samples

    crop = audio[start:end]

    return CropResult(
        audio=crop.astype(np.float32),
        start_sample=start,
        end_sample=end,
        num_samples=crop_samples,
    )


def center_crop_array(
    audio: np.ndarray,
    crop_samples: int,
    pad_if_short: bool = False,
    pad_mode: str = "zero",
) -> CropResult:
    """
    Take a centered crop.

    Useful for debugging or deterministic validation.

    Args:
        audio: Mono source audio.
        crop_samples: Exact number of samples to return.
        pad_if_short: Whether shorter audio should be padded instead of rejected.
        pad_mode: Padding mode passed to ``pad_audio``.

    Returns:
        ``CropResult`` centered in the source audio when possible.

    Raises:
        ValueError: If audio is non-mono, too short without padding, or invalid.
    """
    audio = ensure_float32(audio)

    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio, got shape {audio.shape}")

    audio_len = len(audio)

    if audio_len < crop_samples:
        if not pad_if_short:
            raise ValueError(
                f"Audio too short: {audio_len} samples, expected {crop_samples}."
            )

        padded = pad_audio(audio, crop_samples, mode=pad_mode)

        return CropResult(
            audio=padded,
            start_sample=0,
            end_sample=crop_samples,
            num_samples=crop_samples,
        )

    start = (audio_len - crop_samples) // 2
    end = start + crop_samples

    crop = audio[start:end]

    return CropResult(
        audio=crop.astype(np.float32),
        start_sample=start,
        end_sample=end,
        num_samples=crop_samples,
    )


def aligned_random_crop_pair(
    clean: np.ndarray,
    noisy: np.ndarray,
    crop_samples: int,
    rng: np.random.Generator,
    pad_if_short: bool = False,
    pad_mode: str = "zero",
) -> tuple[CropResult, CropResult]:
    """
    Take aligned random crops from an already paired clean/noisy example.

    Clean and noisy must have the same duration. The same start sample is used for both.

    Args:
        clean: Clean target audio.
        noisy: Noisy input audio aligned with ``clean``.
        crop_samples: Exact number of samples to return for each signal.
        rng: NumPy generator used to draw the shared crop start.
        pad_if_short: Whether short pairs should be padded.
        pad_mode: Padding mode passed to ``pad_audio``.

    Returns:
        Tuple ``(clean_crop, noisy_crop)`` with matching offsets.

    Raises:
        ValueError: If arrays are not mono, not aligned, too short, or invalid.
    """
    clean = ensure_float32(clean)
    noisy = ensure_float32(noisy)

    if clean.ndim != 1 or noisy.ndim != 1:
        raise ValueError("clean and noisy must be mono arrays.")

    if len(clean) != len(noisy):
        raise ValueError(
            f"Alignment error: clean has {len(clean)} samples, noisy has {len(noisy)} samples."
        )

    audio_len = len(clean)

    if audio_len < crop_samples:
        if not pad_if_short:
            raise ValueError(
                f"Pair too short: {audio_len} samples, expected {crop_samples}."
            )

        clean_crop = pad_audio(clean, crop_samples, mode=pad_mode)
        noisy_crop = pad_audio(noisy, crop_samples, mode=pad_mode)

        return (
            CropResult(clean_crop, 0, crop_samples, crop_samples),
            CropResult(noisy_crop, 0, crop_samples, crop_samples),
        )

    max_start = audio_len - crop_samples
    start = int(rng.integers(0, max_start + 1))
    end = start + crop_samples

    clean_crop = clean[start:end]
    noisy_crop = noisy[start:end]

    return (
        CropResult(clean_crop.astype(np.float32), start, end, crop_samples),
        CropResult(noisy_crop.astype(np.float32), start, end, crop_samples),
    )


def random_crop_for_mix(
    clean: np.ndarray,
    noise: np.ndarray,
    crop_samples: int,
    rng: np.random.Generator,
    pad_short_clean: bool = False,
    pad_short_noise: bool = True,
) -> tuple[CropResult, CropResult]:
    """
    Crop clean and noise audio for speech enhancement generation.

    clean :
    - should generally be long enough
    - if too short, it is better to skip unless pad_short_clean=True

    noise :
    - can be randomly cropped
    - if too short, it can be repeated with pad_mode='repeat'

    Args:
        clean: Clean speech source audio.
        noise: Background noise source audio.
        crop_samples: Exact output length for both signals.
        rng: NumPy generator used for crop selection.
        pad_short_clean: Whether short speech can be zero-padded.
        pad_short_noise: Whether short noise can be repeated.

    Returns:
        Tuple ``(clean_crop, noise_crop)`` ready for SNR mixing.
    """
    clean_crop = random_crop_array(
        clean,
        crop_samples=crop_samples,
        rng=rng,
        pad_if_short=pad_short_clean,
        pad_mode="zero",
    )

    noise_crop = random_crop_array(
        noise,
        crop_samples=crop_samples,
        rng=rng,
        pad_if_short=pad_short_noise,
        pad_mode="repeat",
    )

    return clean_crop, noise_crop


def fixed_chunks_array(
    audio: np.ndarray,
    chunk_samples: int,
    drop_last: bool = True,
    pad_last: bool = False,
) -> list[CropResult]:
    """
    Split audio deterministically into fixed chunks.

    Useful for preprocessing:
    - clean_chunks
    - noise_chunks

    Unlike random cropping, this walks through the full file.

    Args:
        audio: Mono source audio.
        chunk_samples: Exact length of each output chunk.
        drop_last: Whether to discard the final incomplete chunk.
        pad_last: Whether to zero-pad an incomplete final chunk when kept.

    Returns:
        List of ``CropResult`` items ordered by source position.

    Raises:
        ValueError: If input is non-mono or contains invalid values.
    """
    audio = ensure_float32(audio)

    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio, got shape {audio.shape}")

    chunks = []
    audio_len = len(audio)

    if audio_len < chunk_samples:
        if pad_last:
            padded = pad_audio(audio, chunk_samples, mode="zero")
            chunks.append(
                CropResult(
                    audio=padded,
                    start_sample=0,
                    end_sample=chunk_samples,
                    num_samples=chunk_samples,
                )
            )
        return chunks

    num_full_chunks = audio_len // chunk_samples

    for idx in range(num_full_chunks):
        start = idx * chunk_samples
        end = start + chunk_samples

        chunks.append(
            CropResult(
                audio=audio[start:end].astype(np.float32),
                start_sample=start,
                end_sample=end,
                num_samples=chunk_samples,
            )
        )

    remaining = audio_len % chunk_samples

    if remaining > 0 and not drop_last and pad_last:
        start = num_full_chunks * chunk_samples
        last = audio[start:]
        padded = pad_audio(last, chunk_samples, mode="zero")

        chunks.append(
            CropResult(
                audio=padded,
                start_sample=start,
                end_sample=start + chunk_samples,
                num_samples=chunk_samples,
            )
        )

    return chunks


def get_wav_info(path: str | Path) -> dict:
    """
    Read audio metadata without loading the full file into memory.

    Args:
        path: WAV/FLAC/audio path readable by soundfile.

    Returns:
        Dictionary containing path, sample rate, channel count, frame count,
        duration, format, and subtype.
    """
    path = Path(path)
    info = sf.info(path)

    return {
        "path": str(path),
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "duration_sec": info.frames / info.samplerate,
        "format": info.format,
        "subtype": info.subtype,
    }


def random_crop_wav_file(
    path: str | Path,
    crop_samples: int,
    expected_sample_rate: int,
    rng: np.random.Generator,
    pad_if_short: bool = False,
    pad_mode: str = "zero",
) -> CropResult:
    """
    Randomly crop from a WAV/FLAC file without loading the whole file.

    Useful for large datasets: read only the requested crop instead of loading a full file.

    Args:
        path: Prepared audio file path.
        crop_samples: Exact number of samples to read.
        expected_sample_rate: Required file sample rate.
        rng: NumPy generator used to draw the crop start.
        pad_if_short: Whether short files should be padded instead of rejected.
        pad_mode: Padding strategy passed to ``pad_audio``.

    Returns:
        ``CropResult`` with audio and source offsets.

    Raises:
        ValueError: If sample rate is wrong, the file is too short without
            padding, or loaded samples are invalid.
    """
    path = Path(path)
    info = sf.info(path)

    if info.samplerate != expected_sample_rate:
        raise ValueError(
            f"Invalid sample rate for {path}: {info.samplerate}, expected {expected_sample_rate}."
        )

    total_samples = info.frames

    if total_samples < crop_samples:
        if not pad_if_short:
            raise ValueError(
                f"File too short: {path}, {total_samples} samples, expected {crop_samples}."
            )

        audio, _ = sf.read(path, dtype="float32", always_2d=False)
        audio = to_mono(audio)
        audio = pad_audio(audio, crop_samples, mode=pad_mode)

        return CropResult(
            audio=audio.astype(np.float32),
            start_sample=0,
            end_sample=crop_samples,
            num_samples=crop_samples,
        )

    if total_samples == crop_samples:
        start = 0
    else:
        max_start = total_samples - crop_samples
        start = int(rng.integers(0, max_start + 1))

    audio, _ = sf.read(
        path,
        start=start,
        frames=crop_samples,
        dtype="float32",
        always_2d=False,
    )

    audio = to_mono(audio)
    audio = ensure_float32(audio)

    if len(audio) != crop_samples:
        audio = pad_audio(audio, crop_samples, mode=pad_mode)

    return CropResult(
        audio=audio.astype(np.float32),
        start_sample=start,
        end_sample=start + crop_samples,
        num_samples=crop_samples,
    )


def random_crop_clean_noise_files(
    clean_path: str | Path,
    noise_path: str | Path,
    sample_rate: int = 16000,
    duration_sec: float = 3.0,
    seed: int | None = None,
) -> tuple[CropResult, CropResult]:
    """
    High-level helper for noisy dataset generation.

    Args:
        clean_path: Prepared clean mono file.
        noise_path: Prepared noise mono file.
        sample_rate: Expected sample rate for both files.
        duration_sec: Requested crop duration in seconds.
        seed: Optional deterministic crop seed.

    Returns:
        Tuple ``(clean_crop, noise_crop)``. Clean files must be long enough;
        short noise files are repeated.
    """
    crop_samples = duration_to_samples(duration_sec, sample_rate)
    rng = make_rng(seed)

    clean_crop = random_crop_wav_file(
        clean_path,
        crop_samples=crop_samples,
        expected_sample_rate=sample_rate,
        rng=rng,
        pad_if_short=False,
        pad_mode="zero",
    )

    noise_crop = random_crop_wav_file(
        noise_path,
        crop_samples=crop_samples,
        expected_sample_rate=sample_rate,
        rng=rng,
        pad_if_short=True,
        pad_mode="repeat",
    )

    return clean_crop, noise_crop
