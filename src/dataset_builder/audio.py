from pathlib import Path
import subprocess
import shutil
import math

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


EPS = 1e-12


def db_to_amp(db: float) -> float:
    return float(10 ** (db / 20))


def amp_to_db(amp: float) -> float:
    return float(20 * math.log10(max(amp, EPS)))


def rms_db(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(audio ** 2) + EPS))
    return amp_to_db(rms)


def peak(audio: np.ndarray) -> float:
    return float(np.max(np.abs(audio))) if len(audio) else 0.0


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio

    if audio.ndim == 2:
        return np.mean(audio, axis=1)

    raise ValueError(f"Audio avec dimensions invalides : {audio.shape}")


def resample_audio(audio: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    if original_sr == target_sr:
        return audio.astype(np.float32)

    gcd = math.gcd(original_sr, target_sr)
    up = target_sr // gcd
    down = original_sr // gcd

    audio = resample_poly(audio, up, down)
    return audio.astype(np.float32)


def load_with_soundfile(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    return audio, sr


def load_with_ffmpeg(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "Impossible de lire ce fichier avec soundfile, et ffmpeg n'est pas installé ou pas dans le PATH."
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
    path = Path(path)

    try:
        audio, sr = load_with_soundfile(path)
    except Exception:
        audio, sr = load_with_ffmpeg(path, target_sr)

    if audio.size == 0:
        raise ValueError("Fichier audio vide.")

    if mono:
        audio = to_mono(audio)

    audio = np.asarray(audio, dtype=np.float32)

    if not np.all(np.isfinite(audio)):
        raise ValueError("Le fichier contient des valeurs NaN ou infinies.")

    audio = resample_audio(audio, sr, target_sr)

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio invalide après resampling.")

    return audio.astype(np.float32), target_sr


def limit_peak(audio: np.ndarray, peak_limit: float = 0.98) -> np.ndarray:
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    audio = limit_peak(audio, peak_limit=0.999)
    sf.write(path, audio, sample_rate, subtype="PCM_16")