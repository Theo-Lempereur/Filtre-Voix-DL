from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np
import soundfile as sf


@dataclass
class CropResult:
    audio: np.ndarray
    start_sample: int
    end_sample: int
    num_samples: int


def duration_to_samples(duration_sec: float, sample_rate: int) -> int:
    return int(round(duration_sec * sample_rate))


def make_rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(seed)


def stable_seed_from_text(text: str, base_seed: int = 42) -> int:
    """
    Crée une seed stable à partir d'un texte.
    Utile pour le multiprocessing : chaque fichier/sample peut avoir une seed unique.
    """
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
    value = int(h[:8], 16)
    return (value + base_seed) % (2**32)


def ensure_float32(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains NaN or Inf.")

    return audio


def to_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convertit un audio stéréo/multicanal en mono.
    Shape attendue :
    - mono : (samples,)
    - stereo : (samples, channels)
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
    Complète un audio trop court jusqu'à target_samples.

    mode:
    - zero : ajoute du silence
    - repeat : répète l'audio
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
    Prend un crop aléatoire dans un tableau audio.

    Cas possibles :
    - audio plus long que crop_samples : crop aléatoire
    - audio exactement crop_samples : retourne l'audio
    - audio plus court :
        - pad_if_short=False : erreur
        - pad_if_short=True : padding
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
    Crop centré.
    Utile pour debug ou validation déterministe.
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
    Crop aligné pour un couple clean/noisy déjà appairé.

    Important :
    clean et noisy doivent avoir la même durée.
    On utilise le même start_sample pour les deux.
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
    Crop pour génération speech enhancement.

    clean :
    - doit généralement être assez long
    - si trop court, mieux vaut ignorer sauf si pad_short_clean=True

    noise :
    - peut être crop aléatoirement
    - si trop court, on peut le répéter avec pad_mode='repeat'
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
    Découpe déterministe en chunks fixes.

    Utile pour preprocessing :
    - clean_chunks
    - noise_chunks

    Contrairement au random crop, ici on découpe tout le fichier.
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
    Récupère les infos sans charger tout le fichier en RAM.
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
    Random crop directement depuis un fichier WAV/FLAC sans charger tout le fichier.

    Très utile pour gros dataset :
    au lieu de charger 3 minutes d'audio,
    on lit seulement 3 secondes.
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
    Version haut niveau pour génération noisy.

    clean_path :
    - fichier clean mono 16 kHz

    noise_path :
    - fichier noise mono 16 kHz

    Retour :
    - clean_crop de 3s
    - noise_crop de 3s
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