"""Connecteur NVIDIA Maxine (BNR + Studio Voice) via NVCF gRPC.

Reproduit fidèlement le client officiel NVIDIA-Maxine/nim-clients (licence MIT)
en mode « transactionnel » (fichier complet en une requête) :

    * canal TLS vers ``grpc.nvcf.nvidia.com:443`` ;
    * métadonnées ``authorization: Bearer <clé>`` et ``function-id: <id>`` ;
    * méthode ``EnhanceAudio`` (stream de chunks 64 KB en entrée, stream de
      chunks WAV en sortie).

Les modèles Maxine attendent du WAV (par défaut 48 kHz). On décode donc l'audio
entrant (n'importe quel format lisible par librosa) vers le bon sample rate
avant l'envoi, et on renvoie le WAV produit tel quel.

Les ``function-id`` ne sont PAS codés en dur : ils viennent de l'environnement
(chaque modèle a le sien, copié depuis sa page sur build.nvidia.com). Voir
serve/NVIDIA.md.
"""
from __future__ import annotations

import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

# --- Rend les stubs gRPC vendorisés importables ------------------------------
# Les fichiers générés font des imports top-level (``import bnr_pb2``), donc on
# ajoute leurs dossiers au sys.path. Les deux protos vivent dans des packages
# distincts (nvidia.ai4m.bnr.v1 / nvidia.ai4m.studiovoice.v1) : pas de collision.
_VENDOR = Path(__file__).resolve().parent / "_vendor"
for _sub in ("studio_voice", "bnr"):
    _p = str(_VENDOR / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)


class NvidiaError(RuntimeError):
    """Erreur générique côté NVIDIA."""


class NvidiaAuthError(NvidiaError):
    """Clé invalide / expirée / non autorisée (HTTP 401)."""


class NvidiaUnavailable(NvidiaError):
    """Dépendance gRPC absente ou configuration incomplète (HTTP 503)."""


@dataclass(frozen=True)
class NvidiaModel:
    id: str               # identifiant exposé dans /models
    label: str            # libellé UI
    function_id_env: str  # nom de la variable d'env contenant le function-id
    sample_rate: int      # sample rate attendu par le modèle
    description: str


# Registre des modèles NVIDIA branchés (les deux, choix dans l'UI).
NVIDIA_MODELS: dict[str, NvidiaModel] = {
    "nvidia:bnr": NvidiaModel(
        id="nvidia:bnr",
        label="NVIDIA · Background Noise Removal",
        function_id_env="NVIDIA_BNR_FUNCTION_ID",
        sample_rate=48000,
        description="Débruitage fidèle : retire le bruit de fond, conserve la voix.",
    ),
    "nvidia:studiovoice": NvidiaModel(
        id="nvidia:studiovoice",
        label="NVIDIA · Studio Voice",
        function_id_env="NVIDIA_STUDIOVOICE_FUNCTION_ID",
        sample_rate=48000,
        description="Qualité studio : débruite + dé-réverbère + restaure la voix.",
    ),
}

_DATA_CHUNK = 64 * 1024  # 64 KB, comme le client officiel


def is_nvidia_model(model_id: str | None) -> bool:
    return bool(model_id) and model_id in NVIDIA_MODELS


def list_models() -> list[dict]:
    """Modèles NVIDIA au même format que les checkpoints locaux (pour /models)."""
    out: list[dict] = []
    for m in NVIDIA_MODELS.values():
        out.append({
            "id": m.id,
            "label": m.label,
            "provider": "nvidia",
            "requires_key": True,
            "description": m.description,
            "configured": bool(os.environ.get(m.function_id_env, "").strip()),
            "is_loaded": False,
        })
    return out


def _grpc_target() -> str:
    return os.environ.get("NVIDIA_GRPC_TARGET", "grpc.nvcf.nvidia.com:443").strip()


def _grpc_timeout() -> float:
    try:
        return float(os.environ.get("NVIDIA_GRPC_TIMEOUT", "60"))
    except ValueError:
        return 60.0


def _function_id(model: NvidiaModel) -> str:
    fid = os.environ.get(model.function_id_env, "").strip()
    if not fid:
        raise NvidiaUnavailable(
            f"function-id manquant pour {model.label} : définissez "
            f"{model.function_id_env} (voir serve/NVIDIA.md)."
        )
    return fid


def _to_wav_bytes(audio_bytes: bytes, sample_rate: int) -> bytes:
    """Décode l'audio entrant et le ré-encode en WAV mono au sample rate cible."""
    try:
        wav, _ = librosa.load(io.BytesIO(audio_bytes), sr=sample_rate, mono=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Audio illisible : {exc}") from exc

    wav = np.asarray(wav, dtype=np.float32)
    if wav.size == 0:
        raise ValueError("Audio vide après décodage.")

    buf = io.BytesIO()
    sf.write(buf, wav, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def _request_iter(wav_bytes: bytes, pb2, intensity: float | None):
    """Génère le flux de requêtes (config BNR optionnelle + chunks 64 KB)."""
    if intensity is not None and hasattr(pb2, "EnhanceAudioConfig"):
        yield pb2.EnhanceAudioRequest(
            config=pb2.EnhanceAudioConfig(intensity_ratio=float(intensity))
        )
    stream = io.BytesIO(wav_bytes)
    while True:
        chunk = stream.read(_DATA_CHUNK)
        if not chunk:
            break
        yield pb2.EnhanceAudioRequest(audio_stream_data=chunk)


def enhance(
    model_id: str,
    audio_bytes: bytes,
    api_key: str,
    intensity: float | None = None,
) -> bytes:
    """Envoie l'audio au NIM Maxine et renvoie le WAV nettoyé (bytes).

    Raises
    ------
    ValueError        : audio illisible/vide.
    NvidiaAuthError   : clé refusée (401).
    NvidiaUnavailable : grpc absent / function-id manquant (503).
    NvidiaError       : autre erreur côté NVIDIA.
    """
    model = NVIDIA_MODELS.get(model_id)
    if model is None:
        raise NvidiaError(f"Modèle NVIDIA inconnu : {model_id}")

    try:
        import grpc
    except ImportError as exc:
        raise NvidiaUnavailable(
            "Le paquet 'grpcio' n'est pas installé : "
            "pip install -r requirements-serve.txt"
        ) from exc

    function_id = _function_id(model)
    wav_bytes = _to_wav_bytes(audio_bytes, model.sample_rate)

    if model_id == "nvidia:bnr":
        import bnr_pb2 as pb2
        import bnr_pb2_grpc as pb2_grpc
        stub_cls = pb2_grpc.BNRStub
    else:  # nvidia:studiovoice
        import studiovoice_pb2 as pb2
        import studiovoice_pb2_grpc as pb2_grpc
        stub_cls = pb2_grpc.StudioVoiceStub

    metadata = (
        ("authorization", f"Bearer {api_key}"),
        ("function-id", function_id),
    )
    credentials = grpc.ssl_channel_credentials()

    try:
        with grpc.secure_channel(_grpc_target(), credentials) as channel:
            stub = stub_cls(channel)
            responses = stub.EnhanceAudio(
                _request_iter(wav_bytes, pb2, intensity),
                metadata=metadata,
                timeout=_grpc_timeout(),
            )
            out = io.BytesIO()
            for resp in responses:
                if resp.HasField("audio_stream_data"):
                    out.write(resp.audio_stream_data)
    except grpc.RpcError as exc:  # type: ignore[attr-defined]
        code = exc.code()
        if code in (grpc.StatusCode.UNAUTHENTICATED, grpc.StatusCode.PERMISSION_DENIED):
            raise NvidiaAuthError("Clé NVIDIA invalide, expirée ou non autorisée.") from exc
        if code == grpc.StatusCode.RESOURCE_EXHAUSTED:
            raise NvidiaError("Limite NVIDIA atteinte (rate limit / crédits épuisés).") from exc
        raise NvidiaError(f"Erreur NVIDIA ({code.name}): {exc.details()}") from exc

    data = out.getvalue()
    if not data:
        raise NvidiaError("Réponse vide du service NVIDIA.")
    return data
