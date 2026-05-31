"""API FastAPI pour servir le modèle de débruitage voix.

Charge un checkpoint au démarrage (via la variable d'environnement ``CKPT_PATH``)
et expose deux endpoints :

    GET  /health   -> état du serveur + infos checkpoint
    POST /denoise  -> reçoit un fichier audio, renvoie le WAV débruité

Lancement (depuis la racine du repo, venv activé) :

    CKPT_PATH="/chemin/absolu/checkpoints/mon_run/best.pt" \\
    DEVICE="cpu" \\
    CORS_ORIGINS="https://monsite.com" \\
    uvicorn serve.server:app --host 127.0.0.1 --port 8000 --workers 1

Variables d'environnement :
    CKPT_PATH     (requis)  chemin absolu vers le .pt
    DEVICE        (option)  "cuda" / "cpu" ; auto-détecté si absent
    CORS_ORIGINS  (option)  origines autorisées, séparées par virgule (défaut "*")
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from serve.inference import denoise_bytes, load_model

# Charge le .env à la racine du repo (CKPT_PATH, DEVICE, CORS_ORIGINS).
# Les variables déjà définies dans l'environnement ont priorité (override=False).
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("serve")


# État global du serveur. Un dict module-level plutôt que app.state : explicite,
# testable, et sans surprise avec un worker unique (cas GPU recommandé).
_state: dict = {
    "model": None,
    "device": None,
    "input_repr": None,
    "ckpt_path": None,
}


# Content-types acceptés (filtre indicatif côté requête ; la vraie validation se
# fait au décodage dans denoise_bytes).
_ALLOWED_CONTENT_TYPES = {
    "audio/wav", "audio/wave", "audio/x-wav",
    "audio/mpeg", "audio/mp3",
    "audio/flac", "audio/x-flac",
    "audio/ogg", "audio/x-ogg",
    "application/octet-stream",  # certains navigateurs n'envoient pas de type
}


def _resolve_device() -> str:
    env = os.environ.get("DEVICE", "").strip()
    if env:
        return env
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_cors_origins() -> list[str]:
    raw = os.environ.get("CORS_ORIGINS", "*").strip()
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins or ["*"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au démarrage ; libère au shutdown."""
    ckpt_path = os.environ.get("CKPT_PATH", "").strip()
    if not ckpt_path:
        raise RuntimeError("Environment variable CKPT_PATH is required.")

    device = _resolve_device()
    logger.info(f"Loading checkpoint {ckpt_path} on device={device}")

    model, input_repr, resolved_path = load_model(ckpt_path, device)

    _state.update(
        model=model,
        device=device,
        input_repr=input_repr,
        ckpt_path=resolved_path,
    )
    logger.info(
        f"Model ready | device={device} | input_repr={input_repr} | ckpt={resolved_path}"
    )

    yield

    # PyTorch libère la VRAM via le GC ; rien d'explicite requis.
    logger.info("Server shutting down.")


app = FastAPI(
    title="Filtre-Voix Denoising API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_resolve_cors_origins(),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    """Sonde de liveness. 503 si le modèle n'est pas chargé."""
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "status": "ok",
        "device": _state["device"],
        "ckpt": _state["ckpt_path"],
        "input_repr": _state["input_repr"],
    }


@app.post("/denoise")
async def denoise_endpoint(file: UploadFile = File(...)) -> StreamingResponse:
    """Débruite le fichier audio envoyé et renvoie un WAV (16 kHz mono PCM16)."""
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if file.content_type and file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported content type: {file.content_type}. "
                f"Expected an audio file (WAV/FLAC/MP3/OGG)."
            ),
        )

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        wav_bytes = denoise_bytes(
            audio_bytes,
            _state["model"],
            _state["device"],
            _state["input_repr"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    return StreamingResponse(
        iter([wav_bytes]),
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="denoised.wav"'},
    )
