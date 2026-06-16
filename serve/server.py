"""API FastAPI pour servir le modèle de débruitage voix.

Charge un checkpoint au démarrage si possible, liste les modèles disponibles,
et expose les endpoints utiles au mini site :

    GET  /          -> mini site de test
    GET  /health   -> état du serveur + infos checkpoint
    GET  /models   -> checkpoints disponibles
    POST /denoise  -> reçoit un fichier audio + modèle, renvoie le WAV débruité

Lancement (depuis la racine du repo, venv activé) :

    CKPT_PATH="/chemin/absolu/checkpoints/mon_run/best.pt" \\
    DEVICE="cpu" \\
    CORS_ORIGINS="https://monsite.com" \\
    uvicorn serve.server:app --host 127.0.0.1 --port 8000 --workers 1

Variables d'environnement :
    CKPT_PATH     (option)  chemin absolu vers le .pt ou dossier de run par défaut
    DEVICE        (option)  "cuda" / "cpu" ; auto-détecté si absent
    CORS_ORIGINS  (option)  origines autorisées, séparées par virgule (défaut "*")
    MODEL_DIRS    (option)  dossiers de checkpoints séparés par point-virgule
"""
from __future__ import annotations

import hashlib
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import RLock

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from serve import keys, quota
from serve.inference import denoise_bytes, load_model, resolve_ckpt_path
from serve.providers import nvidia_maxine

# Charge le .env à la racine du repo (CKPT_PATH, DEVICE, CORS_ORIGINS).
# Les variables déjà définies dans l'environnement ont priorité (override=False).
REPO_ROOT = Path(__file__).resolve().parents[1]
SERVE_DIR = Path(__file__).resolve().parent
load_dotenv(REPO_ROOT / ".env")


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
_state_lock = RLock()


# Content-types acceptés (filtre indicatif côté requête ; la vraie validation se
# fait au décodage dans denoise_bytes).
_ALLOWED_CONTENT_TYPES = {
    "audio/wav", "audio/wave", "audio/x-wav",
    "audio/mpeg", "audio/mp3",
    "audio/flac", "audio/x-flac",
    "audio/ogg", "audio/x-ogg",
    "application/octet-stream",  # certains navigateurs n'envoient pas de type
}

_CHECKPOINT_PRIORITY = {
    "best_si_sdri.pt": 0,
    "best.pt": 1,
    "last.pt": 2,
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


def _model_search_roots() -> list[Path]:
    """Dossiers scannés pour alimenter la liste déroulante du mini site."""
    roots: list[Path] = []

    raw_dirs = os.environ.get("MODEL_DIRS", "").strip()
    if raw_dirs:
        roots.extend(Path(part.strip()) for part in raw_dirs.split(";") if part.strip())

    roots.extend([
        REPO_ROOT / "checkpoints_runpod",
        REPO_ROOT / "checkpoints",
    ])

    ckpt_path = os.environ.get("CKPT_PATH", "").strip()
    if ckpt_path:
        p = Path(ckpt_path)
        try:
            roots.append(p if p.is_dir() else p.parent)
        except OSError:
            roots.append(p.parent)

    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            key = str(root.resolve(strict=False)).lower()
        except OSError:
            continue
        if key not in seen:
            seen.add(key)
            deduped.append(root)
    return deduped


def _checkpoint_sort_key(path: Path) -> tuple:
    name = path.name.lower()
    priority = _CHECKPOINT_PRIORITY.get(name, 3)
    epoch_rank = 0
    if name.startswith("epoch_") and name.endswith(".pt"):
        try:
            epoch_rank = -int(name.removeprefix("epoch_").removesuffix(".pt"))
        except ValueError:
            epoch_rank = 0
    return (path.parent.name.lower(), priority, epoch_rank, name)


def _model_id(path: Path) -> str:
    resolved = str(path.resolve(strict=False)).replace("\\", "/")
    return hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:16]


def _model_label(path: Path) -> str:
    try:
        rel = path.relative_to(REPO_ROOT)
    except ValueError:
        rel = path
    if path.parent.name:
        return f"{path.parent.name} / {path.name}"
    return str(rel)


def _list_models() -> list[dict]:
    candidates: dict[str, Path] = {}
    for root in _model_search_roots():
        try:
            if root.is_file() and root.suffix == ".pt":
                paths = [root]
            elif root.is_dir():
                paths = list(root.rglob("*.pt"))
            else:
                paths = []
        except OSError:
            paths = []

        for path in paths:
            try:
                resolved = path.resolve(strict=True)
            except OSError:
                continue
            candidates[str(resolved).lower()] = resolved

    with _state_lock:
        loaded_path = _state.get("ckpt_path")

    models: list[dict] = []
    for path in sorted(candidates.values(), key=_checkpoint_sort_key):
        try:
            stat = path.stat()
        except OSError:
            continue
        models.append({
            "id": _model_id(path),
            "label": _model_label(path),
            "path": str(path),
            "size_bytes": stat.st_size,
            "modified": stat.st_mtime,
            "is_loaded": _same_path(loaded_path, str(path)) if loaded_path else False,
        })
    return models


def _same_path(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    try:
        return Path(left).resolve(strict=False) == Path(right).resolve(strict=False)
    except OSError:
        return left == right


def _default_model_path() -> str | None:
    ckpt_path = os.environ.get("CKPT_PATH", "").strip()
    if ckpt_path:
        try:
            return resolve_ckpt_path(ckpt_path)
        except (FileNotFoundError, OSError):
            logger.warning("CKPT_PATH introuvable, fallback sur les checkpoints scannés.")

    models = _list_models()
    return models[0]["path"] if models else None


def _resolve_model_selection(model_id: str) -> str:
    for model_info in _list_models():
        if model_info["id"] == model_id:
            return model_info["path"]
    raise HTTPException(status_code=400, detail="Modèle introuvable.")


def _activate_model(ckpt_path: str, device: str) -> None:
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


def _get_model_snapshot(model_id: str | None) -> tuple:
    """Charge le modèle demandé si besoin, puis renvoie une copie stable."""
    with _state_lock:
        device = _state.get("device") or _resolve_device()

        if model_id:
            selected_path = _resolve_model_selection(model_id)
            if not _same_path(_state.get("ckpt_path"), selected_path):
                _activate_model(selected_path, device)
        elif _state["model"] is None:
            selected_path = _default_model_path()
            if selected_path is None:
                raise HTTPException(status_code=503, detail="Aucun modèle .pt trouvé.")
            _activate_model(selected_path, device)

        if _state["model"] is None:
            raise HTTPException(status_code=503, detail="Model not loaded.")

        return (
            _state["model"],
            _state["device"],
            _state["input_repr"],
            _state["ckpt_path"],
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge un modèle par défaut si un checkpoint est disponible."""
    device = _resolve_device()
    _state["device"] = device
    ckpt_path = _default_model_path()
    if ckpt_path:
        try:
            with _state_lock:
                _activate_model(ckpt_path, device)
        except Exception:  # noqa: BLE001
            logger.exception("Impossible de charger le modèle initial.")
    else:
        logger.warning("Aucun checkpoint .pt trouvé au démarrage.")

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


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """Mini site de test."""
    return FileResponse(SERVE_DIR / "test_client.html")


@app.get("/health")
async def health() -> dict:
    """Sonde de liveness. 503 si le modèle n'est pas chargé."""
    with _state_lock:
        if _state["model"] is None:
            raise HTTPException(status_code=503, detail="Model not loaded.")
        device = _state["device"]
        ckpt_path = _state["ckpt_path"]
        input_repr = _state["input_repr"]
    return {
        "status": "ok",
        "device": device,
        "ckpt": ckpt_path,
        "input_repr": input_repr,
    }


@app.get("/models")
async def models() -> dict:
    """Liste les checkpoints locaux + les modèles NVIDIA pour la liste déroulante."""
    return {"models": _list_models() + nvidia_maxine.list_models()}


def _request_ip(request: Request) -> str:
    peer = request.client.host if request.client else None
    return keys.client_ip(request.headers.get("x-forwarded-for"), peer)


@app.get("/quota")
async def quota_status(request: Request) -> dict:
    """État du quota gratuit pour l'IP appelante (mode clé partagée)."""
    ip = _request_ip(request)
    return {
        "limit": quota.free_daily_limit(),
        "used": quota.get_count(ip),
        "remaining": quota.remaining(ip),
        "shared_key_configured": keys.shared_key() is not None,
    }


@app.post("/denoise")
async def denoise_endpoint(
    request: Request,
    file: UploadFile = File(...),
    model_id: str | None = Form(None),
    intensity: float | None = Form(None),
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> StreamingResponse:
    """Débruite le fichier audio envoyé et renvoie un WAV.

    - Modèle local : comportement historique (16 kHz mono PCM16), sans clé.
    - Modèle NVIDIA : nécessite une clé. Si l'utilisateur fournit la sienne via
      l'en-tête ``X-API-Key`` -> illimité ; sinon on utilise notre clé partagée
      avec une limite journalière par IP.
    """
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

    # --- Branche NVIDIA (modèles hébergés, clé requise) ----------------------
    if nvidia_maxine.is_nvidia_model(model_id):
        try:
            decision = keys.resolve_key(x_api_key)
        except keys.InvalidUserKey as exc:
            raise HTTPException(status_code=401, detail=str(exc))
        except keys.NoKeyAvailable as exc:
            raise HTTPException(status_code=503, detail=str(exc))

        ip = _request_ip(request)
        if decision.enforce_quota and not quota.has_quota(ip):
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Limite gratuite atteinte ({quota.free_daily_limit()}/jour). "
                    f"Réessayez demain, ou utilisez votre propre clé NVIDIA pour "
                    f"un accès illimité."
                ),
            )

        try:
            wav_bytes = nvidia_maxine.enhance(
                model_id, audio_bytes, decision.api_key, intensity
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except nvidia_maxine.NvidiaAuthError as exc:
            raise HTTPException(status_code=401, detail=str(exc))
        except nvidia_maxine.NvidiaUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except nvidia_maxine.NvidiaError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

        headers = {"Content-Disposition": 'attachment; filename="denoised.wav"'}
        if decision.enforce_quota:
            quota.increment(ip)  # on ne décompte qu'en cas de succès
            headers["X-Quota-Remaining"] = str(quota.remaining(ip))
        return StreamingResponse(iter([wav_bytes]), media_type="audio/wav", headers=headers)

    # --- Branche locale (checkpoints .pt, inchangée) -------------------------
    try:
        model, device, input_repr, _ckpt_path = _get_model_snapshot(model_id)
        wav_bytes = denoise_bytes(
            audio_bytes,
            model,
            device,
            input_repr,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    return StreamingResponse(
        iter([wav_bytes]),
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="denoised.wav"'},
    )
