"""Inférence HTTP — fine couche au-dessus de ``src.denoiser.Wallace``.

Ce module conserve l'API publique historique (``load_model`` /
``denoise_bytes`` / ``resolve_ckpt_path``) attendue par ``serve/server.py``,
mais **délègue tout** à la « boîte » unique ``src.denoiser.Wallace`` :

  - chargement du checkpoint avec **détection du mode** (mask ↔ complex) — ce qui
    corrige l'incompatibilité de l'ancienne implémentation avec les modèles
    complex spectral mapping (qui construisait un U-Net 1 canal en dur) ;
  - **contrat d'entrée** mono 16 kHz ;
  - **traitement des sons > 4 s** (fenêtrage 4 s + overlap-add 50 %), là où
    l'ancienne version rognait brutalement à 4 s (`fix_length(center)`).

Le « modèle » manipulé par ``server.py`` est désormais une instance de
``Wallace`` ; ``server.py`` la traite de façon opaque (la passe à
``denoise_bytes``, teste seulement ``is None``), donc aucune modification de
``server.py`` n'est nécessaire.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Garantit que `src/` est importable quelle que soit la façon dont uvicorn est
# lancé (depuis la racine du repo, via systemd avec un autre CWD, etc.).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.denoiser import Wallace, resolve_ckpt_path  # noqa: E402,F401  (re-export)


def load_model(ckpt_path: str, device: str) -> tuple[Wallace, str, str]:
    """Charge un checkpoint dans une boîte ``Wallace``.

    Renvoie ``(denoiser, input_repr, resolved_path)`` — signature conservée pour
    ``serve/server.py``. Le premier élément (le « modèle ») est désormais un
    ``Wallace`` qui encapsule modèle + pipeline.

    Raises
    ------
    FileNotFoundError
        Si aucun checkpoint exploitable n'est trouvé.
    RuntimeError
        Si l'architecture du checkpoint ne correspond pas (load_state_dict strict).
    """
    box = Wallace(ckpt_path, device=device)
    return box, box.input_repr, box.ckpt_path


def denoise_bytes(
    audio_bytes: bytes,
    model: Wallace,
    device: str | None = None,
    input_repr: str | None = None,
) -> bytes:
    """Pipeline complet de débruitage en mémoire : ``bytes`` -> ``bytes``.

    Accepte tout format décodable par librosa (WAV/FLAC nativement, MP3/OGG si
    ffmpeg est installé). Renvoie toujours un WAV mono 16 kHz PCM 16 bits, de
    **durée quelconque** (plus de troncature à 4 s).

    ``device`` et ``input_repr`` sont ignorés (portés par le ``Wallace``) ; ils
    restent dans la signature pour la compatibilité avec ``serve/server.py``.

    Raises
    ------
    ValueError
        Si l'audio ne peut pas être décodé, est vide / trop court, ou si la
        sortie du modèle contient des NaN/Inf.
    """
    return model.denoise_bytes(audio_bytes)
