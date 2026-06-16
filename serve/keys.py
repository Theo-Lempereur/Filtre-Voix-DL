"""Résolution de la clé API et de la stratégie d'usage.

Règle métier (demandée) :
    - L'utilisateur fournit SA clé NVIDIA  -> on l'utilise, AUCUNE limite de
      notre côté (NVIDIA applique la sienne).
    - L'utilisateur ne fournit RIEN        -> on utilise NOTRE clé partagée
      (si elle est configurée), avec une limite journalière par IP.
    - Pas de clé du tout (ni la sienne, ni la nôtre) -> on refuse proprement.

Aucune vraie valeur de clé ne vit dans le code ni dans le dépôt : la clé
partagée est lue depuis la variable d'environnement ``NVIDIA_API_KEY`` (placée
dans le ``.env`` du serveur, ignoré par git). La clé utilisateur, elle, arrive
par l'en-tête HTTP ``X-API-Key`` et n'est jamais journalisée ni stockée.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class KeySource(str, Enum):
    USER = "user"      # clé fournie par l'utilisateur -> illimité
    SHARED = "shared"  # notre clé partagée -> quota journalier


@dataclass(frozen=True)
class KeyDecision:
    api_key: str
    source: KeySource
    # True quand on doit faire respecter le quota journalier (mode partagé).
    enforce_quota: bool


def shared_key() -> str | None:
    """Notre clé NVIDIA partagée, depuis l'environnement. None si absente."""
    val = os.environ.get("NVIDIA_API_KEY", "").strip()
    return val or None


def looks_like_nvidia_key(value: str | None) -> bool:
    """Validation de forme bon marché (ne consomme aucun crédit).

    Les clés NVIDIA NIM / NGC sont préfixées ``nvapi-``. La validation
    *autoritaire* se fait au premier vrai appel gRPC (un code UNAUTHENTICATED
    est remonté en 401 par le connecteur). On évite ainsi de brûler un crédit
    juste pour tester une clé.
    """
    if not value:
        return False
    return value.strip().startswith("nvapi-")


class NoKeyAvailable(Exception):
    """Ni clé utilisateur valide, ni clé partagée configurée."""


class InvalidUserKey(Exception):
    """L'utilisateur a fourni une clé, mais elle n'a pas le bon format."""


def resolve_key(user_key: str | None) -> KeyDecision:
    """Choisit la clé à utiliser et le régime (illimité vs quota).

    Raises
    ------
    InvalidUserKey  : clé utilisateur fournie mais malformée.
    NoKeyAvailable  : aucune clé utilisable.
    """
    user_key = (user_key or "").strip()

    if user_key:
        if not looks_like_nvidia_key(user_key):
            raise InvalidUserKey(
                "La clé fournie ne ressemble pas à une clé NVIDIA "
                "(préfixe attendu : 'nvapi-')."
            )
        return KeyDecision(api_key=user_key, source=KeySource.USER, enforce_quota=False)

    shared = shared_key()
    if shared:
        return KeyDecision(api_key=shared, source=KeySource.SHARED, enforce_quota=True)

    raise NoKeyAvailable(
        "Aucune clé disponible : ce modèle nécessite une clé NVIDIA. "
        "Renseignez la vôtre, ou contactez l'administrateur du service."
    )


def client_ip(forwarded_for: str | None, peer: str | None) -> str:
    """IP client pour le comptage de quota.

    Derrière nginx (cf. INTEGRATION.md), l'en-tête ``X-Forwarded-For`` contient
    l'IP réelle en première position. En accès direct, on retombe sur l'IP du
    pair TCP. Note sécurité : ``X-Forwarded-For`` n'est fiable que DERRIÈRE un
    proxy de confiance ; n'exposez pas l'API uvicorn directement à Internet.
    """
    if forwarded_for:
        first = forwarded_for.split(",")[0].strip()
        if first:
            return first
    return peer or "unknown"
