"""Métriques d'entraînement, de validation et indicateurs d'overfitting.

Ce module centralise tout ce qui est numérique pour qu'un même calcul soit
utilisé partout (train.py, notebooks d'analyse, futur evaluate.py).

Trois familles :

1. **Loss** — pour la rétropropagation pendant l'entraînement.
2. **Métriques d'écoute** — pour interpréter la qualité, exprimées en dB :
   plus c'est haut, mieux c'est. Calculées sur la magnitude pour rester
   rapides ; le SI-SDR strict sur waveform sera dans `evaluate.py` (semaine 3).
3. **OverfitMonitor** — accumule (train_loss, val_loss) au fil des epochs et
   en déduit des drapeaux interprétables + un signal d'early stopping.
"""
from __future__ import annotations

from collections import deque
from typing import Iterable

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def magnitude_mse_loss(pred_mag: torch.Tensor, clean_mag: torch.Tensor) -> torch.Tensor:
    """MSE entre la magnitude prédite et la magnitude propre cible.

    pred_mag, clean_mag : (B, 1, F, T), valeurs >= 0.
    Sortie : scalaire torch (moyennée sur tous les éléments du batch).
    """
    return F.mse_loss(pred_mag, clean_mag)


# ---------------------------------------------------------------------------
# Métriques d'écoute (approximations sur magnitude)
# ---------------------------------------------------------------------------

def si_sdr_mag(pred_mag: torch.Tensor, clean_mag: torch.Tensor,
               eps: float = 1e-8) -> torch.Tensor:
    """SI-SDR (Scale-Invariant Signal-to-Distortion Ratio) approximé sur magnitude.

    Définition standard du SI-SDR (en dB) :
        s_target = <pred, target> / ||target||^2 * target
        e_noise  = pred - s_target
        SI-SDR   = 10 * log10( ||s_target||^2 / ||e_noise||^2 )

    On l'applique ici aux magnitudes (et non aux waveforms) — c'est un proxy
    rapide pour piloter l'entraînement. Le SI-SDR strict (sur signal temporel
    reconstruit) sera calculé dans `evaluate.py` à la semaine 3.

    Retourne : scalaire torch en dB, moyenné sur le batch.
    """
    # On aplatit chaque exemple en un vecteur 1D et on calcule le SI-SDR par
    # exemple, puis on moyenne.
    B = pred_mag.shape[0]
    pred = pred_mag.reshape(B, -1)
    target = clean_mag.reshape(B, -1)

    dot = (pred * target).sum(dim=1, keepdim=True)
    target_energy = (target * target).sum(dim=1, keepdim=True) + eps
    s_target = (dot / target_energy) * target
    e_noise  = pred - s_target

    num = (s_target * s_target).sum(dim=1) + eps
    den = (e_noise * e_noise).sum(dim=1) + eps
    si_sdr = 10.0 * torch.log10(num / den)
    return si_sdr.mean()


def mask_statistics(pred_mag: torch.Tensor, noisy_mag: torch.Tensor,
                    eps: float = 1e-8) -> dict[str, float]:
    """Statistiques du masque effectif `pred_mag / noisy_mag`.

    Diagnostic visuel important :
    - Si la moyenne tend vers 1 : le réseau "ne fait rien" (laisse tout passer).
    - Si la moyenne tend vers 0 : il coupe tout (sortie quasi muette).
    - Si l'écart-type est très faible : le masque est uniforme — le réseau
      n'a rien appris de discriminant.

    Retourne un dict de floats Python (déjà détachés du graphe).
    """
    with torch.no_grad():
        mask = pred_mag / (noisy_mag + eps)
        mask = mask.clamp(0.0, 1.0)
        return {
            "mask_mean": float(mask.mean().item()),
            "mask_std":  float(mask.std().item()),
            "mask_min":  float(mask.min().item()),
            "mask_max":  float(mask.max().item()),
        }


# ---------------------------------------------------------------------------
# Suivi de l'overfitting
# ---------------------------------------------------------------------------

class OverfitMonitor:
    """Accumule (train_loss, val_loss) et expose des drapeaux interprétables.

    Utilisation type dans la boucle d'entraînement :

        monitor = OverfitMonitor(patience=7)
        for epoch in range(...):
            ...
            monitor.update(epoch, train_loss, val_loss)
            flags = monitor.flags()
            logger.log(epoch, {**flags, ...})
            if monitor.should_early_stop():
                break

    Les drapeaux retournés sont volontairement simples à interpréter et à
    afficher dans un tableau (notebook 05).
    """

    def __init__(self, patience: int = 7, smooth_window: int = 3,
                 min_delta: float = 1e-4,
                 secondary_min_delta: float = 1e-3):
        """
        Args:
            patience            : nb d'epochs sans amélioration de val avant early stop.
            smooth_window       : fenêtre pour le lissage du gap (réduit le bruit).
            min_delta           : amélioration minimale pour réinitialiser `patience`
                                  sur la métrique primaire (val_loss).
            secondary_min_delta : idem pour la métrique secondaire (par défaut
                                  val_si_sdri, en dB). Plus tolérant que `min_delta`
                                  car le SI-SDR est plus bruité.
        """
        self.patience = patience
        self.smooth_window = smooth_window
        self.min_delta = min_delta
        self.secondary_min_delta = secondary_min_delta

        self.train_losses: list[float] = []
        self.val_losses:   list[float] = []
        self.epochs:       list[int]   = []
        self._best_val: float = float("inf")
        self._epochs_since_improve: int = 0

        # Métrique secondaire (optionnelle). Tant que `update_secondary` n'a
        # pas été appelée, `_secondary_used` reste False et `should_early_stop`
        # retombe sur la logique val_loss seule (rétrocompatibilité).
        self._best_secondary: float | None = None
        self._secondary_higher_is_better: bool = True
        self._epochs_since_secondary_improve: int = 0
        self._secondary_used: bool = False

    # ---- mise à jour ----

    def update(self, epoch: int, train_loss: float, val_loss: float) -> None:
        self.epochs.append(int(epoch))
        self.train_losses.append(float(train_loss))
        self.val_losses.append(float(val_loss))

        if val_loss < self._best_val - self.min_delta:
            self._best_val = float(val_loss)
            self._epochs_since_improve = 0
        else:
            self._epochs_since_improve += 1

    def update_secondary(self, epoch: int, value: float,
                         higher_is_better: bool = True) -> None:
        """Met à jour le compteur de plateau d'une métrique secondaire.

        Le premier appel initialise `best_secondary` à `value` (peu importe
        `higher_is_better`). Les appels suivants comparent avec
        `secondary_min_delta` pour décider si on a "amélioré".

        Tant qu'on n'appelle pas cette méthode, `should_early_stop()` reste
        sur la logique val_loss seule.
        """
        self._secondary_used = True
        self._secondary_higher_is_better = bool(higher_is_better)
        value = float(value)

        if self._best_secondary is None:
            self._best_secondary = value
            self._epochs_since_secondary_improve = 0
            return

        if higher_is_better:
            improved = value > self._best_secondary + self.secondary_min_delta
        else:
            improved = value < self._best_secondary - self.secondary_min_delta

        if improved:
            self._best_secondary = value
            self._epochs_since_secondary_improve = 0
        else:
            self._epochs_since_secondary_improve += 1

    # ---- introspection ----

    @property
    def best_val_loss(self) -> float:
        return self._best_val

    @property
    def epochs_since_improve(self) -> int:
        return self._epochs_since_improve

    @property
    def best_secondary(self) -> float | None:
        return self._best_secondary

    @property
    def epochs_since_secondary_improve(self) -> int:
        return self._epochs_since_secondary_improve

    @property
    def secondary_used(self) -> bool:
        return self._secondary_used

    def _smoothed(self, values: list[float]) -> float:
        """Moyenne des `smooth_window` dernières valeurs (ou de tout, si moins)."""
        if not values:
            return float("nan")
        w = min(self.smooth_window, len(values))
        return sum(values[-w:]) / w

    def flags(self) -> dict:
        """Drapeaux d'overfitting (utilisables tels quels pour logger / afficher).

        - gap              : val_loss - train_loss lissé sur `smooth_window`.
        - gap_ratio        : val_loss / train_loss lissé. >> 1 → overfitting.
        - val_plateau      : nb d'epochs sans amélioration significative de val.
        - val_increasing   : True si val_loss monte de façon monotone sur les
                             `smooth_window` dernières epochs.
        - diverging        : True si train descend mais val monte → overfit franc.
        """
        if len(self.val_losses) == 0:
            return {
                "gap": float("nan"),
                "gap_ratio": float("nan"),
                "val_plateau_epochs": 0,
                "secondary_plateau_epochs": int(self._epochs_since_secondary_improve),
                "val_increasing": False,
                "diverging": False,
            }

        train_s = self._smoothed(self.train_losses)
        val_s   = self._smoothed(self.val_losses)
        gap = val_s - train_s
        gap_ratio = val_s / train_s if train_s > 0 else float("inf")

        val_inc = _is_increasing(self.val_losses, self.smooth_window)
        train_dec = _is_decreasing(self.train_losses, self.smooth_window)
        diverging = bool(val_inc and train_dec)

        return {
            "gap":                      float(gap),
            "gap_ratio":                float(gap_ratio),
            "val_plateau_epochs":      int(self._epochs_since_improve),
            "secondary_plateau_epochs": int(self._epochs_since_secondary_improve),
            "val_increasing":          bool(val_inc),
            "diverging":               bool(diverging),
        }

    def should_early_stop(self) -> bool:
        """True si les conditions d'arrêt sont remplies.

        - Si aucune métrique secondaire n'a été enregistrée : logique
          historique (val_loss seul).
        - Sinon : logique "ET" — les deux métriques doivent plateauer
          depuis au moins `patience` epochs.
        """
        primary_plateau = self._epochs_since_improve >= self.patience
        if not self._secondary_used:
            return primary_plateau
        secondary_plateau = self._epochs_since_secondary_improve >= self.patience
        return primary_plateau and secondary_plateau


# ---------------------------------------------------------------------------
# Helpers privés
# ---------------------------------------------------------------------------

def _is_increasing(values: Iterable[float], window: int) -> bool:
    """True si les `window` dernières valeurs sont strictement croissantes."""
    vals = list(values)
    if len(vals) < window or window < 2:
        return False
    tail = vals[-window:]
    return all(tail[i] < tail[i + 1] for i in range(len(tail) - 1))


def _is_decreasing(values: Iterable[float], window: int) -> bool:
    """True si les `window` dernières valeurs sont strictement décroissantes."""
    vals = list(values)
    if len(vals) < window or window < 2:
        return False
    tail = vals[-window:]
    return all(tail[i] > tail[i + 1] for i in range(len(tail) - 1))
