"""Scheduler de learning rate avec critère hybride à deux métriques.

PyTorch fournit `torch.optim.lr_scheduler.ReduceLROnPlateau`, mais il ne
surveille qu'une seule métrique. Ici on veut ne réduire le LR **que si DEUX**
métriques plateauent simultanément :

  - `val_loss`     (à minimiser)  → "primary"
  - `val_si_sdri`  (à maximiser)  → "secondary"

Tant qu'au moins une des deux métriques progresse, le LR reste inchangé.
Quand les deux plateauent depuis `patience` epochs, on multiplie le LR
par `factor` et on remet les deux compteurs à 0.

API volontairement proche de `ReduceLROnPlateau` :
  - `step(val_loss, val_si_sdri)`            : à appeler à chaque epoch
  - `state_dict()` / `load_state_dict(...)`   : sauvegarde dans les checkpoints
  - `get_last_lr()`                           : LR courant des param_groups
"""
from __future__ import annotations

from typing import Any

import torch


class DualMetricPlateauScheduler:
    """ReduceLROnPlateau à deux métriques, avec logique "ET" pour le déclenchement.

    Parameters
    ----------
    optimizer :
        Optimizer dont le LR sera ajusté.
    patience :
        Nombre d'epochs sans amélioration (sur les DEUX métriques) avant
        de réduire le LR.
    factor :
        Multiplicateur appliqué au LR au déclenchement (0.1 = ÷10).
    min_lr :
        Plancher du LR ; au-delà, plus aucune réduction n'est appliquée.
    primary_min_delta :
        Amélioration minimale de `val_loss` pour réinitialiser son compteur.
    secondary_min_delta :
        Amélioration minimale de `val_si_sdri` (en dB) pour réinitialiser
        son compteur. Plus élevé que `primary_min_delta` car SI-SDRi est
        une métrique plus bruitée.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 patience: int = 3,
                 factor: float = 0.1,
                 min_lr: float = 0.0,
                 primary_min_delta: float = 1e-4,
                 secondary_min_delta: float = 1e-3):
        self.optimizer = optimizer
        self.patience = int(patience)
        self.factor = float(factor)
        self.min_lr = float(min_lr)
        self.primary_min_delta = float(primary_min_delta)
        self.secondary_min_delta = float(secondary_min_delta)

        # État interne
        self._best_loss: float = float("inf")
        self._best_sdri: float = float("-inf")
        self._loss_plateau: int = 0
        self._sdri_plateau: int = 0
        self._num_reductions: int = 0

    # ------------------------------------------------------------------ step

    def step(self, val_loss: float, val_si_sdri: float) -> bool:
        """Met à jour les compteurs et réduit le LR si les deux plateauent.

        Retourne `True` si une réduction de LR vient d'être appliquée à cet
        appel — utile pour logger un événement.
        """
        val_loss = float(val_loss)
        val_si_sdri = float(val_si_sdri)

        # --- Primary : val_loss (minimisée) ---
        if val_loss < self._best_loss - self.primary_min_delta:
            self._best_loss = val_loss
            self._loss_plateau = 0
        else:
            self._loss_plateau += 1

        # --- Secondary : val_si_sdri (maximisée) ---
        if val_si_sdri > self._best_sdri + self.secondary_min_delta:
            self._best_sdri = val_si_sdri
            self._sdri_plateau = 0
        else:
            self._sdri_plateau += 1

        # --- Logique ET : les deux doivent plateauer ---
        if (self._loss_plateau >= self.patience
                and self._sdri_plateau >= self.patience):
            self._reduce_lr()
            # Reset des deux compteurs pour repartir sur un cycle de mesure
            self._loss_plateau = 0
            self._sdri_plateau = 0
            self._num_reductions += 1
            return True
        return False

    def _reduce_lr(self) -> None:
        for group in self.optimizer.param_groups:
            new_lr = max(group["lr"] * self.factor, self.min_lr)
            group["lr"] = new_lr

    # ------------------------------------------------------------ accesseurs

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]

    @property
    def loss_plateau_epochs(self) -> int:
        return self._loss_plateau

    @property
    def sdri_plateau_epochs(self) -> int:
        return self._sdri_plateau

    @property
    def num_reductions(self) -> int:
        return self._num_reductions

    # ------------------------------------------------------------ persistance

    def state_dict(self) -> dict[str, Any]:
        """Sérialisable dans un checkpoint torch.

        On ne sauvegarde QUE l'état interne (compteurs + bests). Les
        hyperparamètres (`patience`, `factor`, etc.) viennent de la config
        et sont passés au constructeur à la reconstruction.
        """
        return {
            "version":        1,
            "best_loss":      self._best_loss,
            "best_sdri":      self._best_sdri,
            "loss_plateau":   self._loss_plateau,
            "sdri_plateau":   self._sdri_plateau,
            "num_reductions": self._num_reductions,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restaure l'état interne. Tolérant aux clés manquantes.

        Si `state` n'est pas reconnu (ex : ancien checkpoint qui stockait
        un `ReduceLROnPlateau` standard), on garde l'état actuel (frais)
        — pas d'exception, juste un comportement dégradé acceptable.
        """
        if not isinstance(state, dict) or "version" not in state:
            # Ancien format (ReduceLROnPlateau natif) : on ignore proprement.
            return
        self._best_loss      = float(state.get("best_loss", float("inf")))
        self._best_sdri      = float(state.get("best_sdri", float("-inf")))
        self._loss_plateau   = int(state.get("loss_plateau", 0))
        self._sdri_plateau   = int(state.get("sdri_plateau", 0))
        self._num_reductions = int(state.get("num_reductions", 0))
