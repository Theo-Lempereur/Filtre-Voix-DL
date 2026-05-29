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


# ---------------------------------------------------------------------------
# Warmup linéaire optionnel — wrapping du scheduler principal
# ---------------------------------------------------------------------------

class WarmupWrapper:
    """Warmup linéaire du LR sur N epochs, puis délègue au scheduler interne.

    Pendant les `warmup_epochs` premières epochs, le LR monte linéairement de
    `warmup_start_factor * base_lr` jusqu'à `base_lr`. Ensuite, on bascule sur
    le scheduler enveloppé (typiquement `DualMetricPlateauScheduler`).

    Si `warmup_epochs == 0`, le wrapper est totalement transparent : il forward
    chaque appel à l'inner sans rien faire. C'est le comportement par défaut
    pour préserver la rétrocompatibilité.

    API conservée :
      - `step(val_loss, val_si_sdri)` → bool (True si LR vient de baisser)
      - `state_dict()` / `load_state_dict()` (compose avec celui de l'inner)
      - `get_last_lr()`
    """

    def __init__(self,
                 inner: "DualMetricPlateauScheduler",
                 optimizer: torch.optim.Optimizer,
                 warmup_epochs: int = 0,
                 warmup_start_factor: float = 0.1):
        self.inner = inner
        self.optimizer = optimizer
        self.warmup_epochs = int(warmup_epochs)
        self.warmup_start_factor = float(warmup_start_factor)
        # On capture le LR de base pour pouvoir reconstruire la rampe.
        self._base_lrs: list[float] = [g["lr"] for g in optimizer.param_groups]
        self._current_epoch: int = 0
        # Si warmup activé, on positionne directement le LR de départ.
        if self.warmup_epochs > 0:
            self._apply_warmup_lr(epoch=0)

    def _apply_warmup_lr(self, epoch: int) -> None:
        # Ramp linéaire : f(0) = start_factor, f(warmup_epochs) = 1.0
        frac = (epoch + 1) / max(self.warmup_epochs, 1)
        factor = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * min(frac, 1.0)
        for g, base in zip(self.optimizer.param_groups, self._base_lrs):
            g["lr"] = base * factor

    def step(self, val_loss: float, val_si_sdri: float) -> bool:
        """Avance d'une epoch. Pendant le warmup, ne consulte pas l'inner.

        Retourne True si une réduction de LR vient d'être faite (inner uniquement).
        """
        self._current_epoch += 1
        if self._current_epoch <= self.warmup_epochs:
            self._apply_warmup_lr(self._current_epoch - 1)
            return False
        # Après warmup : l'inner prend la main.
        return self.inner.step(val_loss, val_si_sdri)

    def get_last_lr(self) -> list[float]:
        return [g["lr"] for g in self.optimizer.param_groups]

    @property
    def loss_plateau_epochs(self) -> int:
        return self.inner.loss_plateau_epochs

    @property
    def sdri_plateau_epochs(self) -> int:
        return self.inner.sdri_plateau_epochs

    @property
    def num_reductions(self) -> int:
        return self.inner.num_reductions

    def state_dict(self) -> dict[str, Any]:
        return {
            "version":         1,
            "warmup_epochs":   self.warmup_epochs,
            "current_epoch":   self._current_epoch,
            "base_lrs":        list(self._base_lrs),
            "inner":           self.inner.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        # Format wrapper : on reconnaît la présence de la clé "inner".
        if "inner" in state and isinstance(state["inner"], dict):
            self._current_epoch = int(state.get("current_epoch", 0))
            base_lrs = state.get("base_lrs")
            if isinstance(base_lrs, list) and base_lrs:
                self._base_lrs = [float(x) for x in base_lrs]
            self.inner.load_state_dict(state["inner"])
            return
        # Format ancien : c'est l'état direct de l'inner (avant que WarmupWrapper
        # existe). On le délègue tel quel à l'inner.
        self.inner.load_state_dict(state)
