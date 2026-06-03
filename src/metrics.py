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

Factory `build_loss(config)` (cf. bas du module) : retourne une closure qui
combine `MSE`, `L1(mag^0.3)` et `MR-STFT` selon `LOSS_WEIGHTS`. La closure
accepte un dict `batch` (clés : pred_mag, clean_mag, pred_wav, clean_wav)
et retourne un scalaire torch.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Callable, Iterable

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


def l1_compressed_loss(pred_mag: torch.Tensor, clean_mag: torch.Tensor,
                       power: float = 0.3, eps: float = 1e-6) -> torch.Tensor:
    """L1 sur magnitude compressée par `mag^power`.

    La MSE/L1 sur magnitude linéaire est dominée par les bins haute énergie
    (basses fréquences voisées). Compresser par `^0.3` (proche de la loi
    psychoacoustique de Stevens pour la sonie) ramène les bins faibles dans
    le budget de gradient → le réseau apprend à effacer le bruit résiduel
    basse-énergie qu'on entend pourtant.

    pred_mag, clean_mag : (B, 1, F, T), valeurs >= 0.
    Sortie : scalaire torch.
    """
    # La dérivée de x^p (p<1) tend vers l'infini en x=0. Un simple clamp(min=0)
    # ne suffit PAS : il laisse passer x=0 exact (cas réel quand un bin de
    # magnitude est nul — ex. frame de silence, accentué par le cache float16),
    # et le backward produit alors un gradient inf -> NaN. On ajoute donc un
    # plancher eps à la base pour garantir une dérivée finie partout.
    pred_c  = (torch.clamp(pred_mag,  min=0.0) + eps).pow(power)
    clean_c = (torch.clamp(clean_mag, min=0.0) + eps).pow(power)
    return F.l1_loss(pred_c, clean_c)


def mr_stft_loss(pred_wav: torch.Tensor, clean_wav: torch.Tensor,
                 fft_sizes: Iterable[int] = (256, 512, 1024),
                 eps: float = 1e-7) -> torch.Tensor:
    """Multi-Resolution STFT loss : moyenne L1 de log-magnitudes sur plusieurs n_fft.

    Standard moderne en débruitage / vocoding (Yamamoto et al., Yang et al.).
    En sommant des STFT à différentes résolutions, on capture à la fois la
    structure fine fréquentielle (n_fft grand) et la structure temporelle
    rapide (n_fft petit). Cible directement les artefacts spectraux audibles.

    pred_wav, clean_wav : (B, T) ou (B, 1, T) — waveform.
    Sortie : scalaire torch.
    """
    if pred_wav.dim() == 3:
        pred_wav = pred_wav.squeeze(1)
    if clean_wav.dim() == 3:
        clean_wav = clean_wav.squeeze(1)

    total = pred_wav.new_zeros(())
    n_scales = 0
    for n_fft in fft_sizes:
        hop = max(n_fft // 4, 1)
        window = torch.hann_window(n_fft, device=pred_wav.device, dtype=pred_wav.dtype)
        # return_complex=True : moderne et sans warning depuis torch >= 1.8.
        pred_spec  = torch.stft(pred_wav,  n_fft=n_fft, hop_length=hop,
                                window=window, return_complex=True)
        clean_spec = torch.stft(clean_wav, n_fft=n_fft, hop_length=hop,
                                window=window, return_complex=True)
        pred_mag   = pred_spec.abs()
        clean_mag  = clean_spec.abs()
        # log-magnitude pour cibler aussi les composantes faibles
        l_log = F.l1_loss(torch.log(pred_mag + eps), torch.log(clean_mag + eps))
        # convergence (magnitude L1) — composante standard de MR-STFT
        l_mag = F.l1_loss(pred_mag, clean_mag)
        total = total + l_log + l_mag
        n_scales += 1
    return total / max(n_scales, 1)


def si_sdr_wav(pred_wav: torch.Tensor, clean_wav: torch.Tensor,
               eps: float = 1e-8) -> torch.Tensor:
    """SI-SDR (en dB) sur signal temporel — métrique de validation.

    Contrairement à `si_sdr_mag`, calcule la VRAIE SI-SDR sur waveform. Loggué
    en validation quand `pred_wav` est disponible (i.e. dès que la loss
    MR-STFT est active, l'ISTFT est déjà calculée).
    """
    if pred_wav.dim() == 3:
        pred_wav = pred_wav.squeeze(1)
    if clean_wav.dim() == 3:
        clean_wav = clean_wav.squeeze(1)
    B = pred_wav.shape[0]
    pred   = pred_wav.reshape(B, -1)
    target = clean_wav.reshape(B, -1)
    dot           = (pred * target).sum(dim=1, keepdim=True)
    target_energy = (target * target).sum(dim=1, keepdim=True) + eps
    s_target = (dot / target_energy) * target
    e_noise  = pred - s_target
    num = (s_target * s_target).sum(dim=1) + eps
    den = (e_noise  * e_noise ).sum(dim=1) + eps
    return (10.0 * torch.log10(num / den)).mean()


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


# ---------------------------------------------------------------------------
# Factory de loss combinée
# ---------------------------------------------------------------------------

# Convention pour le batch d'entrée :
#   batch["pred_mag"]   : (B, 1, F, T)  — magnitude prédite (toujours présent)
#   batch["clean_mag"]  : (B, 1, F, T)  — magnitude propre cible
#   batch["pred_wav"]   : (B, T)        — waveform prédit (si mr_stft active)
#   batch["clean_wav"]  : (B, T)        — waveform propre cible

LossFn = Callable[[dict[str, Any]], torch.Tensor]


def build_loss(config: dict) -> LossFn:
    """Construit la closure de loss combinée à partir de la config.

    La loss totale est une somme pondérée :
        L = w_mse · MSE(pred_mag, clean_mag)
          + w_l1c · L1(pred_mag^0.3, clean_mag^0.3)
          + w_mr  · MR-STFT(pred_wav, clean_wav)

    Les poids sont lus dans `config["loss_weights"]` (clés : "mse", "l1_comp",
    "mr_stft"). Une composante à poids 0 est skippée (économie de compute).
    Si tous les poids sont nuls, on retombe sur la MSE seule par sécurité.

    Le retour est une closure `(batch) -> scalar` que train.py appelle après
    avoir préparé le dict (forward + éventuelle ISTFT pour MR-STFT).
    """
    weights   = dict(config.get("loss_weights", {"l1_comp": 1.0, "mr_stft": 1.0}))
    fft_sizes = tuple(config.get("mr_stft_fft_sizes", (256, 512, 1024)))

    def _mse(batch):    return magnitude_mse_loss(batch["pred_mag"], batch["clean_mag"])
    def _l1c(batch):    return l1_compressed_loss(batch["pred_mag"], batch["clean_mag"])
    def _mrstft(batch):
        if batch.get("pred_wav") is None or batch.get("clean_wav") is None:
            raise ValueError(
                "mr_stft loss requiert batch['pred_wav'] et batch['clean_wav']. "
                "Vérifie que le forward fournit les waveforms (loss_requires_waveform)."
            )
        return mr_stft_loss(batch["pred_wav"], batch["clean_wav"], fft_sizes=fft_sizes)

    funcs = {"mse": _mse, "l1_comp": _l1c, "mr_stft": _mrstft}
    active = [(name, float(w)) for name, w in weights.items()
              if name in funcs and w and w > 0]
    if not active:
        # Sécurité : aucun poids actif → on retombe sur MSE pour ne pas planter.
        return _mse

    def _combo(batch):
        total = None
        for name, w in active:
            term = funcs[name](batch) * w
            total = term if total is None else (total + term)
        return total
    return _combo


def loss_requires_waveform(config: dict) -> bool:
    """True si la loss configurée a besoin des waveforms (pred/clean).

    train.py utilise ce flag pour décider s'il doit faire l'ISTFT pendant
    le forward — coûteuse, donc évitée si pas nécessaire. Avec la baseline
    p2 (mr_stft > 0), c'est True.
    """
    weights = dict(config.get("loss_weights", {}))
    w = weights.get("mr_stft", 0)
    return bool(w and w > 0)
