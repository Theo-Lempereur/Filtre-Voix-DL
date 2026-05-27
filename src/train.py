"""Boucle d'entraînement du U-Net de débruitage.

Orchestration uniquement — la logique vit dans les modules dédiés :
    - src.model        : architecture du réseau
    - src.dataset      : PairedAudioDataset
    - src.checkpoint   : save / load / rotation
    - src.metrics      : loss, SI-SDR, OverfitMonitor
    - src.logging_utils: Logger (wandb + JSONL)
    - src.config       : hyperparamètres et chemins

API publique :
    train(config, resume_from=None, max_train_samples=None, max_val_samples=None)

Le dict `config` accepte (et complète depuis `src.config` si absent) :
    run_id          : identifiant unique du run (obligatoire en pratique)
    lr, weight_decay, batch_size, num_workers, num_epochs, base_channels, seed,
    grad_clip_norm, early_stop_patience, lr_patience, lr_factor,
    keep_last_n_ckpt, ckpt_every_n_epochs,
    train_clean, train_noisy, val_clean, val_noisy   (chemins)
    use_wandb       : booléen
"""
from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from . import config as cfg
from . import checkpoint as ckpt_io
from . import stop_signal
from .dataset import PairedAudioDataset
from .lock import TrainingLock, LockStolenError
from .logging_utils import Logger
from .metrics import (
    magnitude_mse_loss,
    si_sdr_mag,
    mask_statistics,
    OverfitMonitor,
)
from .model import UNet
from .scheduler import DualMetricPlateauScheduler


# Fréquence de rafraîchissement du heartbeat du lock (en secondes). Doit être
# nettement plus court que `src.lock.STALE_AFTER_SECONDS` (300 s) pour qu'une
# session active ne soit jamais considérée comme morte.
_LOCK_HEARTBEAT_SECONDS = 60


class _StopRequested(Exception):
    """Levée depuis la boucle de training quand le flag stop est détecté."""


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _seed_everything(seed: int) -> None:
    """Fixe la graine globale pour python, numpy et torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pick_device() -> torch.device:
    """Sélectionne le meilleur device disponible : CUDA > MPS (Mac Silicon) > CPU.

    MPS (Metal Performance Shaders) est l'équivalent CUDA sur Mac M1/M2/M3/M4
    et est inclus dans torch stable depuis 2.0. Pas de dépendance à installer.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_config(user_config: dict) -> dict:
    """Complète `user_config` avec les valeurs par défaut de `src.config`."""
    defaults = {
        "lr":                  cfg.LR,
        "weight_decay":        cfg.WEIGHT_DECAY,
        "batch_size":          cfg.BATCH_SIZE,
        "num_workers":         cfg.NUM_WORKERS,
        "num_epochs":          cfg.NUM_EPOCHS,
        "base_channels":       cfg.BASE_CHANNELS,
        "seed":                cfg.SEED,
        "grad_clip_norm":      cfg.GRAD_CLIP_NORM,
        "early_stop_patience": cfg.EARLY_STOP_PATIENCE,
        "lr_patience":         cfg.LR_PATIENCE,
        "lr_factor":           cfg.LR_FACTOR,
        "keep_last_n_ckpt":    cfg.KEEP_LAST_N_CKPT,
        "ckpt_every_n_epochs": cfg.CKPT_EVERY_N_EPOCHS,
        "train_clean":         cfg.DATA_TRAIN_CLEAN,
        "train_noisy":         cfg.DATA_TRAIN_NOISY,
        "val_clean":           cfg.DATA_VAL_CLEAN,
        "val_noisy":           cfg.DATA_VAL_NOISY,
        "checkpoints_root":    cfg.CHECKPOINTS,
        "logs_root":           cfg.LOGS,
        "use_wandb":           True,
        "early_stop_enabled":  True,
        # Sauvegarde et log intra-epoch (anti-coupure Colab)
        "intra_epoch_save_seconds": cfg.INTRA_EPOCH_SAVE_SECONDS,
        "intra_epoch_log_every":    cfg.INTRA_EPOCH_LOG_EVERY,
    }
    return {**defaults, **dict(user_config)}


def _build_loader(noisy_dir: str, clean_dir: str, *, batch_size: int,
                  num_workers: int, shuffle: bool, crop_mode: str,
                  max_samples: int | None) -> DataLoader:
    ds: torch.utils.data.Dataset = PairedAudioDataset(
        noisy_dir=noisy_dir,
        clean_dir=clean_dir,
        crop_mode=crop_mode,
        return_spectrogram=True,
    )
    if max_samples is not None and max_samples < len(ds):
        # Subset déterministe (sans random ici : on prend les N premiers triés)
        ds = Subset(ds, list(range(max_samples)))

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,   # entraînement : on évite un dernier batch nain
    )


# ---------------------------------------------------------------------------
# Une epoch d'entraînement / une epoch de validation
# ---------------------------------------------------------------------------

def _train_one_epoch(model, loader, optimizer, device, grad_clip_norm,
                     *,
                     epoch: int,
                     logger=None,
                     log_every: int = 0,
                     save_every_seconds: float = 0.0,
                     save_fn=None,
                     heartbeat_fn=None,
                     heartbeat_every_seconds: float = 0.0,
                     stop_check_fn=None,
                     global_step_start: int = 0) -> tuple[float, int]:
    """Une epoch complète d'entraînement.

    Si `logger` et `log_every > 0` : on push `train_loss_step` (moyenne glissante
    sur `log_every` batches) à wandb / au JSONL pour avoir une courbe live.

    Si `save_fn` et `save_every_seconds > 0` : on appelle `save_fn(global_step)`
    périodiquement pour sauvegarder un `last.pt` en cours d'epoch — sécurité
    anti-coupure Colab.

    Retourne (train_loss_epoch_moyenne, nouveau_global_step).
    """
    model.train()
    total_loss = 0.0
    n_batches  = 0
    window_loss_sum = 0.0
    window_count    = 0
    global_step     = global_step_start
    last_save_at    = time.time()
    last_heartbeat_at = time.time()

    for batch in loader:
        # Stop demandé depuis l'extérieur (GUI / utilitaire) : on sort proprement
        # avant de consommer ce batch. Le `finally` du caller sauvegardera
        # `last.pt` via _intra_save.
        if stop_check_fn is not None and stop_check_fn():
            raise _StopRequested()

        noisy = batch["noisy_mag"].to(device, non_blocking=True)
        clean = batch["clean_mag"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(noisy)
        loss = magnitude_mse_loss(pred, clean)
        loss.backward()
        if grad_clip_norm and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        loss_val = float(loss.item())
        total_loss      += loss_val
        window_loss_sum += loss_val
        window_count    += 1
        n_batches       += 1
        global_step     += 1

        # --- Log intra-epoch (moyenne glissante de train_loss) ---
        if logger is not None and log_every > 0 and window_count >= log_every:
            avg = window_loss_sum / window_count
            try:
                # On utilise logger.log avec une "epoch fractionnaire" pour que
                # wandb empile les points dans l'ordre. Le JSONL marque ce point
                # comme intra-epoch via la clé `is_step`.
                logger.log(epoch, {
                    "train_loss_step": avg,
                    "global_step":     global_step,
                    "is_step":         True,
                })
            except Exception:
                pass
            window_loss_sum = 0.0
            window_count    = 0

        # --- Sauvegarde intra-epoch (toutes les N secondes) ---
        if save_fn is not None and save_every_seconds > 0:
            if time.time() - last_save_at >= save_every_seconds:
                try:
                    save_fn(global_step)
                except Exception as e:
                    print(f"[train] intra-save ignorée ({e})")
                last_save_at = time.time()

        # --- Heartbeat du lock partagé ---
        if heartbeat_fn is not None and heartbeat_every_seconds > 0:
            if time.time() - last_heartbeat_at >= heartbeat_every_seconds:
                try:
                    heartbeat_fn()
                except LockStolenError:
                    # Quelqu'un nous a pris le lock via --force : on stoppe.
                    raise
                except Exception as e:
                    print(f"[train] heartbeat ignoré ({e})")
                last_heartbeat_at = time.time()

    return total_loss / max(n_batches, 1), global_step


@torch.no_grad()
def _validate(model, loader, device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_sdr  = 0.0
    total_noisy_sdr = 0.0
    n_batches  = 0
    acc_mask = {"mean": 0.0, "std": 0.0, "min": 1.0, "max": 0.0}
    for batch in loader:
        noisy = batch["noisy_mag"].to(device, non_blocking=True)
        clean = batch["clean_mag"].to(device, non_blocking=True)

        pred = model(noisy)
        loss = magnitude_mse_loss(pred, clean)
        sdr  = si_sdr_mag(pred, clean)
        # Baseline : SI-SDR du signal bruité brut (sans passage par le modèle).
        # Le val set est en crop centré donc cette valeur est constante d'une
        # epoch à l'autre ; on la calcule quand même chaque fois pour avoir
        # `val_si_sdri = val_si_sdr - noisy_si_sdr` directement dans l'historique.
        noisy_sdr = si_sdr_mag(noisy, clean)

        total_loss += float(loss.item())
        total_sdr  += float(sdr.item())
        total_noisy_sdr += float(noisy_sdr.item())
        n_batches  += 1

        ms = mask_statistics(pred, noisy)
        acc_mask["mean"] += ms["mask_mean"]
        acc_mask["std"]  += ms["mask_std"]
        acc_mask["min"]  = min(acc_mask["min"], ms["mask_min"])
        acc_mask["max"]  = max(acc_mask["max"], ms["mask_max"])

    n = max(n_batches, 1)
    val_si_sdr   = total_sdr / n
    noisy_si_sdr = total_noisy_sdr / n
    return {
        "val_loss":     total_loss / n,
        "val_si_sdr":   val_si_sdr,
        "noisy_si_sdr": noisy_si_sdr,
        "val_si_sdri":  val_si_sdr - noisy_si_sdr,
        "mask_mean":  acc_mask["mean"] / n,
        "mask_std":   acc_mask["std"]  / n,
        "mask_min":   acc_mask["min"],
        "mask_max":   acc_mask["max"],
    }


# ---------------------------------------------------------------------------
# Fonction principale
# ---------------------------------------------------------------------------

def train(
    config: dict,
    resume_from: str | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    progress: bool = True,
    force_lock: bool = False,
) -> dict[str, list]:
    """Lance un entraînement complet et retourne l'historique sous forme de dict.

    `history` est un dict {nom_metrique: [valeur_par_epoch, ...]} contenant
    au minimum : "epoch", "train_loss", "val_loss", "val_si_sdr", "lr",
    et les flags d'overfit.

    `max_train_samples` / `max_val_samples` permettent les mini-entraînements
    pédagogiques (notebook 03) sans toucher au reste du code.

    À la reprise (``resume_from`` non None), la config sauvegardée dans le
    checkpoint sert de baseline et le dict ``config`` passé en argument la
    surcharge. Permet de reprendre un run en ne re-spécifiant que les flags
    qu'on veut changer (ex: bumper ``num_epochs`` ou baisser ``lr``), sans
    perdre les autres hyperparamètres (ex: ``batch_size``).
    """
    # `config` reçu = uniquement les overrides explicites de l'appelant (CLI ou
    # notebook). Si on reprend, on charge d'abord la config sauvegardée comme
    # baseline ; les overrides passent par-dessus.
    user_overrides = dict(config)
    if resume_from:
        saved_config = ckpt_io.peek_checkpoint_config(resume_from)
        if saved_config:
            base = dict(saved_config)
            base.update(user_overrides)
            config = base
            # Repère les valeurs surchargées par rapport au checkpoint pour
            # informer l'utilisateur (utile pour debug "pourquoi c'est rapide").
            overridden = {
                k: (saved_config.get(k), user_overrides[k])
                for k in user_overrides
                if k in saved_config and saved_config[k] != user_overrides[k]
            }
            if overridden:
                print("[train] config restaurée depuis le checkpoint, "
                      "surcharges CLI :")
                for k, (old, new) in overridden.items():
                    print(f"  {k}: {old} -> {new}")
            else:
                print("[train] config restaurée à l'identique depuis le checkpoint.")
        else:
            config = user_overrides

    config = _resolve_config(config)
    run_id = config.get("run_id") or time.strftime("run-%Y%m%d-%H%M%S")
    config["run_id"] = run_id

    with TrainingLock(run_id=run_id, force=force_lock) as lock:
        return _train_locked(
            config=config,
            run_id=run_id,
            resume_from=resume_from,
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples,
            progress=progress,
            lock=lock,
        )


def _train_locked(
    *,
    config: dict,
    run_id: str,
    resume_from: str | None,
    max_train_samples: int | None,
    max_val_samples: int | None,
    progress: bool,
    lock: TrainingLock,
) -> dict[str, list]:
    """Corps de la boucle d'entraînement, exécuté sous lock acquis."""
    _seed_everything(config["seed"])

    device = _pick_device()
    print(f"[train] device={device} | run_id={run_id}")

    # --- Data ---
    train_loader = _build_loader(
        config["train_noisy"], config["train_clean"],
        batch_size=config["batch_size"], num_workers=config["num_workers"],
        shuffle=True, crop_mode="random", max_samples=max_train_samples,
    )
    val_loader = _build_loader(
        config["val_noisy"], config["val_clean"],
        batch_size=config["batch_size"], num_workers=config["num_workers"],
        shuffle=False, crop_mode="center", max_samples=max_val_samples,
    )
    print(f"[train] batches : train={len(train_loader)}  val={len(val_loader)}")

    # --- Modèle / optim / scheduler ---
    model = UNet(base_channels=config["base_channels"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    # Scheduler hybride : on ne réduit le LR que si val_loss ET val_si_sdri
    # plateauent toutes les deux (cf. src.scheduler).
    scheduler = DualMetricPlateauScheduler(
        optimizer,
        patience=config["lr_patience"],
        factor=config["lr_factor"],
    )

    history: dict[str, list] = {
        "epoch": [], "train_loss": [], "val_loss": [], "val_si_sdr": [],
        "noisy_si_sdr": [], "val_si_sdri": [],
        "lr": [], "mask_mean": [], "mask_std": [],
        "gap": [], "gap_ratio": [], "val_plateau_epochs": [],
        "secondary_plateau_epochs": [],
        "val_increasing": [], "diverging": [], "global_step": [],
    }
    start_epoch = 0
    best_val = float("inf")
    best_val_si_sdri = float("-inf")

    # --- Reprise éventuelle ---
    if resume_from:
        info = ckpt_io.load_checkpoint(
            resume_from, model, optimizer, scheduler, device=device,
        )
        start_epoch = info["epoch"]
        best_val    = info["best_val_loss"]
        best_val_si_sdri = info.get("best_val_si_sdri", float("-inf"))
        if info.get("history"):
            # Restaure l'historique pour qu'OverfitMonitor garde sa mémoire
            for k, v in info["history"].items():
                if k in history and isinstance(v, list):
                    history[k] = list(v)
            # Clés ajoutées après-coup (ex: noisy_si_sdr, val_si_sdri) :
            # si elles manquent dans l'historique repris, on rembourre avec
            # NaN pour rester alignés avec `epoch`.
            n_epochs_logged = len(history["epoch"])
            for k, vs in history.items():
                if isinstance(vs, list) and len(vs) < n_epochs_logged:
                    history[k] = [float("nan")] * (n_epochs_logged - len(vs)) + list(vs)
        print(f"[train] reprise depuis {resume_from} "
              f"(epoch terminée = {start_epoch}, best_val = {best_val:.6f})")

    # --- OverfitMonitor (réalimenté depuis l'historique repris) ---
    monitor = OverfitMonitor(patience=config["early_stop_patience"])
    for i, ep in enumerate(history["epoch"]):
        monitor.update(ep, history["train_loss"][i], history["val_loss"][i])
        # Réalimente aussi la métrique secondaire si présente dans l'historique
        # (anciens runs sans val_si_sdri : les valeurs sont NaN, on saute).
        if i < len(history.get("val_si_sdri", [])):
            v = history["val_si_sdri"][i]
            if isinstance(v, (int, float)) and v == v:  # filtre NaN
                monitor.update_secondary(ep, float(v), higher_is_better=True)

    ckpt_dir = Path(config["checkpoints_root"]) / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Logger ---
    logger = Logger(run_id=run_id, config_dict=config,
                    use_wandb=config["use_wandb"], log_dir=config["logs_root"])

    # Global step (compteur de batches cumulés sur tout le run) — utilisé pour
    # ordonner les logs intra-epoch dans wandb. Restauré depuis l'historique
    # repris s'il existe.
    global_step = int(history["global_step"][-1]) if history["global_step"] else 0

    # --- Save fn pour les sauvegardes intra-epoch ---
    def _intra_save(gstep: int) -> None:
        """Sauvegarde `last.pt` au milieu d'une epoch.

        Note : `epoch` est la VARIABLE de boucle ci-dessous (capturée par
        closure). On stocke `epoch - 1` car l'epoch courante n'est pas
        terminée — à la reprise, on recommence cette epoch depuis le début
        avec les poids actuels.
        """
        ckpt_io.save_checkpoint(
            ckpt_dir / "last.pt", model, optimizer, scheduler,
            epoch=max(epoch - 1, 0),   # epoch en cours non terminée
            best_val_loss=best_val,
            best_val_si_sdri=best_val_si_sdri,
            run_id=run_id, config=config, history=history,
        )

    # Au démarrage on nettoie un éventuel flag rémanent (run précédent avec le
    # même run_id qui aurait crashé sans clear). Sinon le training s'arrêterait
    # immédiatement à la première itération.
    stop_signal.clear_stop_flag(run_id)

    stop_was_requested = False
    try:
        for epoch in range(start_epoch + 1, config["num_epochs"] + 1):
            t0 = time.time()

            try:
                train_loss, global_step = _train_one_epoch(
                    model, train_loader, optimizer, device,
                    grad_clip_norm=config["grad_clip_norm"],
                    epoch=epoch,
                    logger=logger,
                    log_every=config["intra_epoch_log_every"],
                    save_every_seconds=config["intra_epoch_save_seconds"],
                    save_fn=_intra_save,
                    heartbeat_fn=lock.heartbeat,
                    heartbeat_every_seconds=_LOCK_HEARTBEAT_SECONDS,
                    stop_check_fn=lambda: stop_signal.is_stop_requested(run_id),
                    global_step_start=global_step,
                )
            except _StopRequested:
                stop_was_requested = True
                print(f"[train] stop demandé via flag, sauvegarde de last.pt puis exit.")
                _intra_save(global_step)
                break
            val_metrics = _validate(model, val_loader, device)
            lr_reduced = scheduler.step(
                val_metrics["val_loss"], val_metrics["val_si_sdri"]
            )
            current_lr = optimizer.param_groups[0]["lr"]

            monitor.update(epoch, train_loss, val_metrics["val_loss"])
            monitor.update_secondary(
                epoch, val_metrics["val_si_sdri"], higher_is_better=True
            )
            flags = monitor.flags()

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["val_loss"])
            history["val_si_sdr"].append(val_metrics["val_si_sdr"])
            history["noisy_si_sdr"].append(val_metrics["noisy_si_sdr"])
            history["val_si_sdri"].append(val_metrics["val_si_sdri"])
            history["lr"].append(current_lr)
            history["mask_mean"].append(val_metrics["mask_mean"])
            history["mask_std"].append(val_metrics["mask_std"])
            history["gap"].append(flags["gap"])
            history["gap_ratio"].append(flags["gap_ratio"])
            history["val_plateau_epochs"].append(flags["val_plateau_epochs"])
            history["secondary_plateau_epochs"].append(
                flags["secondary_plateau_epochs"]
            )
            history["val_increasing"].append(flags["val_increasing"])
            history["diverging"].append(flags["diverging"])
            history["global_step"].append(global_step)

            # --- Checkpoints ---
            # On évalue les deux critères AVANT de logger, pour pouvoir mettre
            # un marqueur ★ dans le print et tracer is_best_* dans wandb.
            is_best_loss = val_metrics["val_loss"] < best_val
            is_best_sdri = val_metrics["val_si_sdri"] > best_val_si_sdri
            if is_best_loss:
                best_val = val_metrics["val_loss"]
            if is_best_sdri:
                best_val_si_sdri = val_metrics["val_si_sdri"]

            logger.log(epoch, {
                "train_loss":     train_loss,
                "val_loss":       val_metrics["val_loss"],
                "val_si_sdr":     val_metrics["val_si_sdr"],
                "noisy_si_sdr":   val_metrics["noisy_si_sdr"],
                "val_si_sdri":    val_metrics["val_si_sdri"],
                "lr":             current_lr,
                "lr_reduced":     bool(lr_reduced),
                "mask_mean":      val_metrics["mask_mean"],
                "mask_std":       val_metrics["mask_std"],
                "best_val_loss":  best_val,
                "best_val_si_sdri": best_val_si_sdri,
                "is_best_loss":   bool(is_best_loss),
                "is_best_sdri":   bool(is_best_sdri),
                **flags,
                "epoch_seconds":  time.time() - t0,
            })

            if progress:
                star = ""
                if is_best_loss and is_best_sdri:
                    star = " ★ both"
                elif is_best_loss:
                    star = " ★ loss"
                elif is_best_sdri:
                    star = " ★ sdri"
                print(f"  ep {epoch:3d} | train {train_loss:.4f}  "
                      f"val {val_metrics['val_loss']:.4f}  "
                      f"SI-SDR {val_metrics['val_si_sdr']:+.2f} dB  "
                      f"SI-SDRi {val_metrics['val_si_sdri']:+.2f} dB  "
                      f"(noisy {val_metrics['noisy_si_sdr']:+.2f})  "
                      f"plat L{flags['val_plateau_epochs']}/S{flags['secondary_plateau_epochs']}  "
                      f"lr {current_lr:.2e}  "
                      f"({time.time() - t0:.1f}s){star}")

            # Sauvegardes (le `best.pt` garde sa sémantique historique = best val_loss)
            if is_best_loss:
                ckpt_io.save_checkpoint(
                    ckpt_dir / "best.pt", model, optimizer, scheduler,
                    epoch=epoch, best_val_loss=best_val,
                    best_val_si_sdri=best_val_si_sdri,
                    run_id=run_id, config=config, history=history,
                )
            if is_best_sdri:
                ckpt_io.save_checkpoint(
                    ckpt_dir / "best_si_sdri.pt", model, optimizer, scheduler,
                    epoch=epoch, best_val_loss=best_val,
                    best_val_si_sdri=best_val_si_sdri,
                    run_id=run_id, config=config, history=history,
                )
            ckpt_io.save_checkpoint(
                ckpt_dir / "last.pt", model, optimizer, scheduler,
                epoch=epoch, best_val_loss=best_val,
                best_val_si_sdri=best_val_si_sdri,
                run_id=run_id, config=config, history=history,
            )
            if epoch % config["ckpt_every_n_epochs"] == 0:
                ckpt_io.save_checkpoint(
                    ckpt_dir / f"epoch_{epoch:03d}.pt", model, optimizer,
                    scheduler, epoch=epoch, best_val_loss=best_val,
                    best_val_si_sdri=best_val_si_sdri,
                    run_id=run_id, config=config, history=history,
                )
                ckpt_io.rotate_checkpoints(ckpt_dir, config["keep_last_n_ckpt"])

            # --- Early stopping ---
            if config["early_stop_enabled"] and monitor.should_early_stop():
                print(f"[train] early stop (val plateau "
                      f"{monitor.epochs_since_improve} epochs).")
                break
    finally:
        logger.finish()
        # Nettoyage du flag, qu'il ait été consommé (stop demandé) ou non
        # (fin normale après early stop / num_epochs).
        stop_signal.clear_stop_flag(run_id)

    if stop_was_requested:
        print(f"[train] arrêté à la demande de l'utilisateur. ckpt_dir={ckpt_dir}")
    else:
        print(f"[train] terminé. best_val={best_val:.6f}  ckpt_dir={ckpt_dir}")
    return history
