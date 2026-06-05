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
from . import audio as A
from . import stop_signal
from .dataset import PairedAudioDataset
from .lock import TrainingLock, LockStolenError
from .logging_utils import Logger
from .metrics import (
    si_sdr_mag,
    si_sdr_wav,
    mask_statistics,
    build_loss,
    loss_requires_waveform,
    OverfitMonitor,
)
from .model import UNet
from .scheduler import DualMetricPlateauScheduler, WarmupWrapper


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


def _configure_backends(device: torch.device) -> None:
    """Active les optimisations backend CUDA (sans effet hors CUDA).

    - TF32 + matmul 'high' : matmuls/convs fp32 accélérées sur Tensor Cores (gratuit).
    - cudnn.benchmark : laissé à FALSE. Mesuré : sur la RTX 5070 12 Go il gonfle la
      VRAM de ~1,4 Go (workspaces d'autotune) et fait déborder en mémoire partagée
      (lent). Le gain d'autotune est marginal à taille d'entrée fixe. Sur un GPU
      large (ex: 5090 32 Go sur RunPod) on peut le réactiver pour un petit bonus.
    """
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


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
        # --- Recette p2 (figée + modifiables) ---
        # Le reste de la recette (input_repr=log1p, GroupNorm, AdamW, masque
        # sigmoid) est figé dans le code et ne passe pas par la config.
        "loss_weights":        dict(cfg.LOSS_WEIGHTS),
        "mr_stft_fft_sizes":   list(cfg.MR_STFT_FFT_SIZES),
        "norm_groups":         cfg.NORM_GROUPS,
        "lr_warmup_epochs":    cfg.LR_WARMUP_EPOCHS,
        # --- Mode de sortie : "mask" (p2) ou "complex" (complex spectral mapping) ---
        "output_mode":         cfg.OUTPUT_MODE,
        "csm_compress":        cfg.CSM_COMPRESS,
        "csm_loss_weights":    dict(cfg.CSM_LOSS_WEIGHTS),
        # --- Performance / précision (CUDA, auto-désactivés sinon) ---
        "amp":                 cfg.AMP,
        "compile":             cfg.COMPILE,
        "use_wav_cache":       cfg.USE_WAV_CACHE,
        "cache_dir":           cfg.CACHE_DIR,
    }
    return {**defaults, **dict(user_config)}


def _build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """Construit AdamW (figé pour la baseline p2).

    AdamW applique correctement le weight decay comme régularisation découplée
    — recommandé à 1e-4 avec GroupNorm pour stabiliser l'entraînement.
    """
    lr = float(config["lr"])
    wd = float(config.get("weight_decay", 0.0))
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


def _build_loader(noisy_dir: str, clean_dir: str, *, batch_size: int,
                  num_workers: int, shuffle: bool, crop_mode: str,
                  max_samples: int | None,
                  use_wav_cache: bool = False,
                  cache_dir: str | None = None,
                  cache_split: str | None = None) -> DataLoader:
    # Le dataset renvoie toujours des waveforms (waveform_only / cache) ; la STFT
    # est calculée sur GPU dans le forward (cf. `_build_forward_fn`).
    if use_wav_cache:
        ds: torch.utils.data.Dataset = PairedAudioDataset(
            cache_dir=cache_dir, cache_split=cache_split, crop_mode=crop_mode,
        )
    else:
        ds = PairedAudioDataset(
            noisy_dir=noisy_dir, clean_dir=clean_dir,
            crop_mode=crop_mode, waveform_only=True,
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
        # Workers gardés vivants entre epochs + préchargement : évite de
        # reconstruire le pool à chaque epoch (gros gain en mode ajustement).
        persistent_workers=(num_workers > 0),
        prefetch_factor=(4 if num_workers > 0 else None),
    )


# ---------------------------------------------------------------------------
# Factory du forward (compression log1p figée + ISTFT optionnelle pour MR-STFT)
# ---------------------------------------------------------------------------

def _build_forward_fn(model, config: dict, *, needs_waveform: bool):
    """Renvoie une fonction `forward_fn(batch, device) -> dict`.

    Le dict retourné contient au minimum :
      pred_mag, clean_mag, noisy_mag
    Et, si `needs_waveform` (MR-STFT loss active) :
      pred_wav, clean_wav, noisy_wav

    L'entrée du réseau est toujours `log1p(noisy_mag)` (compression p2 figée).
    Le masque s'applique à la magnitude linéaire originale.

    Les batches fournissent des **waveforms** (noisy_wav / clean_wav) : la STFT
    est calculée ici sur le device (GPU) via `audio.stft_torch`, en fp32. Seul
    le forward du modèle tourne sous autocast bf16 (si `config['amp']`) ; l'ISTFT
    et la loss MR-STFT restent en fp32 (FFT non fiable en bf16).
    """
    amp = bool(config.get("amp", False))
    output_mode = str(config.get("output_mode", "mask"))
    c_comp = float(config.get("csm_compress", 0.3))

    def _istft(mag: torch.Tensor, phase: torch.Tensor, length: int) -> torch.Tensor:
        """ISTFT différentiable : combine magnitude prédite + phase noisy."""
        # (B, 1, F, T) → (B, F, T)
        if mag.dim() == 4 and mag.shape[1] == 1:
            mag = mag.squeeze(1)
        spec = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
        n_fft = cfg.N_FFT
        hop   = cfg.HOP_LENGTH
        window = torch.hann_window(n_fft, device=spec.device, dtype=spec.real.dtype)
        return torch.istft(spec, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                           window=window, length=length, return_complex=False)

    def forward_fn(batch, device):
        noisy_wav = batch["noisy_wav"].to(device, non_blocking=True)
        clean_wav = batch["clean_wav"].to(device, non_blocking=True)

        # STFT sur GPU, en fp32 (source de vérité unique, hors autocast).
        # (B, samples) -> (B, F, T) -> (B, 1, F, T)
        noisy_mag, noisy_phase = A.stft_torch(noisy_wav)
        clean_mag, _           = A.stft_torch(clean_wav)
        noisy_mag = noisy_mag.unsqueeze(1)
        clean_mag = clean_mag.unsqueeze(1)

        # Compression log1p : rend la distribution plus gaussienne en entrée
        # du réseau. La cible reste linéaire pour conserver la sémantique
        # du masque sigmoid × noisy_mag.
        model_input = torch.log1p(noisy_mag)

        # Seul le forward modèle tourne en bf16 (Tensor Cores). Le masque
        # s'applique à la magnitude originale (non compressée).
        use_amp = amp and device.type == "cuda"
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            pred_mag = model(model_input, noisy_mag)
        pred_mag = pred_mag.float()   # repasse en fp32 pour ISTFT + loss

        pred_wav = None
        if needs_waveform:
            pred_wav = _istft(pred_mag, noisy_phase, length=cfg.CLIP_SAMPLES)

        out = {
            "pred_mag":  pred_mag,
            "clean_mag": clean_mag,
            "noisy_mag": noisy_mag,
        }
        if pred_wav is not None:
            out["pred_wav"]  = pred_wav
            out["clean_wav"] = clean_wav
            out["noisy_wav"] = noisy_wav
        return out

    # -----------------------------------------------------------------------
    # Forward "complex spectral mapping" : le réseau prédit Re/Im du spectre
    # propre (compressés). Récupère la phase -> dépasse le plafond magnitude.
    # -----------------------------------------------------------------------
    def _stft_complex(wav: torch.Tensor) -> torch.Tensor:
        win = torch.hann_window(cfg.WIN_LENGTH, device=wav.device, dtype=wav.dtype)
        return torch.stft(wav, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH,
                          win_length=cfg.WIN_LENGTH, window=win, center=True,
                          pad_mode="constant", return_complex=True)

    def forward_fn_complex(batch, device):
        eps = 1e-8
        noisy_wav = batch["noisy_wav"].to(device, non_blocking=True)
        clean_wav = batch["clean_wav"].to(device, non_blocking=True)

        # STFT complexe en fp32 (hors autocast).
        ns = _stft_complex(noisy_wav)       # (B, F, T) complexe
        cs = _stft_complex(clean_wav)
        n_mag = ns.abs(); c_mag = cs.abs()

        # Compression puissance : Z = |S|^c · e^{jθ} = S · |S|^(c-1).
        # (les entrées/cibles n'ont pas de gradient -> pas de souci de dérivée en 0)
        n_scale = (n_mag + eps).pow(c_comp - 1.0)
        nz_re = ns.real * n_scale; nz_im = ns.imag * n_scale
        c_scale = (c_mag + eps).pow(c_comp - 1.0)
        cz_re = cs.real * c_scale; cz_im = cs.imag * c_scale
        clean_ri = torch.stack([cz_re, cz_im], dim=1)        # (B, 2, F, T)

        model_input = torch.stack([nz_re, nz_im], dim=1)     # (B, 2, F, T)

        use_amp = amp and device.type == "cuda"
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            pred_ri = model(model_input)                     # (B, 2, F, T)
        pred_ri = pred_ri.float()
        pred_re = pred_ri[:, 0]; pred_im = pred_ri[:, 1]

        # Décompression sans atan2 : S = Z · |Z|^((1-c)/c)  (stable, différentiable).
        pred_cmag = torch.sqrt(pred_re ** 2 + pred_im ** 2 + eps)
        up = pred_cmag.pow((1.0 - c_comp) / c_comp)
        s_re = pred_re * up; s_im = pred_im * up
        pred_spec = torch.complex(s_re, s_im)
        win = torch.hann_window(cfg.WIN_LENGTH, device=device, dtype=torch.float32)
        pred_wav = torch.istft(pred_spec, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH,
                               win_length=cfg.WIN_LENGTH, window=win, center=True,
                               length=cfg.CLIP_SAMPLES, return_complex=False)

        # Magnitude linéaire prédite (pour les métriques de validation existantes).
        pred_mag = pred_cmag.pow(1.0 / c_comp)

        return {
            "pred_ri":   pred_ri,
            "clean_ri":  clean_ri,
            "pred_mag":  pred_mag.unsqueeze(1),
            "clean_mag": c_mag.unsqueeze(1),
            "noisy_mag": n_mag.unsqueeze(1),
            "pred_wav":  pred_wav,
            "clean_wav": clean_wav,
            "noisy_wav": noisy_wav,
        }

    if output_mode == "complex":
        return forward_fn_complex
    return forward_fn


# ---------------------------------------------------------------------------
# Une epoch d'entraînement / une epoch de validation
# ---------------------------------------------------------------------------

def _train_one_epoch(model, loader, optimizer, device, grad_clip_norm,
                     *,
                     epoch: int,
                     loss_fn,
                     forward_fn,
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

        optimizer.zero_grad(set_to_none=True)
        # `forward_fn` encapsule tout : déplacement device, forward modèle,
        # éventuelle ISTFT pour produire pred_wav. Retourne le dict que
        # `loss_fn` attend (au minimum pred_mag + clean_mag).
        loss_batch = forward_fn(batch, device)
        loss = loss_fn(loss_batch)
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
def _validate(model, loader, device, *, loss_fn, forward_fn) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_sdr  = 0.0
    total_noisy_sdr = 0.0
    total_sdr_wav = 0.0
    total_noisy_sdr_wav = 0.0
    have_wav = 0
    n_batches  = 0
    acc_mask = {"mean": 0.0, "std": 0.0, "min": 1.0, "max": 0.0}
    for batch in loader:
        loss_batch = forward_fn(batch, device)
        loss = loss_fn(loss_batch)

        pred  = loss_batch["pred_mag"]
        clean = loss_batch["clean_mag"]
        noisy = loss_batch["noisy_mag"]
        sdr   = si_sdr_mag(pred, clean)
        # Baseline : SI-SDR du signal bruité brut (sans passage par le modèle).
        # Le val set est en crop centré donc cette valeur est constante d'une
        # epoch à l'autre ; on la calcule quand même chaque fois pour avoir
        # `val_si_sdri = val_si_sdr - noisy_si_sdr` directement dans l'historique.
        noisy_sdr = si_sdr_mag(noisy, clean)

        total_loss += float(loss.item())
        total_sdr  += float(sdr.item())
        total_noisy_sdr += float(noisy_sdr.item())
        n_batches  += 1

        # SI-SDR waveform : seulement si les waveforms sont dispo dans le batch.
        # Métrique perceptive de référence (vraie SI-SDR, pas approximée mag).
        if loss_batch.get("pred_wav") is not None and loss_batch.get("clean_wav") is not None:
            total_sdr_wav += float(si_sdr_wav(loss_batch["pred_wav"], loss_batch["clean_wav"]).item())
            total_noisy_sdr_wav += float(si_sdr_wav(loss_batch["noisy_wav"], loss_batch["clean_wav"]).item())
            have_wav += 1

        ms = mask_statistics(pred, noisy)
        acc_mask["mean"] += ms["mask_mean"]
        acc_mask["std"]  += ms["mask_std"]
        acc_mask["min"]  = min(acc_mask["min"], ms["mask_min"])
        acc_mask["max"]  = max(acc_mask["max"], ms["mask_max"])

    n = max(n_batches, 1)
    val_si_sdr   = total_sdr / n
    noisy_si_sdr = total_noisy_sdr / n
    out: dict[str, float] = {
        "val_loss":     total_loss / n,
        "val_si_sdr":   val_si_sdr,
        "noisy_si_sdr": noisy_si_sdr,
        "val_si_sdri":  val_si_sdr - noisy_si_sdr,
        "mask_mean":  acc_mask["mean"] / n,
        "mask_std":   acc_mask["std"]  / n,
        "mask_min":   acc_mask["min"],
        "mask_max":   acc_mask["max"],
    }
    if have_wav > 0:
        wav_sdr = total_sdr_wav / have_wav
        wav_noisy = total_noisy_sdr_wav / have_wav
        out["val_si_sdr_wav"]   = wav_sdr
        out["noisy_si_sdr_wav"] = wav_noisy
        out["val_si_sdri_wav"]  = wav_sdr - wav_noisy
    return out


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
    _configure_backends(device)

    # --- Data ---
    # Les loaders renvoient toujours des waveforms ; la STFT (et l'ISTFT côté
    # prédiction pour MR-STFT) est faite sur GPU dans le forward. Le cache
    # memmap supprime le coût librosa.load/STFT côté CPU.
    use_cache = bool(config.get("use_wav_cache", False))
    cache_dir = config.get("cache_dir")
    if use_cache:
        print(f"[train] cache waveforms : {cache_dir}")
    train_loader = _build_loader(
        config["train_noisy"], config["train_clean"],
        batch_size=config["batch_size"], num_workers=config["num_workers"],
        shuffle=True, crop_mode="random", max_samples=max_train_samples,
        use_wav_cache=use_cache, cache_dir=cache_dir, cache_split="train",
    )
    val_loader = _build_loader(
        config["val_noisy"], config["val_clean"],
        batch_size=config["batch_size"], num_workers=config["num_workers"],
        shuffle=False, crop_mode="center", max_samples=max_val_samples,
        use_wav_cache=use_cache, cache_dir=cache_dir, cache_split="val",
    )
    print(f"[train] batches : train={len(train_loader)}  val={len(val_loader)}")

    # --- Modèle / optim / scheduler ---
    output_mode = str(config.get("output_mode", "mask"))
    if output_mode == "complex":
        # Complex spectral mapping : entrée Re/Im (2 canaux), sortie Re/Im
        # (2 canaux), pas de masquage (tête linéaire).
        model = UNet(
            base_channels=config["base_channels"],
            norm_groups=config.get("norm_groups", 8),
            in_channels=2, out_channels=2, head="linear",
        ).to(device)
    else:
        model = UNet(
            base_channels=config["base_channels"],
            norm_groups=config.get("norm_groups", 8),
        ).to(device)
    print(f"[train] output_mode={output_mode}")
    # torch.compile (CUDA only) : fusionne les kernels. Construit AVANT
    # l'optimizer (qui partage les mêmes paramètres). Le state_dict est géré par
    # src.checkpoint qui déballe `_orig_mod` -> compat checkpoints compiled/non.
    if config.get("compile", False) and device.type == "cuda":
        try:
            # La compilation est lazy (au 1er forward). suppress_errors=True fait
            # retomber dynamo en eager si le backend échoue (ex: Triton absent
            # sous Windows) au lieu de planter l'entraînement.
            import torch._dynamo as _dynamo
            _dynamo.config.suppress_errors = True
            model = torch.compile(model)
            print("[train] torch.compile activé (fallback eager si backend indisponible).")
        except Exception as e:
            print(f"[train] torch.compile indisponible, ignoré ({e}).")
    optimizer = _build_optimizer(model, config)
    # Scheduler hybride : on ne réduit le LR que si val_loss ET val_si_sdri
    # plateauent toutes les deux (cf. src.scheduler). Optionnellement enveloppé
    # par un WarmupWrapper si `lr_warmup_epochs > 0` (no-op sinon).
    inner_scheduler = DualMetricPlateauScheduler(
        optimizer,
        patience=config["lr_patience"],
        factor=config["lr_factor"],
    )
    scheduler = WarmupWrapper(
        inner=inner_scheduler,
        optimizer=optimizer,
        warmup_epochs=int(config.get("lr_warmup_epochs", 0) or 0),
    )

    # --- Factory de loss combinée (MSE + L1 compressed + MR-STFT) ---
    loss_fn = build_loss(config)
    needs_waveform = loss_requires_waveform(config)

    # --- Factory de forward (log1p + ISTFT si MR-STFT actif) ---
    forward_fn = _build_forward_fn(model, config, needs_waveform=needs_waveform)

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
                    loss_fn=loss_fn,
                    forward_fn=forward_fn,
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
            val_metrics = _validate(model, val_loader, device,
                                    loss_fn=loss_fn, forward_fn=forward_fn)
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

            log_payload = {
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
            }
            # Métriques waveform (vraies SI-SDR) si dispo (Phase 1+ avec MR-STFT
            # ou Phase 4 cIRM). Précieuses pour suivre le ressenti perceptif.
            for k in ("val_si_sdr_wav", "noisy_si_sdr_wav", "val_si_sdri_wav"):
                if k in val_metrics:
                    log_payload[k] = val_metrics[k]
            logger.log(epoch, log_payload)

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
