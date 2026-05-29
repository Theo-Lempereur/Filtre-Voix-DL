"""Description déclarative des hyperparamètres exposés dans la GUI.

C'est la source de vérité du formulaire : ajouter un champ ici suffit à le
faire apparaître dans `ParamForm` avec son aide contextuelle et sa traduction
en argument CLI pour ``scripts/train_local.py``.

Les défauts et bornes sont alignés sur ``src.config`` et le parser argparse
de ``scripts/train_local.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ParamKind = Literal["str", "int", "float", "bool", "choice"]


@dataclass
class ParamSpec:
    name: str
    cli_flag: str
    label: str
    kind: ParamKind
    default: Any = None
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    decimals: int = 0
    choices: list[str] = field(default_factory=list)
    help: str = ""
    advanced: bool = False
    # Si True : valeur optionnelle (case à cocher "activer" + champ).
    # Utile pour `--max-train-samples` (off par défaut).
    optional: bool = False
    # Pour kind="bool", si True, le drapeau CLI signifie "désactiver"
    # (cas de --no-wandb : la GUI montre "W&B activé" et coche par défaut).
    inverted: bool = False


PARAM_SPECS: list[ParamSpec] = [
    ParamSpec(
        name="resume",
        cli_flag="--resume",
        label="Reprise",
        kind="choice",
        default="",
        choices=["", "last", "best", "best_si_sdri"],
        help=(
            "Reprend depuis un checkpoint existant. Si non vide, le champ "
            "Run ID devient une liste des runs ayant ce checkpoint, et les "
            "autres paramètres se remplissent automatiquement depuis la "
            "config sauvegardée."
        ),
    ),
    ParamSpec(
        name="run_id",
        cli_flag="--run-id",
        label="Run ID",
        kind="str",
        default="",
        help="Identifiant unique du run. Sert de sous-dossier checkpoints/<run-id>/. Obligatoire.",
    ),
    ParamSpec(
        name="epochs",
        cli_flag="--epochs",
        label="Epochs",
        kind="int",
        default=50,
        minimum=1,
        maximum=10000,
        step=1,
        help="Nombre maximum d'epochs. L'early stopping peut couper avant.",
    ),
    ParamSpec(
        name="batch_size",
        cli_flag="--batch-size",
        label="Batch size",
        kind="int",
        default=8,
        minimum=1,
        maximum=1024,
        step=1,
        help="Échantillons par batch. 8 tient en VRAM sur T4 avec base_channels=32.",
    ),
    ParamSpec(
        name="lr",
        cli_flag="--lr",
        label="Learning rate",
        kind="float",
        default=1e-3,
        minimum=1e-7,
        maximum=1.0,
        step=1e-4,
        decimals=7,
        help="LR initial de l'optimiseur Adam. Typique : 1e-3 à 1e-4.",
    ),
    ParamSpec(
        name="num_workers",
        cli_flag="--num-workers",
        label="Num workers",
        kind="int",
        default=2,
        minimum=0,
        maximum=32,
        step=1,
        help="Workers DataLoader. 2 à 4 sur Colab (RAM limitée), plus en local.",
    ),
    ParamSpec(
        name="base_channels",
        cli_flag="--base-channels",
        label="Base channels (U-Net)",
        kind="int",
        default=32,
        minimum=4,
        maximum=256,
        step=4,
        help="Largeur initiale du U-Net. Double = ~4x la mémoire et les params (32 ≈ 7,85M).",
    ),
    ParamSpec(
        name="seed",
        cli_flag="--seed",
        label="Seed",
        kind="int",
        default=42,
        minimum=0,
        maximum=2**31 - 1,
        step=1,
        help="Graine globale (torch, numpy, random). Change pour avoir un run différent.",
    ),
    ParamSpec(
        name="max_train_samples",
        cli_flag="--max-train-samples",
        label="Max train samples",
        kind="int",
        default=256,
        minimum=1,
        maximum=10**7,
        step=1,
        help="Smoke test : limite le nombre d'échantillons d'entraînement.",
        advanced=True,
        optional=True,
    ),
    ParamSpec(
        name="max_val_samples",
        cli_flag="--max-val-samples",
        label="Max val samples",
        kind="int",
        default=64,
        minimum=1,
        maximum=10**7,
        step=1,
        help="Smoke test : limite le nombre d'échantillons de validation.",
        advanced=True,
        optional=True,
    ),
    ParamSpec(
        name="use_wandb",
        cli_flag="--no-wandb",
        label="Activer W&B",
        kind="bool",
        default=True,
        help="Suivi d'entraînement Weights & Biases. Décocher si pas de clé API locale.",
        inverted=True,
    ),
    ParamSpec(
        name="force_lock",
        cli_flag="--force",
        label="Forcer le lock",
        kind="bool",
        default=False,
        help="Override le lock partagé même si un autre run semble en cours. À éviter.",
        advanced=True,
    ),

    # ---------------------------------------------------------- Recette p2
    # Le reste de la recette (log1p, GroupNorm, AdamW, masque sigmoid, loss
    # combo) est figé dans le code et n'apparaît plus comme paramètre GUI.
    # Seuls les paramètres potentiellement intéressants à explorer restent.
    ParamSpec(
        name="loss_w_mse",
        cli_flag="--loss-w-mse",
        label="Poids MSE",
        kind="float",
        default=0.0,
        minimum=0.0,
        maximum=10.0,
        step=0.1,
        decimals=3,
        help="Poids de la composante MSE dans la loss combinée. 0 par défaut.",
        advanced=True,
    ),
    ParamSpec(
        name="loss_w_l1comp",
        cli_flag="--loss-w-l1comp",
        label="Poids L1 compressed",
        kind="float",
        default=1.0,
        minimum=0.0,
        maximum=10.0,
        step=0.1,
        decimals=3,
        help="Poids de L1(mag^0.3) dans la loss combinée. 1.0 par défaut.",
        advanced=True,
    ),
    ParamSpec(
        name="loss_w_mrstft",
        cli_flag="--loss-w-mrstft",
        label="Poids MR-STFT",
        kind="float",
        default=1.0,
        minimum=0.0,
        maximum=10.0,
        step=0.1,
        decimals=3,
        help="Poids de MR-STFT loss dans la loss combinée. 1.0 par défaut.",
        advanced=True,
    ),
    ParamSpec(
        name="weight_decay",
        cli_flag="--weight-decay",
        label="Weight decay",
        kind="float",
        default=1e-4,
        minimum=0.0,
        maximum=1.0,
        step=1e-5,
        decimals=6,
        help="L2 weight decay (AdamW). 1e-4 par défaut.",
        advanced=True,
    ),
    ParamSpec(
        name="lr_warmup_epochs",
        cli_flag="--lr-warmup-epochs",
        label="Warmup LR (epochs)",
        kind="int",
        default=2,
        minimum=0,
        maximum=100,
        step=1,
        help="Nombre d'epochs de warmup linéaire du LR. 2 par défaut, 0 = désactivé.",
        advanced=True,
    ),
]


def by_name(name: str) -> ParamSpec:
    for s in PARAM_SPECS:
        if s.name == name:
            return s
    raise KeyError(name)
