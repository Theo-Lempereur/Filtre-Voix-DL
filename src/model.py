"""U-Net pour le débruitage audio sur spectrogrammes de magnitude.

Architecture figée pour la baseline "p2" :
    - 4 niveaux d'encodeur / décodeur, base_channels configurable
    - GroupNorm dans tous les DoubleConv (stable à petit batch_size)
    - Masque réel `sigmoid × noisy_mag` en sortie (bornage historique [0, 1])

Interface attendue par train.py :

    from src.model import UNet, count_parameters

    model = UNet(base_channels=32)
    pred_mag = model(model_input, noisy_mag)        # (B, 1, F, T)

`model_input` est la magnitude éventuellement compressée (log1p) ; `noisy_mag`
est la magnitude linéaire à laquelle s'applique le masque (cf. train.py).

Dépendances : uniquement PyTorch — ce module n'importe rien de src/.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper : GroupNorm avec un nombre de groupes sûr
# ---------------------------------------------------------------------------

def _group_norm(channels: int, groups: int) -> nn.GroupNorm:
    """GroupNorm dont `num_groups` divise `num_channels`.

    On rabote `groups` vers le bas si nécessaire pour respecter la contrainte
    `channels % groups == 0` — sinon nn.GroupNorm lève à l'init.
    """
    g = min(groups, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=channels)


# ---------------------------------------------------------------------------
# Bloc de base : DoubleConv
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """Deux convolutions Conv→GroupNorm→ReLU empilées."""

    def __init__(self, in_channels: int, out_channels: int, norm_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            _group_norm(out_channels, norm_groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            _group_norm(out_channels, norm_groups),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Bloc encodeur : DoubleConv + MaxPool
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    """Un niveau de l'encodeur (DoubleConv + MaxPool2)."""

    def __init__(self, in_channels: int, out_channels: int, norm_groups: int = 8):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, norm_groups=norm_groups)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip   = self.conv(x)
        pooled = self.pool(skip)
        return skip, pooled


# ---------------------------------------------------------------------------
# Bloc décodeur : upsample + concat skip + DoubleConv
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """Un niveau du décodeur (upsample bilinéaire + concat skip + DoubleConv)."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 norm_groups: int = 8):
        super().__init__()
        self.conv = DoubleConv(in_channels + skip_channels, out_channels,
                               norm_groups=norm_groups)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:],
                          mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Réseau complet : UNet
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """U-Net 4 niveaux pour le débruitage de spectrogrammes.

    Paramètres :
      base_channels : largeur initiale (32 → ~7.85M params).
      norm_groups   : nombre de groupes pour les GroupNorm internes.
    """

    def __init__(self, base_channels: int = 32, norm_groups: int = 8):
        super().__init__()
        b = base_channels
        self.norm_groups = norm_groups

        # --- Encodeur ---
        self.enc1 = EncoderBlock(1,     b,     norm_groups=norm_groups)
        self.enc2 = EncoderBlock(b,     b * 2, norm_groups=norm_groups)
        self.enc3 = EncoderBlock(b * 2, b * 4, norm_groups=norm_groups)
        self.enc4 = EncoderBlock(b * 4, b * 8, norm_groups=norm_groups)

        # --- Bottleneck ---
        self.bottleneck = DoubleConv(b * 8, b * 16, norm_groups=norm_groups)

        # --- Décodeur ---
        self.dec4 = DecoderBlock(b * 16, b * 8, b * 8, norm_groups=norm_groups)
        self.dec3 = DecoderBlock(b * 8,  b * 4, b * 4, norm_groups=norm_groups)
        self.dec2 = DecoderBlock(b * 4,  b * 2, b * 2, norm_groups=norm_groups)
        self.dec1 = DecoderBlock(b * 2,  b,     b,     norm_groups=norm_groups)

        # --- Sortie : 1 canal (masque réel ∈ [0, 1] via sigmoid) ---
        self.out_conv = nn.Conv2d(b, 1, kernel_size=1)

    def forward(self, model_input: torch.Tensor,
                noisy_mag: torch.Tensor | None = None) -> torch.Tensor:
        """Forward du U-Net.

        Args:
            model_input : (B, 1, F, T) — entrée du réseau, éventuellement
                          compressée (log1p) par train.py. C'est ce que voient
                          les convolutions.
            noisy_mag   : (B, 1, F, T) — magnitude **non compressée**, sur
                          laquelle on applique le masque (`sigmoid × noisy_mag`).
                          Si None : on utilise `model_input` (cas où la
                          compression n'est pas activée → équivalent).

        Returns:
            (B, 1, F, T) — magnitude prédite, bornée à `noisy_mag` par sigmoid.
        """
        if noisy_mag is None:
            noisy_mag = model_input

        skip1, x = self.enc1(model_input)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        raw = self.out_conv(x)
        return torch.sigmoid(raw) * noisy_mag


# ---------------------------------------------------------------------------
# Utilitaire
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Compte les paramètres entraînables d'un modèle PyTorch."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
