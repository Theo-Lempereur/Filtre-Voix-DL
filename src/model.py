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

    def __init__(self, base_channels: int = 32, norm_groups: int = 8,
                 in_channels: int = 1, out_channels: int = 1,
                 head: str = "mask"):
        """U-Net configurable.

        Args:
            in_channels  : nb de canaux d'entrée. 1 pour la recette magnitude
                           (log1p), 2 pour le complex spectral mapping (Re, Im).
            out_channels : nb de canaux de sortie. 1 (masque magnitude) ou 2
                           (Re, Im du spectre propre estimé).
            head         : "mask"   → sortie = sigmoid(raw) × noisy_ref ∈ [0, noisy]
                                       (recette p2, magnitude).
                           "linear" → sortie = raw (Re/Im bruts, complex mapping).
        """
        super().__init__()
        b = base_channels
        self.norm_groups = norm_groups
        self.head = head

        # --- Encodeur ---
        self.enc1 = EncoderBlock(in_channels, b, norm_groups=norm_groups)
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

        # --- Sortie : out_channels (1 = masque, 2 = Re/Im) ---
        self.out_conv = nn.Conv2d(b, out_channels, kernel_size=1)

    def forward(self, model_input: torch.Tensor,
                noisy_mag: torch.Tensor | None = None) -> torch.Tensor:
        """Forward du U-Net.

        Args:
            model_input : (B, C_in, F, T) — entrée du réseau (log1p magnitude en
                          mode masque ; Re/Im compressés en mode complex).
            noisy_mag   : (B, 1, F, T) — magnitude **non compressée** sur laquelle
                          s'applique le masque (mode "mask" uniquement). Ignoré
                          en mode "linear". Si None en mode "mask" : on retombe
                          sur `model_input`.

        Returns:
            mode "mask"   : (B, 1, F, T) — magnitude prédite ∈ [0, noisy_mag].
            mode "linear" : (B, C_out, F, T) — sortie brute (Re/Im du spectre).
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
        if self.head == "mask":
            return torch.sigmoid(raw) * noisy_mag
        # head == "linear" : sortie brute (Re/Im), pas de masquage.
        return raw


# ---------------------------------------------------------------------------
# Utilitaire
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Compte les paramètres entraînables d'un modèle PyTorch."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
