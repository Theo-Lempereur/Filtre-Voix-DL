"""U-Net pour le débruitage audio sur spectrogrammes de magnitude.

Ce fichier définit uniquement l'architecture du réseau.
Il ne contient PAS la boucle d'entraînement, la loss, l'optimizer,
ni les checkpoints — tout cela appartient à `src/train.py`.

Interface attendue par train.py :

    from src.model import UNet, count_parameters

    model = UNet(base_channels=32)
    predicted_clean_mag = model(batch["noisy_mag"])   # (B, 1, 257, T) → (B, 1, 257, T)

Dépendances : uniquement PyTorch — ce module n'importe rien de src/.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Bloc de base : DoubleConv
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """Deux convolutions consécutives, chacune suivie d'une BatchNorm et d'un ReLU.

    C'est la brique fondamentale du U-Net, utilisée à chaque niveau de
    l'encodeur, au bottleneck et dans chaque bloc du décodeur.

    Schéma :
        x → Conv(3×3) → BN → ReLU → Conv(3×3) → BN → ReLU → sortie

    Pourquoi deux convolutions et pas une ?
        Une seule conv apprend des "détecteurs simples" (bords, taches...).
        Deux convolutions empilées ont un champ récepteur plus large et peuvent
        apprendre des combinaisons de features plus complexes, sans exploser
        le nombre de paramètres comme le ferait une conv 5×5.

    Pourquoi padding=1 ?
        Une conv 3×3 sans padding réduit la taille spatiale de 2 pixels (1 de
        chaque côté). Avec padding=1 on ajoute un bord de zéros qui compense
        exactement ce rétrécissement → les dimensions (H, W) restent inchangées.

    Pourquoi bias=False ?
        La BatchNorm qui suit apprend déjà un biais (paramètre beta). Avoir un
        biais dans la conv ET dans la BN serait redondant et gaspillerait des
        paramètres.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            # Première conv : change le nombre de canaux
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),   # inplace=True économise de la mémoire GPU
            # Deuxième conv : raffine les features à nombre de canaux fixe
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Bloc encodeur : DoubleConv + MaxPool
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    """Un niveau de l'encodeur du U-Net.

    L'encodeur "descend" dans le U : à chaque niveau, il réduit la taille
    spatiale (÷2 dans chaque dimension) tout en augmentant le nombre de canaux.
    Cette compression force le réseau à apprendre une représentation de plus en
    plus abstraite et compacte du spectrogramme.

    Le bloc retourne UN TUPLE (skip, pooled) :
    - `skip`   : les features à pleine résolution → stockées pour le décodeur
    - `pooled` : les features downsamplees    → passées au niveau suivant

    Pourquoi stocker les features avant le pool (skip connection) ?
        MaxPool est une opération destructive : elle garde 1 valeur sur 4.
        Les détails fins (arêtes spectrales, transitoires) sont perdus.
        La skip connection permet au décodeur de les récupérer directement
        depuis l'encodeur, sans devoir les "deviner" depuis le bottleneck.

    Pourquoi MaxPool et pas une conv avec stride=2 ?
        MaxPool2d(2) n'a aucun paramètre apprenables → pas de surcoût.
        Il sélectionne la valeur maximale dans chaque bloc 2×2, ce qui
        équivaut à conserver le "pic d'activation" de chaque zone.
        Pour un spectrogramme de magnitude (≥ 0), MaxPool est très adapté.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip   = self.conv(x)   # features pleine résolution → skip connection
        pooled = self.pool(skip)  # features réduites → niveau suivant
        return skip, pooled


# ---------------------------------------------------------------------------
# Bloc décodeur : upsample + concat skip + DoubleConv
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """Un niveau du décodeur du U-Net.

    Le décodeur "remonte" dans le U : il agrandit progressivement les features
    (upsample ×2) et les combine avec les skip connections de l'encodeur pour
    récupérer les détails fins perdus lors du downsampling.

    Séquence :
        1. Upsample bilinéaire → colle la taille exacte de la skip connection
        2. Concaténation (cat) avec la skip → double le nombre de canaux
        3. DoubleConv → réduit les canaux et raffine

    Pourquoi interpolation bilinéaire plutôt que ConvTranspose2d ?
        ConvTranspose2d (upsample appris) a un paramètre `output_padding` qui
        doit être fixé selon la taille d'entrée exacte. Or nos spectrogrammes
        ont 257 fréquences (pas une puissance de 2) :
            MaxPool(2) sur 257 → 128   (et non 128.5)
            ConvTranspose × 2 → 256    (pas 257 !)
        Ce décalage d'un pixel provoquerait un crash silencieux lors de la
        concaténation. Avec F.interpolate(size=skip.shape[-2:]), on colle
        toujours exactement à la taille de la skip, quelle que soit la parité.

    Pourquoi align_corners=False ?
        Le comportement recommandé par PyTorch pour une interpolation d'image /
        feature map. Il aligne les cellules (centers) plutôt que les coins,
        ce qui donne de meilleurs résultats en pratique.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        """
        Args:
            in_channels   : canaux provenant du niveau inférieur (upsamplé)
            skip_channels : canaux de la skip connection correspondante
            out_channels  : canaux après DoubleConv
        """
        super().__init__()
        # La conv reçoit in_channels + skip_channels (après concaténation)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # 1. Upsample bilinéaire pour coller exactement à la taille de la skip
        #    (gère proprement les dimensions impaires comme 257 fréquences)
        x = F.interpolate(x, size=skip.shape[-2:],
                          mode='bilinear', align_corners=False)
        # 2. Concat sur la dimension canal (dim=1 = B, C, H, W)
        x = torch.cat([x, skip], dim=1)
        # 3. DoubleConv pour apprendre à fusionner les deux sources
        return self.conv(x)


# ---------------------------------------------------------------------------
# Réseau complet : UNet
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """U-Net 4 niveaux pour le débruitage de spectrogrammes de magnitude.

    Architecture d'ensemble :

        Entrée (B, 1, F, T)
            │
            ├─ EncoderBlock 1 ─── skip1 ───────────────────────┐
            ├─ EncoderBlock 2 ─── skip2 ──────────────────┐    │
            ├─ EncoderBlock 3 ─── skip3 ─────────────┐    │    │
            ├─ EncoderBlock 4 ─── skip4 ────────┐    │    │    │
            │                                   │    │    │    │
            └─ Bottleneck                        │    │    │    │
                    │                            │    │    │    │
                    ├─ DecoderBlock 4 ←─── skip4 ┘    │    │    │
                    ├─ DecoderBlock 3 ←─── skip3 ─────┘    │    │
                    ├─ DecoderBlock 2 ←─── skip2 ──────────┘    │
                    └─ DecoderBlock 1 ←─── skip1 ───────────────┘
                            │
                        Conv1×1 + Sigmoid = masque ∈ [0, 1]
                            │
                        masque × noisy_mag = sortie (B, 1, F, T)

    Sortie : masque IRM (Ideal Ratio Mask)
    ----------------------------------------
    Au lieu de prédire directement le spectrogramme propre, le réseau prédit
    un masque dans [0, 1]. Ce masque est multiplié élément par élément avec
    le spectrogramme bruité en entrée :

        mask = sigmoid(sortie_brute)       # valeurs entre 0 et 1
        clean_pred = mask × noisy_mag      # atténuation sélective

    Avantages :
    - Physiquement juste : débruiter = supprimer des fréquences, pas en inventer
    - La sortie est forcément non-négative (magnitude ≥ 0)
    - La sortie ne peut pas dépasser l'entrée (masque ≤ 1)
    - Le masque est visualisable et pédagogique

    Paramètre base_channels
    -----------------------
    Contrôle la largeur de tout le réseau. Doubler base_channels ≈ ×4 paramètres.

        base=16 → ~1,97M params   (rapide, bon pour tester)
        base=32 → ~7,85M params   (par défaut, bon équilibre)
        base=64 → ~31,4M params   (lourd, pour dataset large)
    """

    def __init__(self, base_channels: int = 32):
        """
        Args:
            base_channels : nombre de canaux au premier niveau de l'encodeur.
                            Tous les autres niveaux sont des multiples de cette valeur.
        """
        super().__init__()
        b = base_channels  # raccourci lisible

        # --- Encodeur (4 niveaux, descend dans le U) ---
        # Chaque EncoderBlock divise la résolution spatiale par 2
        # et multiplie le nombre de canaux par 2
        self.enc1 = EncoderBlock(1,   b)      # (B, 1, F, T)    → (B, b,   F/2,  T/2)
        self.enc2 = EncoderBlock(b,   b*2)    # (B, b, F/2,T/2) → (B, b*2, F/4,  T/4)
        self.enc3 = EncoderBlock(b*2, b*4)    # ...              → (B, b*4, F/8,  T/8)
        self.enc4 = EncoderBlock(b*4, b*8)    # ...              → (B, b*8, F/16, T/16)

        # --- Bottleneck (niveau le plus profond) ---
        # La représentation la plus compressée : encode l'essence du son
        # Exemple base=32 : (B, 256, 16, 31) → (B, 512, 16, 31)
        self.bottleneck = DoubleConv(b*8, b*16)

        # --- Décodeur (4 niveaux, remonte dans le U) ---
        # Chaque DecoderBlock double la résolution spatiale et
        # fusionne avec la skip connection correspondante
        self.dec4 = DecoderBlock(b*16, b*8,  b*8)   # reçoit bottleneck + skip4
        self.dec3 = DecoderBlock(b*8,  b*4,  b*4)   # reçoit dec4       + skip3
        self.dec2 = DecoderBlock(b*4,  b*2,  b*2)   # reçoit dec3       + skip2
        self.dec1 = DecoderBlock(b*2,  b,    b)      # reçoit dec2       + skip1

        # --- Couche de sortie ---
        # Conv 1×1 : compresse b canaux → 1 canal (la valeur du masque)
        # kernel_size=1 n'affecte pas les dimensions spatiales, juste les canaux
        self.out_conv = nn.Conv2d(b, 1, kernel_size=1)

    def forward(self, noisy_mag: torch.Tensor) -> torch.Tensor:
        """Passe avant du U-Net.

        Args:
            noisy_mag : spectrogramme de magnitude bruité, shape (B, 1, F, T)
                        valeurs ≥ 0 (magnitudes)

        Returns:
            Spectrogramme de magnitude prédit (propre), même shape (B, 1, F, T)
            Valeurs dans [0, noisy_mag] grâce au masque IRM.
        """
        # --- Phase descendante : encodeur ---
        # On collecte les skip connections pour les réutiliser en remontant
        skip1, x = self.enc1(noisy_mag)   # skip1: (B, b,   F,   T)
        skip2, x = self.enc2(x)           # skip2: (B, b*2, F/2, T/2)
        skip3, x = self.enc3(x)           # skip3: (B, b*4, F/4, T/4)
        skip4, x = self.enc4(x)           # skip4: (B, b*8, F/8, T/8)

        # --- Bottleneck ---
        x = self.bottleneck(x)            # x: (B, b*16, F/16, T/16)

        # --- Phase ascendante : décodeur ---
        # Chaque niveau reçoit les features du dessous + la skip connection
        # du niveau correspondant dans l'encodeur
        x = self.dec4(x, skip4)           # x: (B, b*8,  F/8, T/8)
        x = self.dec3(x, skip3)           # x: (B, b*4,  F/4, T/4)
        x = self.dec2(x, skip2)           # x: (B, b*2,  F/2, T/2)
        x = self.dec1(x, skip1)           # x: (B, b,    F,   T)

        # --- Sortie : masque IRM ---
        # sigmoid ramène les valeurs dans ]0, 1[ → masque d'atténuation
        mask = torch.sigmoid(self.out_conv(x))   # (B, 1, F, T) ∈ ]0, 1[

        # Multiplication élément par élément : on atténue sélectivement le bruit
        # Les zones où mask ≈ 1 : fréquences "propres" → conservées
        # Les zones où mask ≈ 0 : fréquences de bruit  → supprimées
        return mask * noisy_mag                  # (B, 1, F, T) ∈ [0, noisy_mag]


# ---------------------------------------------------------------------------
# Utilitaire
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Compte le nombre de paramètres apprenables d'un modèle PyTorch.

    Utile pour comparer des architectures ou vérifier que le modèle a la
    taille attendue avant de lancer un entraînement.

    Usage :
        model = UNet(base_channels=32)
        print(f"{count_parameters(model):,} paramètres")   # → 7,849,825
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
