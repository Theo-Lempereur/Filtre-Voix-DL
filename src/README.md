# src/ — Code Python réutilisable

Ce dossier contient les modules Python importés dans les notebooks.

## Fichiers à créer

| Fichier | Rôle |
|---|---|
| `model.py` | Définition de l'architecture U-Net (encoder, decoder, skip connections) |
| `dataset.py` | Classe `Dataset` PyTorch pour charger les paires (audio bruité, audio propre) |
| `utils.py` | Fonctions utilitaires : STFT/ISTFT, visualisation spectrogrammes, métriques |

## Utilisation dans un notebook

```python
# Ajouter src/ au path si besoin
import sys
sys.path.append('/content/drive/MyDrive/Filtre-Voix-DL/src')

from model import UNet
from dataset import AudioDataset
```
