# Memoire technique - Filtre-Voix-DL

Ce document decrit le fonctionnement technique du projet Filtre-Voix-DL : le
modele, les donnees, la generation du dataset, l'utilisation de Google Drive,
l'entrainement, l'inference et les limites actuelles.

Il se base sur le code Python, les configurations YAML et les documents Markdown
du depot. Les notebooks et fichiers d'infrastructure type RunPod ou shell ne
sont pas pris comme source principale ici, afin de documenter le fonctionnement
reutilisable du projet.

## 1. Vision Generale

Filtre-Voix-DL est un systeme de reduction de bruit pour la voix. Le probleme
est formule de facon supervisee : pour chaque exemple, le modele recoit une voix
bruitée et apprend a reconstruire la voix propre correspondante.

La chaine complete est la suivante :

```txt
audios de voix propres
        +
audios de bruits de fond
        |
        v
pretraitement en chunks fixes
        |
        v
melange voix + bruit avec SNR controle
        |
        v
paires noisy/clean alignees
        |
        v
entrainement U-Net sur spectrogrammes
        |
        v
checkpoint PyTorch
        |
        v
API d'inference audio -> audio debruite
```

Le projet ne depend donc pas d'un dataset public deja appaire. Il fabrique ses
propres exemples d'apprentissage a partir de deux banques separees : parole
propre et bruit de fond.

## 2. Organisation Des Donnees

### 2.1 Donnees Brutes

Les donnees brutes sont considerees comme volumineuses et non versionnees dans
Git. La configuration active pointe vers Google Drive via la variable logique
`${DRIVE_PROJECT}` :

```yaml
clean_preprocessing:
  input_dir: "${DRIVE_PROJECT}/data/jules/wav"

noise_preprocessing:
  input_dir: "${DRIVE_PROJECT}/data/jules/noise/free-sound"
```

Le dossier clean contient des enregistrements de voix. Le dossier noise contient
des bruits de fond : rue, pluie, foule, voitures, ambiance cafe, bruit blanc,
ou tout autre materiau sonore non vocal utile au denoising.

### 2.2 Google Drive

`src/config.py` detecte automatiquement la racine du projet Drive. L'ordre de
priorite est :

1. variable d'environnement `FILTRE_VOIX_DL_ROOT` ;
2. Google Drive monte dans Colab sous `/content/drive/MyDrive` ;
3. Google Drive for Desktop sur Windows ou macOS ;
4. fallback vers `/content/drive/MyDrive/Filtre-Voix-DL`.

Cette racine est stockee dans `DRIVE_PROJECT`.

Le projet separe volontairement :

- les artefacts a conserver et partager : checkpoints, logs, sorties ;
- les artefacts regenerables et tres volumineux : dataset genere, cache waveform.

Dans `src/config.py`, les checkpoints et logs pointent vers Drive :

```txt
CHECKPOINTS = <DRIVE_PROJECT>/checkpoints
OUTPUTS     = <DRIVE_PROJECT>/outputs
LOGS        = <DRIVE_PROJECT>/logs
```

Le dataset genere pointe vers le depot local :

```txt
data/processed/generated/
```

Ce choix evite de saturer Drive avec plusieurs gigaoctets de fichiers
reproductibles, tout en gardant les resultats importants d'entrainement sur un
support synchronise.

### 2.3 Configuration Dataset

Toute la generation est pilotee par :

```txt
configs/dataset_config.yaml
```

La configuration actuelle cible :

```txt
sample rate : 16000 Hz
duree       : 4.0 s
train       : 64000 paires
val         : 8000 paires
test        : 8000 paires
SNR global  : -5 a 20 dB
```

Les splits sources sont isoles avec :

```yaml
enforce_split_isolation: true
split_ratios: [0.8, 0.1, 0.1]
```

Cela signifie que les chunks clean et noise sont partitionnes avant generation.
Un meme chunk source ne sert donc pas a la fois au train et a l'evaluation, ce
qui limite les fuites de donnees.

## 3. Pipeline De Preparation Audio

Le pipeline dataset comprend trois etapes principales.

### 3.1 Preparation Des Voix Propres

La preparation clean transforme les fichiers sources en chunks WAV mono de
duree fixe.

Regles principales :

- decodage avec `soundfile`, puis fallback FFmpeg si necessaire ;
- conversion mono ;
- reechantillonnage a 16 kHz ;
- normalisation RMS optionnelle vers `-25 dB` ;
- gain maximum de `20 dB` pour eviter de gonfler du quasi-silence ;
- limitation de peak a `0.98` ;
- decoupage en chunks de 4 secondes ;
- rejet des segments trop silencieux.

La configuration active utilise notamment :

```yaml
normalize_rms: true
target_rms_db: -25.0
silence_threshold_db: -50.0
min_non_silent_ratio: 0.05
pad_short_files: false
```

Les fichiers courts ne sont pas completes par defaut : une voix trop courte est
ignoree plutot que transformee en exemple contenant beaucoup de silence.

### 3.2 Preparation Des Bruits

La preparation noise suit la meme logique generale, mais avec des seuils plus
adaptés aux ambiances sonores.

Differences importantes :

- les bruits tres courts peuvent etre repetes ;
- le seuil de silence est plus bas (`-60 dB`) ;
- le ratio non silencieux minimum est `0.03`.

Cette tolerance est logique : un bruit de fond peut etre plus faible ou plus
stationnaire qu'une voix, et le repeter est acceptable pour fabriquer un fond
sonore continu.

### 3.3 Generation Des Paires Noisy/Clean

La generation finale selectionne aleatoirement un chunk clean et un chunk noise,
les croppe a 4 secondes, applique eventuellement des augmentations, puis produit :

```txt
noisy = clean + noise_scaled
target = clean
```

Les sorties sont :

```txt
data/processed/generated/train/noisy/
data/processed/generated/train/clean/
data/processed/generated/val/noisy/
data/processed/generated/val/clean/
data/processed/generated/test/noisy/
data/processed/generated/test/clean/
```

Chaque fichier noisy et clean porte le meme nom :

```txt
train_00000042.wav
```

Le dataset PyTorch peut donc apparier les fichiers par nom.

## 4. Melange Et SNR

Le mixage est controle par le rapport signal/bruit.

La formule utilisee est :

```txt
SNR dB = 20 * log10(clean_rms / noise_rms)
target_noise_rms = clean_rms / 10^(SNR / 20)
```

Le bruit est donc mis a l'echelle pour atteindre le SNR cible, puis additionne a
la voix propre.

Avant d'ecrire le fichier, le pipeline verifie :

- longueur exacte ;
- mono ;
- absence de NaN ou Inf ;
- RMS clean au-dessus du seuil ;
- RMS noise au-dessus du seuil ;
- peak final sous la limite.

Si le melange risque de clipper, un gain commun est applique. Quand
`apply_gain_to_target` vaut `true`, le meme gain est applique au noisy, au clean
target et au bruit scale. Cela preserve l'alignement physique entre l'entree et
la cible.

## 5. Templates De Generation

Le mode novice de la GUI ecrit `generation.template_mix` dans le YAML. La config
actuelle contient trois profils.

### Classic

Profil simple : voix propre + bruit, sans degradation microphone.

```txt
count : 48000
SNR   : -5 a 20 dB
augmentations : desactivees
```

### Microphone Mode

Profil plus realiste pour simuler des captures imparfaites :

```txt
count : 10667
SNR   : -3 a 18 dB
augmentations clean/post-noisy : activees
```

Il active notamment compression, EQ, filtrage telephone, saturation, codec,
dropout et quantization avec des probabilites plus elevees.

### Very Noisy

Profil plus difficile :

```txt
count : 21333
SNR   : -10 a 12 dB
augmentations : desactivees
```

Le bruit peut y etre beaucoup plus present que dans le profil classique.

Le choix du template est fait proportionnellement dans chaque split. Ainsi train,
validation et test conservent la meme repartition de styles.

## 6. Augmentations Audio

Les augmentations cherchent a rapprocher les donnees synthetiques de conditions
reelles de capture.

Effets disponibles :

- gain global ;
- egalisation trois bandes ;
- filtre telephone/VoIP ;
- compression statique ;
- saturation douce ;
- hard clipping ;
- reverb legere ;
- quantization basse resolution ;
- dropouts courts ;
- simulation codec FFmpeg (`opus`, `mp3`).

Chaque augmentation est probabiliste. Les parametres tires sont sauvegardes dans
les metadonnees de generation, par exemple gain, ratio de compression, frequence
de filtrage, codec choisi ou bitrate.

Le codec n'est autorise que sur l'audio final noisy dans la generation actuelle,
afin d'eviter de degrader directement la cible clean.

## 7. Metadonnees Et Reproductibilite

La generation ecrit :

```txt
data/metadata/generated_metadata.csv
data/metadata/generated_errors.csv
data/processed/generated/config_used.yaml
```

`generated_metadata.csv` contient notamment :

- `sample_id` ;
- split ;
- seed ;
- chemins noisy, clean, noise ;
- fichier clean source ;
- fichier noise source ;
- offsets de crop ;
- sample rate, duree, nombre d'echantillons ;
- SNR reel ;
- RMS clean/noise/noisy ;
- peaks ;
- gain global anti-clipping ;
- augmentations appliquees ;
- metadonnees JSON des augmentations ;
- nom du template.

`config_used.yaml` est essentiel : il fige la configuration exacte qui a servi a
produire un dataset donne. Meme si `configs/dataset_config.yaml` change ensuite,
on peut toujours retrouver les conditions de generation d'un run.

## 8. Dataset PyTorch

`src/dataset.py` fournit `PairedAudioDataset`.

Deux modes existent :

1. lecture directe des fichiers audio noisy/clean ;
2. lecture depuis cache memmap.

En lecture directe, le dataset :

- liste les fichiers audio ;
- apparie noisy et clean par nom identique si possible ;
- bascule sur un appairage par index si aucun nom ne matche ;
- charge en mono 16 kHz ;
- croppe ou pad a `CLIP_SAMPLES` ;
- renvoie soit les waveforms, soit les spectrogrammes selon le mode demande.

Dans le pipeline d'entrainement optimise, le dataset renvoie surtout :

```python
{
    "name": name,
    "noisy_wav": noisy_tensor,
    "clean_wav": clean_tensor,
}
```

La STFT est ensuite calculee sur GPU dans `src/train.py`. Cela evite de faire la
transformation spectrale dans les workers CPU.

### Cache Waveform

`scripts/build_wav_cache.py` cree, pour chaque split :

```txt
train_noisy.npy
train_clean.npy
train_names.json
val_noisy.npy
val_clean.npy
val_names.json
```

Les tableaux `.npy` sont des memmaps `float16` de forme :

```txt
(nombre_exemples, CLIP_SAMPLES)
```

Le cache supprime le cout de decodage audio pendant l'entrainement. La variable
`FILTRE_VOIX_DL_CACHE` peut deplacer ce cache vers un disque persistant ou plus
rapide.

## 9. Representation Audio Du Modele

Le projet travaille en 16 kHz mono.

Constantes principales :

```txt
SAMPLE_RATE   = 16000
CLIP_DURATION = 4.0 s
CLIP_SAMPLES  = 64000
N_FFT         = 512
HOP_LENGTH    = 128
WIN_LENGTH    = 512
```

La STFT produit une magnitude et une phase :

```txt
waveform -> STFT complexe -> magnitude + phase
```

En mode baseline, l'entree reseau est :

```txt
model_input = log1p(noisy_mag)
```

La compression `log1p` rend la distribution des magnitudes plus facile a
apprendre. Les grandes amplitudes dominent moins le gradient, et les zones de
faible energie, souvent audibles comme du bruit residuel, restent visibles pour
le modele.

La cible de sortie reste en magnitude lineaire.

## 10. Architecture Du U-Net

Le modele principal est defini dans `src/model.py`.

La baseline est un U-Net 2D sur spectrogrammes :

```txt
input : (B, 1, F, T)
output: (B, 1, F, T)
```

Avec `base_channels = 32`, les largeurs sont :

```txt
encodeur 1 : 32
encodeur 2 : 64
encodeur 3 : 128
encodeur 4 : 256
bottleneck : 512
decodeur   : 256 -> 128 -> 64 -> 32
```

Chaque bloc `DoubleConv` applique :

```txt
Conv2d 3x3 -> GroupNorm -> ReLU
Conv2d 3x3 -> GroupNorm -> ReLU
```

L'encodeur ajoute un `MaxPool2d(2)`. Le decodeur fait un upsampling bilineaire,
concatene le skip connection correspondant, puis applique un `DoubleConv`.

La sortie est une convolution `1x1`.

### Pourquoi GroupNorm

`GroupNorm` ne depend pas des statistiques de batch. C'est important pour ce
projet car les batch sizes peuvent etre limites par la VRAM, surtout avec des
spectrogrammes 4 secondes et une loss MR-STFT.

Le nombre de groupes est ajuste pour diviser le nombre de canaux, ce qui evite
les erreurs d'initialisation quand `base_channels` change.

### Tete De Sortie Baseline

En mode `mask`, le reseau produit un tenseur brut, puis :

```txt
pred_mag = sigmoid(raw) * noisy_mag
```

Le modele apprend donc un masque reel entre 0 et 1. Il ne peut pas amplifier un
bin spectral au-dessus de la magnitude bruitée. Cette contrainte stabilise
l'apprentissage mais impose une limite : le modele travaille surtout par
attenuation et conserve la phase du signal bruite.

## 11. Mode Complexe Experimental

`src/train.py` supporte aussi :

```txt
output_mode = "complex"
```

Dans ce mode :

- l'entree contient deux canaux : reel et imaginaire du spectre bruite ;
- la sortie contient deux canaux : reel et imaginaire du spectre propre estime ;
- la tete du U-Net devient lineaire ;
- la phase peut etre corrigee, pas seulement reutilisee.

La compression utilise :

```txt
Z = |S|^c * exp(j theta)
c = 0.3
```

Puis le signal est decompresse avant l'ISTFT. Ce mode peut theoriquement
depasser le plafond du masque de magnitude, car il apprend aussi la phase. Il
est cependant plus sensible et l'API actuelle reconstruit principalement la
baseline magnitude classique.

## 12. Loss Et Metriques

La loss baseline est combinee :

```txt
L = w_mse * MSE(pred_mag, clean_mag)
  + w_l1c * L1(pred_mag^0.3, clean_mag^0.3)
  + w_mr * MR-STFT(pred_wav, clean_wav)
```

Poids actuels :

```txt
MSE      : 0.0
L1 comp  : 1.0
MR-STFT  : 1.0
```

### L1 Sur Magnitude Compressee

La composante `mag^0.3` reduit l'avantage des bins a haute energie. Elle force
le modele a traiter aussi les composantes faibles, qui peuvent correspondre a du
bruit audible.

Un epsilon est ajoute avant la puissance pour eviter des gradients infinis en
zero.

### MR-STFT

La Multi-Resolution STFT loss compare les log-magnitudes et magnitudes sur
plusieurs resolutions :

```txt
FFT sizes : 256, 512, 1024
hop       : n_fft / 4
```

Elle capture a la fois :

- les details temporels rapides ;
- la structure frequentielle fine ;
- les artefacts spectraux audibles.

### SI-SDR Et SI-SDRi

Le training loggue :

- `val_si_sdr` : SI-SDR approximee sur magnitude ;
- `noisy_si_sdr` : baseline du signal bruite ;
- `val_si_sdri` : amelioration par rapport au bruité ;
- `val_si_sdr_wav` si une waveform predite est disponible.

`SI-SDRi` est particulierement utile car elle mesure le gain du modele par
rapport a l'entree bruitée, pas seulement la qualite absolue.

## 13. Boucle D'entrainement

`src/train.py` orchestre l'entrainement.

Valeurs principales par defaut :

```txt
batch_size       = 24
learning rate    = 1e-3
weight decay     = 1e-4
epochs max       = 50
base_channels    = 32
grad clip        = 1.0
early patience   = 7
LR patience      = 3
LR factor        = 0.5
warmup epochs    = 2
```

Le device est choisi dans cet ordre :

```txt
CUDA -> MPS -> CPU
```

Sur CUDA :

- TF32 est active ;
- la mixed precision BF16 est activee par defaut ;
- `torch.compile` est optionnel.

### Optimiseur

L'optimiseur est `AdamW`. Le weight decay est decouple, ce qui est plus propre
que la penalisation L2 classique dans Adam.

### Scheduler

Le scheduler est specifique au projet : `DualMetricPlateauScheduler`.

Il reduit le learning rate uniquement si deux metriques plateauent :

```txt
val_loss   ne s'ameliore plus
val_si_sdri ne s'ameliore plus
```

Cette logique evite de baisser le LR trop tot quand la loss stagne mais que le
gain perceptif continue de progresser, ou inversement.

### Early Stopping

`OverfitMonitor` surveille :

- gap train/validation ;
- ratio val/train ;
- validation qui monte ;
- divergence train qui baisse pendant que val monte ;
- plateau de loss ;
- plateau de metrique secondaire.

L'arret anticipe exige aussi la stagnation des signaux suivis, ce qui rend la
decision moins brutale.

## 14. Checkpoints Et Logs

Chaque run ecrit :

```txt
checkpoints/<run_id>/last.pt
checkpoints/<run_id>/best.pt
checkpoints/<run_id>/best_si_sdri.pt
checkpoints/<run_id>/epoch_XXX.pt
```

Un checkpoint contient :

- poids du modele ;
- etat optimizer ;
- etat scheduler ;
- epoch ;
- meilleure `val_loss` ;
- meilleure `val_si_sdri` ;
- `run_id` ;
- config ;
- historique.

Les sauvegardes sont atomiques : le fichier est d'abord ecrit en `.tmp`, puis
remplace l'ancien checkpoint. Cela limite le risque de fichier corrompu en cas
de coupure.

Les logs vont vers deux destinations :

- Weights & Biases si disponible ;
- JSONL local dans `<DRIVE_PROJECT>/logs/<run_id>/history.jsonl`.

Le JSONL reste exploitable meme sans connexion reseau.

## 15. Reprise, Lock Et Arret Propre

L'entrainement peut reprendre depuis :

```txt
last
best
best_si_sdri
chemin absolu vers un .pt
```

Lors d'une reprise, la config sauvegardee dans le checkpoint sert de base. Les
flags CLI fournis ensuite la surchargent.

Un lock d'entrainement evite de lancer deux runs concurrents qui se marcheraient
dessus. Il contient un heartbeat et devient stale apres plusieurs minutes sans
mise a jour.

Un systeme de stop flag permet aussi a la GUI ou a un outil externe de demander
un arret propre. Le run sauvegarde `last.pt`, puis sort.

## 16. Interface Graphique

Le depot contient deux familles d'interfaces :

- GUI dataset : configuration novice/expert du YAML de generation ;
- GUI training : formulaire d'hyperparametres, reprise de runs, file d'attente.

Le mode novice du dataset demande surtout le nombre d'exemples par style :

- Classic ;
- Microphone mode ;
- Very noisy.

La GUI traduit ces choix en `template_mix` dans `configs/dataset_config.yaml`.

Le mode expert expose les chemins, splits, SNR, durees, probabilites
d'augmentation, compression, codec et parametres techniques.

## 17. Inference Locale Et API

L'inference est dans `serve/inference.py` et `serve/server.py`.

Pipeline :

```txt
bytes audio
-> decode librosa
-> mono 16 kHz
-> crop/pad centre a 4 s
-> STFT
-> magnitude + phase
-> log1p(magnitude)
-> U-Net
-> magnitude predite
-> reconstruction avec phase noisy
-> WAV PCM16 en memoire
```

L'API FastAPI expose :

```txt
GET  /          mini client HTML
GET  /health    etat serveur + checkpoint
GET  /models    checkpoints detectes
POST /denoise   upload audio -> WAV debruite
```

Formats d'entree acceptes :

- WAV ;
- FLAC ;
- MP3 si decodeur disponible ;
- OGG si decodeur disponible.

La sortie est toujours :

```txt
WAV mono, 16 kHz, PCM 16 bits
```

Le serveur peut scanner plusieurs dossiers de modeles via `MODEL_DIRS`, utiliser
`CKPT_PATH` comme modele par defaut, et choisir `cpu` ou `cuda` via `DEVICE`.

## 18. Limites Actuelles

Les limites importantes sont :

- duree fixe de 4 secondes en inference ;
- pas encore de fenetrage glissant pour les clips longs ;
- reconstruction baseline avec phase bruitée ;
- le masque `sigmoid` ne peut qu'attenuer la magnitude ;
- l'API actuelle cible surtout les checkpoints U-Net magnitude ;
- qualite dependante de la diversite des voix et bruits sources ;
- dataset synthetique : il faut comparer regulierement avec des captures reelles.

La limite de phase est la plus structurante. En mode `mask`, le modele peut
nettoyer beaucoup de bruit, mais ne reconstruit pas une phase propre. Le mode
complexe est une piste pour lever cette contrainte.

## 19. Commandes De Reference

Installer :

```bash
pip install -r requirements.txt
```

Ouvrir la GUI dataset :

```bash
python scripts/run_full_pipeline.py --gui
```

Generer tout le dataset :

```bash
python scripts/run_full_pipeline.py --compile --reset-all
```

Verifier les donnees :

```bash
python scripts/check_dataset.py
```

Construire le cache :

```bash
python scripts/build_wav_cache.py --splits train val
```

Entrainer :

```bash
python scripts/train_local.py --run-id p2_baseline --epochs 50
```

Reprendre :

```bash
python scripts/train_local.py --run-id p2_baseline --resume last
```

Servir :

```bash
python -m uvicorn serve.server:app --host 127.0.0.1 --port 8000 --workers 1
```

Tester l'API :

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/denoise -F "file=@mon_audio.wav" -o sortie.wav
```

## 20. Lecture Rapide Des Fichiers

```txt
src/model.py
  U-Net, blocs encodeur/decodeur, tete mask ou linear.

src/train.py
  boucle d'entrainement, STFT GPU, loss, scheduler, checkpoints.

src/dataset.py
  appairage noisy/clean, lecture WAV ou memmap, crop/pad.

src/metrics.py
  MSE, L1 compressee, MR-STFT, SI-SDR, OverfitMonitor.

src/config.py
  chemins Drive, constantes audio, hyperparametres par defaut.

src/dataset_builder/
  chargement audio, chunking, augmentations, mixage SNR, generation.

configs/dataset_config.yaml
  source de verite du dataset genere.

gui/
  interfaces de configuration dataset et entrainement.

serve/
  inference en memoire et API FastAPI.
```

## 21. Conclusion

Filtre-Voix-DL est construit comme une chaine complete, pas seulement comme un
modele PyTorch isole. Sa force principale est la tracabilite : chaque dataset
est genere depuis une configuration sauvegardee, chaque sample garde ses
metadonnees, chaque run conserve sa config et son historique.

La baseline U-Net magnitude est stable, simple a servir et adaptee a une premiere
version de filtre vocal. Les prochaines ameliorations naturelles sont le
traitement de clips longs, l'evaluation sur audios reels, et la consolidation du
mode complexe pour depasser les limites de la phase bruitée.
