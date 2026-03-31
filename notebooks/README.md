# notebooks/ — Notebooks du projet

Chaque notebook correspond à une étape du pipeline.

| Notebook | Étape |
|---|---|
| `01_exploration_audio.ipynb` | Charger un fichier audio, visualiser la waveform et le spectrogramme, comprendre la STFT |
| `02_generation_dataset.ipynb` | Générer le dataset synthétique : mélanger voix propres + bruits à différents SNR |
| `03_entrainement.ipynb` | Entraîner le U-Net sur le dataset généré, sauvegarder les checkpoints |
| `04_evaluation.ipynb` | Évaluer le modèle : métriques, écoute des résultats, comparaison avant/après |
| `05_demo.ipynb` | Démo interactive : charger un audio bruité → débruiter → écouter le résultat |
