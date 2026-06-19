# Intégration de l'API de débruitage voix

Guide pour appeler l'API de débruitage depuis ton site web.

L'API prend un fichier audio (voix bruitée) et renvoie un WAV débruité.

---

## 1. Démarrer l'API

### Configuration (`.env` à la racine du projet)

```ini
CKPT_PATH=G:\Mon Drive\Filtre-Voix-DL\checkpoints\p2_combo_mrstft\best.pt
DEVICE=cpu
CORS_ORIGINS=*
```

| Variable | Rôle |
|---|---|
| `CKPT_PATH` | Chemin du modèle. Accepte un fichier `.pt`, un nom sans extension (`best`), ou un dossier de run (prend `best.pt` automatiquement). |
| `DEVICE` | `cpu` ou `cuda`. Vide = auto-détection. |
| `CORS_ORIGINS` | Origines autorisées, séparées par virgule. `*` = tout (dev). En prod : `https://monsite.com`. |

### Lancer le serveur

```bash
# Dev (rechargement auto, surveille seulement serve/) :
.venv/Scripts/python.exe -m uvicorn serve.server:app --port 8000 --reload --reload-dir serve

# Prod (un worker, pas de reload) :
.venv/Scripts/python.exe -m uvicorn serve.server:app --host 127.0.0.1 --port 8000 --workers 1
```

Au démarrage, tu dois voir :
```
Model ready | device=cpu | input_repr=log1p | ckpt=...\best.pt
```

---

## 2. Endpoints

### `GET /health`

Vérifie que le serveur tourne et quel modèle est chargé.

**Réponse `200` :**
```json
{
  "status": "ok",
  "device": "cpu",
  "ckpt": "G:\\Mon Drive\\...\\best.pt",
  "input_repr": "log1p"
}
```
`503` si le modèle n'est pas chargé.

---

### `POST /denoise`

Débruite un fichier audio.

| | |
|---|---|
| **Méthode** | `POST` |
| **Content-Type** | `multipart/form-data` |
| **Champ** | `file` — le fichier audio |
| **Formats entrée** | WAV, FLAC (natif) · MP3, OGG (si ffmpeg installé) |
| **Réponse** | `200` — corps = WAV binaire (`audio/wav`), mono 16 kHz PCM 16 bits |

> ⚠️ **L'audio est tronqué/complété à 4 secondes** (limitation actuelle). Un clip plus long est coupé au centre ; plus court, il est complété par du silence.

**Codes d'erreur :**

| Code | Cause |
|---|---|
| `400` | Fichier illisible, vide, trop court, ou format non supporté |
| `405` | Mauvaise méthode (ex : `GET /denoise` au lieu de `POST`) |
| `422` | Champ `file` manquant |
| `500` | Erreur interne du modèle |
| `503` | Modèle pas encore chargé |

Le corps d'erreur est un JSON : `{"detail": "message explicatif"}`.

---

## 3. Intégration JavaScript (vanilla)

```js
/**
 * Envoie un fichier audio à l'API et renvoie un Blob WAV débruité.
 * @param {File|Blob} audioFile  fichier issu d'un <input type="file"> ou d'un enregistrement
 * @param {string} apiBase       URL de base de l'API (ex: "/api" derrière nginx)
 * @returns {Promise<Blob>}      le WAV débruité
 */
async function denoiseAudio(audioFile, apiBase = "/api") {
  const formData = new FormData();
  formData.append("file", audioFile); // le champ DOIT s'appeler "file"

  const res = await fetch(`${apiBase}/denoise`, {
    method: "POST",
    body: formData,
    // Ne PAS définir Content-Type : le navigateur le règle avec la bonne boundary.
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: `Erreur ${res.status}` }));
    throw new Error(err.detail);
  }

  return await res.blob(); // audio/wav
}
```

### Exemple d'usage : upload + lecture du résultat

```html
<input type="file" id="audioInput" accept="audio/*">
<button id="denoiseBtn">Débruiter</button>
<audio id="result" controls></audio>

<script>
  document.getElementById("denoiseBtn").addEventListener("click", async () => {
    const file = document.getElementById("audioInput").files[0];
    if (!file) return alert("Choisis un fichier audio.");

    try {
      const cleanBlob = await denoiseAudio(file, "http://127.0.0.1:8000");
      const url = URL.createObjectURL(cleanBlob);
      document.getElementById("result").src = url; // lecture dans le navigateur
    } catch (e) {
      alert("Échec : " + e.message);
    }
  });
</script>
```

### Proposer le téléchargement du fichier débruité

```js
const cleanBlob = await denoiseAudio(file);
const url = URL.createObjectURL(cleanBlob);
const a = Object.assign(document.createElement("a"), {
  href: url,
  download: "voix_debruitee.wav",
});
a.click();
URL.revokeObjectURL(url);
```

---

## 4. Intégration React

```jsx
import { useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE ?? "/api";

export function Denoiser() {
  const [busy, setBusy] = useState(false);
  const [resultUrl, setResultUrl] = useState(null);
  const [error, setError] = useState(null);

  async function handleFile(e) {
    const file = e.target.files?.[0];
    if (!file) return;

    setBusy(true);
    setError(null);
    setResultUrl(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${API_BASE}/denoise`, { method: "POST", body: formData });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: `Erreur ${res.status}` }));
        throw new Error(err.detail);
      }

      const blob = await res.blob();
      setResultUrl(URL.createObjectURL(blob));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <input type="file" accept="audio/*" onChange={handleFile} disabled={busy} />
      {busy && <p>Traitement…</p>}
      {error && <p style={{ color: "red" }}>{error}</p>}
      {resultUrl && <audio src={resultUrl} controls />}
    </div>
  );
}
```

---

## 5. Test rapide en ligne de commande

```bash
# Santé
curl http://127.0.0.1:8000/health

# Débruitage (sauvegarde la sortie)
curl -X POST http://127.0.0.1:8000/denoise -F "file=@mon_audio.wav" -o sortie.wav
```

Tu peux aussi tester visuellement avec `serve/test_client.html` (double-clic) ou via la doc
interactive : `http://127.0.0.1:8000/docs`.

---

## 6. Mise en production (même serveur que le site)

Schéma recommandé : le site et l'API tournent sur la même machine, nginx route `/api/` vers l'API.

```
Navigateur ──► nginx (443) ──┬──► site web (fichiers statiques)
                             └──► /api/ ──► 127.0.0.1:8000 (API FastAPI)
```

### Nginx

Dans ton bloc `server { }` existant :

```nginx
location /api/ {
    proxy_pass         http://127.0.0.1:8000/;
    proxy_http_version 1.1;
    proxy_set_header   Host $host;
    proxy_set_header   X-Real-IP $remote_addr;
    proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header   X-Forwarded-Proto $scheme;

    client_max_body_size 10m;   # uploads audio
    proxy_read_timeout   30s;   # l'inférence peut prendre 1-3 s sur CPU
}
```

> La barre oblique finale de `proxy_pass http://127.0.0.1:8000/;` retire `/api/` avant de
> transmettre : `/api/denoise` → `/denoise`. Côté JS, utilise donc `apiBase = "/api"` et
> tu n'as plus de souci de CORS (même origine).

### Lancer l'API en service (systemd, Linux)

`/etc/systemd/system/filtre-voix-api.service` :

```ini
[Unit]
Description=Filtre-Voix API
After=network.target

[Service]
User=www-data
WorkingDirectory=/chemin/vers/Filtre-Voix-DL
ExecStart=/chemin/vers/.venv/bin/uvicorn serve.server:app --host 127.0.0.1 --port 8000 --workers 1
Restart=on-failure
RestartSec=5
Environment="CKPT_PATH=/chemin/absolu/best.pt"
Environment="DEVICE=cpu"
Environment="CORS_ORIGINS=https://monsite.com"

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now filtre-voix-api
sudo systemctl status filtre-voix-api
```

> Sur Windows, utilise plutôt **NSSM** ou le Planificateur de tâches pour lancer la même
> commande uvicorn au démarrage.

### CORS

- **API derrière `/api/` sur le même domaine** (recommandé) → même origine, aucun réglage CORS nécessaire, garde `CORS_ORIGINS=*` ou retire-le.
- **API sur un domaine/port différent** → mets `CORS_ORIGINS=https://monsite.com` pour autoriser ton site.

---

## 7. Notes & limites

- **Durée fixe 4 s** : limitation actuelle. Le traitement de clips plus longs (fenêtrage glissant) est prévu plus tard.
- **Sortie toujours en WAV 16 kHz mono**, quel que soit le format d'entrée.
- **MP3 / OGG en entrée** nécessitent **ffmpeg** installé sur le serveur (`apt install ffmpeg`). WAV et FLAC fonctionnent sans.
- **Un seul worker** : les requêtes sont traitées séquentiellement. Suffisant pour un usage modéré ; pour de la charge, voir mise à l'échelle (plusieurs workers CPU ou file d'attente).
- **Compatibilité modèle** : `CKPT_PATH` doit pointer vers un checkpoint entraîné avec le `model.py` actuel (architecture GroupNorm). Les runs `p3_bigru_softplus` / `p4_cirm` (archi étendue) ne se chargeront pas.
```
