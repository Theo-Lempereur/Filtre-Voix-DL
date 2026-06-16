# Modèles NVIDIA, clés API et quota

Ce document décrit la couche ajoutée pour servir les modèles **NVIDIA Maxine**
(Background Noise Removal + Studio Voice) à côté des checkpoints locaux, avec
gestion de clé et limite d'usage.

## Logique d'usage (en bref)

| Cas | Clé utilisée | Limite |
|---|---|---|
| L'utilisateur fournit **sa** clé (`X-API-Key`) | la sienne | **aucune** de notre côté (NVIDIA applique la sienne) |
| L'utilisateur ne fournit **rien** | notre clé partagée `NVIDIA_API_KEY` | **`FREE_DAILY_LIMIT` / jour / IP** (défaut : 5) |
| Aucune clé (ni la sienne, ni la nôtre) | — | requête refusée (503) |

Les modèles **locaux** (`.pt`) ne sont pas concernés : ni clé, ni quota.

## 1. Obtenir une clé NVIDIA gratuite

1. Créer un compte gratuit sur <https://build.nvidia.com> (NVIDIA Developer
   Program, sans carte bancaire).
2. Ouvrir un modèle, cliquer **Get API Key**. La clé est préfixée `nvapi-`.
   Le tier gratuit donne ~1 000 crédits (jusqu'à 5 000 sur demande) et ~40
   requêtes/minute — calibré pour des tests, pas pour de la production en boucle.

> ⚠️ La clé partagée est *votre* clé. Comme tous les visiteurs « sans clé »
> tapent dessus, gardez `FREE_DAILY_LIMIT` bas pour ne pas épuiser le budget.

## 2. Récupérer les function-id

Chaque NIM a un `function-id` propre, affiché sur la page **API** du modèle :

- BNR : <https://build.nvidia.com/nvidia/bnr/api>
- Studio Voice : <https://build.nvidia.com/nvidia/studiovoice/api>

Copiez-les dans `NVIDIA_BNR_FUNCTION_ID` / `NVIDIA_STUDIOVOICE_FUNCTION_ID`.

## 3. Configuration (`.env` du serveur)

Le serveur lit ces variables via `load_dotenv` (voir `.env.example`) :

```ini
NVIDIA_API_KEY=nvapi-xxxxxxxx            # notre clé partagée (repli gratuit)
NVIDIA_BNR_FUNCTION_ID=........-....-....
NVIDIA_STUDIOVOICE_FUNCTION_ID=........-....-....
FREE_DAILY_LIMIT=5                       # quota/jour/IP en mode partagé
```

Le `.env` est **ignoré par git** : aucune clé ne doit être committée.

## 4. Où vivent les clés (et le malentendu « GitHub Secrets »)

Au **runtime**, un serveur lit ses secrets depuis **son environnement**, pas
depuis GitHub. Les **GitHub Secrets** (Actions) ne sont accessibles que dans un
workflow CI/CD.

- **Déploiement manuel (git pull sur le VPS)** → mettez les clés dans le `.env`
  du serveur, ou dans les `Environment=` du service systemd
  (`serve/INTEGRATION.md`). **N'utilisez pas** les GitHub Secrets ici.
- **Déploiement via GitHub Actions** → là, stockez les clés en *repository
  secrets* (Settings → Secrets and variables → Actions) et faites injecter par
  le workflow dans l'environnement du serveur (ex. génération du `.env` ou
  variables d'environnement du service). C'est le seul cas où GitHub Secrets
  est pertinent.

## 5. Vérifier une clé utilisateur

Une validation de **forme** (préfixe `nvapi-`) est faite immédiatement. La
validation **autoritaire** se fait au premier appel : si NVIDIA renvoie
`UNAUTHENTICATED`, l'API répond **401** (« Clé NVIDIA invalide… »). On évite
ainsi de brûler un crédit juste pour tester une clé.

> Le **solde restant** d'une clé utilisateur n'est pas exposé par l'API NVIDIA
> (seulement sur son dashboard) : on ne peut donc pas l'afficher. Le compteur de
> quota ne concerne que **notre** clé partagée, lisible via `GET /quota`.

## 6. Endpoints

- `GET /models` — checkpoints locaux **+** modèles NVIDIA (`provider: "nvidia"`,
  `requires_key: true`, `configured: <function-id présent ?>`).
- `GET /quota` — `{limit, used, remaining, shared_key_configured}` pour l'IP
  appelante.
- `POST /denoise` — champs `file`, `model_id`, `intensity` (option BNR 0–1),
  en-tête optionnel `X-API-Key`. Réponse : WAV. En mode partagé, l'en-tête
  `X-Quota-Remaining` indique le reste du jour.

## 7. Dépendances

```bash
pip install -r requirements-serve.txt   # ajoute grpcio + protobuf
```

Les stubs gRPC sont vendorisés dans `serve/providers/_vendor/` (générés depuis
NVIDIA-Maxine/nim-clients, licence MIT) : pas besoin de recompiler les protos.

## 8. Limites connues

- Identification du quota **par IP** (zéro inscription) : contournable, et une
  IP partagée (entreprise, 4G) compte pour un seul utilisateur. Acceptable pour
  du « test gratuit ».
- `X-Forwarded-For` n'est fiable que **derrière** un proxy de confiance (nginx).
  N'exposez pas uvicorn directement à Internet.
- Maxine attend du 48 kHz : l'audio entrant est ré-échantillonné avant envoi.
