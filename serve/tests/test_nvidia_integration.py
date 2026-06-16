"""Test d'intégration de la couche clé+quota+NVIDIA, sans torch ni réseau.

Lancer depuis la racine du repo :

    python serve/tests/test_nvidia_integration.py

On mocke les modules lourds (torch, librosa) et serve.inference pour pouvoir
importer serve.server, puis on monkeypatche nvidia_maxine.enhance.
"""
import os
import sys
import types
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# --- Mocks des dépendances lourdes AVANT d'importer serve.server ------------
for name in ("torch", "librosa"):
    mod = types.ModuleType(name)
    if name == "torch":
        mod.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules[name] = mod

fake_inf = types.ModuleType("serve.inference")
fake_inf.denoise_bytes = lambda *a, **k: b"LOCALWAV"
fake_inf.load_model = lambda *a, **k: (object(), "log1p", "/x/best.pt")
fake_inf.resolve_ckpt_path = lambda p: p
sys.modules["serve.inference"] = fake_inf

# --- Environnement de test ---------------------------------------------------
os.environ["QUOTA_DB_PATH"] = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False).name
os.environ["FREE_DAILY_LIMIT"] = "2"
os.environ["NVIDIA_API_KEY"] = "nvapi-SHARED"
os.environ["NVIDIA_BNR_FUNCTION_ID"] = "fid-bnr"
os.environ["NVIDIA_STUDIOVOICE_FUNCTION_ID"] = "fid-sv"

from fastapi.testclient import TestClient  # noqa: E402
from serve import server, keys  # noqa: E402
from serve.providers import nvidia_maxine  # noqa: E402

client = TestClient(server.app)
FILE = {"file": ("a.wav", b"RIFFfake", "audio/wav")}

passed = []
def check(name, cond):
    passed.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")

r = client.get("/models")
ids = [m["id"] for m in r.json()["models"]]
check("/models contient les 2 modèles NVIDIA", "nvidia:bnr" in ids and "nvidia:studiovoice" in ids)

q = client.get("/quota").json()
check("/quota limite=2 + clé partagée configurée", q["limit"] == 2 and q["shared_key_configured"])

calls = {"keys": []}
nvidia_maxine.enhance = lambda mid, audio, api_key, intensity=None: (calls["keys"].append(api_key) or b"NVIDIAWAV")

r1 = client.post("/denoise", data={"model_id": "nvidia:bnr"}, files=FILE)
check("appel gratuit 1 -> 200, remaining=1", r1.status_code == 200 and r1.headers.get("x-quota-remaining") == "1")
r2 = client.post("/denoise", data={"model_id": "nvidia:bnr"}, files=FILE)
check("appel gratuit 2 -> 200, remaining=0", r2.status_code == 200 and r2.headers.get("x-quota-remaining") == "0")
r3 = client.post("/denoise", data={"model_id": "nvidia:bnr"}, files=FILE)
check("appel gratuit 3 -> 429", r3.status_code == 429)

r4 = client.post("/denoise", data={"model_id": "nvidia:bnr"}, files=FILE, headers={"X-API-Key": "nvapi-USER"})
check("clé perso -> 200 malgré quota épuisé", r4.status_code == 200 and calls["keys"][-1] == "nvapi-USER")

r5 = client.post("/denoise", data={"model_id": "nvidia:bnr"}, files=FILE, headers={"X-API-Key": "pas-une-cle"})
check("clé malformée -> 401", r5.status_code == 401)

nvidia_maxine.enhance = lambda *a, **k: (_ for _ in ()).throw(nvidia_maxine.NvidiaAuthError("x"))
r6 = client.post("/denoise", data={"model_id": "nvidia:bnr"}, files=FILE, headers={"X-API-Key": "nvapi-USER"})
check("NvidiaAuthError -> 401", r6.status_code == 401)

keys.shared_key = lambda: None
r7 = client.post("/denoise", data={"model_id": "nvidia:studiovoice"}, files=FILE)
check("aucune clé dispo -> 503", r7.status_code == 503)

server._get_model_snapshot = lambda mid: (object(), "cpu", "log1p", "/x/best.pt")
r8 = client.post("/denoise", data={"model_id": "local-xyz"}, files=FILE)
check("modèle local -> 200 (pas de clé/quota)", r8.status_code == 200 and r8.content == b"LOCALWAV")

print(f"\n{sum(passed)}/{len(passed)} tests OK")
sys.exit(0 if all(passed) else 1)
