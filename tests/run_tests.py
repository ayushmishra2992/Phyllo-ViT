"""Self-contained test suite (no pytest needed):  python tests/run_tests.py

Checks: AQI standard maths, quality gate, and end-to-end API behaviour on real test images
(accepted leaf, non-leaf, blurred image, dark image, bad upload), plus backward-compatible response fields.
"""
import io
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

from phyllo import quality  # noqa: E402
from phyllo.aqi_standard import compute_aqi, sub_index  # noqa: E402
from phyllo.paths import SPLIT_4  # noqa: E402

RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, True, ""))
        print(f"  PASS  {name}")
    except Exception as e:  # noqa: BLE001
        RESULTS.append((name, False, str(e)))
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc()


def png(img, fmt="JPEG"):
    b = io.BytesIO()
    img.save(b, format=fmt)
    return b.getvalue()


# ----------------------------------------------------------------- unit tests
def t_aqi():
    assert sub_index("PM2.5", 0) == 0
    assert sub_index("PM2.5", 30) == 50
    assert sub_index("PM2.5", 60) == 100
    assert sub_index("PM2.5", 90) == 200
    assert sub_index("PM10", 100) == 100
    assert sub_index("CO", 1.0) == 50
    assert sub_index("PM2.5", 1000) == 500
    r = compute_aqi({"PM2.5": 92, "PM10": 160, "NO2": 45})
    assert r["valid"] and r["dominant_pollutant"] == "PM2.5" and r["category"] == "Poor", r
    r = compute_aqi({"NO2": 45, "O3": 30})
    assert not r["valid"]


def t_quality():
    leaf = next((SPLIT_4 / "test" / "Whorled").iterdir())
    im = Image.open(leaf).convert("RGB")
    assert quality.assess(im).passed
    assert not quality.assess(im.filter(ImageFilter.GaussianBlur(40))).passed
    assert not quality.assess(Image.new("RGB", (400, 400), (5, 5, 5))).passed
    assert not quality.assess(im.resize((40, 40))).passed


# ----------------------------------------------------------------- API tests
def api_tests():
    from fastapi.testclient import TestClient
    from api.main import app

    with TestClient(app) as c:
        def health():
            r = c.get("/health").json()
            assert r["status"] == "healthy", r

        def leaf_accept():
            n_acc = 0
            for cls in ("Alternate", "Opposite", "Whorled"):
                for p in sorted((SPLIT_4 / "test" / cls).iterdir())[:3]:
                    r = c.post("/predict", files={"file": (p.name, p.read_bytes(), "image/jpeg")}).json()
                    for k in ("filename", "phyllotaxy", "aqi", "model", "decision", "reliability", "quality"):
                        assert k in r, k
                    assert r["phyllotaxy"]["class"] == cls or r["decision"]["status"] != "ACCEPT", (p, r["phyllotaxy"])
                    n_acc += r["decision"]["status"] == "ACCEPT"
                    if r["decision"]["status"] == "ACCEPT" and cls != "Whorled":
                        assert r["aqi"]["status"] == "NOT_APPLICABLE"
                    if r["decision"]["status"] == "ACCEPT" and cls == "Whorled":
                        assert r["aqi"]["status"] in ("WITHHELD", "EXPLORATORY")
            assert n_acc >= 7, n_acc

        def nonleaf():
            p = sorted((SPLIT_4 / "test" / "Not_a_leaf").iterdir())[0]
            r = c.post("/predict", files={"file": (p.name, p.read_bytes(), "image/jpeg")}).json()
            assert r["decision"]["status"] == "NOT_A_LEAF", r["decision"]
            assert r["aqi"]["value"] is None

        def blurred():
            p = sorted((SPLIT_4 / "test" / "Opposite").iterdir())[0]
            im = Image.open(p).convert("RGB").filter(ImageFilter.GaussianBlur(30))
            r = c.post("/predict", files={"file": ("b.jpg", png(im), "image/jpeg")}).json()
            assert r["decision"]["status"] == "REJECT_QUALITY", r["decision"]

        def random_noise():
            im = Image.fromarray(np.random.default_rng(0).integers(0, 255, (300, 300, 3), dtype=np.uint8))
            r = c.post("/predict", files={"file": ("n.png", png(im, "PNG"), "image/png")}).json()
            assert r["decision"]["status"] != "ACCEPT", r["decision"]

        def bad_upload():
            assert c.post("/predict", files={"file": ("x.txt", b"hello", "text/plain")}).status_code == 400
            assert c.post("/predict", files={"file": ("x.jpg", b"notanimage", "image/jpeg")}).status_code == 400

        def heatmap():
            p = sorted((SPLIT_4 / "test" / "Whorled").iterdir())[0]
            r = c.post("/predict", files={"file": (p.name, p.read_bytes(), "image/jpeg")}).json()
            assert len(r["explanation"]["heatmap_png_base64"]) > 1000

        def aqi_endpoint():
            r = c.post("/aqi/compute", json={"PM2.5": 92, "PM10": 160, "NO2": 45}).json()
            assert r["aqi"] == sub_index("PM2.5", 92)
            assert c.post("/aqi/compute", json={"XYZ": 3}).status_code == 400

        def evidence():
            r = c.get("/evidence").json()
            assert r["reliability"] and r["pollution"] and r["classification"]

        def ui():
            assert "Phyllo-ViT" in c.get("/").text

        def cors():
            for origin in ("http://localhost:3000", "http://localhost:3001"):
                r = c.options("/predict", headers={"Origin": origin, "Access-Control-Request-Method": "POST"})
                assert r.headers.get("access-control-allow-origin") == origin, (origin, r.headers)
            r = c.options("/predict", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"})
            assert "access-control-allow-origin" not in r.headers

        for name, fn in [("health", health), ("known leaves accepted & correct", leaf_accept), ("non-leaf rejected", nonleaf),
                         ("blurred image rejected by quality gate", blurred), ("random noise not accepted", random_noise),
                         ("invalid uploads -> 400", bad_upload), ("explanation heatmap", heatmap),
                         ("AQI standard endpoint", aqi_endpoint), ("evidence endpoint", evidence), ("web UI served", ui), ("CORS (localhost ports only)", cors)]:
            check("api: " + name, fn)


if __name__ == "__main__":
    print("Phyllo-ViT test suite")
    check("CPCB AQI sub-index maths", t_aqi)
    check("image quality gate", t_quality)
    api_tests()
    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    sys.exit(1 if failed else 0)
