"""Phyllo-ViT 2.0 tests. Names say what is REAL (real images / real FTIR file / real HTTP via TestClient),
SYNTHETIC (generated test input for the parser) or MOCK (OpenAQ replaced by a mock transport).
Run: python tests/run_tests_v2.py
"""
import csv
import hashlib
import io
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

from phyllo.paths import SPLIT_4  # noqa: E402
from phyllo_v2 import environment, ftir, imaging  # noqa: E402

REG = ROOT / "v2" / "data_registry"
HEMP = REG / "samples" / "W1_C.xlsx"
RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, True, ""))
        print(f"  PASS  {name}")
    except Exception as e:  # noqa: BLE001
        RESULTS.append((name, False, repr(e)))
        print(f"  FAIL  {name}: {e!r}")
        traceback.print_exc(limit=2)


def first_img(cls, n=0):
    return sorted((SPLIT_4 / "test" / cls).iterdir())[n]


def jpeg_bytes(im):
    b = io.BytesIO(); im.convert("RGB").save(b, "JPEG"); return b.getvalue()


# ------------------------------------------------------------------ FTIR
def synth_spectrum(centres=(2918.0, 1650.0, 1050.0), n=1800):
    x = np.linspace(4000, 650, n)
    y = np.full(n, 0.01)
    for c, a in zip(centres, (0.5, 0.3, 0.8)):
        y += a * np.exp(-((x - c) / 12.0) ** 2)
    return x, y


def t_ftir_synthetic_csv():
    x, y = synth_spectrum()
    csv_text = "wavenumber,absorbance\n" + "\n".join(f"{a:.3f},{b:.6f}" for a, b in zip(x, y))
    res = ftir.analyse(ftir.parse_spectrum("s.csv", csv_text.encode()))
    found = [p["wavenumber_cm-1"] for p in res["peaks"]]
    for c in (2918.0, 1650.0, 1050.0):
        assert any(abs(f - c) < 4 for f in found), (c, found)
    assert res["status"] == "AVAILABLE" and "pollutant" in res["caveat"].lower()
    p = [p for p in res["peaks"] if abs(p["wavenumber_cm-1"] - 2918) < 4][0]
    assert any("C-H" in c["assignment"] for c in p["candidate_groups"])


def t_ftir_semicolon_comma_decimal():
    x, y = synth_spectrum()
    txt = "\n".join(f"{a:.3f};{b:.6f}".replace(".", ",") for a, b in zip(x, y))
    res = ftir.analyse(ftir.parse_spectrum("s.txt", txt.encode()))
    assert res["n_points"] == len(x)


def t_ftir_jcamp_synthetic():
    x, y = synth_spectrum(n=200)
    lines = ["##TITLE=synthetic test", "##JCAMP-DX=4.24", "##DATA TYPE=INFRARED SPECTRUM", "##XUNITS=1/CM", "##YUNITS=ABSORBANCE",
             f"##FIRSTX={x[0]}", f"##LASTX={x[-1]}", f"##NPOINTS={len(x)}", "##XYDATA=(X++(Y..Y))"]
    for i in range(0, len(x), 10):
        lines.append(f"{x[i]:.3f} " + " ".join(f"{v:.5f}" for v in y[i:i + 10]))
    lines.append("##END=")
    sp = ftir.parse_spectrum("s.jdx", "\n".join(lines).encode())
    assert len(sp["wavenumber"]) == len(x) and sp["y_units"] == "absorbance"


def t_ftir_rejects_bad_input():
    for name, data in [("a.csv", b"hello world"), ("a.csv", b"1,2\n3,4\n"), ("a.csv", b""),
                       ("a.pdf", b"%PDF")]:
        try:
            ftir.parse_spectrum(name, data)
        except ValueError:
            continue
        raise AssertionError(f"accepted bad input {name}")
    x, y = synth_spectrum()
    try:
        ftir.parse_spectrum("s.csv", "\n".join(f"{a},{b}" for a, b in zip(x, y)).encode(), y_units="furlongs")
    except ValueError:
        return
    raise AssertionError("accepted bad y_units")


def t_ftir_real_hemp_file():
    sp = ftir.parse_spectrum("W1_C.xlsx", HEMP.read_bytes(), column=1, wavenumber_scale=1000)
    res = ftir.analyse(sp)
    lo, hi = res["range_cm1"]
    assert 399 < lo < 402 and 3999 < hi < 4001 and res["n_points"] > 7000, res["range_cm1"]
    assert res["n_peaks"] >= 3
    assert any(2800 <= p["wavenumber_cm-1"] <= 3000 or 1000 <= p["wavenumber_cm-1"] <= 1200 for p in res["peaks"])


def t_registry_and_sample_integrity():
    rows = list(csv.DictReader(open(REG / "dataset_registry.csv", encoding="utf-8")))
    need = {"dataset_id", "source_url", "licence", "licence_verified_from_source", "project_status", "checked_on"}
    assert need <= set(rows[0]), set(rows[0])
    ids = [r["dataset_id"] for r in rows]
    assert len(ids) == len(set(ids))
    for r in rows:
        if r["project_status"].startswith("ACQUIRED"):
            assert r["licence_verified_from_source"].lower().startswith("yes"), r["dataset_id"]
    md5 = {"W1_C.xlsx": "4a78939d654423c962e43b90a9f53382", "Sample_name_codification.xlsx": "101d63861690d11b5c2b7ae827402a11"}
    for f, h in md5.items():
        assert hashlib.md5((REG / "samples" / f).read_bytes()).hexdigest() == h, f


# ------------------------------------------------------------------ imaging
def t_imaging_real_leaves():
    for cls in ("Alternate", "Opposite", "Whorled"):
        im = Image.open(first_img(cls)).convert("RGB")
        m = imaging.morphology_section(im)
        h = imaging.health_section(im)
        assert m["status"] in ("AVAILABLE", "UNAVAILABLE"), m
        if m["status"] == "AVAILABLE":
            assert 0 < m["leaf_area_fraction"] <= 1 and m["aspect_ratio"] >= 1
            assert h["status"] == "EXPERIMENTAL" and "not validated" in h["reason"].lower() and "disease" in h["reason"].lower()
            assert abs(sum(h["colour_fractions"][k] for k in ("green", "yellowish", "brownish")) - 1) <= 1.0


def t_imaging_blank_image_unavailable():
    im = Image.new("RGB", (300, 300), (128, 128, 128))
    assert imaging.morphology_section(im)["status"] == "UNAVAILABLE"
    assert imaging.health_section(im)["status"] == "UNAVAILABLE"


def t_unsupported_modules_are_honest():
    for s in (imaging.species_section(), imaging.dust_section()):
        assert s["status"] == "NOT_SUPPORTED" and "value" not in s and "species" not in s
    b = imaging.biological_section()
    assert b["status"] == "DISABLED"


# ------------------------------------------------------------------ environment (MOCK transport)
def mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


LOC = {"id": 42, "name": "Mock Station", "coordinates": {"latitude": 28.46, "longitude": 77.03},
       "provider": {"name": "MockProvider"}, "licenses": [{"name": "CC BY 4.0"}],
       "sensors": [{"id": 1, "parameter": {"name": "pm25", "units": "µg/m³"}},
                   {"id": 2, "parameter": {"name": "no2", "units": "µg/m³"}},
                   {"id": 3, "parameter": {"name": "temperature", "units": "c"}}]}


def ok_handler(req):
    if req.url.path.endswith("/locations"):
        assert req.headers["x-api-key"] == "k"
        return httpx.Response(200, json={"results": [LOC]})
    if req.url.path.endswith("/latest"):
        return httpx.Response(200, json={"results": [
            {"sensorsId": 1, "value": 92.0, "datetime": {"utc": "2026-11-14T05:00:00Z"}},
            {"sensorsId": 2, "value": 45.0, "datetime": {"utc": "2026-11-14T05:00:00Z"}},
            {"sensorsId": 3, "value": 20.0, "datetime": {"utc": "2026-11-14T05:00:00Z"}}]})
    return httpx.Response(404)


def t_env_mock_success():
    r = environment.fetch_ambient(28.45, 77.02, collected_at="2026-03-01", client=mock_client(ok_handler), api_key="k")
    assert r["status"] == "AVAILABLE"
    st = r["nearest_station"]
    assert st["name"] == "Mock Station" and st["distance_km"] < 5 and st["licences"] == ["CC BY 4.0"]
    assert {x["parameter"] for x in st["readings"]} == {"PM2.5", "NO2"}      # temperature is not a pollutant row
    assert st["readings"][0]["timestamp_utc"] and st["readings"][0]["unit"]
    assert r["indicative_index_from_latest"]["aqi"] == 204 and "not an official AQI" in r["indicative_index_from_latest"]["note"]
    assert "not measured from the leaf" in r["reason"] and r["time_note"]


def t_env_missing_inputs():
    assert environment.fetch_ambient(None, None)["status"] == "USER_INPUT_REQUIRED"
    assert environment.fetch_ambient(95, 10, api_key="k")["status"] == "USER_INPUT_REQUIRED"
    import os
    old = os.environ.pop("OPENAQ_API_KEY", None)
    try:
        assert environment.fetch_ambient(28, 77)["status"] == "USER_INPUT_REQUIRED"
    finally:
        if old: os.environ["OPENAQ_API_KEY"] = old


def t_env_mock_failures():
    cases = {"none": lambda q: httpx.Response(200, json={"results": []}), "401": lambda q: httpx.Response(401),
             "429": lambda q: httpx.Response(429), "500": lambda q: httpx.Response(500),
             "garbage": lambda q: httpx.Response(200, text="<html>")}
    for k, h in cases.items():
        r = environment.fetch_ambient(28, 77, client=mock_client(h), api_key="k")
        assert r["status"] == "UNAVAILABLE" and "value" not in r and "nearest_station" not in r, (k, r)

    def boom(req): raise httpx.ConnectError("offline")
    assert environment.fetch_ambient(28, 77, client=mock_client(boom), api_key="k")["status"] == "UNAVAILABLE"


# ------------------------------------------------------------------ API (real HTTP via TestClient, real models)
def api_tests():
    from fastapi.testclient import TestClient
    from api.v2_app import app

    with TestClient(app) as c:
        def post(path, img_bytes, name="x.jpg", ctype="image/jpeg", **kw):
            return c.post(path, files={"file": (name, img_bytes, ctype), **kw.pop("files", {})}, data=kw.pop("data", {}))

        def modules():
            m = c.get("/v2/modules").json()["modules"]
            assert m["species"]["status"] == "NOT_SUPPORTED" and m["biological"]["status"] == "DISABLED"
            assert m["dust_deposition"]["status"] == "NOT_SUPPORTED" and m["leaf_health"]["status"] == "EXPERIMENTAL"

        def v1_unchanged():
            assert c.get("/health").json()["status"] == "healthy"
            p = first_img("Opposite")
            r = c.post("/predict", files={"file": (p.name, p.read_bytes(), "image/jpeg")}).json()
            assert r["phyllotaxy"]["class"] == "Opposite" and r["decision"]["status"] == "ACCEPT"
            assert "/v2/analyze" in c.get("/openapi.json").json()["paths"]

        def per_class():
            n_ok = 0
            for cls in ("Alternate", "Opposite", "Whorled"):
                p = first_img(cls, 1)
                r = post("/v2/analyze", p.read_bytes(), p.name).json()
                for k in ("image_quality", "phyllotaxy", "species", "morphology", "leaf_health", "dust_deposition",
                          "leaf_based_aqi", "ambient_environment", "ftir", "biological"):
                    assert k in r and "status" in r[k], k
                ph = r["phyllotaxy"]
                assert ph["label"] == cls or ph["label"] == "Unknown/Uncertain", (cls, ph["label"])
                n_ok += ph["label"] == cls
                assert r["leaf_based_aqi"]["value"] is None
                assert r["species"]["status"] == "NOT_SUPPORTED" and r["biological"]["status"] == "DISABLED"
                assert r["ambient_environment"]["status"] == "USER_INPUT_REQUIRED"      # no location given
                assert r["ftir"]["status"] == "USER_INPUT_REQUIRED"                      # no spectrum given
            assert n_ok >= 2, n_ok

        def non_leaf():
            p = first_img("Not_a_leaf")
            r = post("/v2/analyze", p.read_bytes(), p.name).json()
            assert r["phyllotaxy"]["label"] == "Not_a_leaf"
            assert r["morphology"]["status"] == "UNAVAILABLE" and r["leaf_health"]["status"] == "UNAVAILABLE"
            assert r["leaf_based_aqi"]["value"] is None

        def low_quality():
            im = Image.open(first_img("Opposite")).convert("RGB").filter(ImageFilter.GaussianBlur(30))
            r = post("/v2/analyze", jpeg_bytes(im)).json()
            assert r["image_quality"]["status"] == "UNAVAILABLE" and r["phyllotaxy"]["label"] == "Unknown"
            assert r["morphology"]["status"] == "UNAVAILABLE"

        def invalid_inputs():
            assert post("/v2/analyze", b"hello", "x.txt", "text/plain").status_code == 400
            assert post("/v2/analyze", b"notanimage").status_code == 400
            assert post("/v2/analyze", b"").status_code == 400
            assert post("/v2/analyze", b"\0" * (16 * 1024 * 1024)).status_code == 413
            assert c.post("/v2/analyze").status_code == 422

        def location_without_key():
            import os
            old = os.environ.pop("OPENAQ_API_KEY", None)
            try:
                p = first_img("Whorled")
                r = post("/v2/analyze", p.read_bytes(), p.name, data={"lat": "28.46", "lon": "77.03"}).json()
                assert r["ambient_environment"]["status"] == "USER_INPUT_REQUIRED" and "API key" in r["ambient_environment"]["reason"]
            finally:
                if old: os.environ["OPENAQ_API_KEY"] = old

        def ftir_real_file_through_api():
            p = first_img("Whorled")
            r = post("/v2/analyze", p.read_bytes(), p.name,
                     files={"ftir_file": ("W1_C.xlsx", HEMP.read_bytes(), "application/octet-stream")},
                     data={"ftir_wavenumber_scale": "1000"}).json()
            assert r["ftir"]["status"] == "AVAILABLE" and r["ftir"]["n_peaks"] >= 3 and len(r["ftir"]["spectrum"]["wavenumber"]) > 100

        def ftir_bad_file_400():
            p = first_img("Whorled")
            r = post("/v2/analyze", p.read_bytes(), p.name, files={"ftir_file": ("bad.csv", b"not a spectrum", "text/csv")})
            assert r.status_code == 400 and "FTIR" in r.json()["detail"]

        def ftir_only_endpoint():
            x, y = synth_spectrum()
            txt = "\n".join(f"{a:.3f},{b:.6f}" for a, b in zip(x, y)).encode()
            r = c.post("/v2/ftir/analyse", files={"ftir_file": ("s.csv", txt, "text/csv")}).json()
            assert r["status"] == "AVAILABLE"

        def cors_v2():
            r = c.options("/v2/analyze", headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"})
            assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"

        for name, fn in [("v2 module map honest", modules), ("v1 routes unchanged under v2 app", v1_unchanged),
                         ("REAL held-out images: Alt/Opp/Whorled through /v2/analyze", per_class),
                         ("REAL non-leaf image: no morphology/health/AQI", non_leaf), ("REAL blurred image: low quality handled", low_quality),
                         ("invalid uploads 400/413/422", invalid_inputs), ("location without API key: honest message", location_without_key),
                         ("REAL FTIR hemp file through /v2/analyze", ftir_real_file_through_api),
                         ("bad FTIR file -> 400", ftir_bad_file_400), ("FTIR-only endpoint (SYNTHETIC input)", ftir_only_endpoint),
                         ("CORS on v2 route", cors_v2)]:
            check("api: " + name, fn)


if __name__ == "__main__":
    print("Phyllo-ViT 2.0 test suite")
    for n, f in [("ftir: SYNTHETIC csv, peaks + band candidates", t_ftir_synthetic_csv),
                 ("ftir: SYNTHETIC semicolon/comma-decimal", t_ftir_semicolon_comma_decimal),
                 ("ftir: SYNTHETIC JCAMP-DX", t_ftir_jcamp_synthetic), ("ftir: rejects bad input", t_ftir_rejects_bad_input),
                 ("ftir: REAL Zenodo hemp-leaf file", t_ftir_real_hemp_file),
                 ("dataset registry schema + sample md5", t_registry_and_sample_integrity),
                 ("imaging: REAL held-out leaves", t_imaging_real_leaves), ("imaging: blank image unavailable", t_imaging_blank_image_unavailable),
                 ("unsupported modules never invent values", t_unsupported_modules_are_honest),
                 ("env: MOCK OpenAQ success", t_env_mock_success), ("env: missing location/key", t_env_missing_inputs),
                 ("env: MOCK failures (empty/401/429/500/garbage/offline)", t_env_mock_failures)]:
        check(n, f)
    api_tests()
    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    sys.exit(1 if failed else 0)
