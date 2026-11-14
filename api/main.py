"""Phyllo-ViT v3 API.

Run:  python -m uvicorn api.main:app --host 127.0.0.1 --port 8000      (from the project root)
UI:   http://127.0.0.1:8000/        Docs: http://127.0.0.1:8000/docs
"""
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from phyllo import __version__  # noqa: E402
from phyllo.aqi_standard import BREAKPOINTS, compute_aqi  # noqa: E402
from phyllo.paths import ARTIFACTS, RESULTS  # noqa: E402

ALLOWED = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"}
MAX_BYTES = 15 * 1024 * 1024
STATE = {}


@asynccontextmanager
async def lifespan(app):
    missing = [p.name for p in (ARTIFACTS / "reliability_ft4.json", ARTIFACTS / "aqi_evidence.json") if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing artifacts {missing}. Run RUN_EXPERIMENTS.bat (or python experiments/run_all.py) first.")
    from phyllo.pipeline import PhylloEngine
    print("Loading Phyllo-ViT models ...", flush=True)
    STATE["engine"] = PhylloEngine(load_aqi_model=os.environ.get("PHYLLO_LOAD_AQI_MODEL", "1") != "0")
    print(f"Ready on {STATE['engine'].info()['device']} in {STATE['engine'].load_seconds}s", flush=True)
    yield
    STATE.clear()


app = FastAPI(title="Phyllo-ViT API", version=__version__, lifespan=lifespan,
              description="Reliability-aware phyllotaxy recognition with abstention, explanations and "
                          "evidence-gated AQI output.")
# Extra allowed frontend origins for deployment, comma separated, e.g.
#   PHYLLO_CORS_ORIGINS=https://phyllo-vit.vercel.app
_origins = ["http://localhost:3000", "http://127.0.0.1:3000"] + [
    o.strip() for o in os.environ.get("PHYLLO_CORS_ORIGINS", "").split(",") if o.strip()]
# Any localhost/127.0.0.1 port is allowed so `next dev` still works when port 3000 is taken (it moves to 3001...).
app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
                   allow_methods=["*"], allow_headers=["*"])

STATIC = ROOT / "api" / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")
RESULTS.mkdir(exist_ok=True)
app.mount("/results", StaticFiles(directory=RESULTS), name="results")


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(STATIC / "index.html")


@app.get("/api")
def root():
    e = STATE.get("engine")
    return {"project": "Phyllo-ViT", "version": __version__, "status": "running",
            "classes": e.classes if e else None, "device": e.info()["device"] if e else None,
            "endpoints": ["/health", "/predict", "/aqi/compute", "/evidence", "/model-card", "/docs"]}


@app.get("/health")
def health():
    e = STATE.get("engine")
    return {"status": "healthy" if e else "loading", "version": __version__, **(e.info() if e else {})}


@app.post("/predict")
async def predict(file: UploadFile = File(...),
                  explain: bool = Query(True, description="Return an attention-rollout heatmap"),
                  show_exploratory_aqi: bool = Query(False, description="Also return the withheld exploratory AQI value")):
    if file.content_type not in ALLOWED:
        raise HTTPException(400, "Unsupported file type. Upload a JPG, PNG, WEBP or BMP image.")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image larger than 15 MB.")
    e = STATE["engine"]
    try:
        image = e.read_image(data)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Unable to read image: {exc}")
    result = e.predict(image, explain=explain, show_exploratory_aqi=show_exploratory_aqi)
    return {"filename": file.filename, **result}


@app.post("/aqi/compute")
def aqi_compute(concentrations: dict = Body(..., examples=[{"PM2.5": 92, "PM10": 160, "NO2": 45, "O3": 30, "CO": 1.4}])):
    """Compute India's National AQI (CPCB 2014) from MEASURED pollutant concentrations.
    Units µg/m³ (CO in mg/m³). This is a formula, not a model."""
    try:
        return compute_aqi(concentrations)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/aqi/breakpoints")
def aqi_breakpoints():
    return {"standard": "CPCB National AQI (2014)", "breakpoints": {k: {"low": v[0], "high": v[1]} for k, v in BREAKPOINTS.items()}}


def _load(p):
    p = RESULTS / p
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@app.get("/evidence")
def evidence():
    """Key experiment results (from results/), so the UI can show the evidence behind every output."""
    return {"audit": _load("e01_audit/audit_summary.json"),
            "classification": _load("e03_classification/classification_summary.json"),
            "reliability": _load("e04_reliability/reliability_summary.json"),
            "pollution": _load("e05_pollution_audit/pollution_summary.json")}


@app.get("/model-card")
def model_card():
    p = ROOT / "docs" / "MODEL_CARD.md"
    return {"markdown": p.read_text(encoding="utf-8") if p.exists() else ""}
