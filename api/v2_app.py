"""Phyllo-ViT 2.0 API = the unchanged v1.0.1 app (api.main) + independent /v2/* routes.

Run:  python -m uvicorn api.v2_app:app --host 127.0.0.1 --port 8000
v1 routes (/predict, /aqi/compute, /evidence, /health, /model-card, UI) behave exactly as in v1.0.1.
"""
import logging
import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.main import ALLOWED, MAX_BYTES, STATE, app  # noqa: F401  (importing api.main leaves v1 untouched)
from phyllo_v2 import AVAILABLE, DISABLED, EXPERIMENTAL, NOT_SUPPORTED, USER_INPUT_REQUIRED, UNAVAILABLE, __version__, section
from phyllo_v2 import environment, ftir, imaging

log = logging.getLogger("phyllo.v2")
router = APIRouter(prefix="/v2", tags=["v2"])
MAX_FTIR_BYTES = 10 * 1024 * 1024


def _phyllotaxy_section(r):
    st = r["decision"]["status"]
    p = r["phyllotaxy"]
    base = dict(decision_status=st, calibrated_confidence=p["confidence"], probabilities=p["probabilities"],
                supported_classes=["Alternate", "Opposite", "Whorled", "Not_a_leaf"],
                spiral="NOT_SUPPORTED: no genuinely labelled Spiral data has been acquired, so Spiral cannot be predicted.",
                context_note="Phyllotaxy is defined by how leaves attach along a stem. An isolated leaf blade or a close-up may not "
                             "show the nodes needed to judge it, so treat any answer as dataset-specific, not a botanical determination.",
                validation_note="Validated only on the project's own small held-out set (64/65 test images); not tested on new species or cameras.")
    if st == "ACCEPT":
        return section(AVAILABLE, r["decision"]["headline"], label=p["class"], **base)
    if st == "NOT_A_LEAF":
        return section(AVAILABLE, r["decision"]["headline"], label="Not_a_leaf", **base)
    if st == "REJECT_QUALITY":
        return section(UNAVAILABLE, r["decision"]["headline"], label="Unknown", **base)
    return section(AVAILABLE, r["decision"]["headline"], label="Unknown/Uncertain", **base)


def _leaf_aqi_section(r):
    a = r["aqi"]
    return section(NOT_SUPPORTED if a["value"] is None else EXPERIMENTAL,
                   a["message"] + " Leaf-image AQI estimation has no demonstrated skill in this project and is not an ambient or official AQI.",
                   v1_status=a["status"], value=a["value"], evidence=a["evidence"])


async def _read_image(file: UploadFile):
    if file.content_type not in ALLOWED:
        raise HTTPException(400, "Unsupported file type. Upload a JPG, PNG, WEBP or BMP image.")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image larger than 15 MB.")
    try:
        return STATE["engine"].read_image(data)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Unable to read image: {exc}")


async def _read_spectrum(ftir_file, y_units, column, wavenumber_scale):
    data = await ftir_file.read()
    if not data:
        raise HTTPException(400, "Empty FTIR file.")
    if len(data) > MAX_FTIR_BYTES:
        raise HTTPException(413, "FTIR file larger than 10 MB.")
    try:
        sp = ftir.parse_spectrum(ftir_file.filename, data, y_units=y_units, column=column, wavenumber_scale=wavenumber_scale)
        return ftir.analyse(sp)
    except ValueError as exc:
        raise HTTPException(400, f"FTIR file rejected: {exc}")


@router.get("/modules")
def modules():
    """Honest capability map: what is implemented, experimental, or unsupported - and why."""
    return {"version": __version__, "modules": {
        "phyllotaxy": section(AVAILABLE, "4-class ViT with calibrated abstention (v1.0.1 engine)."),
        "species": imaging.species_section(),
        "morphology": section(AVAILABLE, "Classical pixel-space leaf measurements."),
        "leaf_health": section(EXPERIMENTAL, "Colour heuristics; not validated against labelled data."),
        "dust_deposition": imaging.dust_section(),
        "ambient_environment": section(USER_INPUT_REQUIRED, "Needs a location and an OpenAQ API key (OPENAQ_API_KEY); live retrieval is unverified without a key."),
        "ftir": section(USER_INPUT_REQUIRED, "Needs a user-supplied spectrum (CSV/TSV/TXT, XLSX, JCAMP-DX). Indicative band ranges only."),
        "biological": imaging.biological_section(),
        "leaf_based_aqi": section(NOT_SUPPORTED, "No demonstrated skill (46 pairs, permutation p = 0.45); withheld.")}}


@router.post("/ftir/analyse")
async def ftir_only(ftir_file: UploadFile = File(...), y_units: str = Form("absorbance"), column: int = Form(1),
                    wavenumber_scale: float | None = Form(None)):
    return await _read_spectrum(ftir_file, y_units, column, wavenumber_scale)


@router.post("/analyze")
async def analyze(file: UploadFile = File(...), lat: float | None = Form(None), lon: float | None = Form(None),
                  collected_at: str | None = Form(None), ftir_file: UploadFile | None = File(None),
                  ftir_y_units: str = Form("absorbance"), ftir_column: int = Form(1), ftir_wavenumber_scale: float | None = Form(None)):
    t0 = time.time()
    img = await _read_image(file)
    engine = STATE["engine"]
    r = engine.predict(img, explain=False)
    accepted_leaf = r["decision"]["status"] == "ACCEPT"
    quality = r["quality"]
    usable = quality["passed"] and r["decision"]["status"] != "NOT_A_LEAF"

    out = {"version": __version__, "filename": file.filename,
           "image_quality": section(AVAILABLE if quality["passed"] else UNAVAILABLE,
                                    "Image passed the quality gate." if quality["passed"] else "Image failed the quality gate: " + "; ".join(quality["issues"]),
                                    sharpness=quality["sharpness"], noise=quality["noise"], warnings=quality["warnings"]),
           "phyllotaxy": _phyllotaxy_section(r), "species": imaging.species_section()}
    if usable:
        out["morphology"] = imaging.morphology_section(img)
        out["leaf_health"] = imaging.health_section(img)
    else:
        why = "Skipped: the image is not a usable leaf photo (failed quality gate or not a leaf)."
        out["morphology"] = section(UNAVAILABLE, why)
        out["leaf_health"] = section(UNAVAILABLE, why)
    out["dust_deposition"] = imaging.dust_section()
    out["leaf_based_aqi"] = _leaf_aqi_section(r)
    out["ambient_environment"] = environment.fetch_ambient(lat, lon, collected_at)
    out["ftir"] = (await _read_spectrum(ftir_file, ftir_y_units, ftir_column, ftir_wavenumber_scale)
                   if ftir_file is not None and ftir_file.filename else
                   section(USER_INPUT_REQUIRED, "No FTIR spectrum supplied. No spectrum is generated or assumed."))
    out["biological"] = imaging.biological_section()
    out["separation_notice"] = ("Ambient station readings, any leaf-based estimate, visible-condition indicators and FTIR band ranges are different "
                                "kinds of evidence and are never combined into one number.")
    out["latency_ms"] = round((time.time() - t0) * 1000, 1)
    log.info("v2 analyze %s status=%s latency=%sms", file.filename, r["decision"]["status"], out["latency_ms"])
    return out


app.include_router(router)
