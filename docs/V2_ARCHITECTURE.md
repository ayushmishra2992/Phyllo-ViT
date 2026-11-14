# Phyllo-ViT 2.0 — architecture and status

v1.0.1 is frozen (git tag `v1.0.1`). v2 is additive: new code in `phyllo_v2/`, `api/v2_app.py`, `frontend/app/v2/`, `frontend/lib/v2.ts`, `v2/data_registry/`, `tests/run_tests_v2.py`. `api/main.py`, `phyllo/`, models and the 13 v1 tests are unchanged. `api.v2_app:app` = the v1 app + `/v2/*` routes; v1 routes behave as before.

## Module status (what is actually implemented and tested)
| Module | Status | Basis |
|---|---|---|
| Phyllotaxy (Alternate / Opposite / Whorled / Not_a_leaf, + `Unknown/Uncertain` and `Unknown` on abstain/low quality) | **Available** | v1.0.1 ViT-B/16, calibrated, OOD abstention; validated only on the project's small held-out set |
| Spiral | **Not supported** | no genuinely labelled Spiral data found |
| Species identification | **Not supported** | no licensed labelled dataset verified/acquired |
| Morphology (area fraction, aspect ratio, solidity, eccentricity) | **Available** (pixel space only) | classical segmentation; no physical scale; can fail on cluttered backgrounds |
| Visible leaf condition (yellowish / brownish / very dark fractions) | **Experimental** | colour heuristics, not validated against labelled data; "disease-like appearance" only |
| Dust / deposition | **Not supported** | no labelled masks; no coverage %, particle count, PM or chemistry is reported |
| Ambient AQI and pollutants | **Available when configured** | OpenAQ v3 client (needs `OPENAQ_API_KEY` + location); station, distance, timestamp, unit, provider, licence shown; **live retrieval not verified here (no key); tested with a mock transport only** |
| Leaf-image AQI | **Not supported** | no demonstrated skill (46 pairs, p = 0.45) |
| FTIR | **Available for user-supplied spectra** | CSV/TSV/TXT, XLSX, JCAMP-DX parsing, smoothing, peak detection, *indicative* group ranges; no pollutant attribution |
| Biological (bacteria/fungi) | **Disabled** | cannot be determined from RGB photographs; needs lab/microscopy data |
| Multimodal fusion | **Not built** | no genuinely paired image + FTIR + pollution records exist |

## API
`GET /v2/modules` · `POST /v2/analyze` (multipart: `file`, optional `lat`, `lon`, `collected_at`, `ftir_file`, `ftir_y_units`, `ftir_column`, `ftir_wavenumber_scale`) · `POST /v2/ftir/analyse`. Every response section is `{status, reason, ...}` with status in AVAILABLE / EXPERIMENTAL / USER_INPUT_REQUIRED / UNAVAILABLE / NOT_SUPPORTED / DISABLED.
Four kinds of evidence are never merged: official/ambient station data, measured concentrations, leaf-based estimates (withheld) and visible-condition indicators.

## Data
`v2/data_registry/dataset_registry.csv` + `DATASET_SOURCES.md` (only sources opened and licence-read this session; unverified ones are marked). One real dataset sample was acquired (CC BY 4.0 hemp-leaf IR spectra, 2 files, md5 verified) for parser testing. The unified sample database, near-duplicate tooling and group-aware splitting of the original plan were **not built**: with no new image datasets acquired there was nothing to manage beyond the v1 split.

## Not done (honest list)
Spiral, species, dust masks, leaf-health classifier training, bacterial/fungal module, multimodal fusion, historical (time-matched) ambient retrieval, CPCB portal integration, live OpenAQ verification, Docker image for v2, real-browser click-through of the `/v2` page.