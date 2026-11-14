# Phyllo-ViT v1.0.1

**Reliability-aware phyllotaxy recognition (Alternate / Opposite / Whorled / Not_a_leaf) with open-set abstention, explanations, and an evidence audit of leaf-based air-quality claims.**

Bennett University Â· B.Tech CSE final-year research project Â· Ayush Mishra (ML/backend) & Riya Katiyar (literature/frontend/docs)

## Overview and problem statement
A reference paper proposes predicting street-level AQI from roadside leaf photographs, FTIR spectra and CPCB readings. This project (1) builds a **reliable phyllotaxy classifier** that knows when *not* to answer, and (2) **audits** the paper's imageâ†’AQI claims against the data that came with it. The audit finds no demonstrated imageâ†’AQI skill, so AQI is **withheld** by default; the app never presents a leaf photo as a measurement of AQI, PM2.5, chemical composition or microbes.

## Features
- Four-class ViT-B/16 classifier: Alternate, Opposite, Whorled, Not_a_leaf.
- Image-quality gate, temperature-calibrated confidence, out-of-distribution abstention (Mahalanobis + kNN), attention-rollout heatmap.
- Decision statuses: `ACCEPT`, `ABSTAIN_UNCERTAIN`, `ABSTAIN_UNFAMILIAR`, `NOT_A_LEAF`, `REJECT_QUALITY`.
- CPCB (2014) AQI **calculator** for *measured* concentrations (a formula, not a model).
- Evidence-gated AQI output: `NOT_APPLICABLE` for non-Whorled/abstained images, `WITHHELD` for Whorled (evidence does not support a value).
- Built-in web UI (FastAPI) and a Next.js frontend; 13 automated tests; 7 reproducible experiments.

## Architecture
```
photo -> Quality gate -> ViT-B/16 (4-class) -> Temperature calibration -> OOD check -> Decision
         (size, exposure, blur, contrast)      (T fit on validation)      (Mahalanobis + kNN on CLS embedding)
      -> Attention-rollout heatmap -> Evidence-gated AQI
```
Thresholds are fit on training/validation data only and stored in `artifacts/`.

## Tech stack
Python 3.12 Â· PyTorch (CUDA 12.6 build; CPU works) Â· Hugging Face Transformers 5.18 Â· scikit-learn Â· FastAPI + Uvicorn Â· Next.js 16 / React 18 / TypeScript / Tailwind Â· Git LFS for model weights.

## Models
| Folder | Role |
|---|---|
| `models/vit_phyllotaxy_4class` | Deployed 4-class classifier (fine-tuned `google/vit-base-patch16-224-in21k`) |
| `models/vit_phyllotaxy` | Original 3-class model; its embedding space is used by the AQI regressor â€” do not replace it |

## Install and run (Windows)
Keep the project path short (Windows 260-character limit).
1. `INSTALL.bat` â€” creates `.venv`, installs PyTorch + requirements (about 4 GB, 10â€“20 min), and `npm ci` for the frontend if Node.js â‰¥ 20.9 is present.
2. `RUN.bat` â€” backend + built-in UI at http://127.0.0.1:8000; also the Next.js UI at http://localhost:3000/lab if Node is installed. (`RUN_ME.bat` starts the backend only.)

| Script | Purpose |
|---|---|
| `RUN_TESTS.bat` | 13 automated tests (AQI maths, quality gate, API, CORS) |
| `CHECK_CLASSES.bat` | Per-class check on all held-out images + HTTP â†’ `results/class_check.json`, `results/CLASS_VERIFICATION.md` |
| `RUN_EXPERIMENTS.bat` | Re-run experiments E1â€“E7 â†’ `results/RESULTS.md`, `artifacts/` (~5 min on RTX 3050 locally) |
| `START_FRONTEND.bat` | Next.js dev server (backend must be running) |
| `PACKAGE_FINAL.bat` | Build a distributable ZIP |

Manual: `python -m uvicorn api.main:app --host 127.0.0.1 --port 8000` Â· `python tests/run_tests.py`.
Configuration: `NEXT_PUBLIC_API_URL` (frontend, see `frontend/.env.local.example`), `PHYLLO_CORS_ORIGINS` (backend, comma-separated extra origins; any `localhost` port is allowed), `PHYLLO_LOAD_AQI_MODEL=0` to skip the AQI model. GPU: 4 GB card, batch size â‰¤ 8, fp16 for training.

## API
| Endpoint | Description |
|---|---|
| `POST /predict?explain=true` | Multipart image upload â†’ decision, calibrated probabilities, reliability, quality, heatmap, AQI status |
| `POST /aqi/compute` | CPCB AQI from measured concentrations (Âµg/mÂ³; CO mg/mÂ³) |
| `GET /aqi/breakpoints` Â· `GET /evidence` Â· `GET /model-card` Â· `GET /health` Â· `GET /api` | Metadata and health |
Uploads: JPG/PNG/WEBP/BMP, â‰¤ 15 MB; other types/corrupt/empty â†’ 400, oversize â†’ 413. Docs at `/docs`.

## Dataset
`data/processed/clean`: 265 unique leaf images + 150 COCO non-leaf images; `split_4class` is the deployed model's split. Raw images and metadata are under `data/raw` and `data/metadata`. `data/metadata/provenance_annotation_TODO.csv` (species, plant, date) is **still unfilled**, so species-grouped evaluation has not been run. See `docs/DATASET_SUFFICIENCY.md`.

## Evaluation (held-out, deployed pipeline)
| True class | Test correct / n | Val correct / n |
|---|---|---|
| Alternate | 13 / 14 (the miss is abstained) | 13 / 13 |
| Opposite | 13 / 13 | 12 / 12 |
| Whorled | 15 / 15 | 14 / 14 |
| Not_a_leaf | 23 / 23 rejected | 22 / 22 rejected |

Overall 64/65 test (Wilson 95 % CI 91.8â€“99.7 %); with abstention, 41/42 leaves accepted and all accepted correct. These are small sets from the same photo collection as training: they show the pipeline works, **not** that it generalises to new species or cameras.

## Known limitations
- **Imageâ†’AQI has no demonstrated skill** (46 pairs, permutation p = 0.45); AQI is withheld. The only time-paired station file (5 days) shows no out-of-day signal.
- Matched imageâ€“AQI pairs exist only for Whorled, from one site; the two pollution workbooks have identical targets and are not two sites.
- **Shortcut risk:** colour statistics alone classify the three arrangements at 95.6 %; orientation/size alone reach 76.7 % â€” current accuracy does not prove phyllotaxy understanding. Near-duplicates exist (10 of 65 test images have one in train/val).
- Not_a_leaf is validated on COCO-style images; it is not complete OOD detection. Heavy sensor noise is still partly accepted (17/42 noisy images).
- A single isolated leaf may not show enough stem/node context to define phyllotaxy.
- Ordinary photographs cannot measure exact AQI, PM2.5, chemical composition (FTIR) or microbes.
- Docker image is provided but has not been built or run.
- The Next.js build warns that React 18 is deprecated in Next.js 16 (upgrade not applied).

## Future scope
Species- and plant-grouped cross-validation once provenance is annotated; genuinely paired imageâ€“FTIRâ€“station data collection (`docs/DATA_COLLECTION_PROTOCOL.md`); Spiral class with labelled data; React 19 migration; deployment (`docs/DEPLOY.md`).

## Layout
`api/` FastAPI + built-in UI Â· `phyllo/` core library Â· `frontend/` Next.js app Â· `experiments/` e01â€“e07 Â· `artifacts/` deploy-time thresholds Â· `results/` outputs Â· `models/` checkpoints (Git LFS) Â· `data/` Â· `docs/` (RESEARCH_REPORT, MODEL_CARD, DATASET_SUFFICIENCY, DATA_COLLECTION_PROTOCOL, PAPER_PLAN, DEPLOY) Â· `legacy/` untouched v2 Â· `tests/`.

## Troubleshooting
Port 8000 busy â†’ `python -m uvicorn api.main:app --port 8010`. `FileNotFoundError` on images â†’ path too long, move the folder to `Desktop`. Runs on CPU if no CUDA (about 1â€“2 s per image); `/health` shows the device. Experiments need the pretrained backbone `google/vit-base-patch16-224-in21k` (about 350 MB, downloaded once).

See `docs/RESEARCH_REPORT.md` for the full audit and `V1.0.1_RELEASE_VERIFICATION.md` for release evidence.
