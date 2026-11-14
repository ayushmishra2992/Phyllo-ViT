# Phyllo-ViT v3 â€” notes for Claude

Final-year research project (Ayush Mishra: ML/backend; Riya Katiyar: literature/frontend/docs). Reliability-aware phyllotaxy recognition (Alternate / Opposite / Whorled + Not_a_leaf) with abstention, plus an evidence audit of the reference paper's leafâ†’AQI claims.

## Layout
- `phyllo/` core library: `paths.py` (all paths/constants), `pipeline.py` (`PhylloEngine.predict`), `reliability.py` (OOD scores, temperature, metrics), `quality.py` (quality gate), `features.py` (ViT encoder, attention rollout), `aqi_standard.py` (CPCB 2014 breakpoints).
- `api/main.py` FastAPI app (`/predict`, `/aqi/compute`, `/evidence`, `/health`, `/model-card`) + built-in UI `api/static/index.html`.
- `frontend/` Next.js 16 app (Riya's). API base URL from `NEXT_PUBLIC_API_URL`. Keep changes minimal.
- `experiments/e01â€¦e07` + `run_all.py` â†’ `results/RESULTS.md` (tables T1â€“T9) and `artifacts/` (deploy-time thresholds).
- `models/` two fine-tuned ViT-B/16 checkpoints (Git LFS). `data/processed/clean` = 265 unique leaves + 150 COCO non-leaves; `split_4class` = the deployed model's split.
- `docs/` RESEARCH_REPORT (20 sections), MODEL_CARD, DATASET_SUFFICIENCY, DATA_COLLECTION_PROTOCOL, PAPER_PLAN, DEPLOY. `legacy/` = untouched v2.

## Commands (Windows)
- Python env: `RUN_ME.bat` finds one via `scripts/find_python.bat` (project `.venv`, else `%USERPROFILE%\OneDrive\Desktop\Phyllotaxy\.venv`). Use that interpreter; set `PYTHONIOENCODING=utf-8`.
- Run app: `RUN_ME.bat` (http://127.0.0.1:8000). Tests: `RUN_TESTS.bat` (13 tests). Per-class check: `CHECK_CLASSES.bat`. All experiments: `RUN_EXPERIMENTS.bat` (~5 min on RTX 3050 4 GB from a local folder; ~19 min inside OneDrive, which slows file I/O).
- Frontend: `cd frontend && npm ci && npm run build` (Node â‰¥ 20.9). Port 3000 may already be in use on this machine.
- Paths: keep the project path short (Windows 260-char limit). `.bat` files must stay CRLF.

## Rules (from the project brief â€” do not break)
- Never fabricate results, datasets or citations. Every number must come from a script in `experiments/` and be reproducible.
- Never pair images, FTIR values and pollution readings that were not measured together. The two pollution workbooks have identical targets; do not treat them as two sites.
- Do not present 100 % accuracy on small test sets as generalisation; report Wilson CIs. Not_a_leaf â‰  complete OOD detection.
- AQI from images stays WITHHELD unless a cross-validated, permutation-tested result supports it (see `docs/DATASET_SUFFICIENCY.md`). Never output AQI for non-Whorled or abstained images.
- Respect the 4 GB GPU: batch size â‰¤ 8 for ViT-B/16 inference, fp16 for training.
- Separate implemented / verified / proposed / future work in docs. Don't claim something works until it has been run.
- New datasets need source, licence and provenance recorded before use.

## Current state (verified)
- 4-class ViT: 64/65 test (Wilson 91.8â€“99.7 %); with abstention 41/42 leaves accepted, all correct; 23/23 non-leaves rejected.
- Shortcuts: colour-only probe 95.6 %, orientation/size probe 76.7 %; frozen ViT learning curve flat from ~22 images â†’ diversity, not quantity, is missing.
- Imageâ†’AQI: no skill (p = 0.45). Station pilot (Vikas Sadan, 5 days): no out-of-day skill.
- Open task: fill `data/metadata/provenance_annotation_TODO.csv` (species, plant, date) and rerun with species-grouped CV.
