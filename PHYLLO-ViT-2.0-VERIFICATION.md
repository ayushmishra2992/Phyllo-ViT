# Phyllo-ViT 2.0 — verification

The ZIP was built from the repo tree, extracted fresh to `C:\pv2fresh`, installed with `INSTALL.bat` (own `.venv`, CUDA torch 2.14.1+cu126, `npm ci`) and verified there. The released ZIP is that verified tree plus this file (the only difference; hash is in the release notes and `.sha256` sidecar). v1.0.1 (tag `v1.0.1`, ZIP SHA-256 `62292186DF908FAC9DD2FE2D0747596FD2C9F353CF2987027191EF25D75C3666`) is untouched; v2 adds files and changes only `README.md`, `scripts/package_final.ps1` (new optional `-TopDir`) and docs.

Environment: Windows 11, RTX 3050 4 GB, Node 24.21.0, Python 3.12. Test labels: **REAL** = real held-out images / real FTIR file / real HTTP; **SYNTHETIC** = generated parser input; **MOCK** = OpenAQ replaced by a mock transport.

| # | Verification | Result |
|---|---|---|
| 1 | Contents audit (1,732 entries) | PASS — 0 entries for `.venv`, `node_modules`, `.next`, `__pycache__`, `.git`, `.env`, `.env.local`; only `.env.example` / `.env.local.example`. |
| 2 | `INSTALL.bat` in the fresh extraction | PASS — log ends "Install finished"; `pip check`: no broken requirements; `npm ci`: 0 vulnerabilities. (The exit code written to the log by my wrapper is unreliable; completion is proven by the log and the checks below.) |
| 3 | v1 backend tests | PASS — 13/13 (unchanged suite). |
| 4 | v2 tests | PASS — 23/23: FTIR parsing (3 SYNTHETIC formats, bad-input rejection, REAL Zenodo hemp-leaf file: 7,468 points, 400.16–4000.12 cm⁻¹, ≥3 peaks), registry schema + MD5 of acquired files, imaging on REAL held-out leaves, blank image, unsupported modules never invent values, OpenAQ client (MOCK: success, missing location/key, empty/401/429/500/garbage/offline), `/v2/modules`, v1 routes unchanged under the v2 app, REAL Alternate/Opposite/Whorled through `/v2/analyze`, REAL non-leaf, REAL blurred image, invalid uploads (400/413/422), location without key, REAL FTIR through the API, bad FTIR → 400, CORS. |
| 5 | `check_classes.py` | PASS — Alternate 0.988 ACCEPT, Opposite 0.988 ACCEPT, Whorled 0.980 ACCEPT, Not_a_leaf 0.925 NOT_A_LEAF. One image per class is a smoke test; the single held-out miss is abstained. |
| 6 | Frontend production build | PASS — routes `/`, `/lab`, `/v2`. Warning (unchanged): React 18 deprecated in Next.js 16. |
| 7 | Model hashes | PASS — both `model.safetensors` identical to v1.0.1. |
| 8 | `RUN_V2.bat` live run (my process tree only; ports 3000/8000 free before and after) | PASS — backend healthy in ~11 s; `/v2` page 200; `/v2/modules` map; real multipart POST from origin `http://localhost:3000` with image + location + collection time + real FTIR file → phyllotaxy Whorled 0.980, morphology AVAILABLE, visible-condition EXPERIMENTAL, species/dust/leaf-AQI NOT_SUPPORTED, biological DISABLED, ambient USER_INPUT_REQUIRED ("No OpenAQ API key configured"), FTIR AVAILABLE (7,468 points, 12 peaks; top 3332.9, 1633.9 cm⁻¹); non-leaf → Not_a_leaf, morphology UNAVAILABLE; text upload → 400; v1 `/predict` still 200; CORS allowed for `:3000`. |

## What was NOT verified
- **Live OpenAQ retrieval** — no API key; the client is tested against mock responses written from the documented v3 shapes, so real-response compatibility is unproven.
- Real-browser click-through of `/v2` (the page builds and is served; the workflow was driven over HTTP).
- Morphology/visible-condition quality against labelled data (no labels exist; outputs are heuristic measurements, labelled as such).
- Docker image; `INSTALL.bat` exit code; `PUSH_TO_GITHUB.bat`.
- `RUN.bat`, `RUN_TESTS_V2.bat` wrappers (their Python commands were run directly).

## Scientific limitations (unchanged and new)
Image→AQI has no demonstrated skill and stays withheld; phyllotaxy validated only on the project's small held-out set; colour shortcut risk (95.6 % colour-only); species, Spiral, dust masks, a trained leaf-health classifier, fusion and the bacterial/fungal module do not exist because no verified labelled data was acquired. FTIR band ranges are indicative and do not identify pollutants. See `docs/V2_ARCHITECTURE.md` and `v2/data_registry/DATASET_SOURCES.md`.
