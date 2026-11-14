# Phyllo-ViT 2.0 — dataset sources (checked 2026-11-14)

Rule: an entry is `verified` only if its licence was read from the original source in this session. Nothing here was assumed from memory. Machine-readable version: [`dataset_registry.csv`](dataset_registry.csv).

## Acquired
| Dataset | Licence (read) | What we took | Use in v2 |
|---|---|---|---|
| [Cannabis sativa leaf IR spectra, Zenodo 17165736](https://zenodo.org/records/17165736) | CC BY 4.0, Open | 2 of 19 files: `W1_C.xlsx` (8 control spectra × 7,468 points), `Sample_name_codification.xlsx`; MD5 verified against Zenodo's API | Real spectra to test the FTIR parser/viewer. **Not** pollution data; not paired with images or AQI. |

Format quirks found by inspection (not stated on the record): wavenumbers are integers ×1000 (4000123 → 4000.123 cm⁻¹; inferred from a 0.482 cm⁻¹ step and an end value of 400.157), absorbance uses comma decimals. The parser takes `wavenumber_scale=1000` for this file.
Attribution: Jara Villamayor, Herrera, Marsal, Arrúa Alvarenga, DOI 10.5281/zenodo.17165736 (CC BY 4.0).

## Integrated service (no data stored)
**OpenAQ** — terms and licences pages read. Licences are per source (the API exposes them per location). An API key is required (free registration; `X-API-Key` header); general limit 60/min and 2,000/hour. CPCB is not mentioned on the pages read, so CPCB-sourced stations' licence must be read from the API's licence field at run time. **Live retrieval has not been run in this project** (no key); the client is tested against a mock transport only.

## Registered, not acquired
- **PlantDoc** — CC BY 4.0 (LICENSE.txt read), open. Candidate for a future validated leaf-health classifier; not needed for the experimental colour-based module.
- **PlantVillage** — no licence found on the GitHub README or TFDS page; Hugging Face/Mendeley pages not read. **Do not use until confirmed.**
- **Latvian herbal FTIR-ATR (Zenodo 16892975)** — CC BY 4.0 but files restricted (access request) and samples are powdered herbs, not fresh leaves. Manual access request needed; not bypassed.

## Not verified / not found
- Roboflow `phyllotaxie_`: page returned HTTP 403 to the automated fetch; classes and licence unknown — a person must open it in a browser.
- **Spiral phyllotaxy:** no labelled open dataset found → Spiral stays unsupported.
- **Species identification, dust-deposition masks:** no verified dataset → modules return `NOT_SUPPORTED`.
- **CPCB portal terms, bacterial/fungal laboratory datasets:** not investigated.

## Consequences
Phyllotaxy keeps its 4 classes; species, dust, Spiral and biological modules are disclosed as unavailable; no multimodal fusion is attempted (no genuinely paired image + FTIR + pollution records exist).
