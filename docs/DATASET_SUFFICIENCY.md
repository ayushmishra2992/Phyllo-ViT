# Is the data enough? — dataset-by-dataset decisions

All data in this project comes from the reference paper (*A Phyllotaxy based Multimodal Learning via FTIR and Vision Transformers for Fine-Grained Air Pollution Prediction*). No new data has been added. This page lists, for every operation the project performs or the paper claims, whether the available data supports it, and what the project does instead when it does not. Numbers come from `results/RESULTS.md` (tables T1–T9) and can be regenerated with `RUN_EXPERIMENTS.bat`.

## 1. What the paper says vs what the files contain

| Item | Paper says | Files actually contain | Consequence |
|---|---|---|---|
| Leaf photos | Samsung Galaxy Z Fold6, 1856 × 2160 px, Vatika Sector-82 & IMT Manesar, March 2025 | 265 unique leaf photos in 5 other resolutions (1200×1600, 867×1156, 1600×1200, 1156×867, 960×1280), **no EXIF** (camera, date, GPS all stripped) | Capture conditions cannot be verified; species/site/date unknown → `data/metadata/provenance_annotation_TODO.csv` |
| FTIR | KBr pellets, Y-IR500, peak tables (Tables 5.2/5.3) | One `Absorbance` column per workbook row, a smooth sequence (lag-1 autocorrelation 0.9995) — no spectra | Spectral analysis cannot be reproduced |
| Pollution targets | CPCB/HPCB, IMT Manesar (high traffic) and Sector-82 (low traffic) | `AQI data.xlsx` and `AQI data LT.xlsx` have **identical pollutant values in all 455 rows**; no timestamps | The two "sites" are not different data; any model fit is a fit to row order |
| Image → AQI | R² 0.93–0.96 | `data AQI.xlsx`: 51 filename→AQI rows, 46 verifiable; no timestamps/station | Cross-validated R² is negative (T8) |
| Station file | — (not described) | `NISE Gwal Pahari,Gurugram-IMD.xlsx`, sheet titled **Vikas Sadan, Gurugram (HSPCB)**: 75 hourly rows, 23–27 Oct 2022, PM2.5/NO₂/SO₂/CO/O₃ + mean leaf RGB | The only genuinely time-paired leaf↔pollution data; but a different station, year and month from the paper's study |

## 2. Operation-by-operation

| Operation | Data available | Enough? | What the project does |
|---|---|---|---|
| Phyllotaxy classification (Alternate / Opposite / Whorled) | 265 unique images (88/83/94) | **Quantity yes, diversity no.** Frozen ViT reaches 99.5 % with only ~22 training images; a colour-only probe reaches 86 % → 95 %. More photos of the same kind would not help. | Ships the classifier with abstention, calibration and OOD rejection; reports Wilson CIs; never claims generalisation to new species/sites. |
| Not-a-leaf rejection | 150 COCO-2017 images (licence: CC BY 4.0 annotations; Flickr images under their own licences — cite COCO) | Yes for "far" OOD (AUROC 1.0); not tested on near-OOD (other plants, bark, flowers). | Combined with Mahalanobis + kNN distance so unfamiliar leaves are also abstained. |
| Calibration / abstention thresholds | 39 validation leaves | Small: validation accuracy is 100 %, so temperature is unidentifiable. | Temperature constrained to [0.5, 5]; thresholds at 95 % TPR; documented as a limitation. |
| Image → AQI regression | 46 verified pairs | **No.** 46 pairs can only detect R² ≥ 0.16; observed CV R² < 0; permutation p = 0.45. Detecting R² = 0.2 needs ~37 *independent* site-days, R² = 0.1 needs ~77. | AQI is **withheld** by default; the exploratory value is shown only on request and labelled as such. |
| FTIR → pollutants | 2 × 455 rows, identical targets, no spectra | **No.** Shuffled-CV skill comes from row order (row index alone gives R² 0.51–0.66). | Reported as a leakage audit (T7), not as a model. |
| Leaf colour → PM2.5 (station pilot) | 75 hours on 5 days | **No.** 90 % of PM2.5 variance is between days, i.e. effectively 5 samples. Leave-one-day-out R² is negative for PM2.5 and SO₂; within-day correlation not significant (p = 0.29, 0.72). NO₂ shows R² 0.17–0.24, but NO₂ is almost constant within a day, so this is learned from 4 day-means and is not evidence. 27 Oct RGB values are near-grey (R≈G≈B), suggesting non-leaf pixels. | Reported as a pilot (T9) with the exact data-entry fixes (5 "12.00 AM" → noon; 43 comma-less RGB strings parsed, all unambiguous). |
| CPCB AQI calculation | Official breakpoints | Yes. | `/aqi/compute` endpoint and calculator tab; works on measured concentrations. Hourly sub-indices in the pilot are indicative only (official AQI uses 24-h averages; CO units in the file are unknown, so no CO sub-index). |

## 3. What can still be added without fieldwork

* **Public CPCB data (the paper's own source):** hourly data for the paper's stations and dates can be downloaded from the CPCB portal (https://airquality.cpcb.gov.in/AQI_India/ and the CCR data repository). This lets you (a) verify the station file against the official record, and (b) attach real station values to any photo whose date you can recover. The portal requires a manual download (it uses a CAPTCHA), so a team member has to do it.
* **Provenance annotation of the existing photos:** filling species / plant / capture-date in `data/metadata/provenance_annotation_TODO.csv` costs no fieldwork and enables species-grouped CV — the single most important missing test for the classifier.
* **Near-OOD negatives:** non-leaf plant parts from a licensed public dataset (record source and licence in the manifest) would make the abstention test harder and more honest.

## 4. What the paper/report can claim with this data

Supported: a reliability-aware phyllotaxy classifier on this collection (98.5 % test, 41/42 accepted correct, 23/23 non-leaves rejected); the shortcut and leakage audit; the sample-size analysis above; a working CPCB AQI calculator.
Not supported: AQI or pollutant prediction from leaf images or FTIR; differences between high- and low-traffic sites; generalisation to new species, sites or cameras.
