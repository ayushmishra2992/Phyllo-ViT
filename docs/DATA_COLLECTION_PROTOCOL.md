# Paired data-collection protocol (Phase 2) — draft for review with a botanist and the chemistry lab

Goal: a dataset in which **every image, FTIR spectrum and pollution measurement refers to the same documented leaf or tree visit**, so that multimodal claims can be tested honestly.

## 1. Design
| Factor | MVP | Strong |
|---|---|---|
| Sites | 2 (one high-traffic roadside, one low-traffic residential), each ≤ 3 km from a CPCB CAAQMS station | 4 (add an industrial and a park/background site) |
| Species | 3 (one per primary arrangement) | 6 (two per arrangement) |
| Trees per species per site | 5 (tagged, `plant_id`) | 6 |
| Visits | 4 (≥ 7 days apart; record days since last rain) | 8 across two seasons |
| Tree-visits | **120** | **1,152** |

Candidate roadside species common in Delhi-NCR (verify arrangement with a botanist/local flora before use): *Alstonia scholaris* (whorled), *Nerium oleander* (whorled, usually 3 per node), *Psidium guajava* (opposite), *Syzygium cumini* (opposite), *Polyalthia longifolia* (alternate), *Ficus religiosa* (alternate), *Aegle marmelos* (alternate, trifoliate — the species named in the reference paper).

## 2. Sample identity
`sample_id = SITE-SPECIES-TREE-VISIT-LEAF` (e.g. `HT1-ALSC-T03-V2-L1`). Write it on the bag, the photo board and every data file. One CSV row per leaf; one per tree-visit for exposure.

## 3. Per tree-visit procedure (same time window, e.g. 16:00–18:00)
1. **Shoot photo for phyllotaxy:** stem with ≥ 3 nodes, side view and top view, grey card + scale bar + sample-ID card in frame, no flash, phone on fixed settings (record device). 2 photos.
2. **Leaf selection:** 3 fully expanded leaves from the 3rd–5th node, at 2.0–2.5 m height, outer canopy facing the road. Record height and orientation.
3. **Leaf photo (adaxial surface):** on matte black and white boards, colour card, fixed distance; 1 photo per leaf.
4. **Handling:** gloves; cut petiole; place each leaf in a labelled paper envelope; keep cool and dark; lab within 6 h.
5. **Exposure log:** co-located low-cost PM2.5/PM10 sensor (on site ≥ 24 h before the visit, calibrated against the reference station), temperature, RH, wind; download station hourly data for the exposure window.

## 4. Laboratory
* **Leaf area:** scan → area (cm²).
* **Deposited PM (primary target):** sequential wash — water (surface PM) then chloroform (wax-bound PM); filter through pre-weighed filters by size fraction; dry; weigh on a ≥ 0.01 mg microbalance; report µg/cm². Include field and lab blanks.
* **FTIR:** for each leaf, ATR-FTIR on the adaxial surface (3 positions) **before washing** and on the washed leaf (difference spectrum = deposit contribution); 4000–600 cm⁻¹, 4 cm⁻¹ resolution, ≥ 32 scans, background every 10 samples; record instrument, mode, date. Optional KBr pellet of dried powder for a subset.
* **QC:** replicate 10% of leaves; record instrument drift; blind the analyst to site.

## 5. Data files
`samples.csv` (identity, species, site, tree, visit, height, timestamp, GPS), `images/` (filenames = sample_id), `spectra/` (CSV per spectrum, wavenumber, absorbance), `pm_gravimetric.csv`, `exposure_hourly.csv`, `annotations.csv` (primary + secondary arrangement, two annotators, agreement), `README` with licence.

## 6. Analysis rules (pre-registered)
* Splits grouped by **tree**; report additional leave-one-site-out and leave-last-visit-out results.
* Targets: deposited PM (µg/cm²) — primary; exposure-window mean PM2.5 — secondary. AQI is *computed* from concentrations, never regressed directly.
* Compare image-only, FTIR-only, exposure-only, each pair, and all three, with identical folds; Ridge/PLS first, fusion only if justified.
* Report mean ± SD over folds, CIs over trees, and all configurations.
