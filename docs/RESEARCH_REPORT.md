# Phyllo-ViT — Audit, Research Direction and Final System (v3)

Prepared 12 November 2026 for Ayush Mishra and Riya Katiyar. Every quantitative claim about *our* data in this report comes from scripts in `experiments/` (tables T1–T8 in `results/RESULTS.md`). Status labels used throughout:

**[VERIFIED]** implemented and run, numbers reproduced · **[IMPLEMENTED]** code works, not a research result · **[PROPOSED]** designed, not built · **[FUTURE]** needs new data.

---

## 1. Executive assessment

Phyllo-ViT v2 was a working demo (two fine-tuned ViTs, FastAPI, Next.js) whose headline numbers looked excellent (100% / 98.5% classification, R² 0.82 AQI). The audit shows those numbers are **real but do not mean what they appear to mean**:

1. The three arrangement classes are separable by **colour statistics alone (95.6%)** and partly by **image orientation/size alone (76.7%)**. The model can score near-perfectly without understanding leaf arrangement — each class is very likely dominated by one or a few species/photo sessions.
2. The pollution data cannot support the claims built on it: the "high-traffic" and "low-traffic" workbooks have **identical pollutant targets in all 455 rows**, and under random cross-validation a model that sees **only the row number** predicts pollutants as well as the leaf features. The apparent skill is temporal autocorrelation, not leaf information.
3. The image→AQI model (46 Whorled pairs) **does not beat predicting the mean** under repeated CV (permutation p = 0.45). The R² 0.82 came from a 5-image test split.

What *is* strong and novel enough to build a paper on: a **reliability layer** that knows when not to answer (it abstained on exactly the one misclassified test image and on 90% of corrupted images, and rejects 100% of non-leaf images) and a **rigorous evidence audit** that exposes shortcut learning and temporal leakage in leaf-based air-quality sensing — a problem the attached paper (now published) does not address.

**Recommendation:** reframe the project from "predict AQI from leaves" to **"trustworthy leaf-based environmental sensing: recognise only what you can validate, abstain otherwise, and audit the evidence"**, deliver that now (Phase 1, done in v3), and collect a properly paired dataset for the multimodal pollution question (Phase 2).

---

## 2. Audit of the ZIP and codebase

| Item | What exists | Notes |
|---|---|---|
| `Phyllo-ViT-Test.zip` (990 MB) | **Older** snapshot: 3-class ViT, 3-class API, results, raw zips, `.git` | No 4-class model, no Not_a_leaf data, no frontend. **The ZIP is out of date** vs the report. |
| `Desktop\Phyllotaxy` (actual project) | 3-class + 4-class ViT, 150 Not_a_leaf images, `split_4class`, 4-class API (`src/api.py`), `api_3class_backup.py`, Next.js frontend, working `.venv` (torch 2.13+cu130) | Used as the basis of v3 (copied, not modified). |
| Data | 289 raw → 265 unique leaves (88/83/94) + 150 COCO Not_a_leaf; 70/15/15 random splits | Image-level random split; no species/plant/session metadata. |
| Models | `vit_phyllotaxy` (3-class), `vit_phyllotaxy_4class` | Both `google/vit-base-patch16-224-in21k`, 8 epochs, LR 2e-5. |
| AQI | 4 workbooks; 51-row filename→AQI map (paths from another PC); Ridge on 46 Whorled embeddings | AQI pairs carry no date/station/averaging period. |
| Frontend | Next.js 14 + Tailwind; types expected only 3 classes | No CORS on the v2 backend (cross-origin calls from :3000 would be blocked by browsers). Fixed in v3. |

**Discrepancies found:** (a) the ZIP lacks everything added after 2 Oct (4-class work, frontend); (b) the v2 frontend type `PhyllotaxyClass` omitted `Not_a_leaf`; (c) the 4-class split re-randomised the leaves, so **30 of the 65 four-class test images were in the 3-class model's training set** — any cross-model evaluation on that test set leaks; (d) 15 images carry an EXIF rotation that training ignored; (e) the README's "R² 0.822" omits the negative cross-validated R².

## 3. What has genuinely been completed

| Component | Status |
|---|---|
| Dedup (24 exact duplicates), splits, EDA | [VERIFIED] |
| HOG-SVM baseline | [VERIFIED] 95% on the 40-image split; 93.3 ± 3.6% under repeated CV (T2) |
| 3-class ViT | [VERIFIED] 40/40, Wilson 95% CI 91.2–100% (T3) |
| 4-class ViT | [VERIFIED] 64/65 = 98.5%, CI 91.8–99.7% (T3) |
| Not_a_leaf rejection | [VERIFIED] on COCO-style negatives (AUROC 1.000 even for models never trained on them, T4) |
| FastAPI + Next.js | [IMPLEMENTED] |
| AQI regression | Ran, but [NOT SUPPORTED] by evidence (T8) |

## 4. What is incomplete or incorrectly implemented (v2)

1. **Evaluation protocol**: single random split, no CIs, near-duplicates across splits (44 pairs; 10 four-class test images have a near-duplicate in train/val).
2. **Confound not tested**: no shortcut baselines — colour/orientation explain most of the "phyllotaxy" signal.
3. **Confidence is uncalibrated**: the raw 4-class model is *under*-confident (median ~0.80 when always right; ECE 0.205).
4. **No OOD/abstention**: every leaf-like image was forced into a class.
5. **AQI**: model selection on a 5-sample test, no permutation baseline, target provenance undocumented, AQI conflated with pollutant concentration.
6. **Robustness**: blurred photos were confidently classified (blur → 42/42 predicted as a leaf class by the classifier alone).

All six are addressed in v3 (sections 11–13).

## 5. The research paper — methodology and findings

*"A Phyllotaxy based Multimodal Learning via FTIR and Vision Transformers for Fine-Grained Air Pollution Prediction"* (thesis chapter 5; a version titled *"Prediction of streetwise AQI from deposited pollutant on leaf Phyllotaxy using Vision transformer and FTIR Spectral data"* appears as an accepted manuscript in Global NEST Journal, 2026, doi 10.30955/gnj.07998 — verify the final citation before use).

| Element | What the paper does |
|---|---|
| Problem | Street-level AQI is unavailable; area-level stations are coarse. Use roadside leaves as bio-indicators. |
| Sites / time | Vatika Sector-82 (Gurugram, residential low traffic) and IMT Manesar (high traffic); March 2025, evening 5 PM; leaves at ~2.5 m. |
| Data | Smartphone leaf images (50 MP); FTIR (Y-IR500, 400–4000 cm⁻¹, KBr pellets of dried leaf, *Aegle marmelos* named); CPCB/HPCB pollutant values (PM2.5, PM10, NO₂, CO, O₃). |
| Image features | "Contrast, entropy, standard deviation" per leaf image, computed after ViT processing. |
| Model | ViT classifier for Alternate/Opposite/Whorled + Bayesian-optimised multilevel regression (BO-MLR) in MATLAB R2024a. |
| Reported results | 100% test classification; R² 0.93 (PM2.5, high traffic, RMSE 5.7) to 0.96 (NO₂, low traffic, RMSE 3.7); some text also quotes 96%/0.88. |
| Limitations stated | One urban area; species, leaf age, season, meteorology uncontrolled; station data gaps. |

## 6. Paper vs our implementation

| Area | Research paper | Our current project | Actual gap (evidence) | Proposed improvement |
|---|---|---|---|---|
| Ground-truth pollution | Tables 5.4 (IMT Manesar) and 5.5 (Sector-82) list **identical** daily PM2.5/PM10/NO₂/CO/O₃ for all 30 days | Our workbooks reproduce this: HT and LT targets identical for 455/455 rows (T7) | The two "sites" share one target series, so site-level conclusions cannot come from these targets | Site-specific measurements (co-located sensor or nearest station per site), documented in a manifest |
| Validation protocol | Not reported (no split description, no CV, no baseline) | Random splits | Random splits on autocorrelated daily data inflate R²: row index alone → R² 0.51–0.66 (T7) | Blocked / forward-chaining CV, group by tree and site, permutation tests |
| FTIR feature | One absorbance value per day, rising smoothly (0.281→0.342) | "Absorbance" lag-1 autocorr 0.9995 | A slowly drifting scalar mainly encodes *time*, not per-sample chemistry | Full spectra per sample; preprocessing; spectral models (§8, §12) |
| FTIR interpretation | Peaks assigned to gases (CO, O₃, CH₄, NO₂, NH₃), e.g. 3401 cm⁻¹ as C–H | Not implemented | In dried plant tissue these regions are conventionally assigned to O–H/N–H stretch (~3400), CH₂ stretches of waxes/lipids (~2920/2850), ester C=O (~1730), amide/water (~1640), polysaccharide C–O (~1030). Gas-phase pollutants are not retained as such in a dried KBr pellet. Verify with a plant-FTIR band-assignment review before claiming anything. | Treat FTIR as a leaf-surface chemistry descriptor; link to *measured* deposited PM / elemental load, not to gas names |
| Image features | Contrast ≈ entropy ≈ std (e.g. 183/185/187) | Same pattern; contrast = entropy = std in 15% of HT rows | Three nearly identical "features" add no independent information | Learned embeddings + documented hand-crafted features computed by our code |
| Classification | 100% test, 3 classes | 100% (3-class), 98.5% (4-class) | Classes confounded with colour/orientation (T2); no CIs | Shortcut baselines, CIs, species-disjoint test sets (Phase 2) |
| Phyllotaxy taxonomy | "whorled, opposite, spiral, alternate" as 4 types | 3 classes | Spiral is a form of alternate phyllotaxy, not a peer class | Hierarchical labels (§11) |
| Reliability | Not addressed | v2: Not_a_leaf only | No abstention, calibration, OOD | v3 reliability layer [VERIFIED] |
| Reproducibility | MATLAB, no code/data release | Python scripts | — | Full scripted pipeline, fixed seeds, `RESULTS.md` |

## 7. Research gaps that remain unaddressed

Supported by evidence in this project and by a brief literature check (no single study combining these was found; OOD on plant data is active — e.g. an arXiv 2026 systematic study of OOD methods on plant-pathology images, arXiv:2605.08618, and ViT open-set plant species recognition in VNPLANT200 — both are about species/disease, not phyllotaxy or leaf bio-monitoring):

* **G1 — No reliability layer for leaf-based environmental sensing.** Prior leaf/AQI work reports point accuracy only; nothing tells a user when the input is outside validated scope.
* **G2 — No shortcut/leakage audit.** Leaf-bioindicator ML papers rarely test whether colour, camera, session or time index explain the result. Our T2/T7 show they can explain *all* of it here.
* **G3 — Phyllotaxy is a whole-shoot property** but datasets mix isolated leaves and shoots; there is no public benchmark with species-disjoint splits.
* **G4 — No genuinely paired image + spectrum + exposure dataset** with sample-level identity.

We claim novelty only for G1+G2 in combination with phyllotaxy/leaf-bioindicator sensing — not for using ViT, FTIR, OOD methods or multimodal fusion per se.

## 8. Candidate research directions

| ID | Direction | Description |
|---|---|---|
| D1 | **Reliability-aware phyllotaxy recognition** | Quality gate + calibrated confidence + feature-space OOD + abstention; evaluated on non-leaf, unseen arrangement/species, corruptions; risk–coverage. |
| D2 | **Evidence audit of leaf-based air-quality ML** | Shortcut probes, near-duplicate leakage, temporal-leakage tests (row-index control, blocked vs random CV), permutation tests. |
| D3 | **Species-disjoint, stem-visible phyllotaxy benchmark** | ≥3 species per arrangement, hierarchical labels, grouped splits. |
| D4 | **Paired multimodal leaf sensing** | Same-leaf image + FTIR + gravimetric deposited PM + exposure covariates; image/FTIR/metadata ablations. |
| D5 | Hierarchical phyllotaxy (alternate→distichous/spiral; opposite→decussate/distichous; whorled; rosette) | Needs D3 data. |

## 9. Ranking

Scores 1–5 (5 = best).

| | Novelty | Scientific relevance | Feasibility now | Data available | Compute | Contribution | Publication potential | **Total** |
|---|---|---|---|---|---|---|---|---|
| D1 | 3 | 4 | 5 | 4 | 5 | 4 | 3 | **28** |
| D2 | 4 | 5 | 5 | 5 | 5 | 4 | 4 | **32** |
| D3 | 3 | 4 | 3 | 1 | 5 | 4 | 3 | **23** |
| D4 | 4 | 5 | 1 | 0 | 4 | 5 | 5 | **24** |
| D5 | 2 | 3 | 2 | 0 | 5 | 3 | 2 | **17** |

## 10. Recommended research direction

**Paper 1 (Phase 1, now): D1 + D2 together.**
*Working title:* **"Trust Before Inference: Reliability-Aware Phyllotaxy Recognition and an Evidence Audit of Leaf-Based Air-Quality Estimation."**

* **Problem statement.** Leaf-image AI for phyllotaxy and air-quality bio-monitoring reports near-perfect metrics on small, single-site datasets, but provides no mechanism to recognise invalid, unfamiliar or degraded inputs, and its evaluations are vulnerable to shortcut and temporal leakage. We ask whether such a system can be made to *know when not to answer*, and how much of the reported skill survives leakage-aware evaluation.
* **Research questions.** RQ1: Do accuracies survive grouped CV and shortcut controls? RQ2: Which post-hoc OOD scores best detect non-leaf, unseen-arrangement/species and degraded inputs? RQ3: Does a validation-calibrated abstention policy raise accuracy-on-accepted at acceptable coverage? RQ4: Does leaf-derived evidence predict pollutants/AQI beyond time-index and mean baselines under time-respecting CV?
* **Hypotheses.** H1: colour/orientation probes achieve accuracy far above chance (supported: 95.6%, 76.7%). H2: feature-space scores (Mahalanobis/kNN) beat logit scores for near-OOD (supported for unseen arrangement and noise: 0.984 vs 0.976; 0.942 vs 0.842). H3: abstention increases selective accuracy (supported: 97.6% → 100% at 97.6% coverage — on n=42, so modest evidence). H4: leaf features add no skill beyond time under blocked CV (supported for this dataset).
* **Contributions.** (1) an open, reproducible reliability layer for leaf phyllotaxy models; (2) a leakage/shortcut audit protocol with a row-index control; (3) negative but decisive evidence about the current leaf→AQI data; (4) a pre-registered data protocol (Phase 2).

**Paper 2 (Phase 2): D3 + D4** — only if the paired dataset of §12 is collected.

## 11. Final system architecture (v3)

```
                         ┌───────────────────────── artifacts/ (fit on train, thresholds on val) ─────────────────┐
                         │ temperature T · Mahalanobis means/precision · kNN bank · thresholds · AQI evidence      │
                         └───────────────▲───────────────────────────────▲──────────────────────────▲─────────────┘
 upload ─► Quality gate ─► ViT-B/16 4-class ─► softmax(z/T) ─┐  CLS 768-D ─► Mahalanobis + kNN ─┐   │
 (size, exposure,  │        (deployed)                       └──────────────► Decision policy ◄─┘   │
  blur, contrast,  │                                                   ACCEPT / ABSTAIN_UNCERTAIN / │
  noise warning)   │                                                   ABSTAIN_UNFAMILIAR /         │
                   └── REJECT_QUALITY                                  NOT_A_LEAF / REJECT_QUALITY  │
                                                                              │                     │
                        Attention rollout heatmap ◄──────────────────────────┤                     │
                        Evidence-gated AQI: only if ACCEPT ∧ Whorled ∧ evidence.supported ∧ inside applicability domain
                        CPCB AQI calculator: measured concentrations → sub-indices → AQI (separate from models)
```

| Component | Problem solved | Why scientifically relevant | Cost | Measured by | Phase |
|---|---|---|---|---|---|
| Quality gate | Blurred/dark/tiny photos were confidently classified | Garbage-in should not produce a label | ms, CPU | rejection of corrupted inputs (T5: blur/dark 100%) | 1 [VERIFIED] |
| Temperature calibration | Raw confidence under-states reliability (ECE 0.205) | Confidence must mean probability | negligible | ECE/NLL (0.205→0.033; 0.245→0.049) | 1 [VERIFIED] |
| Mahalanobis + kNN OOD | Closed-set model labels anything | Detects unfamiliar inputs without retraining | ms | AUROC/FPR95 (T4, T5) | 1 [VERIFIED] |
| Abstention policy | Users need "insufficient evidence" | Selective classification | none | coverage vs accuracy (T6) | 1 [VERIFIED] |
| Attention rollout | Black-box output | Lets reviewers spot shortcuts (e.g. background) | +~30 ms | qualitative; future: deletion/insertion test | 1 [IMPLEMENTED] |
| Evidence-gated AQI | Unsupported AQI numbers | Claims must follow evidence | none | permutation test (T8) | 1 [VERIFIED] |
| CPCB AQI calculator | AQI ≠ concentration | Documented standard | none | unit tests | 1 [IMPLEMENTED] |
| Evidence page / `/evidence` | Transparency | Every output traceable to a table | none | — | 1 [IMPLEMENTED] |
| Hierarchical phyllotaxy | Taxonomy correctness | Spiral/distichous/decussate are sub-types | retrain | per-level F1 | 2 [FUTURE] |
| FTIR encoder + fusion | Chemical evidence | Only with paired data | small | ablations (§13) | 2 [FUTURE] |
| Prediction history (PostgreSQL) | Field campaigns | Useful once real data collection starts | low | — | 2 [PROPOSED] |

Not added on purpose: cross-attention fusion, large backbones, LLM explanations, dashboards of decorative metrics — none is justified by the current data.

**Why the 3-class ViT is kept:** the legacy Ridge AQI model lives in its embedding space. Since AQI is withheld, the 3-class model only computes the *exploratory* value on request and the AQI applicability check. It can be dropped (saves ~350 MB VRAM) if the team agrees.

## 12. Dataset and data-collection strategy

### 12.1 What can be inferred from a photograph
* An **isolated leaf blade cannot reveal phyllotaxy** — arrangement is defined by how many leaves attach per node and their angular sequence. Images must show **a stem with ≥3 consecutive nodes**.
* Distichous vs spiral (alternate) and decussate vs distichous (opposite) need a view along/around the stem; often ambiguous in one photo → label as sub-type only when visible, otherwise stop at the primary level.
* Rosette/basal arrangement needs the whole plant base.

**Hierarchical label scheme**
```
Input validity: Not_a_leaf | Unusable | Leaf/shoot
Primary:   Alternate | Opposite | Whorled | Rosette/basal | Uncertain
Secondary: Alternate→{Distichous, Spiral}   Opposite→{Decussate, Distichous}   Whorled→{n leaves per node}
```

### 12.2 Candidate public sources (verify licence per image before use)
| Source | Link | Content | Use / limitation |
|---|---|---|---|
| iNaturalist (research-grade) | https://www.inaturalist.org | Geotagged plant photos, many whole shoots; per-photo CC licences | Best source for species-diverse shoots; filter CC-BY/CC0; arrangement must be labelled by us at species level then verified per image |
| GBIF occurrence media | https://www.gbif.org | Aggregated occurrence images incl. herbarium sheets | Herbarium sheets show stems/nodes well; licences vary |
| Pl@ntNet-300K | search "PlantNet-300K Zenodo" | Plant photos, species labels | Many organ close-ups; check licence; species-level labels only |
| CPCB CAAQMS | https://airquality.cpcb.gov.in | Official station data (hourly/daily) | Ground-truth exposure covariates; station ≠ tree location |
| OpenAQ | https://openaq.org | Aggregated government air-quality data, API | Convenient access to CPCB data; check licence/terms |
| Paired leaf image + FTIR + pollution | — | **None found** | Must be collected (§12.3) |

Leaf datasets such as Flavia, Swedish Leaf or LeafSnap are isolated-leaf scans → **not suitable** for phyllotaxy. Do not merge unrelated images, spectra and pollution readings and call them paired.

### 12.3 Paired data-collection protocol
Full protocol in `docs/DATA_COLLECTION_PROTOCOL.md`. Key design decisions:
* **Target = what is measured on the same leaf:** gravimetric deposited PM per leaf area (surface + wax-bound fractions via sequential washing and filter weighing) — not station AQI.
* **Exposure covariates:** co-located low-cost PM sensor (calibrated against the nearest CAAQMS station) + station data averaged over the exposure window (days since last rain).
* **Units of analysis:** tree (plant_id) × visit. Every image, spectrum and PM measurement carries `sample_id = site-tree-visit-leaf`.
* **MVP:** 2 sites (high/low traffic) × 3 species (one per arrangement) × 5 trees × 4 visits = **120 tree-visits** (~360 images, 120 spectra, 120 PM measurements).
* **Strong dataset:** 4 sites × 6 species (2 per arrangement) × 6 trees × 8 visits = **1,152 tree-visits**.
* **Phyllotaxy image set:** ≥10 species per primary arrangement, ≥30 stem-visible images per species, from ≥2 photographers/devices; double annotation with Cohen's κ reported.

## 13. Experimental plan

| Exp | Data | Inputs | Models | Split | Metrics | Baselines / ablations | Status |
|---|---|---|---|---|---|---|---|
| E1 audit | 415 images | pixels, metadata | pHash | — | near-dup pairs, cross-split leakage | — | [VERIFIED] T1 |
| E3 classification | 265 leaves | HOG, colour, metadata, frozen ViT | SVM, LogReg, kNN | stratified & near-dup-grouped 5-fold × 5 | accuracy, macro-F1, ±SD; Wilson CIs | colour/metadata shortcut probes | [VERIFIED] T2–T3 |
| E4 reliability | 4-class splits + corruptions | logits, CLS | MSP, MaxLogit, Energy, Entropy, Mahalanobis, kNN | thresholds on val; test untouched | AUROC, AUPR, FPR95, ECE, NLL, AURC, coverage | leave-one-arrangement-out (5 seeds) | [VERIFIED] T4–T6 |
| E5 pollution audit | workbooks; 46 pairs | 4 leaf features; embeddings | Ridge, RF, kNN | random vs blocked vs forward CV; RKF 5×10 | R², MAE; permutation p | mean baseline, **row-index-only** control | [VERIFIED] T7–T8 |
| E7 species-disjoint | Phase-2 images | shoots | frozen ViT probe, fine-tuned ViT, LoRA/linear probe | leave-species-out | macro-F1 per level | colour probe must drop to near chance | [FUTURE] |
| E8 multimodal | Phase-2 paired | image emb, FTIR features, exposure | PLS/Ridge per modality → late fusion → gated fusion | grouped by tree, held-out visits & site | R², MAE, CI over trees | image-only, FTIR-only, metadata-only, pairs, full | [FUTURE] |
| E9 robustness | all | lighting, background, device | — | per-condition | accuracy, abstention rate | — | [PROPOSED] |

FTIR preprocessing (E8): truncate noisy edges → baseline correction (asymmetric least squares or rubber-band) → Savitzky–Golay smoothing → vector/SNV normalisation → (optional) 1st derivative; features = full spectrum with PCA/PLS, plus band integrals for pre-registered regions. Models: PLS regression (spectroscopy standard), Ridge, 1-D CNN only if n > ~500 spectra.

Rules: no tuning on test; grouped splits by tree/site/session; report mean ± SD and CIs; report every configuration, not only the best.

## 14. Technical stack
Python 3.12, PyTorch (CUDA), Hugging Face Transformers 5.18, scikit-learn, SciPy, FastAPI/Uvicorn, Next.js 14 (optional UI). A built-in HTML UI removes the Node.js requirement for demos. Not needed now: PostgreSQL (until field data collection), MLflow (scripts + JSON outputs are sufficient at this scale), larger backbones. For Phase 2 on the 4 GB RTX 3050: frozen-backbone probes, LoRA/adapter fine-tuning of ViT-B/16 with batch 8 + fp16, or DINOv2-S/ViT-S backbones.

## 15. Phase 1 completion roadmap

| # | Task | Owner | Files | Verify | Status |
|---|---|---|---|---|---|
| 1 | Reproduce everything | Ayush | `RUN_EXPERIMENTS.bat` | `results/RESULTS.md` matches this report | [VERIFIED] here |
| 2 | Provenance annotation of 265 images | Riya + Ayush | `data/metadata/provenance_annotation_TODO.csv` | every row has species/session; κ on 30 double-labelled rows | [PROPOSED] |
| 3 | Species-grouped CV once task 2 is done | Ayush | add `groups=species` to `e03`, `e04` | T2/T4 regenerated | [PROPOSED] |
| 4 | Frontend type-check & build | Riya | `frontend/` | `npm run build` passes | [PROPOSED] (not run here) |
| 5 | Decide: keep or drop 3-class model in API | both | `phyllo/pipeline.py` | — | [PROPOSED] |
| 6 | Paper 1 draft from `RESULTS.md` | both | `docs/PAPER_PLAN.md` | internal review | [PROPOSED] |

## 16. Phase 2 research roadmap
1. Ethics/permissions for sampling roadside trees; lab access for FTIR and microbalance (department of chemistry/environmental science).
2. Pilot (2 weeks): 1 site, 3 species, 3 trees, 2 visits — test protocol timing and PM measurement variance.
3. MVP campaign (6–8 weeks): §12.3.
4. E7 species-disjoint benchmark; hierarchical heads.
5. E8 multimodal ablations; report negative results if fusion does not help.

## 17. Paper-writing roadmap
See `docs/PAPER_PLAN.md`. Order: Methods → Results (tables already generated) → Discussion/limitations → Intro/related work (Riya's literature matrix) → Abstract. Venue choice after internal review; candidate types: an applied ML / ecological-informatics journal or an IEEE conference on environmental sensing / computer vision applications. No acceptance can be predicted.

## 18. Work division
| Ayush Mishra | Riya Katiyar | Together |
|---|---|---|
| Experiments E1–E6 maintenance, species-grouped CV, OOD extensions, FastAPI, Phase-2 models (E7, E8), FTIR preprocessing code | Literature matrix (leaf bio-monitoring, phyllotaxy CV, OOD/selective classification, FTIR of leaves), provenance annotation lead, frontend build/test and evidence page, data-collection logistics & forms, figures, presentation | Research decisions, annotation double-check, field sampling, paper writing, viva preparation (both must be able to explain every block in §11 and every table) |

## 19. Feasibility, risks, compute
* **What can be reused:** both ViT checkpoints, splits, legacy scripts, frontend (lightly updated), the Ridge model (as an exploratory artefact).
* **Rewritten:** API (v3 engine), evaluation (repeated/grouped CV, CIs).
* **Unrealistic now:** street-level AQI prediction from leaf photos; pollutant identification from FTIR peak names; claims for Alternate/Opposite AQI.
* **Likely to yield a meaningful result:** D1+D2 (already have results), species-disjoint benchmark.
* **Cosmetic:** fancier dashboards, more heads, bigger backbones.
* **Risks:** small n (42 test leaves → wide CIs); confounded classes; Phase-2 lab access; temperature unidentifiable when validation is perfect (we constrain T ∈ [0.5, 5] — disclosed); noisy images partially accepted (17/42).
* **Compute:** all of Phase 1 runs in ~5 min on the RTX 3050 (4 GB). Inference ~0.1–0.9 s/image on GPU; works on CPU.

## 20. The exact first task
**Fill `data/metadata/provenance_annotation_TODO.csv`** (species, plant ID, capture date/session, location, "stem and nodes visible?") for all 265 leaf images, starting with whoever photographed them. It is the single change that most increases the scientific value of everything else: it enables species-grouped evaluation (E3/E4 with `groups=species`), tells us whether the 100% accuracy is phyllotaxy or species recognition, and is the foundation of the Phase-2 benchmark.

---
### References for methods used (well-established; verify page numbers when citing)
Dosovitskiy et al., ICLR 2021 (ViT) · Hendrycks & Gimpel, ICLR 2017 (MSP) · Liu et al., NeurIPS 2020 (energy OOD) · Lee et al., NeurIPS 2018 (Mahalanobis) · Sun et al., ICML 2022 (kNN OOD) · Guo et al., ICML 2017 (temperature scaling) · Geifman & El-Yaniv, NeurIPS 2017 (selective classification) · Abnar & Zuidema, ACL 2020 (attention rollout) · Immerkær, CVIU 1996 (noise estimation) · CPCB, National Air Quality Index, 2014.
