# Phyllo-ViT v3 — Results (auto-generated 2026-11-13 09:36)

Every number below is produced by `experiments/` scripts; regenerate with `RUN_EXPERIMENTS.bat`.

## T1. Dataset audit

| Item | Value |
|---|---|
| n_images | 415 |
| near_duplicate_pairs | 44 |
| near_dup_pairs_linking_test_to_trainval_3class | 7 |
| near_dup_pairs_linking_test_to_trainval_4class | 14 |
| n_groups_after_merging_near_dups | 379 |
| fourclass_test_images_in_threeclass_train | 30 |

Leaf images by resolution (camera/orientation proxy):

| Resolution | Alternate | Opposite | Whorled |
|---|---|---|---|
| 1200x1600 | 65 | 19 | 39 |
| 867x1156 | 23 | 3 | 28 |
| 1600x1200 | 0 | 34 | 17 |
| 1156x867 | 0 | 21 | 10 |
| 960x1280 | 0 | 6 | 0 |

## T2. Classification — repeated 5-fold CV on 265 unique leaves (mean ± SD over 25 folds)

| Model | Protocol | Accuracy | Macro-F1 |
|---|---|---|---|
| metadata_probe (orientation/size only) | stratified 5-fold x5 | 0.767 ± 0.037 | 0.761 |
| metadata_probe (orientation/size only) | grouped 5-fold x5 | 0.766 ± 0.054 | 0.759 |
| colour_stats_probe | stratified 5-fold x5 | 0.956 ± 0.024 | 0.956 |
| colour_stats_probe | grouped 5-fold x5 | 0.948 ± 0.038 | 0.947 |
| HOG + SVM (RBF, C=10) | stratified 5-fold x5 | 0.933 ± 0.036 | 0.933 |
| HOG + SVM (RBF, C=10) | grouped 5-fold x5 | 0.925 ± 0.026 | 0.926 |
| Frozen ViT-in21k + LogReg | stratified 5-fold x5 | 1.000 ± 0.000 | 1.000 |
| Frozen ViT-in21k + LogReg | grouped 5-fold x5 | 1.000 ± 0.000 | 1.000 |
| Frozen ViT-in21k + kNN(k=5, cosine) | stratified 5-fold x5 | 0.992 ± 0.009 | 0.992 |
| Frozen ViT-in21k + kNN(k=5, cosine) | grouped 5-fold x5 | 0.992 ± 0.009 | 0.992 |

## T3. Fine-tuned ViTs on their original held-out test sets

| Model | Correct/n | Accuracy | Wilson 95% CI | Test imgs with near-dup in train/val | Accuracy without them |
|---|---|---|---|---|---|
| ViT 3-class | 40/40 | 100.0% | 91.2%–100.0% | 6 | 100.0% (n=34) |
| ViT 4-class (deployed) | 64/65 | 98.5% | 91.8%–99.7% | 10 | 98.2% (n=55) |

## T4. OOD detection (AUROC; higher = better; in-distribution = positive)

| Score | Leaf vs non-leaf (ft 3-class) | Leaf vs non-leaf (frozen probe) | Unseen arrangement (frozen, LOCO, mean ± SD) | FPR@95TPR unseen |
|---|---|---|---|---|
| MSP | 1.000 | 1.000 | 0.975 ± 0.037 | 0.187 |
| MaxLogit | 1.000 | 1.000 | 0.976 ± 0.037 | 0.187 |
| Energy | 1.000 | 1.000 | 0.976 ± 0.037 | 0.187 |
| NegEntropy | 1.000 | 1.000 | 0.976 ± 0.037 | 0.187 |
| Mahalanobis | 1.000 | 1.000 | 0.984 ± 0.023 | 0.107 |
| kNN-cosine | 1.000 | 1.000 | 0.978 ± 0.023 | 0.191 |

Unseen-arrangement AUROC by held-out class:

| Held-out class | Energy | MSP | Mahalanobis | MaxLogit | NegEntropy | kNN-cosine |
|---|---|---|---|---|---|---|
| Alternate | 0.936 | 0.936 | 0.957 | 0.936 | 0.936 | 0.951 |
| Opposite | 0.991 | 0.991 | 1.000 | 0.991 | 0.991 | 1.000 |
| Whorled | 1.000 | 1.000 | 0.995 | 1.000 | 1.000 | 0.984 |

## T5. Deployed 4-class model — calibration, corruptions, selective prediction

* Validation accuracy 100.0% → unconstrained temperature is unidentifiable (optimum at search bound, T=0.050); constrained T ∈ [0.5, 5.0] → T = 0.500.
* Test ECE 0.2053 → 0.0333; NLL 0.2450 → 0.0493.
* AURC: MSP (calibrated) 0.0002, Mahalanobis 0.0070, Energy 0.0002

| Corruption | MSP | Energy | Mahalanobis | kNN | Quality gate rejects | Predicted as |
|---|---|---|---|---|---|---|
| blur | 0.979 | 0.980 | 0.988 | 0.980 | 100.0% | {'Whorled': 16, 'Alternate': 13, 'Opposite': 13} |
| dark | 0.980 | 0.980 | 1.000 | 1.000 | 100.0% | {'Not_a_leaf': 41, 'Whorled': 1} |
| noise | 0.842 | 0.827 | 0.942 | 0.922 | 0.0% | {'Whorled': 16, 'Alternate': 13, 'Opposite': 13} |
| occlusion | 0.988 | 0.979 | 0.993 | 0.986 | 0.0% | {'Not_a_leaf': 13, 'Opposite': 13, 'Alternate': 10, 'Whorled': 6} |

## T6. Full abstention policy on untouched test data

Thresholds (fit on validation, 95% TPR): {"mahalanobis": -6947.8285, "knn": 0.7982, "confidence": 0.9}

| Set | n | Outcome |
|---|---|---|
| Test leaves | 42 | coverage 97.6%, accuracy on accepted 100.0% (without policy 97.6%); {'ACCEPT': 41, 'ABSTAIN_UNFAMILIAR': 1} |
| Test non-leaf | 23 | {'NOT_A_LEAF': 23} |
| Corrupted test leaves | 168 | not accepted 89.9%; {'blur': {'REJECT_QUALITY': 42}, 'dark': {'REJECT_QUALITY': 42}, 'noise': {'ABSTAIN_UNFAMILIAR': 25, 'ACCEPT': 17}, 'occlusion': {'ABSTAIN_UNFAMILIAR': 29, 'NOT_A_LEAF': 13}} |

## T7. Pollution workbooks (455 rows each) — leakage audit

* Fraction of rows with identical targets in high- vs low-traffic workbook: {'PM2.5': 1.0, 'PM10': 1.0, 'NO2': 1.0, 'CO': 1.0, 'O3': 1.0}
* Lag-1 autocorrelation of 'Absorbance': high 0.9995, low 0.9992

Mean R² over both sites:

| CV protocol | Model | PM2.5 | PM10 | NO2 | CO | O3 |
|---|---|---|---|---|---|---|
| blocked 5-fold (contiguous, no shuffle) | Mean baseline | -0.893 | -0.412 | -1.851 | -3.681 | -6.051 |
| blocked 5-fold (contiguous, no shuffle) | RandomForest (4 leaf features) | 0.014 | -0.513 | -3.569 | -5.289 | -19.378 |
| blocked 5-fold (contiguous, no shuffle) | RandomForest (ROW INDEX ONLY) | -2.807 | -0.923 | -1.802 | -9.875 | -1.218 |
| blocked 5-fold (contiguous, no shuffle) | Ridge (4 leaf features) | 0.261 | -0.205 | -2.884 | -2.944 | -12.053 |
| forward-chaining (TimeSeriesSplit 5) | Mean baseline | -3.087 | -0.472 | -0.323 | -4.033 | -0.625 |
| forward-chaining (TimeSeriesSplit 5) | RandomForest (4 leaf features) | -0.085 | -0.549 | -2.037 | -1.689 | -0.564 |
| forward-chaining (TimeSeriesSplit 5) | RandomForest (ROW INDEX ONLY) | -0.262 | -0.159 | -1.687 | -0.744 | -1.548 |
| forward-chaining (TimeSeriesSplit 5) | Ridge (4 leaf features) | -0.409 | -0.382 | -2.506 | -3.129 | -0.801 |
| shuffled 5-fold (random split) | Mean baseline | -0.018 | -0.009 | -0.001 | -0.027 | -0.001 |
| shuffled 5-fold (random split) | RandomForest (4 leaf features) | 0.615 | 0.405 | 0.308 | 0.280 | 0.199 |
| shuffled 5-fold (random split) | RandomForest (ROW INDEX ONLY) | 0.615 | 0.508 | 0.549 | 0.655 | 0.590 |
| shuffled 5-fold (random split) | Ridge (4 leaf features) | 0.511 | 0.223 | 0.046 | 0.078 | 0.019 |

## T8. Image → AQI (46 verified Whorled pairs, repeated 5-fold × 10)

| Features | Model | R² (mean ± SD) | MAE |
|---|---|---|---|
| - | Mean baseline | -0.133 ± 0.155 | 8.91 |
| ft3 ViT embedding (deployed AQI space) | Ridge (inner-CV alpha) | -0.165 ± 0.335 | 9.10 |
| ft3 ViT embedding (deployed AQI space) | kNN (k=5) | -0.145 ± 0.322 | 9.18 |
| frozen ViT-in21k embedding | Ridge (inner-CV alpha) | -0.179 ± 0.321 | 9.10 |
| frozen ViT-in21k embedding | kNN (k=5) | -0.265 ± 0.386 | 9.39 |
| colour statistics (25-D) | Ridge (inner-CV alpha) | -0.290 ± 0.623 | 9.09 |
| colour statistics (25-D) | kNN (k=5) | -0.413 ± 0.472 | 9.81 |

Permutation test (200 permutations), deployed configuration: pooled CV R² = -0.073, p = 0.448. **Decision:** AQI estimate WITHHELD by default: under repeated cross-validation no image model beats a permutation null with useful effect size, and the pairs lack timestamp/station provenance.


## T9. Data sufficiency (E7) — is the data we have enough?

**Learning curve** (near-duplicate-grouped 5-fold × 5; accuracy mean ± SD):

| Model | 22 imgs | 42 imgs | 64 imgs | 106 imgs | 161 imgs | 212 imgs |
|---|---|---|---|---|---|---|
| Colour-statistics probe (shortcut) | 0.860 ± 0.045 | 0.909 ± 0.044 | 0.906 ± 0.039 | 0.931 ± 0.036 | 0.943 ± 0.033 | 0.954 ± 0.034 |
| Frozen ViT-in21k + LogReg | 0.995 ± 0.008 | 0.995 ± 0.008 | 0.995 ± 0.010 | 1.000 ± 0.000 | 0.998 ± 0.005 | 1.000 ± 0.000 |

**Image → AQI sample size:** with 46 pairs the smallest detectable effect is R² ≈ 0.16 (α = 0.05, power 0.8). Independent pairs needed: R² 0.05 → 155, R² 0.1 → 77, R² 0.2 → 37, R² 0.3 → 24, R² 0.5 → 14.

**Station-paired pilot** (Vikas Sadan,Gurugram-HSPCB; 75 hourly rows, 5 days 2022-10-23 … 2022-10-27; 5 '12.00 AM' entries corrected to noon; 90% of PM2.5 variance is between days). Leave-one-day-out R²:

| Features | PM2.5 | NO2 | SO2 | CO | O3 |
|---|---|---|---|---|---|
| mean baseline | -0.505 | -0.483 | -0.541 | -0.149 | -0.213 |
| hour of day | -0.516 | -0.484 | -0.541 | -0.345 | -0.354 |
| leaf RGB | -3.442 | 0.167 | -3.236 | 0.050 | -0.291 |
| leaf brightness + greenness | -1.366 | -2.165 | -1.451 | -0.123 | -0.884 |
| leaf RGB + hour | -4.033 | 0.238 | -4.085 | -1.142 | -0.372 |

Within-day correlation with PM2.5 (circular-shift permutation): brightness r = 0.18, p = 0.291; greenness r = -0.05, p = 0.718.

**Verdicts:**

* **phyllotaxy_classification** — QUANTITY IS SUFFICIENT, DIVERSITY IS NOT: the curve is flat from ~22 training images and the colour-only probe is also high, so more photos of the same kind add nothing; new species/sites/cameras are what is missing (generalisation untested)
* **image_to_AQI** — INSUFFICIENT: 46 pairs can only detect R² ≥ 0.16; observed CV R² is negative. Keep AQI withheld.
* **FTIR_workbooks_to_pollutants** — UNUSABLE for prediction claims: identical targets in both workbooks and row-order leakage (see E5).
* **station_pilot_leaf_RGB_to_PM2.5** — NO evidence that leaf colour predicts PM2.5 on unseen days (5 days is too few; between-day differences dominate). Report as a pilot, not as a result.