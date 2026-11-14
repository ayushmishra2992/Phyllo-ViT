# Paper 1 plan — "Trust Before Inference: Reliability-Aware Phyllotaxy Recognition and an Evidence Audit of Leaf-Based Air-Quality Estimation"

Write only what `results/RESULTS.md` demonstrates. Tables referenced as T1–T8.

| Section | Content | Source | Owner |
|---|---|---|---|
| Abstract | problem, method, 3 key numbers (selective accuracy, shortcut probe, row-index control), implication | T2, T6, T7 | both |
| 1 Introduction | street-level monitoring motivation; bio-indicators; why reliability & leakage matter; RQ1–RQ4; contributions | REPORT §10 | Riya |
| 2 Related work | leaf bio-monitoring & deposited PM; leaf/plant CV; phyllotaxy; OOD/selective classification/calibration; shortcut learning & leakage | literature matrix | Riya |
| 3 Data | image collection, dedup, splits, Not_a_leaf, workbooks, AQI pairs, provenance limits | T1 | Ayush |
| 4 Methods | ViT fine-tuning; frozen probes; quality gate; scores (MSP, Energy, Mahalanobis, kNN — with equations); temperature; abstention policy; leakage audit (row-index control, CV protocols, permutation test) | code | Ayush |
| 5 Results | classification + shortcut probes; OOD; calibration; corruptions; policy; pollution audit; image→AQI | T2–T8, figures | Ayush |
| 6 Discussion | what accuracies mean under confounding; when to abstain; implications for leaf→AQI literature (identical targets, random splits); errors (the abstained image, noise failures) | — | both |
| 7 Limitations | n, single collection, no species labels, constrained T, COCO negatives are far-OOD | REPORT §19 | both |
| 8 Conclusion & future work | Phase-2 protocol | DATA_COLLECTION_PROTOCOL | both |
| Availability | code + scripts; data subject to permissions | — | both |

Key equations: softmax with temperature p = softmax(z/T); energy E(x) = T·log Σ exp(z_k/T); Mahalanobis M(x) = min_k (f−μ_k)ᵀ Σ⁻¹ (f−μ_k) with shrinkage Σ = (1−λ)S + λ(tr S/d)I; kNN score = k-th largest cosine similarity; ECE = Σ_b |B_b|/n · |acc_b − conf_b|; AURC = mean selective risk over coverage; CPCB sub-index I = (I_hi−I_lo)/(B_hi−B_lo)·(C−B_lo)+I_lo.

Figures: pipeline diagram (README), `results/e04_reliability/ood_auroc.png`, `risk_coverage.png`, an attention-rollout panel (accepted vs abstained image), and a bar chart of R² by CV protocol (T7).

Before submission: complete the provenance annotation and re-run with species-grouped CV; get a botanist to check labels; check every citation against the source.
