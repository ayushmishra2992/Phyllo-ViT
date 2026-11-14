"""E6 — Collect every experiment output into results/RESULTS.md (paper-ready tables, all numbers generated)."""
import json
from datetime import datetime

import pandas as pd

from common import banner
from phyllo.paths import RESULTS


def load(p):
    return json.loads((RESULTS / p).read_text(encoding="utf-8"))


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    banner("E6  RESULTS TABLES")
    A = load("e01_audit/audit_summary.json")
    C = load("e03_classification/classification_summary.json")
    R = load("e04_reliability/reliability_summary.json")
    P = load("e05_pollution_audit/pollution_summary.json")
    L = []
    w = L.append
    w(f"# Phyllo-ViT v3 — Results (auto-generated {datetime.now():%Y-%m-%d %H:%M})\n")
    w("Every number below is produced by `experiments/` scripts; regenerate with `RUN_EXPERIMENTS.bat`.\n")

    w("## T1. Dataset audit\n")
    w("| Item | Value |\n|---|---|")
    for k in ("n_images", "near_duplicate_pairs", "near_dup_pairs_linking_test_to_trainval_3class",
              "near_dup_pairs_linking_test_to_trainval_4class", "n_groups_after_merging_near_dups",
              "fourclass_test_images_in_threeclass_train"):
        w(f"| {k} | {A[k]} |")
    w("\nLeaf images by resolution (camera/orientation proxy):\n")
    w("| Resolution | Alternate | Opposite | Whorled |\n|---|---|---|---|")
    for r, d in A["top_leaf_resolutions"].items():
        w(f"| {r} | {d.get('Alternate', 0)} | {d.get('Opposite', 0)} | {d.get('Whorled', 0)} |")

    w("\n## T2. Classification — repeated 5-fold CV on 265 unique leaves (mean ± SD over 25 folds)\n")
    w("| Model | Protocol | Accuracy | Macro-F1 |\n|---|---|---|---|")
    for r in C["cv"]:
        w(f"| {r['model']} | {r['protocol']} | {r['accuracy_mean']:.3f} ± {r['accuracy_std']:.3f} | {r['macro_f1_mean']:.3f} |")
    w("\n## T3. Fine-tuned ViTs on their original held-out test sets\n")
    w("| Model | Correct/n | Accuracy | Wilson 95% CI | Test imgs with near-dup in train/val | Accuracy without them |\n|---|---|---|---|---|---|")
    for k, v in C["finetuned_heldout"].items():
        name = "ViT 3-class" if k == "ft3" else "ViT 4-class (deployed)"
        w(f"| {name} | {v['correct']}/{v['test_n']} | {pct(v['accuracy'])} | {pct(v['wilson95'][0])}–{pct(v['wilson95'][1])} | "
          f"{v['test_images_with_near_dup_in_trainval']} | {pct(v['dedup_accuracy'])} (n={v['dedup_test_n']}) |")

    w("\n## T4. OOD detection (AUROC; higher = better; in-distribution = positive)\n")
    a = R["A_nonleaf_without_nonleaf_training"]
    b = R["B_unseen_arrangement"]["summary"]
    meths = list(next(iter(a.values())))
    w("| Score | Leaf vs non-leaf (ft 3-class) | Leaf vs non-leaf (frozen probe) | Unseen arrangement (frozen, LOCO, mean ± SD) | FPR@95TPR unseen |\n|---|---|---|---|---|")
    for m in meths:
        w(f"| {m} | {a['fine-tuned 3-class ViT'][m]['AUROC']:.3f} | {a['frozen ViT + LogReg'][m]['AUROC']:.3f} | "
          f"{b[m]['AUROC_mean']:.3f} ± {b[m]['AUROC_std']:.3f} | {b[m]['FPR95_mean']:.3f} |")
    w("\nUnseen-arrangement AUROC by held-out class:\n")
    by = pd.DataFrame(R["B_unseen_arrangement"]["auroc_by_heldout"])
    w("| Held-out class | " + " | ".join(by.columns) + " |")
    w("|---" * (len(by.columns) + 1) + "|")
    for idx, row in by.iterrows():
        w(f"| {idx} | " + " | ".join(f"{x:.3f}" for x in row) + " |")

    c = R["C_deployed_4class"]
    w("\n## T5. Deployed 4-class model — calibration, corruptions, selective prediction\n")
    w(f"* Validation accuracy {pct(c['val_accuracy'])} → unconstrained temperature is unidentifiable "
      f"(optimum at search bound, T={c['temperature_unconstrained']:.3f}); constrained T ∈ {c['temperature_bounds']} → T = {c['temperature']:.3f}.")
    w(f"* Test ECE {c['ECE_test_raw']:.4f} → {c['ECE_test_calibrated']:.4f}; NLL {c['NLL_test_raw']:.4f} → {c['NLL_test_calibrated']:.4f}.")
    w(f"* AURC: " + ", ".join(f"{k} {v:.4f}" for k, v in c["AURC"].items()))
    w("\n| Corruption | MSP | Energy | Mahalanobis | kNN | Quality gate rejects | Predicted as |\n|---|---|---|---|---|---|---|")
    for k, v in c["corruption_auroc"].items():
        w(f"| {k} | {v['MSP']:.3f} | {v['Energy']:.3f} | {v['Mahalanobis']:.3f} | {v['kNN-cosine']:.3f} | {pct(v['quality_gate_rejects'])} | {v['predicted_as']} |")
    d = R["D_policy"]
    e = d["evaluation"]
    w("\n## T6. Full abstention policy on untouched test data\n")
    w(f"Thresholds (fit on validation, 95% TPR): {json.dumps({k: round(v, 4) for k, v in d['thresholds'].items()})}\n")
    w("| Set | n | Outcome |\n|---|---|---|")
    w(f"| Test leaves | {e['test_leaves_n']} | coverage {pct(e['test_leaves_coverage'])}, accuracy on accepted {pct(e['test_leaves_accuracy_on_accepted'])} "
      f"(without policy {pct(e['test_leaves_accuracy_without_policy'])}); {e['test_leaf_decisions']} |")
    w(f"| Test non-leaf | {e['test_nonleaf_n']} | {e['test_nonleaf_decisions']} |")
    w(f"| Corrupted test leaves | {e['corrupted_n']} | not accepted {pct(e['corrupted_not_accepted_rate'])}; {e['corrupted_decisions_by_kind']} |")

    wb = P["workbooks"]
    w("\n## T7. Pollution workbooks (455 rows each) — leakage audit\n")
    w(f"* Fraction of rows with identical targets in high- vs low-traffic workbook: {wb['identical_targets_HT_vs_LT']}")
    w(f"* Lag-1 autocorrelation of 'Absorbance': high {wb['lag1_autocorrelation']['high_traffic']['Absorbance']:.4f}, "
      f"low {wb['lag1_autocorrelation']['low_traffic']['Absorbance']:.4f}\n")
    w("Mean R² over both sites:\n")
    w("| CV protocol | Model | PM2.5 | PM10 | NO2 | CO | O3 |\n|---|---|---|---|---|---|---|")
    for r in P["workbook_r2_pivot"]:
        w(f"| {r['cv']} | {r['model']} | " + " | ".join(f"{r[t]:.3f}" for t in ("PM2.5", "PM10", "NO2", "CO", "O3")) + " |")
    w("\n## T8. Image → AQI (46 verified Whorled pairs, repeated 5-fold × 10)\n")
    w("| Features | Model | R² (mean ± SD) | MAE |\n|---|---|---|---|")
    for r in P["image_aqi"]:
        w(f"| {r['features']} | {r['model']} | {r['R2_mean']:.3f} ± {r['R2_std']:.3f} | {r['MAE']:.2f} |")
    v = P["verdict"]
    w(f"\nPermutation test ({v['n_permutations']} permutations), deployed configuration: pooled CV R² = "
      f"{v['pooled_cv_R2_ft3_ridge']:.3f}, p = {v['permutation_p_value']:.3f}. **Decision:** {v['decision']}\n")
    ds_path = RESULTS / "e07_data_sufficiency" / "data_sufficiency.json"
    if ds_path.exists():
        D = load("e07_data_sufficiency/data_sufficiency.json")
        w("\n## T9. Data sufficiency (E7) — is the data we have enough?\n")
        w("**Learning curve** (near-duplicate-grouped 5-fold × 5; accuracy mean ± SD):\n")
        w("| Model | " + " | ".join(f"{r['n_train']:.0f} imgs" for r in D["learning_curve"] if r["model"].startswith("Frozen"))
          + " |\n|---|" + "---|" * sum(r["model"].startswith("Frozen") for r in D["learning_curve"]))
        for name in sorted({r["model"] for r in D["learning_curve"]}):
            w(f"| {name} | " + " | ".join(f"{r['acc_mean']:.3f} ± {r['acc_sd']:.3f}" for r in D["learning_curve"]
                                          if r["model"] == name) + " |")
        pw = D["power_image_to_aqi"]
        w(f"\n**Image → AQI sample size:** with {pw['n_pairs_available']} pairs the smallest detectable effect is "
          f"R² ≈ {pw['min_detectable_R2_at_n46']:.2f} (α = 0.05, power 0.8). Independent pairs needed: "
          + ", ".join(f"R² {k} → {v}" for k, v in pw["n_required_for_R2"].items()) + ".\n")
        s = D["station_pilot"]
        w(f"**Station-paired pilot** ({s['sheet_title']}; {s['n_rows']} hourly rows, {s['n_days']} days "
          f"{s['dates'][0]} … {s['dates'][-1]}; {s['noon_fixes']} '12.00 AM' entries corrected to noon; "
          f"{s['variance_share_between_days_PM2.5']:.0%} of PM2.5 variance is between days). "
          "Leave-one-day-out R²:\n")
        tg = ["PM2.5", "NO2", "SO2", "CO", "O3"]
        w("| Features | " + " | ".join(tg) + " |\n|---|" + "---|" * len(tg))
        res = D["station_pilot_results"]
        for f in dict.fromkeys(r["features"] for r in res):
            w(f"| {f} | " + " | ".join(f"{next(r['LODO_R2'] for r in res if r['features'] == f and r['target'] == t):.3f}"
                                      for t in tg) + " |")
        t = s["relation_tests_PM2.5"]
        w("\nWithin-day correlation with PM2.5 (circular-shift permutation): "
          + "; ".join(f"{k} r = {v['within_day_r']:.2f}, p = {v['within_day_perm_p']:.3f}" for k, v in t.items()) + ".\n")
        w("**Verdicts:**\n")
        for k, v in D["verdict"].items():
            w(f"* **{k}** — {v}")
    (RESULTS / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote", RESULTS / "RESULTS.md")


if __name__ == "__main__":
    main()
