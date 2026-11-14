"""E4 — Reliability: OOD detection, unseen-arrangement detection, calibration, selective prediction,
and construction of the deploy-time abstention policy.

A  Non-leaf rejection by models that NEVER saw non-leaf images (ft3 and a frozen-ViT probe).
B  Unseen-arrangement detection (leave-one-arrangement-out, frozen ViT — no leakage), 5 seeds x 3 held-out classes.
C  Deployed 4-class ViT: temperature calibration, corrupted/low-quality inputs, risk-coverage.
D  Build artifacts/reliability.{json,npz} (thresholds fit on VALIDATION data only) and evaluate the full
   abstention policy on the untouched test set.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.special import softmax  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from common import banner, out_dir, save_json  # noqa: E402
from e02_extract_features import CORRUPTIONS  # noqa: E402
from phyllo import quality  # noqa: E402
from phyllo import reliability as R  # noqa: E402
from phyllo.paths import ALL_CLASSES, ARTIFACTS, CLEAN, FEATURES, LEAF_CLASSES, RESULTS, ensure_dirs  # noqa: E402

TPR = 0.95
T_BOUNDS = (0.5, 5.0)


def logit_scores(logits, T=1.0):
    return {"MSP": R.msp(logits, T), "MaxLogit": R.max_logit(logits), "Energy": R.energy(logits, T),
            "NegEntropy": R.neg_entropy(logits, T)}


def all_scores(logits_id, logits_ood, emb_tr, y_tr, emb_id, emb_ood):
    maha = R.Mahalanobis().fit(emb_tr, y_tr)
    knn = R.KNNScore(5).fit(emb_tr)
    sid, sood = logit_scores(logits_id), logit_scores(logits_ood)
    sid["Mahalanobis"], sood["Mahalanobis"] = maha.score(emb_id), maha.score(emb_ood)
    sid["kNN-cosine"], sood["kNN-cosine"] = knn.score(emb_id), knn.score(emb_ood)
    return {m: R.ood_metrics(sid[m], sood[m]) for m in sid}


def main():
    banner("E4  RELIABILITY / OOD / CALIBRATION")
    ensure_dirs()
    out = out_dir("e04_reliability")
    man = pd.read_csv(RESULTS / "e01_audit" / "manifest.csv").fillna("")
    F = {k: np.load(FEATURES / f"{k}.npz", allow_pickle=True) for k in ("pretrained", "ft3", "ft4")}
    C = {k: np.load(FEATURES / f"corrupt_{k}.npz", allow_pickle=True) for k in ("pretrained", "ft4")}
    lab3 = {c: i for i, c in enumerate(LEAF_CLASSES)}
    report = {}

    # ------------------------------------------------------------------ A. non-leaf rejection without non-leaf training
    tr3 = man.index[(man.split_3class == "train")].values
    te3 = man.index[(man.split_3class == "test")].values
    nl = man.index[man["class"] == "Not_a_leaf"].values
    y_tr3 = man.loc[tr3, "class"].map(lab3).values
    A = {}
    e, lg = F["ft3"]["emb"], F["ft3"]["logits"]
    A["fine-tuned 3-class ViT"] = all_scores(lg[te3], lg[nl], e[tr3], y_tr3, e[te3], e[nl])
    pe = F["pretrained"]["emb"]
    sc = StandardScaler().fit(pe[tr3])
    clf = LogisticRegression(C=0.5, max_iter=5000).fit(sc.transform(pe[tr3]), y_tr3)
    lg_p = clf.decision_function(sc.transform(pe))
    A["frozen ViT + LogReg"] = all_scores(lg_p[te3], lg_p[nl], pe[tr3], y_tr3, pe[te3], pe[nl])
    report["A_nonleaf_without_nonleaf_training"] = A

    # ------------------------------------------------------------------ B. unseen arrangement (leave-one-class-out)
    leaf_idx = man.index[man["class"] != "Not_a_leaf"].values
    B_rows = []
    for held in LEAF_CLASSES:
        known = [c for c in LEAF_CLASSES if c != held]
        kidx = man.index[man["class"].isin(known)].values
        oidx = man.index[man["class"] == held].values
        yk = man.loc[kidx, "class"].map({c: i for i, c in enumerate(known)}).values
        for seed in range(5):
            tr, te = train_test_split(np.arange(len(kidx)), test_size=0.3, stratify=yk, random_state=seed)
            Xtr, Xte, Xood = pe[kidx[tr]], pe[kidx[te]], pe[oidx]
            s = StandardScaler().fit(Xtr)
            m = LogisticRegression(C=0.5, max_iter=5000).fit(s.transform(Xtr), yk[tr])
            l_te, l_ood = m.decision_function(s.transform(Xte)), m.decision_function(s.transform(Xood))
            l_te, l_ood = np.c_[-l_te, l_te], np.c_[-l_ood, l_ood]   # binary -> 2 logits
            res = all_scores(l_te, l_ood, Xtr, yk[tr], Xte, Xood)
            acc = float((l_te.argmax(1) == yk[te]).mean())
            for meth, v in res.items():
                B_rows.append({"held_out": held, "seed": seed, "method": meth, "known_acc": acc, **v})
    B = pd.DataFrame(B_rows)
    B.to_csv(out / "unseen_arrangement_runs.csv", index=False)
    Bs = B.groupby("method")[["AUROC", "FPR@95TPR"]].agg(["mean", "std"]).round(4)
    Bh = B.groupby(["held_out", "method"])["AUROC"].mean().unstack().round(3)
    flat = {m: {"AUROC_mean": float(g.AUROC.mean()), "AUROC_std": float(g.AUROC.std()),
                "FPR95_mean": float(g["FPR@95TPR"].mean())} for m, g in B.groupby("method")}
    report["B_unseen_arrangement"] = {"summary": flat, "auroc_by_heldout": json.loads(Bh.to_json()),
                                      "known_class_accuracy_mean": float(B.known_acc.mean())}
    print("\nB. unseen arrangement (frozen ViT) mean AUROC / FPR95:\n", Bs.to_string())
    print(Bh.to_string())

    # ------------------------------------------------------------------ C/D. deployed 4-class model
    lab4 = {c: i for i, c in enumerate(ALL_CLASSES)}
    sp = man.split_4class
    tr4, va4, te4 = (man.index[sp == s].values for s in ("train", "val", "test"))
    y4 = man["class"].map(lab4).values
    e4, l4 = F["ft4"]["emb"], F["ft4"]["logits"]
    is_leaf = (man["class"] != "Not_a_leaf").values

    T_unconstrained = R.fit_temperature(l4[va4], y4[va4])
    val_acc = float((l4[va4].argmax(1) == y4[va4]).mean())
    # Pre-registered constraint: with 100% validation accuracy the NLL optimum is at the search bound,
    # so we restrict T to [0.5, 5] (at most a 2x sharpening of the logits).
    T = R.fit_temperature(l4[va4], y4[va4], bounds=T_BOUNDS)
    ece_raw, rows_raw = R.ece(softmax(l4[te4], 1), y4[te4])
    ece_cal, rows_cal = R.ece(softmax(l4[te4] / T, 1), y4[te4])
    nll_raw, nll_cal = R.nll(l4[te4], y4[te4], 1.0), R.nll(l4[te4], y4[te4], T)

    leaf_tr = tr4[is_leaf[tr4]]
    leaf_va = va4[is_leaf[va4]]
    maha = R.Mahalanobis().fit(e4[leaf_tr], y4[leaf_tr])
    knn = R.KNNScore(5).fit(e4[leaf_tr])
    conf_va = R.msp(l4[leaf_va], T)
    thr = {
        "mahalanobis": R.threshold_at_tpr(maha.score(e4[leaf_va]), TPR),
        "knn": R.threshold_at_tpr(knn.score(e4[leaf_va]), TPR),
        "confidence": min(R.threshold_at_tpr(conf_va, TPR), 0.90),
    }

    # noise level of clean TRAIN+VAL images -> data-derived (non-blocking) noise warning for the API
    noise_clean = [quality.noise_sigma(Image.open(CLEAN / k)) for k in man.key[np.r_[tr4, va4]]]

    # corrupted inputs (and their quality-gate outcome)
    ce, cl, ckind, ckeys = C["ft4"]["emb"], C["ft4"]["logits"], C["ft4"]["kind"], C["ft4"]["keys"]
    rng = np.random.default_rng(42)
    qpass = []
    order = list(CORRUPTIONS)
    base = {}
    for kind in order:
        for k in ckeys[ckind == kind]:
            if k not in base:
                with Image.open(CLEAN / k) as im:
                    base[k] = im.convert("RGB").copy()
            qpass.append(quality.assess(CORRUPTIONS[kind](base[k], rng)).passed)
    qpass = np.array(qpass)

    te_leaf = te4[is_leaf[te4]]
    corr = {}
    for kind in order:
        m = ckind == kind
        sid = {**logit_scores(l4[te_leaf], T), "Mahalanobis": maha.score(e4[te_leaf]), "kNN-cosine": knn.score(e4[te_leaf])}
        sood = {**logit_scores(cl[m], T), "Mahalanobis": maha.score(ce[m]), "kNN-cosine": knn.score(ce[m])}
        corr[kind] = {meth: R.ood_metrics(sid[meth], sood[meth])["AUROC"] for meth in sid}
        corr[kind]["predicted_as"] = pd.Series(np.array(ALL_CLASSES)[cl[m].argmax(1)]).value_counts().to_dict()
        corr[kind]["quality_gate_rejects"] = float((~qpass[m]).mean())

    # ---------------- full abstention policy, evaluated on untouched test data
    def policy(logits, emb, qp=None):
        p = softmax(logits / T, 1)
        top = p.argmax(1)
        conf = p.max(1)
        ms, ks = maha.score(emb), knn.score(emb)
        dec = np.full(len(top), "ACCEPT", dtype=object)
        dec[(ms < thr["mahalanobis"]) | (ks < thr["knn"])] = "ABSTAIN_UNFAMILIAR"
        dec[(dec == "ACCEPT") & (conf < thr["confidence"])] = "ABSTAIN_UNCERTAIN"
        dec[top == lab4["Not_a_leaf"]] = "NOT_A_LEAF"
        if qp is not None:
            dec[~qp] = "REJECT_QUALITY"
        return dec, top

    d_leaf, top_leaf = policy(l4[te_leaf], e4[te_leaf])
    acc_mask = d_leaf == "ACCEPT"
    te_nl = te4[~is_leaf[te4]]
    d_nl, _ = policy(l4[te_nl], e4[te_nl])
    d_c, _ = policy(cl, ce, qpass)
    pol = {
        "test_leaves_n": int(len(te_leaf)),
        "test_leaves_coverage": float(acc_mask.mean()),
        "test_leaves_accuracy_on_accepted": float((top_leaf[acc_mask] == y4[te_leaf][acc_mask]).mean()),
        "test_leaves_accuracy_without_policy": float((top_leaf == y4[te_leaf]).mean()),
        "test_leaf_decisions": pd.Series(d_leaf).value_counts().to_dict(),
        "test_nonleaf_n": int(len(te_nl)),
        "test_nonleaf_decisions": pd.Series(d_nl).value_counts().to_dict(),
        "corrupted_n": int(len(d_c)),
        "corrupted_not_accepted_rate": float((d_c != "ACCEPT").mean()),
        "corrupted_decisions_by_kind": {k: pd.Series(d_c[ckind == k]).value_counts().to_dict() for k in order},
    }

    # risk-coverage on the 4-class test set
    corr_ok = (l4[te4].argmax(1) == y4[te4])
    rc = {}
    plt.figure(figsize=(5.5, 4))
    for name, conf in (("MSP (calibrated)", R.msp(l4[te4], T)), ("Mahalanobis", maha.score(e4[te4])),
                       ("Energy", R.energy(l4[te4], T))):
        cov, risk, aurc = R.risk_coverage(conf, corr_ok)
        rc[name] = aurc
        plt.plot(cov, risk, label=f"{name} (AURC={aurc:.4f})")
    plt.xlabel("Coverage"); plt.ylabel("Selective risk (error rate)"); plt.title("Risk-coverage, 4-class test (n=65)")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(out / "risk_coverage.png", dpi=200); plt.close()

    # OOD summary figure (setting A, fine-tuned 3-class)
    meths = list(A["fine-tuned 3-class ViT"])
    plt.figure(figsize=(7, 3.6))
    xs = np.arange(len(meths))
    plt.bar(xs - 0.2, [A["fine-tuned 3-class ViT"][m]["AUROC"] for m in meths], 0.4, label="Leaf vs non-leaf (ft 3-class)")
    plt.bar(xs + 0.2, [B[B.method == m].AUROC.mean() for m in meths], 0.4, label="Unseen arrangement (frozen ViT)")
    plt.xticks(xs, meths, rotation=20, fontsize=8); plt.ylim(0.4, 1.0); plt.axhline(0.5, ls=":", c="grey")
    plt.ylabel("AUROC"); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(out / "ood_auroc.png", dpi=200); plt.close()

    report["C_deployed_4class"] = {
        "temperature": T, "temperature_bounds": T_BOUNDS, "temperature_unconstrained": T_unconstrained,
        "val_accuracy": val_acc, "ECE_test_raw": ece_raw, "ECE_test_calibrated": ece_cal,
        "NLL_test_raw": nll_raw, "NLL_test_calibrated": nll_cal,
        "reliability_bins_raw": rows_raw, "reliability_bins_calibrated": rows_cal,
        "corruption_auroc": corr, "AURC": rc,
    }
    report["D_policy"] = {"thresholds": thr, "target_tpr_on_val": TPR, "evaluation": pol}
    save_json(report, out / "reliability_summary.json")

    # ---------------- deploy artifacts
    np.savez_compressed(ARTIFACTS / "reliability_ft4.npz", maha_means=maha.means_, maha_precision=maha.precision_,
                        maha_classes=maha.classes_, knn_bank=knn.bank_)
    save_json({
        "model": "models/vit_phyllotaxy_4class", "classes": ALL_CLASSES, "temperature": T,
        "noise_sigma_warn": float(np.percentile(noise_clean, 99)),
        "thresholds": thr, "knn_k": 5, "fit_on": "4-class TRAIN leaf embeddings; thresholds on VAL leaves (95% TPR)",
        "test_evaluation": pol, "ece_test_raw": ece_raw, "ece_test_calibrated": ece_cal,
    }, ARTIFACTS / "reliability_ft4.json")

    print("\nA. leaf vs non-leaf AUROC (ft3):", {k: round(v["AUROC"], 3) for k, v in A["fine-tuned 3-class ViT"].items()})
    print("A. leaf vs non-leaf AUROC (frozen):", {k: round(v["AUROC"], 3) for k, v in A["frozen ViT + LogReg"].items()})
    print(f"\nC. T={T:.3f}  ECE raw {ece_raw:.4f} -> cal {ece_cal:.4f}   NLL {nll_raw:.4f} -> {nll_cal:.4f}")
    for k, v in corr.items():
        print(f"   corruption {k:10s}", {m: (round(x, 3) if isinstance(x, float) else x) for m, x in v.items()})
    print("\nD. thresholds", thr)
    print("D. policy", json.dumps(pol, indent=1))


if __name__ == "__main__":
    main()
