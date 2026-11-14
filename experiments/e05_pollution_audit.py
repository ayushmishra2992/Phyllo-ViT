"""E5 — Evidence audit of the pollution data and the image->AQI model.

Part A  Workbooks (AQI data.xlsx = high-traffic, AQI data LT.xlsx = low-traffic, 455 rows each):
        - are the pollutant targets of the two sites actually different?
        - temporal autocorrelation of the "FTIR absorbance" and image-statistic columns
        - how much of the apparent predictability is explained by row order alone?
          Compare shuffled K-fold (what random splitting does) vs blocked and forward-chaining CV,
          and a model that sees ONLY the row index.
        Assumption (stated in the report): rows are in chronological order, as in the paper's tables.
Part B  46 verified Whorled image-AQI pairs: repeated CV, baselines, permutation test.
        The verdict decides whether the API is allowed to show an AQI number (artifacts/aqi_evidence.json).
"""
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, RepeatedKFold, TimeSeriesSplit, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import banner, out_dir, save_json
from e03_classification import color_features
from phyllo import reliability as R
from phyllo.paths import AQI_PAIRS, ARTIFACTS, CLEAN, FEATURES, RAW_XLSX, RESULTS, ensure_dirs

TARGETS = ["PM2.5", "PM10", "NO2", "CO", "O3"]
FEATS = ["Contrast", "Entropy", "Standard Deviation", "Absorbance"]
N_PERM = 200


def cv_r2(model_fn, X, y, splitter):
    r2s, maes = [], []
    for tr, te in splitter.split(X):
        m = model_fn().fit(X[tr], y[tr])
        p = m.predict(X[te])
        r2s.append(r2_score(y[te], p))
        maes.append(mean_absolute_error(y[te], p))
    return float(np.mean(r2s)), float(np.std(r2s, ddof=1)), float(np.mean(maes))


def part_a():
    ht = pd.read_excel(RAW_XLSX / "AQI data.xlsx")
    lt = pd.read_excel(RAW_XLSX / "AQI data LT.xlsx").rename(columns={"CH": "PM10"})
    res = {"rows": int(len(ht)),
           "lt_column_named_CH_equals_ht_PM10": True,
           "identical_targets_HT_vs_LT": {t: float((ht[t].values == lt[t].values).mean()) for t in TARGETS},
           "lag1_autocorrelation": {}}
    for name, d in (("high_traffic", ht), ("low_traffic", lt)):
        res["lag1_autocorrelation"][name] = {c: float(np.corrcoef(d[c][:-1], d[c][1:])[0, 1]) for c in FEATS + ["PM2.5"]}

    splitters = {"shuffled 5-fold (random split)": KFold(5, shuffle=True, random_state=0),
                 "blocked 5-fold (contiguous, no shuffle)": KFold(5, shuffle=False),
                 "forward-chaining (TimeSeriesSplit 5)": TimeSeriesSplit(5)}
    rf = lambda: RandomForestRegressor(300, min_samples_leaf=3, random_state=0, n_jobs=-1)
    ridge = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    rows = []
    for site, d in (("high_traffic", ht), ("low_traffic", lt)):
        X = d[FEATS].values.astype(float)
        idx = np.arange(len(d)).reshape(-1, 1).astype(float)
        for t in TARGETS:
            y = d[t].values.astype(float)
            for sname, sp in splitters.items():
                for mname, fn, XX in (("Mean baseline", DummyRegressor, X), ("Ridge (4 leaf features)", ridge, X),
                                      ("RandomForest (4 leaf features)", rf, X), ("RandomForest (ROW INDEX ONLY)", rf, idx)):
                    m, s, mae = cv_r2(fn, XX, y, sp)
                    rows.append({"site": site, "target": t, "cv": sname, "model": mname, "R2_mean": m, "R2_std": s, "MAE": mae})
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["cv", "model"], columns="target", values="R2_mean", aggfunc="mean").round(3)
    return res, df, piv


def part_b():
    pairs = pd.read_csv(AQI_PAIRS)
    keys = ["Whorled/" + f for f in pairs.filename]
    man = pd.read_csv(RESULTS / "e01_audit" / "manifest.csv")
    pos = {k: i for i, k in enumerate(man.key)}
    ok = [k in pos for k in keys]
    pairs, keys = pairs[ok].reset_index(drop=True), [k for k, o in zip(keys, ok) if o]
    y = pairs.AQI.values.astype(float)
    ii = np.array([pos[k] for k in keys])
    X = {"ft3 ViT embedding (deployed AQI space)": np.load(FEATURES / "ft3.npz")["emb"][ii],
         "frozen ViT-in21k embedding": np.load(FEATURES / "pretrained.npz")["emb"][ii],
         "colour statistics (25-D)": color_features([CLEAN / k for k in keys])}
    ridge = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))
    knn = lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(5))
    rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=0)
    rows = []
    for fname, XX in X.items():
        for mname, fn in (("Mean baseline", DummyRegressor), ("Ridge (inner-CV alpha)", ridge), ("kNN (k=5)", knn)):
            if mname == "Mean baseline" and fname != list(X)[0]:
                continue
            m, s, mae = cv_r2(fn, XX, y, rkf)
            rows.append({"features": fname if mname != "Mean baseline" else "-", "model": mname,
                         "R2_mean": m, "R2_std": s, "MAE": mae})
    df = pd.DataFrame(rows)

    # permutation test for the deployed configuration (ft3 + Ridge)
    Xd = X["ft3 ViT embedding (deployed AQI space)"]
    kf = KFold(5, shuffle=True, random_state=0)
    obs = r2_score(y, cross_val_predict(ridge(), Xd, y, cv=kf))
    rng = np.random.default_rng(0)
    null = [r2_score(yp, cross_val_predict(ridge(), Xd, yp, cv=kf)) for yp in (rng.permutation(y) for _ in range(N_PERM))]
    p = (1 + np.sum(np.array(null) >= obs)) / (N_PERM + 1)

    # applicability domain for the API: kNN similarity of a new embedding to the 46 training embeddings
    knn_s = R.KNNScore(3).fit(Xd)
    loo = knn_s.score(Xd, exclude_self=True)
    best = df[df.model != "Mean baseline"].sort_values("R2_mean", ascending=False).iloc[0]
    supported = bool(p < 0.05 and best.R2_mean > 0.1)
    verdict = {
        "n_pairs": int(len(y)), "classes_covered": ["Whorled"], "target": "AQI (undocumented station, date and averaging period)",
        "aqi_range": [float(y.min()), float(y.max())], "aqi_std": float(y.std()),
        "pooled_cv_R2_ft3_ridge": float(obs), "permutation_p_value": float(p), "n_permutations": N_PERM,
        "best_model": best.to_dict(), "supported": supported,
        "decision": ("AQI estimates may be shown as exploratory." if supported else
                     "AQI estimate WITHHELD by default: under repeated cross-validation no image model beats a "
                     "permutation null with useful effect size, and the pairs lack timestamp/station provenance."),
        "applicability_knn_k": 3, "applicability_threshold": float(np.quantile(loo, 0.05)),
    }
    ensure_dirs()
    np.savez_compressed(ARTIFACTS / "aqi_domain_ft3.npz", bank=knn_s.bank_)
    save_json(verdict, ARTIFACTS / "aqi_evidence.json")
    return df, verdict, null


def main():
    banner("E5  POLLUTION DATA & AQI EVIDENCE AUDIT")
    out = out_dir("e05_pollution_audit")
    a, adf, piv = part_a()
    adf.to_csv(out / "workbook_cv_results.csv", index=False)
    piv.to_csv(out / "workbook_cv_r2_pivot.csv")
    print("identical targets HT vs LT:", a["identical_targets_HT_vs_LT"])
    print("lag-1 autocorrelation:", {k: {c: round(v, 3) for c, v in d.items()} for k, d in a["lag1_autocorrelation"].items()})
    print("\nMean R² across both sites:\n", piv.to_string())
    bdf, verdict, null = part_b()
    bdf.to_csv(out / "image_aqi_cv_results.csv", index=False)
    print("\nImage -> AQI (46 Whorled pairs), repeated 5-fold x10:\n", bdf.round(3).to_string(index=False))
    print("\nverdict:", {k: v for k, v in verdict.items() if k != "best_model"})
    save_json({"workbooks": a, "workbook_r2_pivot": piv.reset_index().to_dict("records"),
               "image_aqi": bdf.to_dict("records"), "verdict": verdict,
               "permutation_null_quantiles": np.quantile(null, [0.5, 0.95, 0.99]).tolist()},
              out / "pollution_summary.json")


if __name__ == "__main__":
    main()
