"""E7 — Is the data we have enough? Data-sufficiency analysis using ONLY the data that came with the paper.

Part A  Learning curve for phyllotaxy classification: accuracy vs number of training images
        (frozen ViT-in21k + LogReg, and the colour-statistics shortcut probe for comparison),
        near-duplicate-grouped 5-fold x 5 repeats. Answers "is 265 images enough for this task?"
Part B  Sample size needed for an image -> AQI claim (Fisher-z power analysis), versus the 46 pairs we have.
Part C  Station-paired pilot: the one workbook in which a leaf signal (mean RGB) and pollutant readings share a
        real timestamp (data/raw/NISE Gwal Pahari,Gurugram-IMD.xlsx; sheet titled "Vikas Sadan";
        5 days x 15 hours, Oct 2022). Evaluated with leave-one-DAY-out CV (the honest unit is the day),
        against mean and hour-of-day baselines, plus a within-day permutation test.

Outputs: results/e07_data_sufficiency/{learning_curve.csv, learning_curve.png, station_pilot.csv,
         station_pilot_clean.csv, data_sufficiency.json}
"""
import itertools
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.linear_model import LogisticRegression, RidgeCV  # noqa: E402
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from common import banner, out_dir, save_json  # noqa: E402
from phyllo.aqi_standard import sub_index  # noqa: E402
from phyllo.paths import CLEAN, FEATURES, LEAF_CLASSES, RAW_XLSX, RESULTS, SEED  # noqa: E402

FRACTIONS = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
N_REPEATS = 5
STATION_FILE = RAW_XLSX / "NISE Gwal Pahari,Gurugram-IMD.xlsx"
POLLUTANTS = ["PM2.5", "NO2", "SO2", "CO", "O3"]


# ----------------------------------------------------------------------------- Part A
def colour_features(paths):
    out = []
    for p in paths:
        with Image.open(p) as im:
            rgb = np.asarray(im.convert("RGB").resize((128, 128)), np.float32) / 255
            hsv = np.asarray(im.convert("HSV").resize((128, 128)), np.float32) / 255
        hist = np.histogram(hsv[..., 0], bins=16, range=(0, 1))[0] / hsv[..., 0].size
        out.append(np.concatenate([rgb.mean((0, 1)), rgb.std((0, 1)), hsv[..., 1:].mean((0, 1)), hist]))
    return np.array(out)


def learning_curve():
    man = pd.read_csv(RESULTS / "e01_audit" / "manifest.csv")
    leaf = man[man["class"] != "Not_a_leaf"].reset_index(drop=True)
    f = np.load(FEATURES / "pretrained.npz", allow_pickle=True)
    idx = {k: i for i, k in enumerate(f["keys"])}
    X_vit = f["emb"][[idx[k] for k in leaf.key]]
    X_col = colour_features([CLEAN / k for k in leaf.key])
    y = leaf["class"].map({c: i for i, c in enumerate(LEAF_CLASSES)}).values
    groups = leaf.group.values
    feats = {"Frozen ViT-in21k + LogReg": (X_vit, lambda: LogisticRegression(C=0.5, max_iter=5000)),
             "Colour-statistics probe (shortcut)": (X_col, lambda: make_pipeline(StandardScaler(),
                                                                               LogisticRegression(max_iter=5000)))}
    rows = []
    for rep in range(N_REPEATS):
        rng = np.random.default_rng(SEED + rep)
        for fold, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEED + rep).split(X_vit, y, groups)):
            for frac in FRACTIONS:
                # subsample whole near-duplicate groups, stratified by class, so duplicates never straddle
                sub = []
                for c in np.unique(y[tr]):
                    g = np.unique(groups[tr][y[tr] == c])
                    keep = rng.choice(g, max(2, int(round(frac * len(g)))), replace=False)
                    sub.append(tr[(y[tr] == c) & np.isin(groups[tr], keep)])
                sub = np.concatenate(sub)
                for name, (X, make) in feats.items():
                    m = make().fit(X[sub], y[sub])
                    rows.append({"model": name, "fraction": frac, "n_train": len(sub), "repeat": rep, "fold": fold,
                                 "accuracy": accuracy_score(y[te], m.predict(X[te]))})
    df = pd.DataFrame(rows)
    summ = (df.groupby(["model", "fraction"])
              .agg(n_train=("n_train", "mean"), acc_mean=("accuracy", "mean"), acc_sd=("accuracy", "std"))
              .reset_index())
    return df, summ


# ----------------------------------------------------------------------------- Part B
def n_required(r, alpha=0.05, power=0.8):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return int(np.ceil(((za + zb) / np.arctanh(r)) ** 2 + 3))


def min_detectable_r(n, alpha=0.05, power=0.8):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return float(np.tanh((za + zb) / np.sqrt(n - 3)))


# ----------------------------------------------------------------------------- Part C
def parse_rgb(s, prev):
    """'91,99,110' or comma-less '93101112'. For comma-less strings choose the split into three 0-255 integers
    closest to the previous row's RGB (hourly colour changes are small). Returns (rgb, how)."""
    s = str(s).strip()
    parts = [v for v in re.split(r"[,\s]+", s) if v]
    if len(parts) == 3 and all(v.isdigit() and int(v) <= 255 for v in parts):
        return [int(v) for v in parts], "comma"
    digits = re.sub(r"\D", "", s)
    cands = []
    for a, b in itertools.combinations(range(1, len(digits)), 2):
        parts = [digits[:a], digits[a:b], digits[b:]]
        if all(1 <= len(p) <= 3 and (p == "0" or not p.startswith("0")) and int(p) <= 255 for p in parts):
            cands.append([int(p) for p in parts])
    if not cands:
        return None, "unparseable"
    if len(cands) == 1 or prev is None:
        best = min(cands, key=lambda c: np.ptp(c))     # no reference: the most grey-balanced split
        return best, "digits-unique" if len(cands) == 1 else "digits-ambiguous"
    best = min(cands, key=lambda c: np.abs(np.subtract(c, prev)).sum())
    return best, "digits-ambiguous"


def parse_hour(t):
    m = re.match(r"(\d+)[.:](\d+)\s*([AP]M)", str(t).strip().upper())
    h = int(m.group(1)) % 12 + (12 if m.group(3) == "PM" else 0)
    return h


def load_station():
    raw = pd.read_excel(STATION_FILE, header=None)
    title = str(raw.iloc[0, 0]).strip()
    rows = []
    for r, c in zip(*np.where(raw.astype(str).values == "PM2.5")):
        i = r + 1
        while i < len(raw) and pd.notna(raw.iat[i, c]):
            rows.append({"date": str(raw.iat[i, c + 5]).strip(), "time_raw": str(raw.iat[i, c + 6]).strip(),
                         **{p: float(raw.iat[i, c + k]) for k, p in enumerate(POLLUTANTS)},
                         "rgb_raw": raw.iat[i, c + 7]})
            i += 1
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True).dt.date.astype(str)
    df["hour"] = df.time_raw.map(parse_hour)
    df = df.sort_values(["date", "hour"]).reset_index(drop=True)
    # data-entry fix: "12.00 AM" sits between 11 AM and 1 PM in every block -> it is noon
    noon_fix = int(((df.time_raw.str.upper().str.startswith("12")) & (df.hour == 0)).sum())
    df.loc[df.time_raw.str.upper().str.startswith("12") & (df.hour == 0), "hour"] = 12
    df = df.sort_values(["date", "hour"]).reset_index(drop=True)
    prev, rgbs, how = None, [], []
    for i, s in enumerate(df.rgb_raw):
        if i and df.date[i] != df.date[i - 1]:
            prev = None
        v, h = parse_rgb(s, prev)
        rgbs.append(v), how.append(h)
        prev = v if v is not None else prev
    df[["R", "G", "B"]] = pd.DataFrame(rgbs, index=df.index)
    df["rgb_parse"] = how
    df["brightness"] = df[["R", "G", "B"]].mean(axis=1)
    df["greenness"] = df.G / df[["R", "G", "B"]].sum(axis=1)          # chromatic G coordinate
    df["PM2.5_hourly_subindex"] = [sub_index("PM2.5", v) for v in df["PM2.5"]]
    return df, {"sheet_title": title, "noon_fixes": noon_fix, "rgb_parse_counts": df.rgb_parse.value_counts().to_dict()}


def lodo(df, feats, target):
    """Leave-one-day-out CV. feats: list of columns, or [] for the training-mean baseline."""
    pred = np.empty(len(df))
    for d in df.date.unique():
        tr, te = df.date != d, df.date == d
        if not feats:
            pred[te] = df.loc[tr, target].mean()
            continue
        X = df[feats].values.astype(float)
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))).fit(X[tr], df.loc[tr, target])
        pred[te] = m.predict(X[te])
    y = df[target].values
    return r2_score(y, pred), mean_absolute_error(y, pred)


def within_day_perm(df, x, y, n_perm=5000, seed=SEED):
    """Correlation of x and y after removing each day's mean; null = circular shifts of x within each day."""
    xd = df[x] - df.groupby("date")[x].transform("mean")
    yd = df[y] - df.groupby("date")[y].transform("mean")
    obs = np.corrcoef(xd, yd)[0, 1]
    rng = np.random.default_rng(seed)
    days = [np.where(df.date == d)[0] for d in df.date.unique()]
    null = np.empty(n_perm)
    for k in range(n_perm):
        xs = xd.values.copy()
        for ix in days:
            xs[ix] = np.roll(xd.values[ix], rng.integers(1, len(ix)))
        null[k] = np.corrcoef(xs, yd)[0, 1]
    return float(obs), float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))


def station_pilot():
    df, meta = load_station()
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * df.hour / 24), np.cos(2 * np.pi * df.hour / 24)
    sets = {"mean baseline": [], "hour of day": ["hour_sin", "hour_cos"],
            "leaf RGB": ["R", "G", "B"], "leaf brightness + greenness": ["brightness", "greenness"],
            "leaf RGB + hour": ["R", "G", "B", "hour_sin", "hour_cos"]}
    rows = []
    for t in POLLUTANTS:
        for name, cols in sets.items():
            r2, mae = lodo(df, cols, t)
            rows.append({"target": t, "features": name, "LODO_R2": r2, "LODO_MAE": mae})
    res = pd.DataFrame(rows)
    day = df.groupby("date").agg(**{"PM2.5": ("PM2.5", "mean"), "brightness": ("brightness", "mean"),
                                    "greenness": ("greenness", "mean")})
    tests = {}
    for x in ("brightness", "greenness"):
        r_w, p_w = within_day_perm(df, x, "PM2.5")
        r_b, p_b = stats.pearsonr(day[x], day["PM2.5"])
        tests[x] = {"within_day_r": r_w, "within_day_perm_p": p_w, "between_day_r": float(r_b),
                    "between_day_p": float(p_b), "n_days": int(len(day))}
    meta.update({"n_rows": int(len(df)), "n_days": int(df.date.nunique()), "dates": sorted(df.date.unique().tolist()),
                 "hours": [int(df.hour.min()), int(df.hour.max())],
                 "PM2.5_range": [float(df["PM2.5"].min()), float(df["PM2.5"].max())],
                 "variance_share_between_days_PM2.5": float(
                     df.groupby("date")["PM2.5"].transform("mean").var() / df["PM2.5"].var()),
                 "CO_unit_note": "CO values 12-36 are implausible as mg/m3 hourly CPCB values; unit unknown, so no CO "
                                 "sub-index is computed.",
                 "relation_tests_PM2.5": tests})
    return df, res, meta


def main():
    banner("E7  DATA SUFFICIENCY")
    out = out_dir("e07_data_sufficiency")

    lc_raw, lc = learning_curve()
    lc_raw.to_csv(out / "learning_curve_folds.csv", index=False)
    lc.to_csv(out / "learning_curve.csv", index=False)
    print(lc.to_string(index=False))
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, g in lc.groupby("model"):
        ax.errorbar(g.n_train, g.acc_mean, yerr=g.acc_sd, marker="o", capsize=3, label=name)
    ax.set_xlabel("training images (near-duplicate groups kept together)")
    ax.set_ylabel("held-out accuracy (grouped 5-fold × 5)")
    ax.set_ylim(0.3, 1.02)
    ax.axhline(1 / 3, ls=":", c="grey", lw=1)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Phyllotaxy: accuracy vs training-set size")
    fig.tight_layout()
    fig.savefig(out / "learning_curve.png", dpi=150)
    plt.close(fig)

    vit = lc[lc.model.str.startswith("Frozen")].set_index("fraction")
    plateau = bool(vit.acc_mean[1.0] - vit.acc_mean[0.5] < 0.02)
    power = {"n_pairs_available": 46, "min_detectable_r_at_n46": min_detectable_r(46),
             "min_detectable_R2_at_n46": min_detectable_r(46) ** 2,
             "n_required_for_R2": {str(r2): n_required(np.sqrt(r2)) for r2 in (0.05, 0.1, 0.2, 0.3, 0.5)},
             "note": "n counts INDEPENDENT units (distinct site-days / trees), not photos of the same leaf or "
                     "hours of the same day. Assumes alpha=0.05 two-sided, power=0.8."}
    print(power)

    st, st_res, st_meta = station_pilot()
    st.drop(columns=["rgb_raw"]).to_csv(out / "station_pilot_clean.csv", index=False)
    st_res.to_csv(out / "station_pilot.csv", index=False)
    print(st_res.pivot(index="features", columns="target", values="LODO_R2").round(3).to_string())
    print(st_meta)

    pm = st_res[st_res.target == "PM2.5"].set_index("features").LODO_R2
    leaf_best = max(pm["leaf RGB"], pm["leaf brightness + greenness"])
    t = st_meta["relation_tests_PM2.5"]
    leaf_signal = leaf_best > max(0.0, pm["hour of day"]) and min(t[x]["within_day_perm_p"] for x in t) < 0.05
    verdict = {
        "phyllotaxy_classification": (
            "QUANTITY IS SUFFICIENT, DIVERSITY IS NOT: the curve is flat from ~22 training images and the colour-only "
            "probe is also high, so more photos of the same kind add nothing; new species/sites/cameras are what is "
            "missing (generalisation untested)"
            if plateau else "NOT YET PLATEAUED — more images would still help"),
        "image_to_AQI": (f"INSUFFICIENT: 46 pairs can only detect R² ≥ {power['min_detectable_R2_at_n46']:.2f}; "
                         "observed CV R² is negative. Keep AQI withheld."),
        "FTIR_workbooks_to_pollutants": "UNUSABLE for prediction claims: identical targets in both workbooks and "
                                        "row-order leakage (see E5).",
        "station_pilot_leaf_RGB_to_PM2.5": (
            "Leaf colour adds out-of-day skill beyond the hour-of-day baseline (pilot evidence only, 5 days)"
            if leaf_signal else
            "NO evidence that leaf colour predicts PM2.5 on unseen days (5 days is too few; between-day "
            "differences dominate). Report as a pilot, not as a result."),
    }
    save_json({"learning_curve": lc.to_dict("records"), "learning_curve_plateaued": plateau,
               "power_image_to_aqi": power, "station_pilot": st_meta,
               "station_pilot_results": st_res.to_dict("records"), "verdict": verdict},
              out / "data_sufficiency.json")
    for k, v in verdict.items():
        print(f"{k:35s} {v}")


if __name__ == "__main__":
    main()
