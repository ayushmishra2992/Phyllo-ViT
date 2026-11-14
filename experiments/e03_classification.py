"""E3 — Classification benchmark with repeated, leakage-aware evaluation.

Part A  Repeated cross-validation on the 265 unique leaf images (3 classes), comparing
        shortcut probes (metadata, colour statistics), HOG-SVM, and frozen-ViT probes.
        Two protocols: plain stratified 5-fold x5 repeats, and near-duplicate-GROUPED 5-fold x5.
Part B  Fine-tuned ViTs on their original held-out test sets, with Wilson 95% CIs, and
        re-scored after removing test images that have a near-duplicate in train/val.
"""
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVC

from common import banner, mean_std, out_dir, save_json, wilson_ci
from phyllo.paths import ALL_CLASSES, CLEAN, FEATURES, LEAF_CLASSES, RESULTS

N_REPEATS = 5


def hog_features(paths):
    feats = []
    for p in paths:
        with Image.open(p) as im:
            g = np.asarray(im.convert("L").resize((128, 128)), dtype=np.float32) / 255.0
        feats.append(hog(g, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)))
    return np.array(feats)


def color_features(paths):
    feats = []
    for p in paths:
        with Image.open(p) as im:
            rgb = np.asarray(im.convert("RGB").resize((128, 128)), np.float32) / 255
            hsv = np.asarray(im.convert("HSV").resize((128, 128)), np.float32) / 255
        hist = np.histogram(hsv[..., 0], bins=16, range=(0, 1))[0] / hsv[..., 0].size
        feats.append(np.concatenate([rgb.mean((0, 1)), rgb.std((0, 1)), hsv[..., 1:].mean((0, 1)), hist]))
    return np.array(feats)


def metadata_features(df):
    return np.c_[(df.width > df.height).astype(float), df.width / df.height, df.width, df.height, df.file_size / 1e5]


def models():
    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    return {
        "metadata_probe (orientation/size only)": ("meta", lr),
        "colour_stats_probe": ("color", lr),
        "HOG + SVM (RBF, C=10)": ("hog", lambda: make_pipeline(StandardScaler(), SVC(C=10, gamma="scale"))),
        "Frozen ViT-in21k + LogReg": ("vit", lambda: LogisticRegression(C=0.5, max_iter=5000)),
        "Frozen ViT-in21k + kNN(k=5, cosine)": ("vitn", lambda: KNeighborsClassifier(5)),
    }


def run_cv(X, y, groups, make, grouped, seed):
    if grouped:
        splitter = StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups)
    else:
        splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed).split(X, y)
    accs, f1s = [], []
    for tr, te in splitter:
        m = make().fit(X[tr], y[tr])
        p = m.predict(X[te])
        accs.append(accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p, average="macro"))
    return accs, f1s


def main():
    banner("E3  CLASSIFICATION BENCHMARK")
    out = out_dir("e03_classification")
    man = pd.read_csv(RESULTS / "e01_audit" / "manifest.csv")
    leaf = man[man["class"] != "Not_a_leaf"].reset_index()
    y = leaf["class"].map({c: i for i, c in enumerate(LEAF_CLASSES)}).values
    paths = [CLEAN / k for k in leaf.key]

    pre = np.load(FEATURES / "pretrained.npz", allow_pickle=True)
    vit = pre["emb"][leaf["index"].values]
    feats = {"meta": metadata_features(leaf), "color": color_features(paths), "hog": hog_features(paths),
             "vit": StandardScaler().fit_transform(vit), "vitn": normalize(vit)}

    rows = []
    for name, (fkey, make) in models().items():
        for grouped in (False, True):
            accs, f1s = [], []
            for s in range(N_REPEATS):
                a, f = run_cv(feats[fkey], y, leaf.group.values, make, grouped, seed=s)
                accs += a
                f1s += f
            rows.append({"model": name, "protocol": "grouped 5-fold x5" if grouped else "stratified 5-fold x5",
                         "accuracy_mean": np.mean(accs), "accuracy_std": np.std(accs, ddof=1),
                         "macro_f1_mean": np.mean(f1s), "macro_f1_std": np.std(f1s, ddof=1), "n_folds": len(accs)})
            print(f"{name:42s} {rows[-1]['protocol']:22s} acc {np.mean(accs):.3f}±{np.std(accs, ddof=1):.3f} "
                  f"F1 {np.mean(f1s):.3f}", flush=True)
    cv = pd.DataFrame(rows)
    cv.to_csv(out / "cv_benchmark.csv", index=False)

    # ---------------- Part B: fine-tuned models on their own held-out test sets
    pairs = pd.read_csv(RESULTS / "e01_audit" / "near_duplicate_pairs.csv")
    finetuned = {}
    for enc, split_col, classes in (("ft3", "split_3class", LEAF_CLASSES), ("ft4", "split_4class", ALL_CLASSES)):
        z = np.load(FEATURES / f"{enc}.npz", allow_pickle=True)
        test = man[man[split_col] == "test"]
        lab = test["class"].map({c: i for i, c in enumerate(classes)}).values
        pred = z["logits"][test.index].argmax(1)
        # test images with a near-duplicate in train/val of the same split scheme
        sa, sb = pairs[f"split{split_col[6]}_a"], pairs[f"split{split_col[6]}_b"]
        leaky = set(pairs.a[(sa == "test") & sb.isin(["train", "val"])]) | set(pairs.b[(sb == "test") & sa.isin(["train", "val"])])
        clean_mask = ~test.key.isin(leaky).values
        k, n = int((pred == lab).sum()), len(lab)
        kc, nc = int((pred[clean_mask] == lab[clean_mask]).sum()), int(clean_mask.sum())
        finetuned[enc] = {
            "test_n": n, "correct": k, "accuracy": k / n, "wilson95": wilson_ci(k, n),
            "macro_f1": float(f1_score(lab, pred, average="macro")),
            "confusion_matrix": confusion_matrix(lab, pred).tolist(), "classes": classes,
            "test_images_with_near_dup_in_trainval": int(n - nc),
            "dedup_test_n": nc, "dedup_accuracy": kc / nc, "dedup_wilson95": wilson_ci(kc, nc),
        }
        print(f"{enc}: {k}/{n} = {k / n:.4f}  Wilson95 {wilson_ci(k, n)}  | dedup {kc}/{nc}")
    save_json({"cv": rows, "finetuned_heldout": finetuned}, out / "classification_summary.json")


if __name__ == "__main__":
    main()
