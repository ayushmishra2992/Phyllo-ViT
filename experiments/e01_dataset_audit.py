"""E1 — Dataset audit: manifest, perceptual near-duplicates, cross-split leakage, acquisition clusters.

Outputs (results/e01_audit/):
  manifest.csv            one row per image: class, size, md5, pHash, split membership, near-dup group
  near_duplicate_pairs.csv
  audit_summary.json
"""
import hashlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dct

from common import banner, out_dir, save_json
from phyllo.paths import ALL_CLASSES, CLEAN, SPLIT_3, SPLIT_4, list_images

NEAR_DUP_BITS = 6   # Hamming distance (of 64) at or below which two images are treated as near-duplicates


def phash(img):
    g = np.asarray(img.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    d = dct(dct(g, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    bits = (d > np.median(d.ravel()[1:])).ravel()   # exclude DC term from the median
    return int("".join("1" if b else "0" for b in bits), 2)


def split_of(root, cls, name):
    for s in ("train", "val", "test"):
        if (root / s / cls / name).exists():
            return s
    return ""


def main():
    banner("E1  DATASET AUDIT")
    out = out_dir("e01_audit")
    rows = []
    for cls in ALL_CLASSES:
        for p in list_images(CLEAN / cls):
            with Image.open(p) as im:
                w, h = im.size
                ph = phash(im)
            rows.append({
                "key": f"{cls}/{p.name}", "class": cls, "filename": p.name, "width": w, "height": h,
                "file_size": p.stat().st_size, "md5": hashlib.md5(p.read_bytes()).hexdigest(),
                "phash": f"{ph:016x}", "split_3class": split_of(SPLIT_3, cls, p.name) if cls != "Not_a_leaf" else "",
                "split_4class": split_of(SPLIT_4, cls, p.name),
            })
    df = pd.DataFrame(rows)
    print(df.groupby("class").size().to_string())

    # ---- near duplicates (all pairs; n≈415 so O(n²) is fine)
    hashes = np.array([int(h, 16) for h in df.phash], dtype=np.uint64)
    n = len(df)
    pairs = []
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        x = np.bitwise_xor(hashes[i], hashes[i + 1:])
        dist = np.array([bin(int(v)).count("1") for v in x])
        for j in np.where(dist <= NEAR_DUP_BITS)[0]:
            j = i + 1 + j
            pairs.append({"a": df.key[i], "b": df.key[j], "hamming": int(dist[j - i - 1]),
                          "same_class": df["class"][i] == df["class"][j],
                          "split3_a": df.split_3class[i], "split3_b": df.split_3class[j],
                          "split4_a": df.split_4class[i], "split4_b": df.split_4class[j]})
            parent[find(i)] = find(j)
    df["group"] = [find(i) for i in range(n)]
    df["group"] = df.group.map({g: k for k, g in enumerate(sorted(set(df.group)))})
    pdf = pd.DataFrame(pairs)
    pdf.to_csv(out / "near_duplicate_pairs.csv", index=False)
    df.to_csv(out / "manifest.csv", index=False)

    def cross(col_a, col_b):
        if pdf.empty:
            return 0
        m = (pdf[col_a] != "") & (pdf[col_b] != "") & (pdf[col_a] != pdf[col_b])
        return int(m.sum())

    def touches_test(col_a, col_b):
        if pdf.empty:
            return 0
        m = ((pdf[col_a] == "test") ^ (pdf[col_b] == "test")) & (pdf[col_a] != "") & (pdf[col_b] != "")
        return int(m.sum())

    # ---- acquisition clusters: identical resolution is a proxy for camera/session
    res = df.assign(res=df.width.astype(str) + "x" + df.height.astype(str))
    leaf = res[res["class"] != "Not_a_leaf"]
    res_table = leaf.groupby(["res", "class"]).size().unstack(fill_value=0)
    res_table = res_table.loc[res_table.sum(axis=1).sort_values(ascending=False).index]
    res_table.to_csv(out / "resolution_by_class.csv")
    # purity: fraction of leaf images whose resolution cluster is dominated (>=90%) by a single class
    pure = res_table[(res_table.max(axis=1) / res_table.sum(axis=1)) >= 0.9].sum(axis=1).sum() / len(leaf)

    # ---- 4-class test images seen by the 3-class ViT during training (matters for any cross-model eval)
    seen = df[(df.split_4class == "test") & (df.split_3class == "train")]

    summary = {
        "n_images": int(n), "per_class": df["class"].value_counts().to_dict(),
        "exact_duplicate_md5_groups": int((df.md5.value_counts() > 1).sum()),
        "near_duplicate_threshold_bits": NEAR_DUP_BITS,
        "near_duplicate_pairs": int(len(pdf)),
        "near_duplicate_pairs_cross_class": int((~pdf.same_class).sum()) if len(pdf) else 0,
        "near_dup_pairs_across_splits_3class": cross("split3_a", "split3_b"),
        "near_dup_pairs_across_splits_4class": cross("split4_a", "split4_b"),
        "near_dup_pairs_linking_test_to_trainval_3class": touches_test("split3_a", "split3_b"),
        "near_dup_pairs_linking_test_to_trainval_4class": touches_test("split4_a", "split4_b"),
        "n_groups_after_merging_near_dups": int(df.group.nunique()),
        "distinct_leaf_resolutions": int(leaf.res.nunique()),
        "leaf_images_in_class_pure_resolution_clusters": round(float(pure), 3),
        "top_leaf_resolutions": {r: res_table.loc[r].to_dict() for r in res_table.index[:6]},
        "fourclass_test_images_in_threeclass_train": int(len(seen)),
    }
    save_json(summary, out / "audit_summary.json")

    # annotation template for the team's first task (never overwritten once it exists)
    from phyllo.paths import DATA
    tmpl = DATA / "metadata" / "provenance_annotation_TODO.csv"
    if not tmpl.exists():
        t = df[df["class"] != "Not_a_leaf"][["key", "class", "width", "height", "group"]].copy()
        for col in ("species", "plant_id", "capture_session_or_date", "location", "stem_and_nodes_visible_yes_no",
                    "arrangement_verified_by", "notes"):
            t[col] = ""
        t.to_csv(tmpl, index=False)
    for k, v in summary.items():
        print(f"{k:55s} {v}")


if __name__ == "__main__":
    main()
