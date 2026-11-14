"""Check that Alternate, Opposite and Whorled (and Not_a_leaf) work end-to-end.

Runs every held-out image (val + test of the 4-class split) through
  1. the deployed PhylloEngine (quality gate -> ViT -> calibration -> OOD -> decision), and
  2. the real HTTP endpoint POST /predict (FastAPI TestClient), one image per class,
and prints per-class accuracy plus a confusion matrix. Writes results/class_check.json.
Usage:  python scripts/check_classes.py
"""
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PIL import Image, ImageOps  # noqa: E402

from phyllo.paths import ALL_CLASSES, RESULTS, SPLIT_4, list_images  # noqa: E402


def main():
    from phyllo.pipeline import PhylloEngine
    eng = PhylloEngine(load_aqi_model=False)
    report = {}
    for split in ("test", "val"):
        conf = defaultdict(Counter)       # true -> predicted top class
        status = defaultdict(Counter)     # true -> decision status
        wrong = []
        for cls in ALL_CLASSES:
            for p in list_images(SPLIT_4 / split / cls):
                with Image.open(p) as im:
                    img = ImageOps.exif_transpose(im).convert("RGB")
                r = eng.predict(img, explain=False)
                pred = r["phyllotaxy"]["class"]
                st = r["decision"]["status"]
                conf[cls][pred] += 1
                status[cls][st] += 1
                if pred != cls:
                    wrong.append({"file": f"{cls}/{p.name}", "predicted": pred, "status": st})
        print(f"\n=== {split.upper()} split ===")
        print(f"{'true/pred':14s}" + "".join(f"{c:>12s}" for c in ALL_CLASSES) + f"{'acc':>8s}")
        per = {}
        for cls in ALL_CLASSES:
            n = sum(conf[cls].values())
            acc = conf[cls][cls] / n if n else float("nan")
            per[cls] = {"n": n, "correct": conf[cls][cls], "accuracy": round(acc, 4),
                        "predicted": dict(conf[cls]), "decision_status": dict(status[cls])}
            print(f"{cls:14s}" + "".join(f"{conf[cls][c]:>12d}" for c in ALL_CLASSES) + f"{acc:>8.1%}")
        print("decision status per true class:")
        for cls in ALL_CLASSES:
            print(f"  {cls:12s} {dict(status[cls])}")
        if wrong:
            print("misclassified:", wrong)
        report[split] = {"per_class": per, "misclassified": wrong}

    # ---- real HTTP round trip, one image per class
    from fastapi.testclient import TestClient
    import api.main as app_mod
    http = {}
    with TestClient(app_mod.app) as client:
        for cls in ALL_CLASSES:
            p = list_images(SPLIT_4 / "test" / cls)[0]
            res = client.post("/predict?explain=true", files={"file": (p.name, p.read_bytes(), "image/jpeg")})
            body = res.json()
            http[cls] = {"file": p.name, "http": res.status_code, "predicted": body["phyllotaxy"]["class"],
                         "confidence": body["phyllotaxy"]["confidence"],
                         "status": body["decision"]["status"], "has_heatmap": bool((body.get("explanation") or {}).get("heatmap_png_base64"))}
            print(f"HTTP /predict  {cls:11s} -> {http[cls]}")
    report["http_predict"] = http

    # ---- final verification table (one real held-out image per class, via the HTTP API)
    mapping = json.loads((Path(__file__).resolve().parent.parent / "models" / "vit_phyllotaxy_4class" / "class_mapping.json").read_text())
    lines = ["# Class verification (real held-out test images through POST /predict)", "",
             f"Model class mapping: `{json.dumps(mapping)}`", "",
             "| Class | Test image | Predicted class | Confidence (calibrated) | Decision | Result |", "|---|---|---|---|---|---|"]
    all_pass = True
    for cls in ALL_CLASSES:
        h = http[cls]
        ok = h["http"] == 200 and h["predicted"] == cls and h["status"] in (("NOT_A_LEAF",) if cls == "Not_a_leaf" else ("ACCEPT",))
        all_pass &= ok
        lines.append(f"| {cls} | {cls}/{h['file']} | {h['predicted']} | {h['confidence']:.3f} | {h['status']} | {'PASS' if ok else 'FAIL'} |")
    lines += ["", "One image per class is a smoke test, not an accuracy claim. Full held-out confusion matrices are printed above "
              "and stored in class_check.json (small test sets: report Wilson CIs, see results/RESULTS.md)."]
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "CLASS_VERIFICATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines[4:]))
    print("\nOVERALL:", "PASS" if all_pass else "FAIL")
    report["overall"] = "PASS" if all_pass else "FAIL"
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "class_check.json").write_text(json.dumps(report, indent=2))
    print("\nwritten results/class_check.json")


if __name__ == "__main__":
    main()
