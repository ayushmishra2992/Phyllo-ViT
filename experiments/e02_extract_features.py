"""E2 — Extract CLS embeddings + logits for every image with three encoders, and for corrupted copies.

Encoders
  pretrained : google/vit-base-patch16-224-in21k, frozen, NEVER trained on our images
               -> the only leakage-free representation for cross-validation / unseen-class experiments
  ft3        : original fine-tuned 3-class ViT (AQI embedding space)
  ft4        : deployed fine-tuned 4-class ViT
Corruptions (applied to the 4-class TEST leaves) simulate low-quality uploads.

Output: results/features/<encoder>.npz  and  results/features/corrupt_<encoder>.npz
"""
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

from common import banner
from phyllo.features import ViTEncoder, load_pretrained, set_seed
from phyllo.paths import CLEAN, FEATURES, MODEL_3CLASS, MODEL_4CLASS, RESULTS, ensure_dirs

CORRUPTIONS = {
    "blur": lambda im, r: im.filter(ImageFilter.GaussianBlur(radius=max(im.size) / 60)),
    "dark": lambda im, r: Image.fromarray((np.asarray(im, np.float32) * 0.12).astype(np.uint8)),
    "noise": lambda im, r: Image.fromarray(np.clip(np.asarray(im, np.float32) + r.normal(0, 70, (im.size[1], im.size[0], 3)), 0, 255).astype(np.uint8)),
    "occlusion": lambda im, r: _occlude(im, r),
}


def _occlude(im, r):
    a = np.asarray(im).copy()
    h, w = a.shape[:2]
    bh, bw = int(h * 0.6), int(w * 0.6)
    y, x = r.integers(0, h - bh), r.integers(0, w - bw)
    a[y:y + bh, x:x + bw] = 0
    return Image.fromarray(a)


def load(paths):
    out = []
    for p in paths:
        with Image.open(p) as im:
            out.append(im.convert("RGB").copy())   # no EXIF transpose: matches how the models were trained
    return out


def main():
    banner("E2  FEATURE EXTRACTION")
    set_seed(42)
    ensure_dirs()
    man = pd.read_csv(RESULTS / "e01_audit" / "manifest.csv")
    paths = [CLEAN / k for k in man.key]
    images = load(paths)
    print(f"{len(images)} images loaded")

    test_leaf = man[(man.split_4class == "test") & (man["class"] != "Not_a_leaf")]
    rng = np.random.default_rng(42)
    base = [images[i] for i in test_leaf.index]
    corrupt_imgs, corrupt_keys, corrupt_kind = [], [], []
    for kind, fn in CORRUPTIONS.items():
        for k, im in zip(test_leaf.key, base):
            corrupt_imgs.append(fn(im, rng))
            corrupt_keys.append(k)
            corrupt_kind.append(kind)

    encoders = {
        "pretrained": lambda: load_pretrained(),
        "ft3": lambda: ViTEncoder(MODEL_3CLASS),
        "ft4": lambda: ViTEncoder(MODEL_4CLASS),
    }
    for name, make in encoders.items():
        enc = make()
        r = enc.encode(images)
        np.savez_compressed(FEATURES / f"{name}.npz", keys=man.key.values, emb=r["embeddings"],
                            logits=r["logits"] if r["logits"] is not None else np.zeros((len(images), 0)))
        rc = enc.encode(corrupt_imgs)
        np.savez_compressed(FEATURES / f"corrupt_{name}.npz", keys=np.array(corrupt_keys), kind=np.array(corrupt_kind),
                            emb=rc["embeddings"],
                            logits=rc["logits"] if rc["logits"] is not None else np.zeros((len(corrupt_imgs), 0)))
        print(f"  {name:10s} emb {r['embeddings'].shape}  corrupt {rc['embeddings'].shape}")
        del enc
        import torch
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
