"""Central paths and constants for Phyllo-ViT v3. Everything is relative to the project root,
so the project works from any folder (Desktop, Documents, ...)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODELS = ROOT / "models"
MODEL_3CLASS = MODELS / "vit_phyllotaxy"          # original 3-class ViT (AQI embedding space)
MODEL_4CLASS = MODELS / "vit_phyllotaxy_4class"   # deployed 4-class ViT (input validation)
PRETRAINED_ID = "google/vit-base-patch16-224-in21k"  # frozen backbone (never fine-tuned on our data)

DATA = ROOT / "data"
CLEAN = DATA / "processed" / "clean"              # 265 unique leaves + 150 Not_a_leaf
SPLIT_3 = DATA / "processed" / "split"            # split used to train the 3-class ViT
SPLIT_4 = DATA / "processed" / "split_4class"     # split used to train the 4-class ViT
RAW_XLSX = DATA / "raw"
AQI_PAIRS = ROOT / "legacy" / "results" / "aqi_regression" / "aqi_vit_metadata.csv"

ARTIFACTS = ROOT / "artifacts"                    # deploy-time reliability artifacts
RESULTS = ROOT / "results"                        # experiment outputs (v3)
FEATURES = RESULTS / "features"

LEAF_CLASSES = ["Alternate", "Opposite", "Whorled"]
ALL_CLASSES = ["Alternate", "Opposite", "Whorled", "Not_a_leaf"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

SEED = 42


def ensure_dirs():
    for p in (ARTIFACTS, RESULTS, FEATURES):
        p.mkdir(parents=True, exist_ok=True)


def list_images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)
