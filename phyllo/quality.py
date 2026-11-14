"""Image-quality gate. Cheap, interpretable checks that run before any model.

Thresholds are deliberately conservative: they only reject images that are clearly unusable
(tiny, nearly black/white, extremely blurred, or almost featureless). Borderline images pass
with a warning so the downstream reliability layer can decide.
"""
from dataclasses import dataclass, field, asdict

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.signal import convolve2d

MIN_SIDE = 64            # px
DARK_MEAN = 25           # 0-255 grey mean
BRIGHT_MEAN = 235
LOW_CONTRAST_STD = 12    # grey std
BLUR_REJECT = 8.0        # variance of Laplacian on a 512px-wide grey image
BLUR_WARN = 40.0


_NOISE_K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)


def noise_sigma(image: Image.Image) -> float:
    """Fast noise standard-deviation estimate (Immerkaer, 1996) on a 512px-wide grey image."""
    g = image.convert("L")
    if g.width != 512:
        g = g.resize((512, max(3, int(g.height * 512 / g.width))))
    a = np.asarray(g, dtype=np.float64)
    h, w = a.shape
    return float(np.sqrt(np.pi / 2) / (6 * (w - 2) * (h - 2)) * np.abs(convolve2d(a, _NOISE_K, "valid")).sum())


@dataclass
class QualityReport:
    passed: bool
    width: int
    height: int
    brightness: float
    contrast: float
    sharpness: float
    noise: float
    issues: list = field(default_factory=list)     # blocking problems
    warnings: list = field(default_factory=list)   # non-blocking

    def to_dict(self):
        d = asdict(self)
        for k in ("brightness", "contrast", "sharpness", "noise"):
            d[k] = round(float(d[k]), 2)
        return d


def assess(image: Image.Image, noise_warn: float = None) -> QualityReport:
    """noise_warn: data-derived noise level (99th pct of clean training images) above which we warn."""
    w, h = image.size
    g = image.convert("L")
    if g.width > 512:
        g = g.resize((512, max(1, int(g.height * 512 / g.width))))
    a = np.asarray(g, dtype=np.float32)
    brightness, contrast = float(a.mean()), float(a.std())
    sharpness = float(ndimage.laplace(a).var())

    issues, warnings = [], []
    if min(w, h) < MIN_SIDE:
        issues.append(f"Image is too small ({w}x{h}); need at least {MIN_SIDE}px per side.")
    if brightness < DARK_MEAN:
        issues.append("Image is too dark to analyse.")
    elif brightness > BRIGHT_MEAN:
        issues.append("Image is over-exposed (almost white).")
    if contrast < LOW_CONTRAST_STD:
        issues.append("Image has almost no visual structure (very low contrast).")
    if sharpness < BLUR_REJECT:
        issues.append("Image is severely blurred.")
    elif sharpness < BLUR_WARN:
        warnings.append("Image looks soft/blurred; prediction reliability may be reduced.")
    noise = noise_sigma(image)
    if noise_warn is not None and noise > noise_warn:
        warnings.append("Image is noisier than 99% of the validated training photos.")
    if min(w, h) < 224:
        warnings.append("Low resolution (<224px); fine leaf/stem structure may be lost.")

    return QualityReport(not issues, w, h, brightness, contrast, sharpness, noise, issues, warnings)
