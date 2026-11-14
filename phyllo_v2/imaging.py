"""Classical (non-learned) leaf measurements and colour-based visible-condition flags.

These are MEASUREMENTS of the photograph. They are not validated against labelled masks or pathology, so:
  * morphology  -> AVAILABLE as pixel-space measurements only (no physical scale),
  * health      -> EXPERIMENTAL colour heuristics, "disease-like" wording only, never a diagnosis,
  * dust        -> NOT_SUPPORTED (no labelled dust/deposition masks were acquired).
"""
import numpy as np
from PIL import Image
from skimage import color, measure, morphology

from . import AVAILABLE, EXPERIMENTAL, NOT_SUPPORTED, UNAVAILABLE, section

MAX_SIDE = 512


def _prep(img: Image.Image):
    img = img.convert("RGB")
    s = MAX_SIDE / max(img.size)
    if s < 1:
        img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.BILINEAR)
    return np.asarray(img)


def leaf_mask(rgb):
    """Foreground = pixels more saturated / greener than the background (Otsu on an excess-green+saturation score)."""
    from skimage.filters import threshold_otsu
    f = rgb.astype(float) / 255.0
    exg = 2 * f[..., 1] - f[..., 0] - f[..., 2]
    sat = color.rgb2hsv(rgb)[..., 1]
    score = 0.6 * (exg - exg.min()) / (np.ptp(exg) + 1e-9) + 0.4 * sat
    m = score > threshold_otsu(score)
    m = morphology.remove_small_objects(m, max_size=200)
    m = morphology.closing(m, morphology.disk(3)).astype(bool)
    lab = measure.label(m)
    if lab.max() == 0:
        return None
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == sizes.argmax()


def morphology_section(img):
    rgb = _prep(img)
    m = leaf_mask(rgb)
    if m is None or m.mean() < 0.02:
        return section(UNAVAILABLE, "No leaf-like foreground region could be segmented from this image.")
    frac = float(m.mean())
    if frac > 0.95:
        return section(UNAVAILABLE, "Foreground covers almost the whole frame; leaf outline cannot be separated from background.")
    p = measure.regionprops(m.astype(np.uint8))[0]
    h, w = m.shape
    minr, minc, maxr, maxc = p.bbox
    return section(AVAILABLE, "Pixel-space measurements of the largest segmented region; no physical scale is known, so no cm/mm values are reported.",
                   leaf_area_fraction=round(frac, 4), aspect_ratio=round(float(p.axis_major_length / max(p.axis_minor_length, 1e-6)), 3),
                   solidity=round(float(p.solidity), 3), eccentricity=round(float(p.eccentricity), 3),
                   touches_image_border=bool(minr == 0 or minc == 0 or maxr == h or maxc == w),
                   mean_rgb_in_leaf=[int(x) for x in rgb[m].mean(axis=0)],
                   caveat="Segmentation is a colour threshold; cluttered backgrounds, shadows or multiple leaves can mislead it.")


def health_section(img):
    rgb = _prep(img)
    m = leaf_mask(rgb)
    if m is None or m.mean() < 0.02:
        return section(UNAVAILABLE, "No leaf region segmented, so visible condition cannot be assessed.")
    hsv = color.rgb2hsv(rgb)
    hh, ss, vv = hsv[..., 0] * 360, hsv[..., 1], hsv[..., 2]
    n = float(m.sum())
    healthy = m & (hh >= 70) & (hh <= 170) & (ss > 0.2)
    yellow = m & (hh >= 35) & (hh < 70) & (ss > 0.25) & (vv > 0.45)
    brown = m & (hh >= 10) & (hh < 45) & (ss > 0.25) & (vv <= 0.55)
    dark = m & (vv < 0.2)
    fr = {k: round(float(v.sum() / n), 4) for k, v in
          dict(green=healthy, yellowish=yellow, brownish=brown, very_dark=dark).items()}
    flags = []
    if fr["yellowish"] > 0.15:
        flags.append("noticeable yellowish area (possible chlorosis, senescence or lighting)")
    if fr["brownish"] > 0.08:
        flags.append("noticeable brownish area (possible drying, necrosis or soil/background)")
    if fr["very_dark"] > 0.05:
        flags.append("very dark patches (possible spots, shadow or occlusion)")
    return section(EXPERIMENTAL, "Colour-fraction heuristics inside the segmented leaf. Not validated against labelled leaf-health data; "
                   "colour alone cannot establish disease, pathogen, nutrient status or pollution exposure.",
                   colour_fractions=fr, flags=flags, wording="disease-like / condition-like appearance only")


def dust_section():
    return section(NOT_SUPPORTED, "Visible dust/deposition needs labelled segmentation masks, which have not been acquired or validated. "
                   "No coverage %, particle count, PM2.5 value or chemical composition is reported or can be inferred from an RGB photograph.")


def species_section():
    return section(NOT_SUPPORTED, "No licensed, labelled species dataset has been acquired and validated yet (see v2/data_registry). "
                   "No species name or confidence is shown rather than a guess.")


def biological_section():
    from . import DISABLED
    return section(DISABLED, "Bacterial/fungal identity cannot be determined from an ordinary RGB leaf photograph. "
                   "This module stays disabled until laboratory/microscopy evidence and a labelled dataset exist.")
