"""FTIR ingestion and indicative peak analysis. Only operates on a spectrum the user supplies - never invents one.

Supported inputs: CSV/TSV/TXT (2 columns: wavenumber, intensity; comma-decimals accepted), XLSX (first sheet,
first column = wavenumber, first spectrum column = intensity; select with `column`), JCAMP-DX (##XYDATA=(X++(Y..Y)) or
##XYPOINTS/##PEAK TABLE). Band assignments are INDICATIVE textbook ranges: overlapping bands mean they cannot identify
individual air pollutants on a leaf (the v1 audit does not support the reference paper's SO2/NO2/O3 assignments).
"""
import io
import re

import numpy as np

from . import AVAILABLE, UNAVAILABLE, section

# (low, high, assignment, note) in cm-1 - generic infrared group-frequency ranges
BANDS = [
    (3600, 3200, "O-H / N-H stretching", "water, alcohols, polysaccharides, proteins"),
    (3000, 2840, "aliphatic C-H stretching", "lipids, waxes, cuticle"),
    (1780, 1700, "C=O stretching (esters/acids)", "cutin, pectin, lipids"),
    (1690, 1600, "C=O stretch / amide I; O-H bend of water", "proteins, water"),
    (1570, 1500, "amide II (N-H bend, C-N stretch)", "proteins"),
    (1470, 1370, "C-H bending / COO- stretching", "aliphatics, carboxylates"),
    (1300, 1200, "C-O stretching / amide III", "esters, proteins"),
    (1180, 1000, "C-O / C-O-C / C-C stretching", "polysaccharides (cellulose, pectin), silicates also absorb near 1000-1100"),
    (900, 700, "C-H out-of-plane bending / fingerprint", "aromatics, carbonates"),
]
CAVEAT = ("Indicative functional-group ranges only. Many compounds overlap in each range, so peaks do not identify "
          "specific pollutants or leaf composition; no pollutant attribution is made.")


def _num(s):
    return float(str(s).strip().replace(",", "."))


def _from_pairs(x, y, source, y_units):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 50:
        raise ValueError(f"Only {len(x)} valid points; a spectrum needs at least 50.")
    if np.any(np.diff(x) == 0):
        raise ValueError("Duplicate wavenumber values found.")
    order = np.argsort(x)
    x, y = x[order], y[order]
    if x[0] < 0 or x[-1] > 20000 or (x[-1] - x[0]) < 200:
        raise ValueError(f"Wavenumber range {x[0]:.1f}-{x[-1]:.1f} cm-1 is not a plausible mid-IR spectrum (expected spanning >= 200 cm-1 within 0-20000).")
    if y_units not in ("absorbance", "transmittance"):
        raise ValueError("y_units must be 'absorbance' or 'transmittance'.")
    return {"wavenumber": x, "intensity": y, "y_units": y_units, "source": source}


def parse_jcamp(text, y_units="absorbance"):
    meta = {}
    xs, ys = [], []
    in_xy = False
    xfactor = yfactor = 1.0
    firstx = None
    for line in text.splitlines():
        line = line.split("$$")[0].strip()
        if not line:
            continue
        if line.startswith("##"):
            k, _, v = line[2:].partition("=")
            k = re.sub(r"[\s_\-/]", "", k).upper()
            meta[k] = v.strip()
            in_xy = k in ("XYDATA", "XYPOINTS", "PEAKTABLE")
            if k == "XFACTOR": xfactor = _num(v)
            if k == "YFACTOR": yfactor = _num(v)
            if k == "FIRSTX": firstx = _num(v)
            continue
        if in_xy:
            vals = [_num(t) for t in re.split(r"[\s,;]+", line) if t]
            if "XYDATA" in meta:  # X++(Y..Y): first value is X, rest are Y
                xs_line, ys_line = vals[0], vals[1:]
                npts = len(ys_line)
                # spacing from DELTAX if present else inferred later from successive X
                for i, yv in enumerate(ys_line):
                    xs.append((xs_line * xfactor, i)); ys.append(yv * yfactor)
            else:
                for i in range(0, len(vals) - 1, 2):
                    xs.append((vals[i] * xfactor, 0)); ys.append(vals[i + 1] * yfactor)
    if not ys:
        raise ValueError("No XYDATA/XYPOINTS/PEAK TABLE block found in the JCAMP-DX file.")
    if "XYDATA" in meta:
        # X++(Y..Y): x is uniform between FIRSTX and LASTX
        if "FIRSTX" not in meta or "LASTX" not in meta:
            raise ValueError("JCAMP XYDATA block needs FIRSTX and LASTX.")
        x = np.linspace(_num(meta["FIRSTX"]), _num(meta["LASTX"]), len(ys))
    else:
        x = [x0 for x0, _ in xs]
    units = meta.get("YUNITS", "").upper()
    if "TRANSMITT" in units: y_units = "transmittance"
    elif "ABSORB" in units: y_units = "absorbance"
    return _from_pairs(x, ys, "jcamp-dx", y_units)


def parse_table_text(text, y_units="absorbance"):
    xs, ys = [], []
    for line in text.splitlines():
        line = line.strip()
        if ";" in line or "\t" in line:      # explicit delimiter -> comma may be a decimal separator
            parts = [p for p in re.split(r"[;\t]", line) if p.strip()]
        elif "," in line and not re.search(r"\s", line):  # plain CSV
            parts = line.split(",")
        else:
            parts = line.split()
        if len(parts) < 2:
            continue
        try:
            xs.append(_num(parts[0])); ys.append(_num(parts[1]))
        except ValueError:
            continue  # header / comment lines
    if not xs:
        raise ValueError("No numeric two-column (wavenumber, intensity) rows found.")
    return _from_pairs(xs, ys, "text-table", y_units)


def parse_xlsx(data, column=1, y_units="absorbance", wavenumber_scale=None):
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    rows = list(wb.worksheets[0].iter_rows(values_only=True))
    xs, ys = [], []
    for r in rows:
        if len(r) <= column or r[0] is None or r[column] is None:
            continue
        try:
            xs.append(_num(r[0])); ys.append(_num(r[column]))
        except ValueError:
            continue
    if not xs:
        raise ValueError("No numeric rows found in the first sheet.")
    xs = np.asarray(xs)
    if wavenumber_scale:
        xs = xs / wavenumber_scale
    return _from_pairs(xs, ys, "xlsx", y_units)


def parse_spectrum(filename, data: bytes, y_units="absorbance", column=1, wavenumber_scale=None):
    name = (filename or "").lower()
    if name.endswith((".xlsx", ".xlsm")):
        return parse_xlsx(data, column, y_units, wavenumber_scale)
    text = data.decode("utf-8", errors="replace")
    if name.endswith((".jdx", ".dx", ".jcm")) or "##TITLE" in text[:500].upper():
        return parse_jcamp(text, y_units)
    if name.endswith((".csv", ".tsv", ".txt", ".dat", "")):
        return parse_table_text(text, y_units)
    raise ValueError("Unsupported spectrum format. Use CSV/TSV/TXT, XLSX or JCAMP-DX.")


def _to_absorbance(y, units):
    return y if units == "absorbance" else -np.log10(np.clip(y / (100.0 if np.nanmax(y) > 2 else 1.0), 1e-6, None))


def analyse(sp, max_points=1500, prominence_frac=0.03):
    from scipy.signal import find_peaks, savgol_filter
    x, y = sp["wavenumber"], _to_absorbance(sp["intensity"], sp["y_units"])
    if len(y) > 11:
        w = min(len(y) // 2 * 2 - 1, 11)
        ys = savgol_filter(y, w, 3)
    else:
        ys = y
    # linear baseline between the ends (display/peak helper only)
    base = np.linspace(ys[0], ys[-1], len(ys))
    z = ys - base
    prom = max(prominence_frac * float(np.ptp(z)), 1e-9)
    idx, props = find_peaks(z, prominence=prom)
    peaks = []
    for i, pr in zip(idx, props["prominences"]):
        wn = float(x[i])
        cands = [{"range_cm-1": [lo, hi], "assignment": a, "typical_of": n} for hi, lo, a, n in
                 [(b[0], b[1], b[2], b[3]) for b in BANDS] if lo <= wn <= hi]
        peaks.append({"wavenumber_cm-1": round(wn, 2), "absorbance": round(float(y[i]), 5), "prominence": round(float(pr), 5),
                      "candidate_groups": cands})
    peaks.sort(key=lambda p: -p["prominence"])
    step = max(1, len(x) // max_points)
    return section(AVAILABLE, "Peaks detected on a Savitzky-Golay smoothed, end-point-baselined copy of the supplied spectrum.",
                   n_points=int(len(x)), range_cm1=[round(float(x.min()), 2), round(float(x.max()), 2)], y_units_input=sp["y_units"],
                   n_peaks=len(peaks), peaks=peaks[:25], caveat=CAVEAT,
                   spectrum={"wavenumber": [round(float(v), 3) for v in x[::step]], "absorbance": [round(float(v), 5) for v in y[::step]]})
