"""India National Air Quality Index (CPCB, 2014) — documented, deterministic AQI computation.

This module exists to keep two things strictly separate:
  * pollutant CONCENTRATION estimation (a modelling problem), and
  * AQI COMPUTATION (a fixed formula applied to measured/estimated concentrations).

Sub-index:  I_p = (I_hi - I_lo) / (B_hi - B_lo) * (C_p - B_lo) + I_lo
Overall AQI = max sub-index, valid only when >= 3 pollutants are given and at least one of
PM2.5 / PM10 is among them (CPCB rule). Units: µg/m³ except CO (mg/m³).
Averaging periods: PM10, PM2.5, NO2, SO2, NH3, Pb = 24 h; CO, O3 = 8 h (max 8-h for O3/CO).
Above the top breakpoint the sub-index is capped at 500.
"""

INDEX_BANDS = [(0, 50), (51, 100), (101, 200), (201, 300), (301, 400), (401, 500)]
CATEGORIES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

BREAKPOINTS = {  # concentration bands aligned with INDEX_BANDS; last band open-ended (upper = cap value)
    "PM10":  ([0, 51, 101, 251, 351, 431], [50, 100, 250, 350, 430, 600]),
    "PM2.5": ([0, 31, 61, 91, 121, 251], [30, 60, 90, 120, 250, 380]),
    "NO2":   ([0, 41, 81, 181, 281, 401], [40, 80, 180, 280, 400, 520]),
    "O3":    ([0, 51, 101, 169, 209, 749], [50, 100, 168, 208, 748, 1000]),
    "CO":    ([0, 1.1, 2.1, 10.1, 17.1, 34.1], [1.0, 2.0, 10, 17, 34, 50]),
    "SO2":   ([0, 41, 81, 381, 801, 1601], [40, 80, 380, 800, 1600, 2000]),
    "NH3":   ([0, 201, 401, 801, 1201, 1801], [200, 400, 800, 1200, 1800, 2400]),
    "Pb":    ([0, 0.6, 1.1, 2.1, 3.1, 3.6], [0.5, 1.0, 2.0, 3.0, 3.5, 5.0]),
}
UNITS = {k: "µg/m³" for k in BREAKPOINTS}
UNITS["CO"] = "mg/m³"
AVERAGING = {"PM10": "24h", "PM2.5": "24h", "NO2": "24h", "SO2": "24h", "NH3": "24h", "Pb": "24h",
             "O3": "8h", "CO": "8h"}


def category(aqi):
    for (lo, hi), name in zip(INDEX_BANDS, CATEGORIES):
        if aqi <= hi:
            return name
    return "Severe"


def sub_index(pollutant, conc):
    if pollutant not in BREAKPOINTS:
        raise ValueError(f"Unsupported pollutant '{pollutant}'. Supported: {list(BREAKPOINTS)}")
    if conc is None or conc < 0:
        raise ValueError(f"Concentration for {pollutant} must be a non-negative number.")
    lows, highs = BREAKPOINTS[pollutant]
    for (blo, bhi), (ilo, ihi) in zip(zip(lows, highs), INDEX_BANDS):
        # bands are contiguous: treat values between bhi of one band and blo of the next as the upper band
        if conc <= bhi:
            c = max(conc, blo)
            return round((ihi - ilo) / (bhi - blo) * (c - blo) + ilo)
    return 500


def compute_aqi(concentrations: dict):
    conc = {k: v for k, v in concentrations.items() if v is not None}
    unknown = set(conc) - set(BREAKPOINTS)
    if unknown:
        raise ValueError(f"Unsupported pollutant(s): {sorted(unknown)}")
    subs = {p: sub_index(p, float(c)) for p, c in conc.items()}
    valid = len(subs) >= 3 and ("PM2.5" in subs or "PM10" in subs)
    result = {
        "standard": "India National Air Quality Index (CPCB, 2014)",
        "sub_indices": {p: {"concentration": conc[p], "unit": UNITS[p], "averaging": AVERAGING[p],
                            "sub_index": s, "category": category(s)} for p, s in subs.items()},
        "valid": valid,
    }
    if subs:
        dom = max(subs, key=subs.get)
        result.update({"aqi": subs[dom], "category": category(subs[dom]), "dominant_pollutant": dom})
    if not valid:
        result["note"] = ("CPCB requires at least 3 pollutants including PM2.5 or PM10 for a valid AQI; "
                          "the value shown is the maximum sub-index only.")
    return result
