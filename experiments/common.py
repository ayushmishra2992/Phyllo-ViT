import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from phyllo.paths import RESULTS  # noqa: E402


def out_dir(name):
    d = RESULTS / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(obj, path):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    Path(path).write_text(json.dumps(obj, indent=2, default=conv), encoding="utf-8")


def banner(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72, flush=True)


def wilson_ci(k, n, z=1.96):
    """95% Wilson score interval for a proportion (e.g. accuracy k/n)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def mean_std(xs):
    xs = np.asarray(xs, float)
    return {"mean": float(xs.mean()), "std": float(xs.std(ddof=1)) if len(xs) > 1 else 0.0, "n": int(len(xs))}
