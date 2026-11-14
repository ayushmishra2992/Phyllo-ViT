"""Run every experiment in order (≈6–10 min on an RTX 3050; slower on CPU).  python experiments/run_all.py"""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["e01_dataset_audit", "e02_extract_features", "e03_classification", "e04_reliability",
         "e05_pollution_audit", "e07_data_sufficiency", "e06_results_tables"]

if __name__ == "__main__":
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "TRANSFORMERS_VERBOSITY": "error"}
    t0 = time.time()
    for s in STEPS:
        print(f"\n>>> {s}", flush=True)
        r = subprocess.run([sys.executable, str(HERE / f"{s}.py")], cwd=HERE, env=env)
        if r.returncode:
            sys.exit(f"{s} failed (exit {r.returncode})")
    print(f"\nAll experiments finished in {(time.time() - t0) / 60:.1f} min. See results/RESULTS.md")
