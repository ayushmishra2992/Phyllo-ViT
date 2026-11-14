"""Build a backend-only folder ready to upload as a Hugging Face Docker Space.

    python scripts/make_hf_space.py              -> creates build_hf_space/
    hf auth login                                 (once; paste a WRITE token from huggingface.co/settings/tokens)
    hf upload <user>/<space-name> build_hf_space . --repo-type space

`hf` comes with huggingface_hub (installed with transformers). It uploads large files automatically.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build_hf_space"
ITEMS = ["Dockerfile", "requirements.txt", "phyllo", "api", "models", "artifacts", "results", "legacy/results"]
HEADER = """---
title: Phyllo-ViT
emoji: 🌿
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Phyllo-ViT API

Reliability-aware phyllotaxy recognition (Alternate / Opposite / Whorled, with abstention).
Web UI at `/`, API docs at `/docs`. Source: see the project's GitHub repository.
"""


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    ign = shutil.ignore_patterns("__pycache__", "*.pyc", "_tmp")
    for item in ITEMS:
        src = ROOT / item
        if not src.exists():
            sys.exit(f"missing {item}")
        dst = OUT / item
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=ign)
        else:
            shutil.copy2(src, dst)
    (OUT / "README.md").write_text(HEADER, encoding="utf-8")
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"Created {OUT}  ({size:.0f} MB)")
    print("Next:  hf auth login   then   hf upload <user>/<space-name> build_hf_space . --repo-type space")


if __name__ == "__main__":
    main()
