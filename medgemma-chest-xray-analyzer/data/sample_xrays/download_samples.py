#!/usr/bin/env python3
"""
Download public-domain sample chest X-rays for testing.
Run from project root:
    python data/sample_xrays/download_samples.py
"""

import io
import os
import sys

import requests
from PIL import Image

_DIR = os.path.dirname(os.path.abspath(__file__))

SAMPLES = {
    "chest_xray_pa.png": (
        "https://upload.wikimedia.org/wikipedia/commons/c/c8/"
        "Chest_Xray_PA_3-8-2010.png"
    ),
}

_USER_AGENT = (
    "MedGemma-XRay-Analyzer/1.0 "
    "(https://github.com/your-repo; research use only)"
)


def download_samples() -> None:
    print(f"📂 Download directory: {_DIR}\n")
    for filename, url in SAMPLES.items():
        dest = os.path.join(_DIR, filename)
        if os.path.exists(dest):
            print(f"  ✅ Already exists: {filename} — skipping.")
            continue

        print(f"  ⬇️  Downloading {filename}...")
        try:
            resp = requests.get(
                url, headers={"User-Agent": _USER_AGENT}, timeout=30
            )
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            img.save(dest)
            print(f"  ✅ Saved: {dest} ({img.size[0]}x{img.size[1]}px)")
        except Exception as exc:
            print(f"  ❌ Failed to download {filename}: {exc}")


if __name__ == "__main__":
    download_samples()
