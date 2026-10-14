"""One-time conversion for fast loading: Data/processed/<id>.npz -> Data/processed_npy/<id>.npy
Keeps only the image (the mask is dropped) and stores it uncompressed (about 10 GB for 619 patients).
Safe to re-run: patients that are already converted are skipped.
Run from the repo root:  python -m src.to_npy"""
import numpy as np

from src.config import PROCESSED, PROCESSED_NPY

PROCESSED_NPY.mkdir(exist_ok=True)
files = sorted(PROCESSED.glob("*.npz"))

for i, f in enumerate(files, 1):
    out = PROCESSED_NPY / f"{f.stem}.npy"
    if out.exists():
        continue
    tmp = PROCESSED_NPY / f"{f.stem}.tmp.npy"        # write to a temp name first, so an interrupted
    with np.load(f) as z:                            # run never leaves a half-written file behind
        np.save(tmp, z["image"])
    tmp.replace(out)
    if i % 100 == 0:
        print(f"{i}/{len(files)}")
print(f"done: {len(files)} patients")
