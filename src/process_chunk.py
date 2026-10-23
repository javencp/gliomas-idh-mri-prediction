"""Process every downloaded patient folder in Data/raw or Data/raw_ucsf that is listed in the manifest.
Can be run in chunks, which is necessary for large datasets that don't fit in memory. 
Specifically, UCSF which is >140 GB. The output is a compressed .npz file per patient in Data/processed. 

Run from the repo root with the venv active (added --dataset as an argument; --delete-raw should work for both datasets):
    python -m src.process_chunk --dataset utsw               # process only; raw folders are kept (using UTSW dataset)
    python -m src.process_chunk --dataset ucsf --delete-raw   # also delete each raw folder once its 
                                                                output is verified (using UCSF dataset)

Safe to re-run: patients with a valid output file are skipped. Failed patients are logged
with a reason and their raw folder is never deleted.
"""
import argparse
import csv
import shutil

import numpy as np
import pandas as pd

from src import preprocessing as pp
from src.config import (LOG, MANIFEST, PROCESSED, RAW, RAW_UCSF, 
                        PROCESSED_UCSF, MANIFEST_UCSF, LOG_UCSF)

# added datasets argument to allow processing UCSF data as well
DATASETS = {
    "utsw": dict(raw=RAW, processed=PROCESSED, manifest=MANIFEST, log=LOG,
                 contrasts=pp.CONTRASTS, seg=pp.SEG, folder_suffix=""), # UTSW values already defined in preprocessing.py
    "ucsf": dict(raw=RAW_UCSF, processed=PROCESSED_UCSF, manifest=MANIFEST_UCSF, log=LOG_UCSF,
                 contrasts=["{pid}_T1", "{pid}_T1c", "{pid}_T2", "{pid}_FLAIR"],  # same channel order as UTSW
                 seg="{pid}_tumor_segmentation", folder_suffix="_nifti"),
}

LOG_FIELDS = ["patient_id", "status", "reason", "bbox_x", "bbox_y", "bbox_z",
              "center_x", "center_y", "center_z", "exceeds_crop", "tumor_in_brain",
              "tumor_in_crop", "n_components", "main_fraction", "mask_voxels"]  # fields in the log file


def verify(path):
    """Reload a saved file and check it looks right."""
    try:
        with np.load(path) as f:
            img = f["image"]
            return (img.shape == (len(pp.CONTRASTS), pp.CROP, pp.CROP, pp.CROP)
                    and img.dtype == np.float16
                    and bool(np.isfinite(img).all())
                    and bool((img != 0).any()))
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--delete-raw", action="store_true")
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    raw, processed, log = cfg["raw"], cfg["processed"], cfg["log"]

    ids = set(pd.read_csv(cfg["manifest"])["patient_id"])
    processed.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)

    folders = sorted(p for p in raw.rglob("*") if p.is_dir() and p.name in ids)
    print(f"{len(folders)} downloaded patient folders listed in the manifest")

    n_done = n_new = n_failed = n_deleted = 0
    write_header = not log.exists()
    with open(log, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()

        for folder in folders:
            pid = folder.name
            stem = pid.removesuffix(cfg["folder_suffix"]) # ex. UCSF-PDGM-0004_nifti -> UCSF-PDFM-0004
            contrasts = [t.format(pid=stem) for t in cfg["contrasts"]]
            seg = cfg["seg"].format(pid=stem)

            out = processed / f"{pid}.npz"
            ok = out.exists() and verify(out)

            if ok:
                n_done += 1  # already processed earlier
            else:
                row = {"patient_id": pid}
                try:
                    r = pp.process_case(folder, contrasts, seg)
                    np.savez_compressed(out, image=r["image"], mask=r["mask"])
                    if not verify(out):
                        raise pp.PreprocessError("saved file failed verification")
                    ok = True
                    n_new += 1
                    row.update(status="ok", reason="", tumor_in_brain=round(r["tumor_in_brain"], 4),
                               bbox_x=r["bbox"][0], bbox_y=r["bbox"][1], bbox_z=r["bbox"][2],
                               center_x=r["center"][0], center_y=r["center"][1], center_z=r["center"][2],
                               exceeds_crop=bool((r["bbox"] > pp.CROP).any()),
                               tumor_in_crop=round(r["tumor_in_crop"], 4),
                               n_components=r["n_components"],
                               main_fraction=round(r["main_fraction"], 4),
                               mask_voxels=r["mask_voxels"])
                except Exception as e:  # keep the loop going; the reason goes in the log
                    out.unlink(missing_ok=True)
                    n_failed += 1
                    reason = str(e) if isinstance(e, pp.PreprocessError) else f"{type(e).__name__}: {e}"
                    row.update(status="failed", reason=reason)
                    print(f"  FAILED {pid}: {reason}")
                writer.writerow(row)
                fh.flush()

            if ok and args.delete_raw and folder.name == pid and raw in folder.parents:
                shutil.rmtree(folder)
                n_deleted += 1

    print(f"already done: {n_done} | newly processed: {n_new} | failed: {n_failed} | raw folders deleted: {n_deleted}")


if __name__ == "__main__":
    main()
