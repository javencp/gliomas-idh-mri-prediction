"""
Build the final analysis cohort: manifest (labels, age, sex, folds) joined with each
patient's latest processing result. 
Computed fresh each time so it never goes stale.
"""
import pandas as pd

from src.config import LOG, MANIFEST


def latest_log_rows():
    """The log is append-only history, so the last row written for a patient is the latest attempt."""
    log = pd.read_csv(LOG)
    return log.groupby("patient_id", sort=False).tail(1) 


def build_cohort(expected_n=None, verbose=True):
    manifest = pd.read_csv(MANIFEST).drop(columns="processed", errors="ignore")  # stale flag, ignored
    latest = latest_log_rows()[["patient_id", "status", "reason", "mask_voxels"]]
    merged = manifest.merge(latest, on="patient_id", how="left")  # how="left" keeps unprocessed patients visible

    cohort = merged[merged["status"] == "ok"].reset_index(drop=True)
    excluded = merged[merged["status"] != "ok"]

    if verbose:
        print(f"manifest: {len(manifest)} | usable: {len(cohort)} | excluded: {len(excluded)}")
        print(excluded[["patient_id", "status", "reason"]].to_string(index=False))
        print(cohort["idh"].value_counts())
        print(cohort.groupby("fold")["idh"].value_counts().unstack())
    if expected_n is not None:
        assert len(cohort) == expected_n, f"expected {expected_n} patients, got {len(cohort)}"
    return cohort