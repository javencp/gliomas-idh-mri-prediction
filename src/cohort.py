"""
Build the final analysis cohort: manifest (labels, age, sex, folds) joined with each
patient's latest processing result. 
Computed fresh each time so it never goes stale.

dataset="utsw" (default) is the training cohort; dataset="ucsf" is the external test set.
"""
import pandas as pd

from src.config import LOG, LOG_UCSF, MANIFEST, MANIFEST_UCSF

# manifest and processing log for each dataset
DATASETS = {"utsw": (MANIFEST, LOG), "ucsf": (MANIFEST_UCSF, LOG_UCSF)}


def latest_log_rows(log_path=LOG):
    """The log is append-only history, so the last row written for a patient is the latest attempt."""
    log = pd.read_csv(log_path)
    return log.groupby("patient_id", sort=False).tail(1) 


def build_cohort(expected_n=None, verbose=True, dataset="utsw", with_labels=True):
    """with_labels=False drops the idh column, so label-blind code (the UCSF scoring
    script) cannot read labels by accident. Labels are joined only in the evaluation notebook."""
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {list(DATASETS)}, got {dataset!r}")
    manifest_path, log_path = DATASETS[dataset]

    manifest = pd.read_csv(manifest_path).drop(columns="processed", errors="ignore")  # stale flag, ignored
    latest = latest_log_rows(log_path)[["patient_id", "status", "reason", "mask_voxels"]]
    merged = manifest.merge(latest, on="patient_id", how="left")  # how="left" keeps unprocessed patients visible

    cohort = merged[merged["status"] == "ok"].reset_index(drop=True)
    excluded = merged[merged["status"] != "ok"]

    if verbose:
        print(f"{dataset} manifest: {len(manifest)} | usable: {len(cohort)} | excluded: {len(excluded)}")
        print(excluded[["patient_id", "status", "reason"]].to_string(index=False))
        if with_labels:
            print(cohort["idh"].value_counts())
            if "fold" in cohort.columns:   # only UTSW has saved folds
                print(cohort.groupby("fold")["idh"].value_counts().unstack())
    if expected_n is not None:
        assert len(cohort) == expected_n, f"expected {expected_n} patients, got {len(cohort)}"
    return cohort if with_labels else cohort.drop(columns="idh")
