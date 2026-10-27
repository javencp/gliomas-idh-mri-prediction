"""Score the UCSF test set with the saved UTSW fold models. Label-blind: labels are never loaded here.

For each patient, every model is the mean of the probabilities of its five fold models
(age-only, image v2, fused), as fixed in the UCSF analysis plan in the project brief.
Labels are joined only in the evaluation notebook.

Run from the repo root:
    python -m src.score_ucsf --check        regression check on UTSW (check before touching UCSF)
    python -m src.score_ucsf --limit 10     timing run on 10 UCSF patients, saves nothing
    python -m src.score_ucsf                the real run, writes results/ucsf_predictions.csv
    python -m src.score_ucsf --overwrite    only to repeat a run; any rerun must be reported
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch

from src.cnn import FusedCNN, SmallCNN
from src.cohort import build_cohort
from src.config import OOF_PATH, PROCESSED, PROCESSED_NPY, PROCESSED_UCSF, RESULTS
from src.fit_age_only import load_age_models
from src.train_cnn import CONFIGS, FINAL_CONFIG

WIDTH = CONFIGS[FINAL_CONFIG]["width"]        # network width of the locked round 2 configuration
N_FOLDS = 5
IMAGE_DIR = RESULTS / "cnn_v2"                # fold{k}.pt = plain state dict of SmallCNN
FUSED_DIR = RESULTS / "cnn_fused"             # fold{k}.pt = dict: state, age_mean, age_sd, sex_code
OUT_PATH = RESULTS / "ucsf_predictions.csv"
CHECK_N = 20                                  # patients used in the regression check
CHECK_TOL = 1e-5                              # image models were trained on a GPU, so allow float noise


def load_crop(patient_id, npy_dir, npz_dir):
    """The 4 x 128 x 128 x 128 image of one patient (float16). Only the "image" array is read, never the mask."""
    if npy_dir is not None and (npy_dir / f"{patient_id}.npy").exists():
        return np.load(npy_dir / f"{patient_id}.npy")
    with np.load(npz_dir / f"{patient_id}.npz") as f:
        return f["image"]


def load_models():
    """The five SmallCNNs and the five FusedCNNs (with each fold's own age mean, age SD and sex dictionary)."""
    image, fused = [], []
    for k in range(N_FOLDS):
        m = SmallCNN(width=WIDTH)
        m.load_state_dict(torch.load(IMAGE_DIR / f"fold{k}.pt", map_location="cpu"))
        image.append(m.eval())                # eval: dropout off, batch norm uses its running statistics
        ckpt = torch.load(FUSED_DIR / f"fold{k}.pt", map_location="cpu")
        f = FusedCNN(n_tab=2, width=WIDTH)
        f.load_state_dict(ckpt["state"])
        fused.append({"model": f.eval(), "age_mean": ckpt["age_mean"], "age_sd": ckpt["age_sd"],
                      "sex_code": ckpt["sex_code"]})
    return image, fused


@torch.no_grad()
def score_patients(cohort, npy_dir, npz_dir, folds):
    """One probability per patient from each fold model in `folds`. Returns a table with
    patient_id and columns age_only_f{k}, image_v2_f{k}, fused_f{k}."""
    image_models, fused_models = load_models()
    age_models = load_age_models()
    age = cohort["age"].to_numpy(dtype=np.float64)
    sex = cohort["sex"].to_numpy()
    assert not np.isnan(age).any(), "missing age"
    for k in folds:                           # every sex value must be in each fold's own dictionary
        unknown = set(sex) - set(fused_models[k]["sex_code"])
        assert not unknown, f"sex values not in fold {k} dictionary: {unknown}"

    out = {"patient_id": cohort["patient_id"].to_numpy()}
    for k in folds:                           # age-only: logistic regression, no images needed
        out[f"age_only_f{k}"] = age_models[k].predict_proba(age[:, None])[:, 1]
        out[f"image_v2_f{k}"] = np.zeros(len(cohort))
        out[f"fused_f{k}"] = np.zeros(len(cohort))

    t0 = time.time()
    for i, pid in enumerate(out["patient_id"]):
        # one patient at a time: the crop is about 17 MB (float16), so memory is never a problem
        x = torch.from_numpy(load_crop(pid, npy_dir, npz_dir)).float().unsqueeze(0)
        for k in folds:
            out[f"image_v2_f{k}"][i] = torch.sigmoid(image_models[k](x)).item()
            fm = fused_models[k]
            tab = torch.tensor([[(age[i] - fm["age_mean"]) / fm["age_sd"], fm["sex_code"][sex[i]]]],
                               dtype=torch.float32)
            out[f"fused_f{k}"][i] = torch.sigmoid(fm["model"](x, tab)).item()
        if (i + 1) % 25 == 0 or i + 1 == len(cohort):
            print(f"{i + 1}/{len(cohort)} patients | {(time.time() - t0) / (i + 1):.2f} s per patient")
    return pd.DataFrame(out)


def run_check():
    """Regression check on UTSW: fold 0's models on fold 0's held-out patients must reproduce the
    predictions saved during training (image v2, fused) and in notebook 03 (age-only)."""
    cohort = build_cohort(expected_n=617, verbose=False).set_index("patient_id")
    ref_image = pd.read_csv(IMAGE_DIR / "fold0_pred.csv").head(CHECK_N).set_index("patient_id")["cnn_prob"]
    ref_fused = pd.read_csv(FUSED_DIR / "fold0_pred.csv").set_index("patient_id")["cnn_prob"]
    ref_age = pd.read_csv(OOF_PATH).set_index("patient_id")["clinical_age_prob"]
    ids = ref_image.index
    sub = cohort.loc[ids].reset_index()
    assert (sub["fold"] == 0).all(), "check patients must all be in fold 0"

    got = score_patients(sub, PROCESSED_NPY, PROCESSED, folds=[0]).set_index("patient_id")
    diffs = {"age_only": (got["age_only_f0"] - ref_age.loc[ids]).abs().max(),
             "image_v2": (got["image_v2_f0"] - ref_image).abs().max(),
             "fused": (got["fused_f0"] - ref_fused.loc[ids]).abs().max()}
    print(f"max absolute difference on {len(ids)} patients:", {k: f"{v:.1e}" for k, v in diffs.items()})
    if max(diffs.values()) > CHECK_TOL:
        raise SystemExit(f"Check failed (tolerance {CHECK_TOL}). Do not score UCSF until this is explained.")
    print("check passed")


def run_scoring(limit, overwrite):
    if limit is None and OUT_PATH.exists() and not overwrite:
        raise SystemExit(f"{OUT_PATH.name} already exists. UCSF is scored once; use --overwrite only to repeat "
                         "a run, and report the rerun.")
    cohort = build_cohort(dataset="ucsf", expected_n=495, with_labels=False, verbose=False)
    assert "idh" not in cohort.columns and cohort["patient_id"].is_unique
    missing = [p for p in cohort["patient_id"] if not (PROCESSED_UCSF / f"{p}.npz").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} crop files not found in {PROCESSED_UCSF}, for example {missing[:3]}")
    print(f"{len(cohort)} patients | age {cohort['age'].min():.0f} to {cohort['age'].max():.0f} | "
          f"sex {cohort['sex'].value_counts().to_dict()}")
    if limit:
        cohort = cohort.head(limit)

    folds = range(N_FOLDS)
    table = score_patients(cohort, None, PROCESSED_UCSF, folds)
    means = {}
    for name in ("age_only", "image_v2", "fused"):    # the plan: mean of the five fold probabilities
        means[f"{name}_prob"] = table[[f"{name}_f{k}" for k in folds]].mean(axis=1)
    table = pd.concat([table[["patient_id"]], pd.DataFrame(means), table.drop(columns="patient_id")], axis=1)

    values = table.drop(columns="patient_id").to_numpy()
    assert np.isfinite(values).all() and values.min() >= 0 and values.max() <= 1, "bad probabilities"
    print(table[list(means)].describe().loc[["mean", "min", "max"]].round(3))   # label-blind summary
    if limit:
        print(f"timing run on {limit} patients: nothing saved")
        return
    table.to_csv(OUT_PATH, index=False)
    print(f"saved {len(table)} rows to {OUT_PATH}")


def main():
    ap = argparse.ArgumentParser(description="Label-blind UCSF scoring with the five saved fold models per model.")
    ap.add_argument("--check", action="store_true", help="regression check on UTSW fold 0, no UCSF")
    ap.add_argument("--limit", type=int, help="timing run on the first N UCSF patients; saves nothing")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing an existing predictions file")
    args = ap.parse_args()
    if args.check:
        run_check()
    else:
        run_scoring(args.limit, args.overwrite)


if __name__ == "__main__":
    main()
