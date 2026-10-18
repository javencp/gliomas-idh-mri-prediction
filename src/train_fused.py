"""Train the fused CNN (image + age + sex) on the saved patient-level folds, UTSW only.

Same recipe as the locked round 2 image-only model: the settings are imported from train_cnn.py,
so they cannot drift. No tuning here (there is no --dev mode). Run from the repo root:
    python -m src.train_fused
    python -m src.train_fused --smoke      (quick local check: 100 patients, 2 epochs, fold 0)
Seeds 43 and 44 are only for measuring retraining noise; seed 42 is the primary result.
"""
import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from src.cnn import FusedCNN, GliomaCrops
from src.cohort import build_cohort
from src.config import PROJECT_ROOT
from src.train_cnn import CONFIGS, FINAL_CONFIG, PATIENCE, SPLIT_SEED, VAL_FRACTION, append_csv

# ---------------------------------- settings ----------------------------------
CFG = CONFIGS[FINAL_CONFIG]                  # locked round 2 settings (batch 8, lr 1e-3, wd 1e-2)
SEED = 42                                    # primary run; 43 and 44 only for retraining-noise runs
SEX_CODE = {"M": 0, "F": 1}                 # sex is categorical, but the model takes a float, so code it 0/1
OUT_NAME = "cnn_fused" if SEED == 42 else f"cnn_fused_s{SEED}"

DRIVE_OUT = Path("/content/drive/MyDrive/glioma") / OUT_NAME
OUT = DRIVE_OUT if DRIVE_OUT.parent.parent.exists() else PROJECT_ROOT / "results" / OUT_NAME
# --------------------------------------------------------------------


def make_tabular(cohort, train_idx):
    """Tabular inputs for every patient: [scaled age, sex coded 0/1], shape (n, 2).
    Age is z-scored with the mean and SD of this fold's training patients only, so nothing
    from the held-out fold influences the scaling. Returns the array and the mean and SD used
    (they are saved with the model, because UCSF scoring needs the same scaling)."""
    age = cohort["age"].to_numpy(dtype=np.float64)
    sex = cohort["sex"].map(SEX_CODE)
    assert not np.isnan(age).any(), "missing age"
    assert sex.notna().all(), "a sex value is not in SEX_CODE"
    mean, sd = age[train_idx].mean(), age[train_idx].std()
    tab = np.column_stack([(age - mean) / sd, sex.to_numpy()]).astype(np.float32)
    return tab, float(mean), float(sd)


def make_loader(cohort, tab, idx, training, device, batch):
    """Loader over the patients at positions idx, with their tabular rows. Training batches are
    shuffled and the last incomplete batch is dropped so batch norm never sees a batch of 1."""
    ds = GliomaCrops(cohort["patient_id"].iloc[idx], cohort["idh"].iloc[idx], tabular=tab[idx])
    workers = 2 if device == "cuda" else 0
    return DataLoader(ds, batch_size=batch if training else 8, shuffle=training, drop_last=training,
                      num_workers=workers, pin_memory=(device == "cuda"), persistent_workers=(workers > 0))


@torch.no_grad()
def predict(model, loader, device):
    """Probability of IDH mutant for every patient in the loader, in loader order."""
    model.eval()                                     # dropout off, batch norm uses its running stats
    logits = [model(*[t.to(device).float() for t in batch[:-1]]).cpu() for batch in loader]
    return torch.sigmoid(torch.cat(logits)).numpy()


def run_fold(cohort, fold, cfg, seed, out, device):
    y = cohort["idh"].to_numpy()
    folds = cohort["fold"].to_numpy()
    train_idx = np.where(folds != fold)[0]           # the other four folds
    test_idx = np.where(folds == fold)[0]            # held out: predicted once, after stopping
    fit_idx, stop_idx = train_test_split(train_idx, test_size=VAL_FRACTION,
                                         stratify=y[train_idx], random_state=SPLIT_SEED)
    assert not set(train_idx) & set(test_idx), "patient appears in both training and held-out sets"
    assert not cfg["cosine"], "the locked config has no scheduler, so none is built here"

    tab, age_mean, age_sd = make_tabular(cohort, train_idx)   # scaling from training patients only

    torch.manual_seed(seed + fold)                   # initial weights and shuffle order
    model = FusedCNN(n_tab=2, width=cfg["width"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    loss_fn = nn.BCEWithLogitsLoss()                 # unweighted, as in the project brief
    fit_loader = make_loader(cohort, tab, fit_idx, True, device, cfg["batch"])
    stop_loader = make_loader(cohort, tab, stop_idx, False, device, cfg["batch"])
    y_stop = y[stop_idx]

    best = {"loss": np.inf, "auroc": np.nan, "epoch": 0, "state": None}
    history, waited, t0 = [], 0, time.time()
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        total, n = 0.0, 0
        for batch in fit_loader:
            *inputs, yb = batch                      # inputs = [image, tabular]; the label is always last
            inputs = [t.to(device).float() for t in inputs]
            yb = yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(*inputs), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
            n += len(yb)
        p = predict(model, stop_loader, device)
        stop_loss = log_loss(y_stop, p, labels=[0, 1])
        stop_auroc = roc_auc_score(y_stop, p)
        history.append({"config": FINAL_CONFIG, "fold": fold, "seed": seed, "epoch": epoch,
                        "train_loss": round(total / n, 4), "stop_loss": round(stop_loss, 4),
                        "stop_auroc": round(stop_auroc, 4)})
        print(f"fused fold {fold} seed {seed} epoch {epoch:2d} | train loss {total / n:.3f} | "
              f"stop-set loss {stop_loss:.3f} | stop-set AUROC {stop_auroc:.3f}")
        if stop_loss < best["loss"]:
            best.update(loss=stop_loss, auroc=stop_auroc, epoch=epoch, state=copy.deepcopy(model.state_dict()))
            waited = 0
        else:
            waited += 1
            if waited >= PATIENCE:
                break
    model.load_state_dict(best["state"])             # go back to the best epoch

    # the only place the held-out fold is ever predicted
    p_test = predict(model, make_loader(cohort, tab, test_idx, False, device, cfg["batch"]), device)
    pd.DataFrame({"patient_id": cohort["patient_id"].iloc[test_idx].to_numpy(), "fold": fold,
                  "idh": y[test_idx], "cnn_prob": p_test}).to_csv(out / f"fold{fold}_pred.csv", index=False)
    torch.save({"state": model.state_dict(), "age_mean": age_mean, "age_sd": age_sd,
                "sex_code": SEX_CODE}, out / f"fold{fold}.pt")

    append_csv(out / "history.csv", pd.DataFrame(history))
    summary = {"model": "fused", "config": FINAL_CONFIG, "fold": fold, "seed": seed, "batch": cfg["batch"],
               "lr": cfg["lr"], "wd": cfg["wd"], "width": cfg["width"], "best_epoch": best["epoch"],
               "epochs_run": epoch, "stop_set_loss": round(best["loss"], 4),
               "stop_set_auroc": round(best["auroc"], 4), "minutes": round((time.time() - t0) / 60, 1)}
    append_csv(out / "runs.csv", pd.DataFrame([summary]))
    print(summary)


def main():
    ap = argparse.ArgumentParser(description="Train the fused CNN (image + age + sex) with patient-level cross-validation.")
    ap.add_argument("--smoke", action="store_true", help="quick code check: 100 patients, 2 epochs, fold 0, separate folder")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cohort = build_cohort(expected_n=617, verbose=False)
    out, cfg, folds = OUT, CFG, sorted(cohort["fold"].unique())
    if args.smoke:
        cohort = cohort.sample(100, random_state=0).reset_index(drop=True)
        out, cfg, folds = OUT.parent / "cnn_smoke_fused", {**CFG, "max_epochs": 2}, [0]
    out.mkdir(parents=True, exist_ok=True)
    print(f"device: {device} | patients: {len(cohort)} | seed: {SEED} | output: {out}")
    print(f"config {FINAL_CONFIG}: {cfg}")      # confirms the settings imported from train_cnn.py

    for fold in folds:
        if (out / f"fold{fold}_pred.csv").exists():
            print(f"fold {fold}: already done, skipping")      # lets a run resume after a Colab disconnect
            continue
        run_fold(cohort, fold, cfg, SEED, out, device)


if __name__ == "__main__":      # required on Windows: loader workers re-import this file
    main()