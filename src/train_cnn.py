"""Train the image-only CNN on the saved patient-level folds (UTSW only).

All settings are in the block below; change them there, commit, and push. Run from the repo root:
  Tuning (tries every pair in TUNING_GRID on one fold; the held-out fold is never predicted):
      python -m src.train_cnn --dev
  Final run (all folds, held-out predictions saved; re-running skips finished folds):
      python -m src.train_cnn
  Quick local check that the code runs (100 patients, 2 epochs, fold 0 only, separate folder):
      python -m src.train_cnn --smoke --dev        and        python -m src.train_cnn --smoke

For each held-out fold: the other folds' patients are split again, ~85% to train and ~15% only to decide
when to stop (the epoch with the lowest stop-set loss). The held-out fold is predicted once, after stopping.
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

from src.cnn import GliomaCrops, SmallCNN
from src.cohort import build_cohort
from src.config import PROJECT_ROOT

# ---------------------------------- settings ----------------------------------
LR = 1e-3            # learning rate used by the final run (set after tuning)
WEIGHT_DECAY = 1e-2  # weight decay used by the final run (set after tuning)
MAX_EPOCHS = 40
PATIENCE = 8         # stop after this many epochs without a better stop-set loss
BATCH = 4
VAL_FRACTION = 0.15  # share of each fold's training patients used only to choose when to stop
SEED = 42
DEV_FOLD = 0         # the one fold used for tuning
TUNING_GRID = [(1e-3, 1e-2), (1e-3, 1e-4), (3e-4, 1e-2), (3e-4, 1e-4)]   # (learning rate, weight decay)

# Results go to Drive when it is mounted (Colab), otherwise to results/cnn in the repo.
DRIVE_OUT = Path("/content/drive/MyDrive/glioma/cnn")
OUT = DRIVE_OUT if DRIVE_OUT.parent.parent.exists() else PROJECT_ROOT / "results" / "cnn"
# ------------------------------------------------------------------------------


def make_loader(cohort, idx, training, device):
    """Loader over the patients at positions idx. Training batches are shuffled; the last
    incomplete batch is dropped so batch norm never sees a batch of 1."""
    ds = GliomaCrops(cohort["patient_id"].iloc[idx], cohort["idh"].iloc[idx])
    workers = 2 if device == "cuda" else 0
    return DataLoader(ds, batch_size=BATCH if training else 8, shuffle=training, drop_last=training,
                      num_workers=workers, pin_memory=(device == "cuda"), persistent_workers=(workers > 0))


@torch.no_grad()
def predict(model, loader, device):
    """Probability of IDH mutant for every patient in the loader, in loader order."""
    model.eval()                                     # turns off dropout, batch norm uses its running stats
    logits = [model(x.to(device).float()).cpu() for x, _ in loader]
    return torch.sigmoid(torch.cat(logits)).numpy()


def run_fold(cohort, fold, lr, wd, max_epochs, final, out, device):
    y = cohort["idh"].to_numpy()
    folds = cohort["fold"].to_numpy()
    train_idx = np.where(folds != fold)[0]           # the other four folds
    test_idx = np.where(folds == fold)[0]            # held out: touched once, at the very end (final mode only)
    fit_idx, stop_idx = train_test_split(train_idx, test_size=VAL_FRACTION,
                                         stratify=y[train_idx], random_state=SEED)
    assert not set(train_idx) & set(test_idx), "patient appears in both training and held-out sets"

    torch.manual_seed(SEED + fold)                   # same start for the same fold in dev and final runs
    model = SmallCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss()                 # unweighted, as in the project brief
    fit_loader = make_loader(cohort, fit_idx, True, device)
    stop_loader = make_loader(cohort, stop_idx, False, device)
    y_stop = y[stop_idx]
    mode = "final" if final else "dev"

    best = {"loss": np.inf, "auroc": np.nan, "epoch": 0, "state": None}
    history, waited, t0 = [], 0, time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, yb in fit_loader:
            x, yb = x.to(device).float(), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
            n += len(yb)
        p = predict(model, stop_loader, device)
        stop_loss = log_loss(y_stop, p, labels=[0, 1])
        stop_auroc = roc_auc_score(y_stop, p)
        history.append({"mode": mode, "fold": fold, "lr": lr, "wd": wd, "epoch": epoch,
                        "train_loss": round(total / n, 4), "stop_loss": round(stop_loss, 4),
                        "stop_auroc": round(stop_auroc, 4)})
        print(f"fold {fold} lr {lr} wd {wd} epoch {epoch:2d} | train loss {total / n:.3f} | "
              f"stop-set loss {stop_loss:.3f} | stop-set AUROC {stop_auroc:.3f}")
        if stop_loss < best["loss"]:
            best.update(loss=stop_loss, auroc=stop_auroc, epoch=epoch, state=copy.deepcopy(model.state_dict()))
            waited = 0
        else:
            waited += 1
            if waited >= PATIENCE:
                break
    model.load_state_dict(best["state"])             # go back to the best epoch

    if final:                                        # the only place the held-out fold is ever predicted
        p_test = predict(model, make_loader(cohort, test_idx, False, device), device)
        pd.DataFrame({"patient_id": cohort["patient_id"].iloc[test_idx].to_numpy(), "fold": fold,
                      "idh": y[test_idx], "cnn_prob": p_test}).to_csv(out / f"fold{fold}_pred.csv", index=False)
        torch.save(model.state_dict(), out / f"fold{fold}.pt")

    append_csv(out / "history.csv", pd.DataFrame(history))      # per-epoch curves (training and stop set only)
    summary = {"mode": mode, "fold": fold, "lr": lr, "wd": wd, "best_epoch": best["epoch"],
               "epochs_run": epoch, "stop_set_loss": round(best["loss"], 4),
               "stop_set_auroc": round(best["auroc"], 4), "minutes": round((time.time() - t0) / 60, 1)}
    append_csv(out / "runs.csv", pd.DataFrame([summary]))       # one row per run: settings + stop-set scores
    print(summary)


def append_csv(path, df):
    """Add rows to a CSV (creating it with a header the first time). No patient information goes in these."""
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def main():
    ap = argparse.ArgumentParser(description="Train the image-only CNN with patient-level cross-validation.")
    ap.add_argument("--dev", action="store_true", help="tuning mode: runs TUNING_GRID on one fold, no held-out predictions")
    ap.add_argument("--smoke", action="store_true", help="quick code check: 100 patients, 2 epochs, fold 0, separate folder")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cohort = build_cohort(expected_n=617, verbose=False)
    out, max_epochs = OUT, MAX_EPOCHS
    if args.smoke:
        cohort = cohort.sample(100, random_state=0).reset_index(drop=True)
        out, max_epochs = OUT.parent / "cnn_smoke", 2
    out.mkdir(parents=True, exist_ok=True)
    print(f"device: {device} | patients: {len(cohort)} | mode: {'dev' if args.dev else 'final'} | output: {out}")

    if args.dev:
        for lr, wd in (TUNING_GRID[:1] if args.smoke else TUNING_GRID):
            run_fold(cohort, DEV_FOLD, lr, wd, max_epochs, False, out, device)
    else:
        for fold in ([DEV_FOLD] if args.smoke else sorted(cohort["fold"].unique())):
            if (out / f"fold{fold}_pred.csv").exists():
                print(f"fold {fold}: already done, skipping")      # lets a run resume after a Colab disconnect
                continue
            run_fold(cohort, fold, LR, WEIGHT_DECAY, max_epochs, True, out, device)


if __name__ == "__main__":      # required on Windows: loader workers re-import this file
    main()
