"""Train the image-only CNN on the saved patient-level folds (UTSW only).

All settings are in the block below; change them there, commit, and push. Run from the repo root:
  Tuning (every configuration in CONFIGS on folds DEV_FOLDS; held-out folds are never predicted;
  re-running skips runs already logged, so it resumes after a Colab disconnect):
      python -m src.train_cnn --dev
  Final run (all folds with the FINAL_CONFIG settings; held-out predictions saved; skips finished folds):
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
# One configuration = one full set of training choices. "baseline" is the Round 1 setup.
BASE = {"batch": 4, "lr": 1e-3, "wd": 1e-2, "width": 1, "cosine": False, "max_epochs": 40}
BATCHES = (4, 8, 16)
LRS = (1e-3, 3e-4)
WDS = (1e-2,)      # add 1e-4 to bring back Round 1's other weight-decay value (doubles the number of runs)
CONFIGS = {}
for batch in BATCHES:
    for lr in LRS:
        for wd in WDS:
            name = "baseline" if (batch, lr, wd) == (4, 1e-3, 1e-2) else f"b{batch}_lr{lr:g}_wd{wd:g}"
            CONFIGS[name] = {**BASE, "batch": batch, "lr": lr, "wd": wd}
CONFIGS["cosine"] = {**BASE, "cosine": True, "max_epochs": 25}   # learning rate decays to ~0 over these 25 epochs
CONFIGS["width2"] = {**BASE, "width": 2}                         # twice the channels in every layer

FINAL_CONFIG = "baseline"          # configuration used by the final run (set after tuning)
DEV_FOLDS = [0, 1]                 # folds used for tuning
BASELINE_SEEDS = [42, 43, 44]      # extra seeds run only for the baseline, to measure run-to-run noise
SEED = 42                          # training seed for every other run
SPLIT_SEED = 42                    # fixed, so every configuration sees the same stop-set patients
PATIENCE = 8                       # stop after this many epochs without a better stop-set loss
VAL_FRACTION = 0.15                # share of each fold's training patients used only to choose when to stop
OUT_NAME = "cnn_v2"                # Round 1 results stay untouched in results/cnn

# Results go to Drive when it is mounted (Colab), otherwise to results/<OUT_NAME> in the repo.
DRIVE_OUT = Path("/content/drive/MyDrive/glioma") / OUT_NAME
OUT = DRIVE_OUT if DRIVE_OUT.parent.parent.exists() else PROJECT_ROOT / "results" / OUT_NAME
# ------------------------------------------------------------------------------


def make_loader(cohort, idx, training, device, batch):
    """Loader over the patients at positions idx. Training batches are shuffled; the last
    incomplete batch is dropped so batch norm never sees a batch of 1."""
    ds = GliomaCrops(cohort["patient_id"].iloc[idx], cohort["idh"].iloc[idx])
    workers = 2 if device == "cuda" else 0
    return DataLoader(ds, batch_size=batch if training else 8, shuffle=training, drop_last=training,
                      num_workers=workers, pin_memory=(device == "cuda"), persistent_workers=(workers > 0))


@torch.no_grad()
def predict(model, loader, device):
    """Probability of IDH mutant for every patient in the loader, in loader order."""
    model.eval()                                     # turns off dropout, batch norm uses its running stats
    logits = [model(x.to(device).float()).cpu() for x, _ in loader]
    return torch.sigmoid(torch.cat(logits)).numpy()


def run_fold(cohort, fold, name, cfg, seed, final, out, device):
    y = cohort["idh"].to_numpy()
    folds = cohort["fold"].to_numpy()
    train_idx = np.where(folds != fold)[0]           # the other four folds
    test_idx = np.where(folds == fold)[0]            # held out: touched once, at the very end (final mode only)
    fit_idx, stop_idx = train_test_split(train_idx, test_size=VAL_FRACTION,
                                         stratify=y[train_idx], random_state=SPLIT_SEED)
    assert not set(train_idx) & set(test_idx), "patient appears in both training and held-out sets"

    torch.manual_seed(seed + fold)                   # initial weights and shuffle order
    model = SmallCNN(width=cfg["width"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["max_epochs"]) if cfg["cosine"] else None
    loss_fn = nn.BCEWithLogitsLoss()                 # unweighted, as in the project brief
    fit_loader = make_loader(cohort, fit_idx, True, device, cfg["batch"])
    stop_loader = make_loader(cohort, stop_idx, False, device, cfg["batch"])
    y_stop = y[stop_idx]
    mode = "final" if final else "dev"

    best = {"loss": np.inf, "auroc": np.nan, "epoch": 0, "state": None}
    history, waited, t0 = [], 0, time.time()
    for epoch in range(1, cfg["max_epochs"] + 1):
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
        if sched is not None:
            sched.step()
        p = predict(model, stop_loader, device)
        stop_loss = log_loss(y_stop, p, labels=[0, 1])
        stop_auroc = roc_auc_score(y_stop, p)
        history.append({"mode": mode, "config": name, "fold": fold, "seed": seed, "epoch": epoch,
                        "train_loss": round(total / n, 4), "stop_loss": round(stop_loss, 4),
                        "stop_auroc": round(stop_auroc, 4)})
        print(f"{name} fold {fold} seed {seed} epoch {epoch:2d} | train loss {total / n:.3f} | "
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
        p_test = predict(model, make_loader(cohort, test_idx, False, device, cfg["batch"]), device)
        pd.DataFrame({"patient_id": cohort["patient_id"].iloc[test_idx].to_numpy(), "fold": fold,
                      "idh": y[test_idx], "cnn_prob": p_test}).to_csv(out / f"fold{fold}_pred.csv", index=False)
        torch.save(model.state_dict(), out / f"fold{fold}.pt")

    append_csv(out / "history.csv", pd.DataFrame(history))      # per-epoch curves (training and stop set only)
    summary = {"mode": mode, "config": name, "fold": fold, "seed": seed, "batch": cfg["batch"], "lr": cfg["lr"],
               "wd": cfg["wd"], "width": cfg["width"], "cosine": cfg["cosine"], "best_epoch": best["epoch"],
               "epochs_run": epoch, "stop_set_loss": round(best["loss"], 4),
               "stop_set_auroc": round(best["auroc"], 4), "minutes": round((time.time() - t0) / 60, 1)}
    append_csv(out / "runs.csv", pd.DataFrame([summary]))       # one row per run: settings + stop-set scores
    print(summary)


def append_csv(path, df):
    """Add rows to a CSV (creating it with a header the first time). No patient information goes in these."""
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def dev_run_done(out, name, fold, seed):
    """True if this tuning run is already in runs.csv (lets a re-run resume after a disconnect)."""
    path = out / "runs.csv"
    if not path.exists():
        return False
    r = pd.read_csv(path)
    return bool(((r["mode"] == "dev") & (r["config"] == name) & (r["fold"] == fold) & (r["seed"] == seed)).any())


def main():
    ap = argparse.ArgumentParser(description="Train the image-only CNN with patient-level cross-validation.")
    ap.add_argument("--dev", action="store_true", help="tuning mode: runs CONFIGS on DEV_FOLDS, no held-out predictions")
    ap.add_argument("--smoke", action="store_true", help="quick code check: 100 patients, 2 epochs, fold 0, separate folder")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cohort = build_cohort(expected_n=617, verbose=False)
    out, configs, dev_folds = OUT, CONFIGS, DEV_FOLDS
    if args.smoke:
        cohort = cohort.sample(100, random_state=0).reset_index(drop=True)
        out, dev_folds = OUT.parent / "cnn_smoke", [0]
        configs = {k: {**v, "max_epochs": 2} for k, v in CONFIGS.items()}
    out.mkdir(parents=True, exist_ok=True)
    print(f"device: {device} | patients: {len(cohort)} | mode: {'dev' if args.dev else 'final'} | output: {out}")

    if args.dev:
        for name, cfg in (list(configs.items())[:1] if args.smoke else configs.items()):
            for seed in ([SEED] if args.smoke or name != "baseline" else BASELINE_SEEDS):
                for fold in dev_folds:
                    if dev_run_done(out, name, fold, seed):
                        print(f"{name} fold {fold} seed {seed}: already done, skipping")
                        continue
                    run_fold(cohort, fold, name, cfg, seed, False, out, device)
    else:
        for fold in ([0] if args.smoke else sorted(cohort["fold"].unique())):
            if (out / f"fold{fold}_pred.csv").exists():
                print(f"fold {fold}: already done, skipping")      # lets a run resume after a Colab disconnect
                continue
            run_fold(cohort, fold, FINAL_CONFIG, configs[FINAL_CONFIG], SEED, True, out, device)


if __name__ == "__main__":      # required on Windows: loader workers re-import this file
    main()
