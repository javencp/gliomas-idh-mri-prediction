"""Evaluation shared by every model (clinical, image, fused, and later UCSF), so the numbers are
computed the same way everywhere. Definitions follow notebook 03. The last three functions
read and write the shared results CSVs.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             precision_score, recall_score, roc_auc_score)

from src.config import CMP_PATH, OOF_PATH

def threshold_predictions(prob, y, folds):
    """0/1 predictions from out-of-fold probabilities. Each fold's cutoff is the mutant rate of the
    patients in the OTHER folds (the training patients); it is never tuned."""
    pred = np.zeros(len(y), dtype=int)
    for k in np.unique(folds):
        pred[folds == k] = (prob[folds == k] >= y[folds != k].mean()).astype(int)
    return pred


def summarize(y, prob, pred):
    """Pooled metrics over all patients."""
    return {"auroc": roc_auc_score(y, prob), "pr_auc": average_precision_score(y, prob),
            "brier": brier_score_loss(y, prob), "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred), "recall": recall_score(y, pred),
            "specificity": recall_score(y, pred, pos_label=0)}


def bootstrap_ci(y, prob, metric, n_boot=1000, seed=42):
    """95% interval from resampling patients with replacement (covers patient sampling only)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = [metric(y[i], prob[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    return np.nanpercentile(stats, [2.5, 97.5])


def paired_difference(y, prob_a, prob_b, metric, n_boot=1000, seed=42):
    """metric(a) - metric(b) and its 95% interval. Both models are scored on the same resampled
    patients in every draw, so the interval reflects the difference, not each model separately."""
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        diffs.append(metric(y[i], prob_a[i]) - metric(y[i], prob_b[i]))
    return metric(y, prob_a) - metric(y, prob_b), np.nanpercentile(diffs, [2.5, 97.5])


def paired_auroc_difference(y, prob_a, prob_b, n_boot=1000, seed=42):
    """AUROC(a) - AUROC(b), the pre-specified model comparison."""
    return paired_difference(y, prob_a, prob_b, roc_auc_score, n_boot, seed)


def stitch_folds(run_dir, column, cohort):
    """Join the five per-fold prediction files, check them against the cohort, and add them to the
    shared out-of-fold file as `column`."""
    pred = pd.concat([pd.read_csv(run_dir / f"fold{k}_pred.csv") for k in range(5)], ignore_index=True)
    n = len(cohort)
    assert len(pred) == n and pred["patient_id"].is_unique
    check = pred.merge(cohort[["patient_id", "fold", "idh"]], on="patient_id", suffixes=("", "_manifest"))
    assert len(check) == n
    assert (check["fold"] == check["fold_manifest"]).all() and (check["idh"] == check["idh_manifest"]).all()
    assert pred["cnn_prob"].between(0, 1).all()
    oof = pd.read_csv(OOF_PATH).drop(columns=column, errors="ignore")
    oof = oof.merge(pred[["patient_id", "cnn_prob"]].rename(columns={"cnn_prob": column}),
                    on="patient_id", how="left")
    assert oof[column].notna().all()
    oof.to_csv(OOF_PATH, index=False)
    print(f"{column}: stitched {n} predictions; ids, folds and labels match the manifest")


def score_model(name, prob, y, folds, save=True):
    """Pooled metrics with bootstrap CIs and per-fold AUROC. With save=True the row goes to
    model_comparison.csv (replacing any row with the same name). Returns the 0/1 predictions."""
    pred = threshold_predictions(prob, y, folds)
    row = {"model": name, "n_patients": len(y), **summarize(y, prob, pred)}
    row["auroc_ci_low"], row["auroc_ci_high"] = bootstrap_ci(y, prob, roc_auc_score)
    row["pr_auc_ci_low"], row["pr_auc_ci_high"] = bootstrap_ci(y, prob, average_precision_score)
    row = pd.DataFrame([row]).round(3)
    if save:
        old = pd.read_csv(CMP_PATH)
        pd.concat([old[old["model"] != name], row], ignore_index=True).to_csv(CMP_PATH, index=False)
    print(row.set_index("model").T)
    print("per-fold AUROC:", np.round([roc_auc_score(y[folds == k], prob[folds == k]) for k in np.unique(folds)], 3))
    return pred


def compare(y, label_a, prob_a, label_b, prob_b, metric=roc_auc_score, metric_name="AUROC"):
    """Paired bootstrap of a metric difference between two models scored on the same patients."""
    diff, (lo, hi) = paired_difference(y, prob_a, prob_b, metric)
    print(f"{metric_name} {label_a} minus {label_b}: {diff:+.3f} (95% CI {lo:+.3f} to {hi:+.3f})")