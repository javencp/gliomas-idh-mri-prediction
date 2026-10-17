"""Evaluation shared by every model (clinical, image, fused, and later UCSF), so the numbers are
computed the same way everywhere. Definitions follow notebook 03."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             precision_score, recall_score, roc_auc_score)


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