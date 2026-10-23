"""Fit the five age-only logistic regressions on the saved UTSW folds and save them.

Why: the age-only model has no saved weights (oof_predictions.csv holds UTSW
out-of-fold probabilities, which cannot score UCSF patients). This script refits
one model per fold, exactly as in notebook 03, so that score_ucsf.py can average
their probabilities on UCSF.

Run from the repo root:  python -m src.fit_age_only
Uses UTSW only. UCSF is not touched.
"""
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.cohort import build_cohort
from src.config import OOF_PATH, RESULTS

AGE_DIR = RESULTS / "age_only"      # fold0.joblib t0 fold4.joblib + age_only_coefficients.csv
TOLERANCE = 1e-8                    # max allowed difference vs clinical_age_prob


def make_model():
    # Identical to notebook 03: the scaler sits inside the pipeline, so its mean and
    # SD come from the training folds only. No class weights, default regularization.
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def load_age_models():
    """Used by score_ucsf.py: the five fitted pipelines, in fold order."""
    return [joblib.load(AGE_DIR / f"fold{k}.joblib") for k in range(5)]


def main():
    # 1) Cohort: the 617 usable UTSW patients with their saved fold labels
    cohort = build_cohort(expected_n=617, verbose=False)
    y = cohort["idh"].to_numpy()
    folds = cohort["fold"].to_numpy()
    X = cohort[["age"]].astype(float).to_numpy()
    print(f"{len(y)} patients, {int(y.sum())} mutant (overall rate {y.mean():.4f})")

    # 2) One model per fold, trained on the other four folds
    models, rows = [], []
    held_out_prob = np.zeros(len(y))
    for k in sorted(np.unique(folds)):
        train, valid = folds != k, folds == k
        model = make_model().fit(X[train], y[train])
        held_out_prob[valid] = model.predict_proba(X[valid])[:, 1]
        scaler, logreg = model[0], model[1]
        rows.append({"fold": k, "n_train": int(train.sum()),
                     "train_mutant_rate": y[train].mean(),
                     "age_mean": scaler.mean_[0], "age_sd": scaler.scale_[0],
                     "coef_per_sd_age": logreg.coef_[0, 0],
                     "intercept": logreg.intercept_[0],
                     "sklearn_version": sklearn.__version__})
        models.append(model)

    # 3) Regression check BEFORE saving: held-out predictions must reproduce
    #    clinical_age_prob from notebook 03 (same data, same settings, same folds)
    saved = pd.read_csv(OOF_PATH).set_index("patient_id")["clinical_age_prob"]
    diff = np.abs(held_out_prob - saved.loc[cohort["patient_id"]].to_numpy()).max()
    print(f"max |refit - saved clinical_age_prob| = {diff:.2e}")
    if diff > TOLERANCE:
        raise SystemExit("Refit does not reproduce notebook 03; nothing saved. "
                         "Check the settings against notebook 03.")

    # 4) Save the models (joblib) and a human-readable coefficient table
    AGE_DIR.mkdir(parents=True, exist_ok=True)
    for k, model in zip(sorted(np.unique(folds)), models):
        joblib.dump(model, AGE_DIR / f"fold{k}.joblib")
    coefs = pd.DataFrame(rows)
    coefs.to_csv(AGE_DIR / "age_only_coefficients.csv", index=False)
    print(coefs.round(4).to_string(index=False))
    print(f"saved to {AGE_DIR}")


if __name__ == "__main__":
    main()
