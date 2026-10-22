# Predicting IDH mutation status in gliomas from MRI

> A reproducible, cross-institution machine learning project.

> **Status: work in progress.** Model development on the training cohort (UTSW-Glioma) is complete. The external test on UCSF-PDGM has **not** been run, so no external result exists yet.
>
> **Not a clinical tool.** This is a research and learning project. Nothing here is validated for clinical use.

## Background

- **Glioma:** a tumor that arises from the supporting cells of the brain or spinal cord.
- **IDH (isocitrate dehydrogenase) mutation:** a change in the *IDH1* or *IDH2* genes. IDH is biomarker that is critical for diagnostic and treatment planning tool defined by the 2021 World Health Organization (WHO) tumor classification. Mutant and wildtype gliomas behave differently and are classed differently. IDH status is normally determined by testing tumor tissue after biopsy or surgery.
    - **IDH mutant**: accounts for 80% of lower-grade gliomas, and thus has better outcomes
    - **IDH Wildtype**: constitutes the majority of primary glioblastomas (grade IV gliomas), hence has poorer outcomes. 
- **Radiogenomics:** the scientific field that connects medical imaging (in this case, MRI )with a patient's genomic and/or molecular data, which would allow for a non-invasively method for disease analysis prior to surgery and more invasive approaches.

## Goal

1. Predict IDH status (mutant vs wildtype) in adult gliomas from preoperative MRI.
2. Test whether a model trained on one institution's data holds up on another's, which is the part that most often fails in practice.

## Data

Both datasets are public on [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/). Due to file size, the data are **not** included in this repository. Use is subject to the TCIA Data Usage Policy and Restrictions; required citations are under [Data citations](#data-citations).

| | UTSW-Glioma | UCSF-PDGM |
|---|---|---|
| Role | Training and development | External test (used once, at the end) |
| Patients | 625 listed, 617 usable (175 mutant, 28.4%) | 495 unique (103 mutant, 20.8%), not yet processed |
| Scanners | Multi-vendor, 0.3 T to 3 T | Single 3 T scanner |
| Contrasts used | T1, post-contrast T1, T2, FLAIR | Same four |

UTSW exclusions: 3 patients lack an IDH label, 3 have an empty tumor segmentation, and 2 have tumors under 50 voxels in the crop. UCSF-PDGM has 501 scans, of which 6 are follow-ups that are not used. Dataset sizes and scanner details are from the TCIA pages; patient counts and exclusions come from this project's own metadata and preprocessing checks.

## Approach

Three models, always reported together:

- **Clinical-only:** age and sex (baseline, simplest).
- **Image-only:** a 3D CNN, trained from scratch on tumor-centered crops of the four MRI contrasts.
- **Fused:** the same CNN with age and sex added, trained end to end.

Design choices to keep the evaluation honest:

- **Preprocessing:** per-scan intensity normalization, then a fixed 128 x 128 x 128 crop (1 mm voxels) centered on the tumor. The main purpose of cropping each scan was due to local storage constraints. For instance, the UTSW dataset is 22.9GB, after preprocessing the dataset became 4.03GB. The tumor segmentation only positions the crop and is never a model input.
- **Splits:** by patient, never by image or slice. Five stratified patient-level folds; every patient is scored by a model that never trained on their fold (out-of-fold).
- **No leakage from tuning:** the decision cutoff comes from training labels only. All tuning happens inside UTSW-Glioma. UCSF-PDGM is run once as a locked test set, and its labels are never used for modeling decisions.
- **Features:** only information available at imaging time (age and sex). Diagnosis, tumor type, WHO grade, 1p/19q, MGMT, survival, and extent of resection are excluded, since they are defined partly by IDH status or unknown before surgery.
- **Metrics:** AUROC (headline), PR-AUC next to its prevalence baseline, sensitivity, specificity and precision at the fixed cutoff, Brier score and calibration. Patient-level bootstrap 95% confidence intervals are reported on AUROC and PR-AUC. A model counts as an improvement only if the paired 95% CI of its AUROC difference excludes 0.
- **Cross-dataset differences:** the two datasets were preprocessed with different pipelines, so intensities are normalized per scan and orientation and shape are checked.

## Results so far (UTSW-Glioma, out-of-fold, 617 patients)

These are **development estimates**: the image-model settings were tuned on these patients, so the scores are somewhat optimistic. UCSF-PDGM is the only unbiased test.

| Model | AUROC (95% CI) |
|---|---|
| Age only | 0.88 (0.85 to 0.91) |
| Image only | 0.907 (0.880 to 0.930) |
| Fused (image + age + sex) | 0.922 (0.894 to 0.947) |

- Age alone is a strong baseline: mutant patients are much younger (median 38 vs 63 years).
- Fused minus age-only is +0.042 (paired 95% CI +0.009 to +0.071), the only primary comparison called an improvement. Image-only minus age-only (+0.027, CI -0.006 to +0.057) is not.
- Within age bands, the image models stay above age-only, which suggests the images carry information beyond age. It does not show what that information is.
- Scores did not differ clearly by scanner make, field strength or IDH testing method (descriptive; intervals about ±0.05).
- Grad-CAM heatmaps (which image regions the score responds to) are descriptive only. Most of their apparent concentration on the tumor comes from the crops being tumor-centered, so they do not show that the model "looks at the tumor".
- Intervals cover patient sampling only. Retraining variability was not measured (one run per model).

## Setup

Developed on Windows (PowerShell) with VS Code (Jupyter extension) and a Python virtual environment, on a CPU-only laptop. Preprocessing runs locally; model training runs on Google Colab GPUs.

### 1. Install

```powershell
git clone https://github.com/<your-username>/gliomas-idh-mri-prediction.git
cd gliomas-idh-mri-prediction
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` installs the repo in editable mode so notebooks can `import src...` from any working directory. If PowerShell blocks the activate script, run `Set-ExecutionPolicy -Scope Process RemoteSigned` first.

### 2. Download the data

Download the NIfTI files from each dataset's TCIA page with the **IBM Aspera desktop client**, following the page's instructions. The datasets are large (about 23 GB for UTSW-Glioma and 142 GB for UCSF-PDGM, per the TCIA pages), so download in chunks of about 100 patients and delete each chunk's raw files after processing (this is done in `src/process_chunk.py`). Paths are set in `src/config.py`; edit them to match your machine.

### 3. Preprocess locally

Run notebook 01 first to build the patient manifest, then, from the repository root:

```powershell
python -m src.process_chunk --delete-raw
```

`src/process_chunk.py` turns each downloaded patient into a small tumor-centered crop (`Data/processed`, one `.npz` per patient), appends a row per patient to `results/processing_log.csv`, and skips patients already done, so it is safe to re-run. `--delete-raw` removes each raw folder after its patient succeeds. It currently handles UTSW-Glioma only; UCSF-PDGM support is planned.

### 4. Train on Google Colab

Training uses the committed launcher notebook, which mounts Google Drive, clones the repo, unpacks the data and runs the training scripts (`src.train_cnn`, `src.train_fused`).

1. Pack the processed crops into one file (Windows 10+ includes `tar`):
```powershell
   tar -cf processed.tar -C Data processed
```
2. Upload `processed.tar` and `manifest.csv` to `MyDrive/glioma/` in Google Drive.
3. Open the launcher notebook in Colab, select a GPU runtime (a T4 was used), and run the setup cells, then the run cells.
4. Outputs are written to `MyDrive/glioma/<folder>`. Download them into `results/` and commit from local machine.

## Repository contents

```
src/         preprocessing, models, training scripts, metrics, config
notebooks/   analysis in order: data inspection, metadata, preprocessing QC,
             clinical baseline, image-only CNN, fused model, subgroup report,
             Grad-CAM; plus the Colab launcher
results/     committed result tables, predictions and trained fold weights
Data/        gitignored; raw and processed data live here locally
```

## Data citations

Copied from each dataset's TCIA "Citations & Data Usage Policy" section.

> **UTSW-Glioma (training):** Reddy, D., Saadat, N., Holcomb, J., Wagner, B., Truong, N., Bowerman, J., Hatanpaa, K., Patel, T., Pinho, M., Yu, F., Zhang, K., Lodhi, S., Madhuranthakam, A., Bangalore Yogananda, C. G., & Maldjian, J. (2026). The University of Texas Southwestern Glioma MRI dataset with molecular marker characterization and segmentations (UTSW-Glioma) (Version 1) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/DFAE-1B86

> **UCSF-PDGM (external test):** Calabrese, E., Villanueva-Meyer, J., Rudie, J., Rauschecker, A., Baid, U., Bakas, S., Cha, S., Mongan, J., Hess, C. (2022). The University of California San Francisco Preoperative Diffuse Glioma MRI (UCSF-PDGM) (Version 5) [dataset]. The Cancer Imaging Archive. https://doi.org/10.7937/tcia.bdgf-8v37

## References

To be added.