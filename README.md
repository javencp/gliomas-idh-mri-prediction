# Predicting IDH mutation status in gliomas from MRI

> A reproducible, cross-institution machine learning project.

> **Status: analysis complete.** Models were developed on UTSW-Glioma and tested once on UCSF-PDGM; results can be found [Here](#results).
>
> **Reproduciblity**: steps for [Setup](#setup)
>
> **Not a clinical tool.** This is a research and learning project. One external institution is not clinical validation.

## Background

- **Glioma:** a tumor that arises from the supporting cells of the brain or spinal cord.
- **IDH (isocitrate dehydrogenase) mutation:** a change in the *IDH1* or *IDH2* genes. IDH status is a defining marker in the 2021 World Health Organization (WHO) classification of central nervous system tumors [1]. It is normally determined by testing tumor tissue after biopsy or surgery.
    - **IDH mutant:** IDH-mutant diffuse astrocytic tumors form a single type, "Astrocytoma, IDH-mutant", graded 2 to 4 [1].
    - **IDH wildtype:** IDH-wildtype diffuse astrocytic tumors in adults that meet specific histologic or molecular criteria are classed as "Glioblastoma, IDH-wildtype", grade 4 [1].
- **Radiogenomics:** the field that connects medical imaging (here, MRI) with a patient's genomic or molecular data, which could allow non-invasive disease analysis before surgery or other more invasive approaches.

## Goal

1. Predict IDH status (mutant vs wildtype) in adult gliomas from preoperative MRI.
2. Test whether a model trained on one institution's data holds up on another's, which is the part that most often fails in practice.

## Data

Both datasets are public on [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/). Due to file size, the data are **not** included in this repository. Use is subject to the TCIA Data Usage Policy and Restrictions; required citations are under [Data citations](#data-citations).

| | UTSW-Glioma | UCSF-PDGM |
|---|---|---|
| Role | Training and development | External test (scored once) |
| Usable patients | 617 (175 mutant, 28.4%) | 495 (103 mutant, 20.8%) |
| Scanners | Multi-vendor, 0.3 T to 3 T | Single 3 T scanner |
| Contrasts used | T1, post-contrast T1, T2, FLAIR | Same four |

UTSW-Glioma: 625 patients listed; 3 lack an IDH label, 3 have an empty tumor segmentation and 2 have under 50 tumor voxels in the crop, all excluded. UCSF-PDGM: 501 scans, of which 6 are follow-ups that are not used; no baseline patient was excluded. One UCSF patient is 17 years old (the UTSW minimum is 18), a minor deviation from "adult". Dataset sizes and scanner details are from the TCIA pages; patient counts and exclusions come from this project's own metadata and preprocessing checks.

## Approach

Three models, always reported together:

- **Clinical-only:** age and sex (baseline, simplest). Adding sex did not change the results, so the version scored on UCSF uses age only.
- **Image-only:** a 3D CNN, trained from scratch on tumor-centered crops of the four MRI contrasts.
- **Fused:** the same CNN with age and sex added, trained end to end.

Design choices:

- **Preprocessing:** per-scan intensity normalization, then a fixed 128 x 128 x 128 crop (1 mm voxels) centered on the tumor, with the same code and rules for both datasets. A tumor-focused crop keeps the model from seeing large amounts of normal brain, whose appearance tracks age. Together with float16 storage and compression, cropping also shrank UTSW-Glioma from 22.9 GB to 4.03 GB and shrank UCSF-PDGM from 142 GB to 3.36 GB, which made local processing on a laptop practical. The tumor segmentation only positions the crop and is never a model input.
- **Splits:** by patient, never by image or slice. Five stratified patient-level folds; every UTSW patient is scored by a model that never trained on their fold (out-of-fold).
- **No leakage from tuning:** the decision cutoff comes from training labels only (0.284, the UTSW mutant rate, for UCSF too). All tuning happens inside UTSW-Glioma. UCSF-PDGM labels are never used for modeling decisions.
- **Features:** only information available at imaging time (age and sex). Diagnosis, tumor type, WHO grade, 1p/19q, MGMT, survival, and extent of resection are excluded, since they are defined partly by IDH status or unknown before surgery.
- **Metrics:** AUROC (headline), PR-AUC next to its prevalence baseline, sensitivity/recall, specificity and precision at the fixed cutoff, Brier score and calibration. Patient-level bootstrap 95% confidence intervals are reported on AUROC and PR-AUC. A model counts as an improvement only if the paired 95% CI of its AUROC difference excludes 0; all other comparisons are exploratory.
- **UCSF analysis plan:** fixed before UCSF was scored. Each model's UCSF prediction is the mean of its five fold models, and no recalibration was applied. The scoring script never reads labels; labels are joined only in the UCSF evaluation notebook, and UCSF was scored once.
- **Cross-dataset differences:** the two datasets were preprocessed with different upstream pipelines, so intensities are normalized per scan and orientation and shape are checked.

## Results

UTSW-Glioma scores are out-of-fold over 617 patients and are **development estimates**: the image-model settings were tuned on these patients, so they are somewhat optimistic. UCSF-PDGM (495 patients) is the only unbiased test.

| Model | UTSW AUROC (95% CI) | UCSF AUROC (95% CI) | UCSF PR-AUC (prevalence 0.208) |
|---|---|---|---|
| Age only | 0.880 (0.853 to 0.909) | 0.905 (0.870 to 0.935) | 0.728 |
| Image only | 0.907 (0.880 to 0.930) | 0.919 (0.885 to 0.952) | 0.749 |
| Fused (image + age + sex) | 0.922 (0.894 to 0.947) | 0.940 (0.905 to 0.969) | 0.878 |

**Main Findings** 
- **Ranking held up on a new institution.** AUROC did not drop on UCSF. But age alone also scored higher there, so UCSF looks like an easier set. This does not show better generalization, and absolute levels should not be compared across datasets.
- **Age alone is a strong baseline.** Mutant patients are much younger (UTSW median 38 vs 63 years).
- **Fused beats age alone, modestly.** Fused minus age-only AUROC is +0.035 on UCSF (paired 95% CI 0.003 to 0.069) and +0.042 on UTSW (0.009 to 0.071). The lower ends are close to 0.
- **Images alone are not shown to beat age.** Image-only minus age-only is +0.014 on UCSF (-0.023 to 0.053) and +0.027 on UTSW (-0.006 to 0.057).

**Extended Findings** 
- **Beyond age (exploratory).** In the youngest age band, where age separates poorly, the image models stay well above age-only on both datasets (UCSF, up to 50 years: age 0.76, image 0.92, fused 0.94). This suggests the images carry information beyond age. It does not show what that information is.
- **Probabilities do not transfer.** At the fixed UTSW cutoff, all models over-call mutants on UCSF, and calibration curves sit below the diagonal in the lower and middle ranges, most clearly for the image-only model.
- **Scanner and testing method (UTSW).** No clear differences across scanner make, field strength or IDH testing method (descriptive; intervals about ±0.05).
- **Grad-CAM (UTSW, descriptive).** Heatmaps show which image regions the score responds to. Most of their apparent concentration on the tumor comes from the crops being tumor-centered, so they do not show that the model "looks at the tumor".

**Limits:** one external institution and 495 patients; UTSW scores are development estimates from single models, while UCSF scores are five-model averages; intervals cover patient sampling only, not retraining variability; the two datasets used different tumor segmentation tools; a few large tumors are partly cut off at the crop edge.

## Future work

- **Measure retraining noise.** Each model was trained once per fold, so seed-to-seed variation is unmeasured. Training extra seeds on UTSW would show how much of a small AUROC difference (such as fused vs image-only) is just retraining luck.
- **Test whether the image model depends on the tumor.** Occlusion (blank the tumor, rescore) would give a direct test that Grad-CAM cannot.
- **Post hoc recalibration on UCSF.** Probabilities did not transfer at the fixed cutoff. A separate exploratory analysis could try a simple prevalence offset or Platt scaling fitted inside UCSF with cross-validation. It would use UCSF labels, so it could not replace the locked results.
- **Reverse direction.** Train on UCSF and test on UTSW as a secondary check on generalization.

## Setup

Developed on Windows (PowerShell) with VS Code (Jupyter extension) and a Python virtual environment, on a CPU-only laptop. Preprocessing and scoring run locally; model training runs on Google Colab GPUs (optional).

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

Download the NIfTI files from each dataset's TCIA page with the **IBM Aspera desktop client**, following the page's instructions. The datasets are large (about 23 GB for UTSW-Glioma and 142 GB for UCSF-PDGM, per the TCIA pages), so download in chunks (ex. 100 patients for UTSW, 50 for UCSF) and delete each chunk's raw files after processing (`--delete-raw` in `src/process_chunk.py`). Paths are set in `src/config.py`; edit them to match your machine.

### 3. Preprocess locally

Build each dataset's patient manifest first (notebook 01 for UTSW, notebook 09 for UCSF), then, from the repository root:

```powershell
python -m src.process_chunk --dataset utsw --delete-raw
python -m src.process_chunk --dataset ucsf --delete-raw
```

`src/process_chunk.py` turns each downloaded patient into a small tumor-centered crop (one `.npz` per patient) and appends a row per patient to a per-dataset log in `results/`. It skips patients already done, so it is safe to re-run. `--dataset` is required, and `--delete-raw` removes a patient's raw folder only after that patient processed successfully.

### 4. Train on Google Colab (Optional)

Training uses the committed launcher notebook, which mounts Google Drive, clones the repo, unpacks the data and runs the training scripts (`src.train_cnn`, `src.train_fused`).

1. Pack the processed UTSW crops into one file (Windows 10+ includes `tar`):
   ```powershell
   tar -cf processed.tar -C Data processed
   ```
2. Upload `processed.tar` and `manifest.csv` to `MyDrive/glioma/` in Google Drive.
3. Open the launcher notebook in Colab, select a GPU runtime (a T4 was used), and run the setup cells, then the run cells.
4. Outputs are written to `MyDrive/glioma/<folder>`. Download them into `results/` and commit from your local machine.

### 5. Score UCSF-PDGM

The trained fold weights are committed in `results/`, so no retraining is needed:

```powershell
python -m src.fit_age_only
python -m src.score_ucsf
```

`src.fit_age_only` fits the five age-only models on the saved UTSW folds. `src.score_ucsf` is label-blind and writes `results/ucsf_predictions.csv`; it refuses to overwrite existing predictions unless you pass `--overwrite`. Labels are joined only in the UCSF evaluation notebook. UCSF was designed as a one-time test, so any rerun is a new run and should be reported as one.

## Repository contents

```
src/         preprocessing, models, training and scoring scripts, metrics, config
notebooks/   analysis in order: data inspection, metadata, preprocessing QC,
             clinical baseline, image-only CNN, fused model, subgroup report,
             Grad-CAM, UCSF inspection, manifest, QC and evaluation;
             plus the Colab launcher
results/     committed result tables, predictions and trained fold weights
Data/        gitignored; raw and processed data live here locally
```

## Data citations

Copied from each dataset's TCIA "Citations & Data Usage Policy" section.

> **UTSW-Glioma (training):** Reddy, D., Saadat, N., Holcomb, J., Wagner, B., Truong, N., Bowerman, J., Hatanpaa, K., Patel, T., Pinho, M., Yu, F., Zhang, K., Lodhi, S., Madhuranthakam, A., Bangalore Yogananda, C. G., & Maldjian, J. (2026). The University of Texas Southwestern Glioma MRI dataset with molecular marker characterization and segmentations (UTSW-Glioma) (Version 1) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/DFAE-1B86

> **UCSF-PDGM (external test):** Calabrese, E., Villanueva-Meyer, J., Rudie, J., Rauschecker, A., Baid, U., Bakas, S., Cha, S., Mongan, J., Hess, C. (2022). The University of California San Francisco Preoperative Diffuse Glioma MRI (UCSF-PDGM) (Version 5) [dataset]. The Cancer Imaging Archive. https://doi.org/10.7937/tcia.bdgf-8v37

## Additional References

1. Louis, D. N., Perry, A., Wesseling, P., Brat, D. J., Cree, I. A., Figarella-Branger, D., Hawkins, C., Ng, H. K., Pfister, S. M., Reifenberger, G., Soffietti, R., von Deimling, A., & Ellison, D. W. (2021). The 2021 WHO Classification of Tumors of the Central Nervous System: a summary. *Neuro-Oncology*, 23(8), 1231–1251. https://doi.org/10.1093/neuonc/noab106