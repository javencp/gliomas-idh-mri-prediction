# Predicting IDH mutation status in gliomas from MRI

> **Status: work in progress.** The repository currently contains the project configuration and a first inspection of a single patient's images. No models have been trained and there are no results yet.
>
> **Not a clinical tool.** This is a research and learning project. Nothing here is validated for clinical use.

## Background in brief

- **Glioma:** a tumor that arises from the supporting cells of the brain or spinal cord.
- **IDH mutation:** a change in the *IDH1* or *IDH2* genes. Mutant ("mutated") and wildtype (unmutated) gliomas behave differently and are classed differently in the WHO tumor classification. IDH status is normally determined by testing tumor tissue after biopsy or surgery.
- **Radiogenomics:** the idea that image appearance (here, MRI) carries signal about a tumor's genetic makeup, which would make it possible to estimate molecular status non-invasively and before surgery.

## Goal
> A reproducible, cross-institution machine learning project. 

1. Predict IDH status (mutant vs wildtype) in adult gliomas from preoperative MRI.
2. Test whether a model trained on one institution's data holds up on another's, which is the part that most often fails in practice.

## Data

Both datasets are public on [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/). The data are **not** included in this repository.

| | UTSW-Glioma | UCSF-PDGM |
|---|---|---|
| Role | Training and validation | External test (used once, at the end) |
| Patients | 625 (622 with an IDH label) | 495 unique (501 scans; 6 are follow-ups) |
| Scanners | Multi-vendor, 0.3 T to 3 T | Single 3 T scanner |
| Contrasts used | T1, post-contrast T1, T2, FLAIR | Same four |

Patient counts and scanner details are from the TCIA dataset pages, except the 622 figure, which comes from this project's own metadata check. UCSF-PDGM has not yet been checked against its metadata files. Please see the TCIA pages for each dataset's terms of use and citation requirements.

 
## Data citations
 
Copied from each dataset's TCIA "Citations & Data Usage Policy" section.
 
> **UTSW-Glioma (training):** Reddy, D., Saadat, N., Holcomb, J., Wagner, B., Truong, N., Bowerman, J., Hatanpaa, K., Patel, T., Pinho, M., Yu, F., Zhang, K., Lodhi, S., Madhuranthakam, A., Bangalore Yogananda, C. G., & Maldjian, J. (2026). The University of Texas Southwestern Glioma MRI dataset with molecular marker characterization and segmentations (UTSW-Glioma) (Version 1) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/DFAE-1B86

> **UCSF-PDGM (external test):** Calabrese, E., Villanueva-Meyer, J., Rudie, J., Rauschecker, A., Baid, U., Bakas, S., Cha, S., Mongan, J., Hess, C. (2022). The University of California San Francisco Preoperative Diffuse Glioma MRI (UCSF-PDGM) (Version 5) [dataset]. The Cancer Imaging Archive. https://doi.org/10.7937/tcia.bdgf-8v37

## Approach

Three models:

- **Clinical-only:** age and sex. (Baseline, Simplest)
- **Image-only:** a 3D CNN on tumor-centered crops of the four MRI contrasts.
- **Fused:** image and clinical features combined.

Design choices to keep the evaluation honest:

- All splits are by patient, never by image or slice.
- All tuning and threshold selection happens inside UTSW-Glioma. UCSF-PDGM is run once as a locked test set.
- Only information available at imaging time is used as a feature (age and sex). Diagnosis, tumor type, WHO grade, 1p/19q, MGMT, survival, and extent of resection are excluded, since several are defined partly by IDH status or are unknown before surgery.
- Reported metrics: AUROC (headline), AUPRC against prevalence, sensitivity and specificity at a fixed threshold. Other potential metrics: PPV/NPV, balanced accuracy, Brier score and calibration, with patient-level bootstrap confidence intervals.
- Because the two datasets were preprocessed with different pipelines, intensities are normalized per scan and orientation and shape are checked before training.

## Setup

Installation and run instructions will be added once there is something to run.

## References

To be added. Entries will be included only after they have been checked against the source.