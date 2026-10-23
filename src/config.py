"""Project paths. Everything is relative to the repo root, so a fresh clone works anywhere."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA = PROJECT_ROOT / "Data"              # gitignored
RAW = DATA / "raw"                        # downloaded TCIA folders (temporary)
PROCESSED = DATA / "processed"            # one BT#### per patient
MANIFEST = DATA / "manifest.csv"          # full manifest (gitignored)


# results folder
RESULTS = PROJECT_ROOT / "results"       
SPLITS = RESULTS / "utsw_folds.csv"   # patient_id + fold only (committed)
LOG = RESULTS / "processing_log.csv"  # per-patient bbox/center/status (committed)
OOF_PATH = RESULTS / "oof_predictions.csv"             # out-of-fold probabilities, one column per model (committed)
CMP_PATH = RESULTS / "model_comparison.csv"            # one row of metrics per model (committed)

# metadata files
METADATA_TSV = DATA / "UTSW_Glioma_Metadata-2-1.tsv"     # path to the UTSW metadata .tsv you downloaded (gitignored)
UCSF_METADATA_CSV = DATA / "UCSF-PDGM-metadata_v5.csv"   # UCSF-PDGM metadata .csv (gitignored)

UCSF_CLEAN = DATA / "ucsf_clean.csv"                     # one row per UCSF patient: id, age, sex, idh (gitignored)

# Segmentations under this many voxels (1 mm voxels, so mm^3) are excluded from modeling, 
# added here to reduce the chance of drift
# Set from the volume distribution and visual review in notebook 02, before looking at IDH.
# Applied unchanged to UCSF. Strict inequality: volume < MIN_TUMOR_VOXELS is excluded.
MIN_TUMOR_VOXELS = 50

PROCESSED_NPY = DATA / "processed_npy"   # uncompressed copy for fast loading (gitignored with Data/)

# UCSF Held Out Test Set Paths
RAW_UCSF = DATA / "raw_ucsf"          # downloaded UCSF-PDGM folders (temporary)
PROCESSED_UCSF = DATA / "processed_ucsf"  # one UCSF-PDGM folder per patient
MANIFEST_UCSF = DATA / "manifest_ucsf.csv"
LOG_UCSF = RESULTS / "processing_log_ucsf.csv"