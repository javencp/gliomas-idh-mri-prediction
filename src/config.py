"""Project paths. Everything is relative to the repo root, so a fresh clone works anywhere."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA = PROJECT_ROOT / "Data"              # gitignored
RAW = DATA / "raw"                        # downloaded TCIA folders (temporary)
PROCESSED = DATA / "processed"            # one BT#### per patient
MANIFEST = DATA / "manifest.csv"          # full manifest (gitignored)

SPLITS = PROJECT_ROOT / "results" / "utsw_folds.csv"   # patient_id + fold only (committed)
LOG = PROJECT_ROOT / "results" / "processing_log.csv"  # per-patient bbox/center/status (committed)
METADATA_TSV = DATA / "UTSW_Glioma_Metadata-2-1.tsv"     # path to the UTSW metadata .tsv you downloaded (gitignored)
UCSF_METADATA_CSV = DATA / "UCSF-PDGM-metadata_v5.csv"   # UCSF-PDGM metadata .csv (gitignored)

UCSF_CLEAN = DATA / "ucsf_clean.csv"                     # one row per UCSF patient: id, age, sex, idh (gitignored)

# Segmentations under this many voxels (1 mm voxels, so mm^3) are excluded from modeling, 
# added here to reduce the chance of drift
# Set from the volume distribution and visual review in notebook 02, before looking at IDH.
# Applied unchanged to UCSF. Strict inequality: volume < MIN_TUMOR_VOXELS is excluded.
MIN_TUMOR_VOXELS = 50