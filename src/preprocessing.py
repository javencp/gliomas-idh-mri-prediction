"""Per-patient preprocessing: z-score each contrast inside the brain, crop a fixed window
around the tumor, and stack the 4 contrasts into one array.

The same code must be used for UTSW (training) and UCSF (external test), so any
difference between the datasets is a real one and not a preprocessing artifact. 
Originally, functions were in Notebooks/00_inspect_one_patient.ipynb, 
but they are now in src/preprocessing.py so they can be imported by src/process_chunk.py. 
"""
from pathlib import Path

import nibabel as nib
import numpy as np

CROP = 96  # voxels per side; revisit after looking at the logged tumor bounding boxes
CONTRASTS = ["brain_t1", "brain_t1ce", "brain_t2", "brain_flair"]  # channel order: 0=T1, 1=T1ce, 2=T2, 3=FLAIR
SEG = "tumorseg_FeTS"  # only used to locate the tumor, never given to the model


class PreprocessError(Exception):
    """A patient that cannot be processed. The message is written to the log as the reason."""


def zscore_brain(volume):
    """Rescale one scan to mean 0, std 1 over brain voxels (nonzero); background stays 0."""
    normalized = np.zeros(volume.shape, dtype=np.float32)
    in_brain = volume != 0
    if not in_brain.any():
        raise PreprocessError("image has no nonzero voxels")
    brain_values = volume[in_brain]
    spread = brain_values.std() or 1.0  # avoids dividing by zero on a constant image
    normalized[in_brain] = (brain_values - brain_values.mean()) / spread
    return normalized


def crop_center(volume, center, size):
    """Cut a size^3 window around center; zero-pad where it leaves the volume."""
    cropped = np.zeros((size, size, size), dtype=volume.dtype)
    read_from, write_to = [], []

    # loop over each axis (x, y, z)
    for center_on_axis, axis_length in zip(center, volume.shape):
        window_start = int(center_on_axis) - size // 2
        window_end = window_start + size

        # clamp the read window to the volume's bounds; shift the write window if it overhangs
        read_start = max(window_start, 0)
        read_end = min(window_end, axis_length)
        read_from.append(slice(read_start, read_end))
        write_to.append(slice(read_start - window_start, read_end - window_start))

    cropped[tuple(write_to)] = volume[tuple(read_from)]  # copy the overlapping data into the window
    return cropped


def _load(case_dir, name):
    path = case_dir / f"{name}.nii.gz"
    if not path.exists():
        raise PreprocessError(f"missing file {path.name}")
    return nib.load(path)


def process_case(case_dir):
    """Return {'image': (4, CROP, CROP, CROP) float16, 'mask': (CROP, CROP, CROP) uint8,
    'bbox': tumor extent in voxels, 'center': tumor center in voxel coordinates,
    'tumor_in_brain': fraction of tumor voxels inside the brain mask}."""
    case_dir = Path(case_dir)
    seg_img = _load(case_dir, SEG)
    scans = [_load(case_dir, name) for name in CONTRASTS]

    # All files must share one grid, otherwise the tumor location would not match the images
    for name, scan in zip(CONTRASTS, scans):
        if scan.shape != seg_img.shape or not np.allclose(scan.affine, seg_img.affine):
            raise PreprocessError(f"{name} shape/affine differs from segmentation")

    tumor = np.asanyarray(seg_img.dataobj) > 0  # whole tumor: any label above 0
    tumor_coords = np.argwhere(tumor)
    if len(tumor_coords) == 0:
        raise PreprocessError("empty tumor segmentation")
    first_voxel, last_voxel = tumor_coords.min(axis=0), tumor_coords.max(axis=0)
    center = (first_voxel + last_voxel) / 2
    bbox = last_voxel - first_voxel + 1  # logged for every patient to choose CROP

    volumes = [scan.get_fdata(dtype=np.float32) for scan in scans]
    brain = volumes[0] != 0  # the plain contrasts share one mask in BT0001; T1 stands in for it
    tumor_in_brain = float((tumor & brain).sum() / tumor.sum())

    # Normalize on the whole brain before cropping, so the statistics don't depend on tumor size or position
    channels = [crop_center(zscore_brain(v), center, CROP) for v in volumes]
    image = np.stack(channels).astype(np.float16)  # float16 halves storage; cast to float32 for training
    if not np.isfinite(image).all():
        raise PreprocessError("non-finite values after normalization")

    return {"image": image,
            "mask": crop_center(tumor.astype(np.uint8), center, CROP),  # same center, so voxel-aligned
            "bbox": bbox, "center": center, "tumor_in_brain": tumor_in_brain}