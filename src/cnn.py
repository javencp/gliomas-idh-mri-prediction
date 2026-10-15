"""Image-only CNN: how patients are loaded and what the network looks like."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from src.config import PROCESSED, PROCESSED_NPY   # compressed .npz crops; optional uncompressed .npy copy


class GliomaCrops(Dataset):
    """One patient per item: a 4 x 128 x 128 x 128 image (float16) and the IDH label (1 = mutant).
    Only the image is ever read, so the tumor mask can never reach the model.
    If an uncompressed copy exists (made by src.to_npy) it is used because it loads much faster;
    otherwise the compressed .npz is read. The caller converts to float32 on the device."""

    def __init__(self, patient_ids, labels):
        self.ids = list(patient_ids)
        self.labels = np.asarray(labels, dtype=np.float32)

    def __len__(self):
        return len(self.ids)

    def _load(self, patient_id):
        fast = PROCESSED_NPY / f"{patient_id}.npy"
        if fast.exists():
            return np.load(fast)
        with np.load(PROCESSED / f"{patient_id}.npz") as f:
            return f["image"]                        # .npz decompresses a key only when it is asked for

    def __getitem__(self, i):
        return torch.from_numpy(self._load(self.ids[i])), torch.tensor(self.labels[i])


def conv_block(c_in, c_out):
    """Conv -> batch norm -> ReLU -> halve the volume. Batch norm keeps training stable."""
    return nn.Sequential(
        nn.Conv3d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(2),
    )


class SmallCNN(nn.Module):
    """Small 3D CNN. The first conv has stride 2, which halves every side (8x fewer voxels)
    before the expensive layers. width multiplies every layer's channel count (1 = 8, 16, 32, 64, 64).
    Output: one logit per patient (apply sigmoid for a probability)."""

    def __init__(self, in_channels=4, dropout=0.3, width=1):
        super().__init__()
        c = [int(w * width) for w in (8, 16, 32, 64, 64)]
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, c[0], kernel_size=3, stride=2, padding=1, bias=False),  # 128 -> 64
            nn.BatchNorm3d(c[0]),
            nn.ReLU(inplace=True),
            conv_block(c[0], c[1]),   # 64 -> 32
            conv_block(c[1], c[2]),   # 32 -> 16
            conv_block(c[2], c[3]),   # 16 -> 8
            conv_block(c[3], c[4]),   # 8 -> 4
            nn.AdaptiveAvgPool3d(1),  # average each channel over the whole volume
            nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c[4], 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)
