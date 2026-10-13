"""Image-only CNN: how patients are loaded and what the network looks like."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from src.config import PROCESSED   # Data/processed, one <patient_id>.npz per patient


class GliomaCrops(Dataset):
    """One patient per item: a 4 x 128 x 128 x 128 image and the IDH label (1 = mutant).
    Only the "image" key is read from the .npz, so the tumor mask can never reach the model."""

    def __init__(self, patient_ids, labels):
        self.ids = list(patient_ids)
        self.labels = np.asarray(labels, dtype=np.float32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        with np.load(PROCESSED / f"{self.ids[i]}.npz") as f:
            image = f["image"]                       # .npz decompresses a key only when it is asked for
        return torch.from_numpy(image.astype(np.float32)), torch.tensor(self.labels[i])


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
    before the expensive layers. Output: one logit per patient (apply sigmoid for a probability)."""

    def __init__(self, in_channels=4, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=3, stride=2, padding=1, bias=False),  # 128 -> 64
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            conv_block(8, 16),    # 64 -> 32
            conv_block(16, 32),   # 32 -> 16
            conv_block(32, 64),   # 16 -> 8
            conv_block(64, 64),   # 8 -> 4
            nn.AdaptiveAvgPool3d(1),  # average each channel over the whole volume
            nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)
