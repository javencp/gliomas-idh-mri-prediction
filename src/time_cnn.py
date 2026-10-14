"""Time a few training steps on real patients to estimate cost on CPU or GPU.
Run from the repo root:  python -m src.time_cnn
Nothing is saved and no patient information is printed."""
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.cnn import GliomaCrops, SmallCNN
from src.cohort import build_cohort

N_PATIENTS, BATCH = 40, 4          # 10 steps; the first is a warm-up and is not timed
N_TRAIN = 420                      # about the training size per fold after holding out a validation set


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    workers = 2 if device == "cuda" else 0     # background processes that load the next batch while the GPU works
    torch.manual_seed(0)
    cohort = build_cohort(expected_n=617, verbose=False).head(N_PATIENTS)
    loader = DataLoader(GliomaCrops(cohort["patient_id"], cohort["idh"]), batch_size=BATCH,
                        shuffle=True, num_workers=workers, pin_memory=(device == "cuda"))

    model = SmallCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()           # unweighted, as in the project brief

    wait_for_data = compute = 0.0
    t = time.time()
    for step, (x, y) in enumerate(loader):
        now = time.time()
        if step > 0:
            wait_for_data += now - t                       # time spent waiting for the next batch
        x, y = x.to(device).float(), y.to(device)          # images travel as float16, become float32 here
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
        if device == "cuda":
            torch.cuda.synchronize()                       # GPU work is asynchronous; wait so timing is real
        t = time.time()
        if step > 0:
            compute += t - now                             # forward + backward + update

    n_timed = N_PATIENTS - BATCH
    per_patient = (wait_for_data + compute) / n_timed
    print(f"device: {device} | torch threads: {torch.get_num_threads()} | loader workers: {workers}")
    print(f"waiting for data: {wait_for_data / n_timed:.2f} s/patient | compute: {compute / n_timed:.2f} s/patient")
    print(f"estimated time per epoch ({N_TRAIN} patients): {per_patient * N_TRAIN / 60:.1f} min")


if __name__ == "__main__":      # required on Windows: loader workers re-import this file
    main()
