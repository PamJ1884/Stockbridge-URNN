#!/usr/bin/env python3
"""Smoke test U-RNN Dynamic2DFlood loader with Stockbridge mini dataset.

Original U-RNN appears to support train/test splits only.
Validation split support will be added later as a Stockbridge-specific pipeline adaptation.
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

U_RNN_CODE = Path.home() / "flood/01_repositories/U-RNN/code"
if not U_RNN_CODE.exists():
    raise FileNotFoundError(f"Expected U-RNN code folder not found: {U_RNN_CODE}")
if str(U_RNN_CODE) not in sys.path:
    sys.path.insert(0, str(U_RNN_CODE))

from src.lib.dataset.Dynamic2DFlood import Dynamic2DFlood, preprocess_inputs


DATA_ROOT = Path.home() / "flood/02_data/CityCAT_Winchester/urnn/stockbridge_urnn_mini_v01"
LOCATION = "stockbridge"

EVENT_LISTS = {
    "train": DATA_ROOT / "stockbridge_train.txt",
    "test": DATA_ROOT / "stockbridge_test.txt",
}


def inspect_split(split: str):
    print(f"\n=== {split.upper()} ===")

    dataset = Dynamic2DFlood(
        data_root=str(DATA_ROOT),
        split=split,
        event_list_file=str(EVENT_LISTS[split]),
        duration=25,
        location=LOCATION,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    print("event list:", EVENT_LISTS[split])
    print("dataset length:", len(dataset))
    print("loader batches:", len(loader))

    batch = next(iter(loader))

    inputs = batch[0]
    label = batch[1]
    event_dir = batch[2]

    print("event_dir:", event_dir)
    print("input keys:", sorted(inputs.keys()))
    print("label shape:", tuple(label.shape))
    print("label dtype:", label.dtype)
    print("label min/max:", float(label.min()), float(label.max()))

    for key in [
        "absolute_DEM",
        "impervious",
        "manhole",
        "rainfall",
        "cumsum_rainfall",
        "max_DEM",
        "min_DEM",
    ]:
        value = inputs[key]
        if torch.is_tensor(value):
            print(
                f"{key:16s}",
                "shape=", tuple(value.shape),
                "dtype=", value.dtype,
                "min=", float(value.min()),
                "max=", float(value.max()),
            )
        else:
            print(key, type(value), value)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for t in [0, 1, 2, 24]:
        x = preprocess_inputs(
            t,
            inputs,
            device=device,
            nums=3,
            rain_max=60.0,
            cumsum_rain_max=250.0,
        )
        print(f"preprocess_inputs(t={t}) shape:", tuple(x.shape), "device:", x.device)


def main():
    print("Data root:", DATA_ROOT)
    print("CUDA available:", torch.cuda.is_available())

    for split in ["train", "test"]:
        inspect_split(split)

    print("\nTrain/test loader + preprocessing smoke test complete.")


if __name__ == "__main__":
    main()
