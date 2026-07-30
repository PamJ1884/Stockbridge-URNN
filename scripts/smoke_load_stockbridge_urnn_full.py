#!/usr/bin/env python3
"""Diagnose loading the full Stockbridge dataset with the original U-RNN loader.

This script only loads one batch per split and preprocesses selected timesteps. It
does not train a model or modify the dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


U_RNN_CODE = Path.home() / "flood/01_repositories/U-RNN/code"
if not U_RNN_CODE.is_dir():
    raise FileNotFoundError(f"Expected original U-RNN code directory: {U_RNN_CODE}")
if str(U_RNN_CODE) not in sys.path:
    sys.path.insert(0, str(U_RNN_CODE))

from src.lib.dataset.Dynamic2DFlood import Dynamic2DFlood, preprocess_inputs


DATA_ROOT = (
    Path.home() / "flood/02_data/CityCAT_Winchester/urnn/stockbridge_urnn_full_v01"
)
LOCATION = "stockbridge"
DURATION = 25
SPLITS = ("train", "valid", "test")
REQUIRED_SPLITS = {"train", "test"}


def event_list_path(split: str) -> Path:
    return DATA_ROOT / f"stockbridge_{split}.txt"


def describe_event(event: object) -> str:
    """Format the path/name returned by the default DataLoader collator."""
    if isinstance(event, (list, tuple)) and len(event) == 1:
        event = event[0]
    event_text = str(event)
    return f"{event_text} (name={Path(event_text).name})"


def inspect_split(split: str, device: torch.device) -> None:
    """Construct one split, load its first batch, and preprocess three steps."""
    print(f"\n=== {split.upper()} ===")
    list_path = event_list_path(split)
    dataset = Dynamic2DFlood(
        data_root=str(DATA_ROOT),
        split=split,
        event_list_file=str(list_path),
        duration=DURATION,
        location=LOCATION,
    )
    print("dataset construction: succeeded")
    print("event list:", list_path)
    print("number of samples:", len(dataset))

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    inputs, label, event = next(iter(loader))

    print("first event:", describe_event(event))
    print(
        "label:",
        "shape=", tuple(label.shape),
        "min=", float(label.min()),
        "max=", float(label.max()),
    )
    for timestep in (0, 1, 24):
        prepared = preprocess_inputs(
            timestep,
            inputs,
            device=device,
            nums=3,
            rain_max=60.0,
            cumsum_rain_max=250.0,
        )
        print(
            f"preprocess_inputs(t={timestep}):",
            "shape=", tuple(prepared.shape),
            "min=", float(prepared.min()),
            "max=", float(prepared.max()),
        )


def main() -> int:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Data root:", DATA_ROOT)
    print("Original U-RNN code:", U_RNN_CODE)
    print("Preprocessing device:", device)

    failures: dict[str, Exception] = {}
    for split in SPLITS:
        try:
            inspect_split(split, device)
        except Exception as exc:
            failures[split] = exc
            print(f"{split} diagnostic: FAILED ({type(exc).__name__}: {exc})")
            if split == "valid":
                print(
                    "The original U-RNN Dynamic2DFlood loader may support only "
                    "train/test split names. The upstream loader was not patched; "
                    "continuing with the remaining splits."
                )

    required_failures = [split for split in SPLITS if split in REQUIRED_SPLITS and split in failures]
    if required_failures:
        print(
            "\nSmoke test failed because required split(s) did not work:",
            ", ".join(required_failures),
            file=sys.stderr,
        )
        return 1

    if "valid" in failures:
        print(
            "\nTrain and test succeeded. Valid was skipped because the original "
            "loader rejected or could not load it."
        )
    else:
        print("\nTrain, valid, and test loader diagnostics succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
