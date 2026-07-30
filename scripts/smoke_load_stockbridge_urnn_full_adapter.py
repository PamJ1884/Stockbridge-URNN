#!/usr/bin/env python3
"""Smoke test all full Stockbridge splits through the local U-RNN adapter.

This diagnostic loads and preprocesses one batch per split. It does not train a
model or modify the converted dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
U_RNN_CODE = Path.home() / "flood/01_repositories/U-RNN/code"
for code_path in (REPO_ROOT, U_RNN_CODE):
    if not code_path.is_dir():
        raise FileNotFoundError(f"Expected code directory not found: {code_path}")
    if str(code_path) not in sys.path:
        sys.path.insert(0, str(code_path))

from src.lib.dataset.Dynamic2DFlood import preprocess_inputs
from stockbridge_urnn.dataset import StockbridgeDynamic2DFlood


DATA_ROOT = (
    Path.home() / "flood/02_data/CityCAT_Winchester/urnn/stockbridge_urnn_full_v01"
)
SPLITS = ("train", "valid", "test")


def inspect_split(split: str, device: torch.device) -> None:
    print(f"\n=== {split.upper()} ===")
    dataset = StockbridgeDynamic2DFlood(DATA_ROOT, split=split, duration=25)
    print("dataset length:", len(dataset))

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    inputs, label, event_path = next(iter(loader))
    if isinstance(event_path, (list, tuple)) and len(event_path) == 1:
        event_path = event_path[0]

    print("event path:", event_path)
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
    print("Repository root:", REPO_ROOT)
    print("Original U-RNN code:", U_RNN_CODE)
    print("Preprocessing device:", device)

    failures: list[tuple[str, Exception]] = []
    for split in SPLITS:
        try:
            inspect_split(split, device)
        except Exception as exc:
            failures.append((split, exc))
            print(
                f"{split} diagnostic: FAILED ({type(exc).__name__}: {exc})",
                file=sys.stderr,
            )

    if failures:
        print("\nAdapter smoke test failed:", file=sys.stderr)
        for split, exc in failures:
            print(f"  {split}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("\nTrain, valid, and test adapter diagnostics succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
