#!/usr/bin/env python3
"""Convert a tiny Stockbridge PT subset into original U-RNN .npy format.

This does NOT modify the U-RNN architecture.
It creates 1 train, 1 valid, and 1 test event in the data structure expected
by Dynamic2DFlood.

Stockbridge PT tensor:
    (H, W, T, C)
    C = [H_depth, Vx/U, Vy/V, rainfall, DEM]

U-RNN expected:
    flood.npy        -> (T, H, W), water depth in metres
    rainfall.npy     -> (T,), rainfall in mm per timestep
    absolute_DEM.npy -> (H, W), metres
    impervious.npy   -> (H, W), [0, 1]
    manhole.npy      -> (H, W), {0, 1}

Stockbridge grid is padded from 305x326 to 308x328 for U-RNN compatibility.
"""

from pathlib import Path
import json
import shutil

import numpy as np
import torch


SOURCE_ROOT = Path.home() / "flood/02_data/CityCAT_Winchester/pt/stockbridge_phase2_core_v01"
OUT_ROOT = Path.home() / "flood/02_data/CityCAT_Winchester/urnn/stockbridge_urnn_mini_v01"

LOCATION = "stockbridge"
ORIG_H, ORIG_W = 305, 326
PAD_H, PAD_W = 308, 328
DT_SECONDS = 300.0

CHANNELS = {
    "H": 0,
    "Vx": 1,
    "Vy": 2,
    "rainfall": 3,
    "DEM": 4,
}


def pad_2d(arr: np.ndarray, pad_value: float = 0.0, mode: str = "constant") -> np.ndarray:
    """Pad 2D array from 305x326 to 308x328."""
    if arr.shape != (ORIG_H, ORIG_W):
        raise ValueError(f"Expected {(ORIG_H, ORIG_W)}, got {arr.shape}")

    pad_bottom = PAD_H - ORIG_H
    pad_right = PAD_W - ORIG_W

    if mode == "edge":
        return np.pad(arr, ((0, pad_bottom), (0, pad_right)), mode="edge").astype(np.float32)

    return np.pad(
        arr,
        ((0, pad_bottom), (0, pad_right)),
        mode="constant",
        constant_values=pad_value,
    ).astype(np.float32)


def pad_3d_time_first(arr: np.ndarray, pad_value: float = 0.0) -> np.ndarray:
    """Pad time-first array (T,H,W) from 305x326 to 308x328."""
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array (T,H,W), got {arr.shape}")
    if arr.shape[1:] != (ORIG_H, ORIG_W):
        raise ValueError(f"Expected spatial {(ORIG_H, ORIG_W)}, got {arr.shape[1:]}")

    pad_bottom = PAD_H - ORIG_H
    pad_right = PAD_W - ORIG_W

    return np.pad(
        arr,
        ((0, 0), (0, pad_bottom), (0, pad_right)),
        mode="constant",
        constant_values=pad_value,
    ).astype(np.float32)


def load_rainfall_vector(pt_file: Path, tensor: torch.Tensor) -> np.ndarray:
    """Return rainfall as mm per 5-min timestep.

    The Stockbridge rainfall channel appears to store rainfall rate in m/s.
    U-RNN expects rainfall depth per timestep in mm.
    """
    json_file = pt_file.with_suffix(pt_file.suffix + ".json")

    if json_file.exists():
        meta = json.loads(json_file.read_text())
        raw = np.asarray(meta["rainfall_vector"], dtype=np.float32)
    else:
        # fallback: rainfall should be spatially uniform, so take one pixel
        raw = tensor[0, 0, :, CHANNELS["rainfall"]].cpu().numpy().astype(np.float32)

    rainfall_mm_per_step = raw * DT_SECONDS * 1000.0
    return rainfall_mm_per_step.astype(np.float32)


def convert_one(split: str, pt_file: Path) -> dict:
    tensor = torch.load(pt_file, map_location="cpu")
    if not torch.is_tensor(tensor):
        raise TypeError(f"{pt_file} did not load as tensor")
    if tuple(tensor.shape) != (ORIG_H, ORIG_W, 25, 5):
        raise ValueError(f"Unexpected tensor shape for {pt_file}: {tuple(tensor.shape)}")

    arr = tensor.numpy().astype(np.float32)

    event_name = pt_file.stem

    # Stockbridge: (H,W,T,C) -> U-RNN: (T,H,W)
    flood = np.transpose(arr[..., CHANNELS["H"]], (2, 0, 1))
    flood = pad_3d_time_first(flood, pad_value=0.0)

    rainfall = load_rainfall_vector(pt_file, tensor)

    dem = arr[:, :, 0, CHANNELS["DEM"]]
    dem = pad_2d(dem, mode="edge")

    # Neutral placeholder rasters.
    # impervious=0.05 maps to 0 under U-RNN MinMaxScaler(impervious, 0.95, 0.05).
    # manhole=0 means no drainage inlet.
    impervious = np.full((PAD_H, PAD_W), 0.05, dtype=np.float32)
    manhole = np.zeros((PAD_H, PAD_W), dtype=np.float32)

    event_dir = OUT_ROOT / split / "flood" / LOCATION / event_name
    geo_dir = OUT_ROOT / split / "geodata" / LOCATION

    event_dir.mkdir(parents=True, exist_ok=True)
    geo_dir.mkdir(parents=True, exist_ok=True)

    np.save(event_dir / "flood.npy", flood)
    np.save(event_dir / "rainfall.npy", rainfall)

    # Save geodata once per split/location. Same DEM for all events.
    np.save(geo_dir / "absolute_DEM.npy", dem)
    np.save(geo_dir / "impervious.npy", impervious)
    np.save(geo_dir / "manhole.npy", manhole)

    return {
        "split": split,
        "source": str(pt_file),
        "event": event_name,
        "flood_shape": list(flood.shape),
        "rainfall_shape": list(rainfall.shape),
        "rainfall_total_mm": float(rainfall.sum()),
        "rainfall_max_mm_per_step": float(rainfall.max()),
        "dem_shape": list(dem.shape),
        "dem_min_m": float(dem.min()),
        "dem_max_m": float(dem.max()),
        "padding": {
            "original": [ORIG_H, ORIG_W],
            "padded": [PAD_H, PAD_W],
            "crop_back_to": [ORIG_H, ORIG_W],
        },
        "placeholder_static_inputs": {
            "impervious": "constant 0.05",
            "manhole": "constant 0.0",
        },
    }


def main() -> None:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)

    manifest = []

    for split in ["train", "valid", "test"]:
        files = sorted((SOURCE_ROOT / split).glob("*.pt"))
        if not files:
            raise FileNotFoundError(f"No PT files found for split {split}")

        # mini dataset: first event only
        summary = convert_one(split, files[0])
        manifest.append(summary)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "stockbridge_urnn_mini_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("Created mini U-RNN Stockbridge dataset:")
    print(OUT_ROOT)
    print()
    for item in manifest:
        print(
            f"{item['split']:5s} {item['event']} "
            f"flood={item['flood_shape']} "
            f"rain={item['rainfall_shape']} "
            f"rain_total={item['rainfall_total_mm']:.3f} mm "
            f"rain_max={item['rainfall_max_mm_per_step']:.3f} mm/step"
        )


if __name__ == "__main__":
    main()
