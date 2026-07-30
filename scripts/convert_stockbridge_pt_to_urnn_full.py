#!/usr/bin/env python3
"""Convert the full Stockbridge PT dataset to the U-RNN dataset layout.

The 305x326 source grid is padded on its bottom and right edges to 308x328.
IMPORTANT: U-RNN predictions must be cropped back to 305x326 before they are
compared with CityCAT targets using DNO-style evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_SOURCE_ROOT = (
    Path.home() / "flood/02_data/CityCAT_Winchester/pt/stockbridge_phase2_core_v01"
)
DEFAULT_OUT_ROOT = (
    Path.home() / "flood/02_data/CityCAT_Winchester/urnn/stockbridge_urnn_full_v01"
)

LOCATION = "stockbridge"
SPLIT_COUNTS = {"train": 70, "valid": 15, "test": 15}
ORIGINAL_HEIGHT = 305
ORIGINAL_WIDTH = 326
PADDED_HEIGHT = 308
PADDED_WIDTH = 328
TIMESTEPS = 25
CHANNEL_COUNT = 5
TIMESTEP_SECONDS = 300.0
METRES_TO_MILLIMETRES = 1000.0
IMPERVIOUS_VALUE = 0.05
MANHOLE_VALUE = 0.0

CHANNELS = {
    "flood_depth": 0,
    "velocity_x": 1,
    "velocity_y": 2,
    "rainfall": 3,
    "dem": 4,
}

MANIFEST_FIELDS = (
    "split",
    "event_name",
    "source_pt_path",
    "output_event_folder",
    "rainfall_total_mm",
    "max_rainfall_per_timestep_mm",
    "max_flood_depth_m",
    "mean_flood_depth_m",
    "dem_min_m",
    "dem_max_m",
    "original_shape",
    "padded_shape",
    "crop_back_shape",
    "rainfall_source",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert all Stockbridge phase-2 PT events to U-RNN format."
    )
    parser.add_argument(
        "--source-root", type=Path, default=DEFAULT_SOURCE_ROOT,
        help=f"Source dataset root (default: {DEFAULT_SOURCE_ROOT})",
    )
    parser.add_argument(
        "--out-root", type=Path, default=DEFAULT_OUT_ROOT,
        help=f"Output dataset root (default: {DEFAULT_OUT_ROOT})",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Remove and recreate an existing output folder",
    )
    return parser.parse_args()


def natural_sort_key(path: Path) -> tuple[tuple[int, object], ...]:
    """Return a deterministic natural-sort key based on a filename."""
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", path.name)
    )


def discover_source_files(source_root: Path) -> dict[str, list[Path]]:
    """Find and count all PT files before creating or deleting output data."""
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source dataset root does not exist: {source_root}")

    split_files: dict[str, list[Path]] = {}
    for split, expected_count in SPLIT_COUNTS.items():
        split_dir = source_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing source split directory: {split_dir}")
        files = sorted(
            (path for path in split_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pt"),
            key=lambda path: (natural_sort_key(path), path.name.casefold(), path.name),
        )
        if len(files) != expected_count:
            raise ValueError(
                f"Expected {expected_count} .pt files in {split_dir}, found {len(files)}"
            )
        split_files[split] = files
    return split_files


def prepare_output_root(source_root: Path, out_root: Path, overwrite: bool) -> None:
    """Create a clean output root without risking deletion of source data."""
    source_resolved = source_root.expanduser().resolve()
    out_resolved = out_root.expanduser().resolve()
    if out_resolved == source_resolved or source_resolved in out_resolved.parents:
        raise ValueError(f"Output root must not be the source root or inside it: {out_root}")
    if out_root.exists():
        if not out_root.is_dir():
            raise FileExistsError(f"Output path exists and is not a directory: {out_root}")
        if not overwrite:
            raise FileExistsError(
                f"Output folder already exists: {out_root}. Pass --overwrite to replace it."
            )
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)


def validate_tensor(tensor: Any, pt_file: Path) -> torch.Tensor:
    expected_shape = (
        ORIGINAL_HEIGHT, ORIGINAL_WIDTH, TIMESTEPS, CHANNEL_COUNT,
    )
    if not torch.is_tensor(tensor):
        raise TypeError(f"Expected a PyTorch tensor in {pt_file}, found {type(tensor).__name__}")
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"Expected tensor shape {expected_shape} in {pt_file}, found {tuple(tensor.shape)}"
        )
    return tensor.detach().cpu()


def load_rainfall_vector(pt_file: Path, tensor: torch.Tensor) -> tuple[np.ndarray, str]:
    """Load rainfall rate (m/s), preferring the .pt.json sidecar."""
    sidecar = pt_file.with_suffix(pt_file.suffix + ".json")
    raw: np.ndarray
    source: str
    if sidecar.is_file():
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read rainfall sidecar {sidecar}: {exc}") from exc
        if "rainfall_vector" in metadata:
            raw = np.asarray(metadata["rainfall_vector"], dtype=np.float64)
            source = str(sidecar)
        else:
            raw = rainfall_from_tensor(tensor, pt_file)
            source = "tensor rainfall channel (sidecar has no rainfall_vector)"
    else:
        raw = rainfall_from_tensor(tensor, pt_file)
        source = "tensor rainfall channel"

    if raw.shape != (TIMESTEPS,):
        raise ValueError(
            f"Expected {TIMESTEPS} rainfall values for {pt_file}, found shape {raw.shape}"
        )
    if not np.isfinite(raw).all():
        raise ValueError(f"Rainfall vector contains non-finite values for {pt_file}")
    if np.any(raw < 0):
        raise ValueError(f"Rainfall vector contains negative rates for {pt_file}")

    rainfall_mm_per_step = raw * TIMESTEP_SECONDS * METRES_TO_MILLIMETRES
    return rainfall_mm_per_step.astype(np.float32), source


def rainfall_from_tensor(tensor: torch.Tensor, pt_file: Path) -> np.ndarray:
    """Extract one spatially uniform rainfall rate for each timestep."""
    rainfall_field = tensor[..., CHANNELS["rainfall"]].numpy().astype(np.float64, copy=False)
    vector = np.empty(TIMESTEPS, dtype=np.float64)
    for timestep in range(TIMESTEPS):
        values = rainfall_field[:, :, timestep]
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            raise ValueError(f"No finite rainfall values at timestep {timestep} in {pt_file}")
        first = finite_values[0]
        if not np.allclose(finite_values, first, rtol=1e-5, atol=1e-12):
            raise ValueError(
                f"Rainfall channel is not spatially uniform at timestep {timestep} in {pt_file}"
            )
        vector[timestep] = first
    return vector


def pad_flood(flood: np.ndarray) -> np.ndarray:
    """Zero-pad time-first flood data on the bottom and right."""
    if flood.shape != (TIMESTEPS, ORIGINAL_HEIGHT, ORIGINAL_WIDTH):
        raise ValueError(f"Unexpected flood shape before padding: {flood.shape}")
    return np.pad(
        flood,
        ((0, 0), (0, PADDED_HEIGHT - ORIGINAL_HEIGHT), (0, PADDED_WIDTH - ORIGINAL_WIDTH)),
        mode="constant",
        constant_values=0.0,
    ).astype(np.float32, copy=False)


def pad_dem(dem: np.ndarray) -> np.ndarray:
    """Pad DEM on the bottom and right by repeating its edge values."""
    if dem.shape != (ORIGINAL_HEIGHT, ORIGINAL_WIDTH):
        raise ValueError(f"Unexpected DEM shape before padding: {dem.shape}")
    return np.pad(
        dem,
        ((0, PADDED_HEIGHT - ORIGINAL_HEIGHT), (0, PADDED_WIDTH - ORIGINAL_WIDTH)),
        mode="edge",
    ).astype(np.float32, copy=False)


def finite_min_max(values: np.ndarray, label: str, pt_file: Path) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"{label} contains no finite values in {pt_file}")
    return float(finite.min()), float(finite.max())


def convert_event(split: str, pt_file: Path, out_root: Path) -> dict[str, Any]:
    """Convert one event and return its manifest record."""
    try:
        loaded = torch.load(pt_file, map_location="cpu")
    except Exception as exc:
        raise ValueError(f"Could not load {pt_file}: {exc}") from exc
    tensor = validate_tensor(loaded, pt_file)
    array = tensor.numpy().astype(np.float32, copy=False)

    original_flood = np.transpose(array[..., CHANNELS["flood_depth"]], (2, 0, 1))
    flood = pad_flood(original_flood)
    rainfall, rainfall_source = load_rainfall_vector(pt_file, tensor)
    original_dem = array[:, :, 0, CHANNELS["dem"]]
    dem = pad_dem(original_dem)

    event_name = pt_file.stem
    event_dir = out_root / split / "flood" / LOCATION / event_name
    geodata_dir = out_root / split / "geodata" / LOCATION
    event_dir.mkdir(parents=True, exist_ok=False)
    geodata_dir.mkdir(parents=True, exist_ok=True)

    np.save(event_dir / "flood.npy", flood)
    np.save(event_dir / "rainfall.npy", rainfall)

    # U-RNN shares static geodata across all events in a split/location.
    dem_path = geodata_dir / "absolute_DEM.npy"
    if dem_path.exists():
        existing_dem = np.load(dem_path)
        if not np.array_equal(existing_dem, dem, equal_nan=True):
            raise ValueError(f"DEM differs between events in split {split}: {pt_file}")
    else:
        np.save(dem_path, dem)
        np.save(
            geodata_dir / "impervious.npy",
            np.full((PADDED_HEIGHT, PADDED_WIDTH), IMPERVIOUS_VALUE, dtype=np.float32),
        )
        np.save(
            geodata_dir / "manhole.npy",
            np.full((PADDED_HEIGHT, PADDED_WIDTH), MANHOLE_VALUE, dtype=np.float32),
        )

    flood_finite = original_flood[np.isfinite(original_flood)]
    if flood_finite.size == 0:
        raise ValueError(f"Flood depth contains no finite values in {pt_file}")
    dem_min, dem_max = finite_min_max(original_dem, "DEM", pt_file)
    return {
        "split": split,
        "event_name": event_name,
        "source_pt_path": str(pt_file.resolve()),
        "output_event_folder": str(event_dir.resolve()),
        "rainfall_total_mm": float(np.sum(rainfall, dtype=np.float64)),
        "max_rainfall_per_timestep_mm": float(np.max(rainfall)),
        "max_flood_depth_m": float(flood_finite.max()),
        "mean_flood_depth_m": float(flood_finite.mean(dtype=np.float64)),
        "dem_min_m": dem_min,
        "dem_max_m": dem_max,
        "original_shape": [ORIGINAL_HEIGHT, ORIGINAL_WIDTH, TIMESTEPS, CHANNEL_COUNT],
        "padded_shape": [TIMESTEPS, PADDED_HEIGHT, PADDED_WIDTH],
        "crop_back_shape": [ORIGINAL_HEIGHT, ORIGINAL_WIDTH],
        "rainfall_source": rainfall_source,
    }


def write_event_list(out_root: Path, split: str, event_names: list[str]) -> None:
    path = out_root / f"stockbridge_{split}.txt"
    path.write_text("\n".join(event_names) + "\n", encoding="utf-8")


def write_manifests(out_root: Path, records: list[dict[str, Any]]) -> None:
    json_path = out_root / "stockbridge_urnn_full_manifest.json"
    json_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")

    csv_path = out_root / "stockbridge_urnn_full_manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for record in records:
            csv_record = record.copy()
            for field in ("original_shape", "padded_shape", "crop_back_shape"):
                csv_record[field] = "x".join(str(value) for value in record[field])
            writer.writerow(csv_record)


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser()
    out_root = args.out_root.expanduser()

    split_files = discover_source_files(source_root)
    prepare_output_root(source_root, out_root, args.overwrite)

    records: list[dict[str, Any]] = []
    for split in SPLIT_COUNTS:
        event_names: list[str] = []
        for pt_file in split_files[split]:
            record = convert_event(split, pt_file, out_root)
            records.append(record)
            event_names.append(record["event_name"])
            print(f"Converted {split}/{record['event_name']}")
        write_event_list(out_root, split, event_names)

    write_manifests(out_root, records)
    print(f"Converted {len(records)} events to {out_root}")
    print(
        "Reminder: crop U-RNN predictions from 308x328 back to 305x326 "
        "before DNO-style evaluation."
    )


if __name__ == "__main__":
    main()
