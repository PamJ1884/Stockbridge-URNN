#!/usr/bin/env python3
"""Validate the complete converted Stockbridge U-RNN dataset."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_DATA_ROOT = (
    Path.home() / "flood/02_data/CityCAT_Winchester/urnn/stockbridge_urnn_full_v01"
)
LOCATION = "stockbridge"
SPLIT_COUNTS = {"train": 70, "valid": 15, "test": 15}
FLOOD_SHAPE = (25, 308, 328)
RAINFALL_SHAPE = (25,)
GEODATA_SHAPE = (308, 328)
EVENT_FILES = ("flood.npy", "rainfall.npy")
GEODATA_FILES = ("absolute_DEM.npy", "impervious.npy", "manhole.npy")


class DatasetCheckError(RuntimeError):
    """Raised when one or more dataset integrity checks fail."""


@dataclass
class SplitSummary:
    """Aggregate values displayed after validation."""

    event_count: int
    min_flood_depth: float = float("inf")
    max_flood_depth: float = float("-inf")
    min_rainfall_total: float = float("inf")
    max_rainfall_total: float = float("-inf")
    max_rainfall_per_timestep: float = float("-inf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the complete converted Stockbridge U-RNN dataset."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Converted dataset root (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each event and array as it is checked",
    )
    return parser.parse_args()


def load_array(
    path: Path,
    expected_shape: tuple[int, ...],
    errors: list[str],
    verbose: bool,
) -> np.ndarray | None:
    """Load an array and validate its shape and finite values."""
    if not path.is_file():
        errors.append(f"missing required file: {path}")
        return None
    try:
        array = np.load(path, allow_pickle=False)
    except Exception as exc:
        errors.append(f"could not load {path}: {exc}")
        return None
    if verbose:
        print(f"  checked {path} shape={array.shape} dtype={array.dtype}")
    if array.shape != expected_shape:
        errors.append(
            f"wrong shape for {path}: expected {expected_shape}, found {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.number):
        errors.append(f"non-numeric dtype for {path}: {array.dtype}")
        return None
    try:
        all_finite = bool(np.isfinite(array).all())
    except TypeError as exc:
        errors.append(f"could not check finite values in {path}: {exc}")
        return None
    if not all_finite:
        errors.append(f"NaN or Inf values found in {path}")
        return None
    return array


def event_folders(data_root: Path, split: str, errors: list[str]) -> list[Path]:
    flood_root = data_root / split / "flood" / LOCATION
    if not flood_root.is_dir():
        errors.append(f"missing event directory: {flood_root}")
        return []
    return sorted(path for path in flood_root.iterdir() if path.is_dir())


def check_event_list(
    data_root: Path,
    split: str,
    folder_names: list[str],
    errors: list[str],
) -> None:
    list_path = data_root / f"stockbridge_{split}.txt"
    if not list_path.is_file():
        errors.append(f"missing event-list file: {list_path}")
        return
    try:
        listed_names = [
            line.strip()
            for line in list_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except OSError as exc:
        errors.append(f"could not read {list_path}: {exc}")
        return
    if len(listed_names) != len(set(listed_names)):
        errors.append(f"duplicate event names found in {list_path}")
    listed = set(listed_names)
    folders = set(folder_names)
    if listed != folders:
        missing = sorted(folders - listed)
        extra = sorted(listed - folders)
        errors.append(
            f"{list_path} does not match event folders; "
            f"missing from list={missing}, not present as folders={extra}"
        )


def check_geodata(
    data_root: Path,
    split: str,
    errors: list[str],
    verbose: bool,
) -> np.ndarray | None:
    geodata_root = data_root / split / "geodata" / LOCATION
    arrays: dict[str, np.ndarray | None] = {}
    for filename in GEODATA_FILES:
        arrays[filename] = load_array(
            geodata_root / filename, GEODATA_SHAPE, errors, verbose
        )
    return arrays["absolute_DEM.npy"]


def check_split(
    data_root: Path,
    split: str,
    errors: list[str],
    verbose: bool,
) -> tuple[SplitSummary, list[str], np.ndarray | None]:
    folders = event_folders(data_root, split, errors)
    names = [path.name for path in folders]
    expected_count = SPLIT_COUNTS[split]
    if len(folders) != expected_count:
        errors.append(
            f"wrong event count for {split}: expected {expected_count}, found {len(folders)}"
        )
    check_event_list(data_root, split, names, errors)
    summary = SplitSummary(event_count=len(folders))

    for event_dir in folders:
        if verbose:
            print(f"Checking {split}/{event_dir.name}")
        flood = load_array(event_dir / EVENT_FILES[0], FLOOD_SHAPE, errors, verbose)
        rainfall = load_array(
            event_dir / EVENT_FILES[1], RAINFALL_SHAPE, errors, verbose
        )
        if flood is not None:
            if np.any(flood < 0):
                errors.append(f"negative flood depth found in {event_dir / EVENT_FILES[0]}")
            else:
                summary.min_flood_depth = min(
                    summary.min_flood_depth, float(np.min(flood))
                )
                summary.max_flood_depth = max(
                    summary.max_flood_depth, float(np.max(flood))
                )
        if rainfall is not None:
            if np.any(rainfall < 0):
                errors.append(f"negative rainfall found in {event_dir / EVENT_FILES[1]}")
            else:
                rainfall_total = float(np.sum(rainfall, dtype=np.float64))
                summary.min_rainfall_total = min(
                    summary.min_rainfall_total, rainfall_total
                )
                summary.max_rainfall_total = max(
                    summary.max_rainfall_total, rainfall_total
                )
                summary.max_rainfall_per_timestep = max(
                    summary.max_rainfall_per_timestep, float(np.max(rainfall))
                )

    dem = check_geodata(data_root, split, errors, verbose)
    return summary, names, dem


def display_value(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.6g}"


def print_summary(summaries: dict[str, SplitSummary]) -> None:
    headers = (
        "split",
        "events",
        "min flood",
        "max flood",
        "min rain total",
        "max rain total",
        "max rain/step",
    )
    rows = [
        (
            split,
            str(summary.event_count),
            display_value(summary.min_flood_depth),
            display_value(summary.max_flood_depth),
            display_value(summary.min_rainfall_total),
            display_value(summary.max_rainfall_total),
            display_value(summary.max_rainfall_per_timestep),
        )
        for split, summary in summaries.items()
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print("\nDataset summary")
    print("  ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def validate_dataset(data_root: Path, verbose: bool) -> None:
    if not data_root.is_dir():
        raise DatasetCheckError(f"dataset root does not exist: {data_root}")

    errors: list[str] = []
    summaries: dict[str, SplitSummary] = {}
    names_by_split: dict[str, list[str]] = {}
    dems: dict[str, np.ndarray | None] = {}
    for split in SPLIT_COUNTS:
        summary, names, dem = check_split(data_root, split, errors, verbose)
        summaries[split] = summary
        names_by_split[split] = names
        dems[split] = dem

    owner: dict[str, str] = {}
    for split, names in names_by_split.items():
        for name in names:
            if name in owner:
                errors.append(
                    f"event name {name!r} appears in both {owner[name]} and {split}"
                )
            else:
                owner[name] = split

    reference_split = "train"
    reference_dem = dems[reference_split]
    if reference_dem is not None:
        for split in ("valid", "test"):
            dem = dems[split]
            if dem is not None and not np.array_equal(reference_dem, dem):
                errors.append(
                    f"absolute_DEM.npy differs between {reference_split} and {split}"
                )

    print_summary(summaries)
    if errors:
        details = "\n".join(f"  {index}. {error}" for index, error in enumerate(errors, 1))
        raise DatasetCheckError(
            f"dataset validation failed with {len(errors)} error(s):\n{details}"
        )
    print(f"\nAll checks passed for {sum(item.event_count for item in summaries.values())} events.")


def main() -> None:
    args = parse_args()
    try:
        validate_dataset(args.data_root.expanduser(), args.verbose)
    except DatasetCheckError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
