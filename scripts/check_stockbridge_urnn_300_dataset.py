#!/usr/bin/env python3
"""Independent QC for the frozen Stockbridge U-RNN 300-event dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch


BASE = Path.home() / "flood/02_data/CityCAT_Winchester"

MASTER = (
    BASE / "pt/stockbridge_phase2plus3_300_v01/"
    "stockbridge_phase2plus3_300_manifest.csv"
)

DATASET = (
    BASE / "urnn/stockbridge_urnn_phase2plus3_300_v01"
)

REPORT_DIR = (
    Path.home()
    / "flood/03_outputs/Stockbridge-URNN/comet/qc_phase2plus3_300_v01"
)

COUNTS = {"train": 210, "valid": 45, "test": 45}
H, W, HP, WP, T = 305, 326, 308, 328, 25


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_npy(path, shape):
    require(path.is_file(), f"Missing file: {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)

    require(
        array.shape == shape,
        f"Wrong shape: {path}: {array.shape}, expected {shape}"
    )
    require(
        array.dtype == np.float32,
        f"Wrong dtype: {path}: {array.dtype}"
    )
    require(
        np.isfinite(array).all(),
        f"Non-finite values: {path}"
    )
    return array


def check_rainfall(tensor, sidecar_path, output_rain):
    require(sidecar_path.is_file(), f"Missing sidecar: {sidecar_path}")

    metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))

    if "rainfall_vector" in metadata:
        rates = np.asarray(metadata["rainfall_vector"], dtype=np.float64)
        source = "sidecar"
    else:
        # Match the documented fallback contract independently.
        rainfall_field = tensor[..., 3]
        rates = np.empty(T, dtype=np.float64)

        for t in range(T):
            field = rainfall_field[:, :, t]
            require(
                np.allclose(field, field.flat[0], rtol=1e-5, atol=1e-12),
                f"Nonuniform fallback rainfall at timestep {t}: {sidecar_path}"
            )
            rates[t] = float(field.flat[0])

        source = "tensor_fallback"

    require(rates.shape == (T,), f"Wrong rainfall vector: {sidecar_path}")
    require(np.isfinite(rates).all(), f"Non-finite rain: {sidecar_path}")
    require((rates >= 0).all(), f"Negative rainfall: {sidecar_path}")

    expected = (rates * 300.0 * 1000.0).astype(np.float32)

    require(
        np.array_equal(output_rain, expected),
        f"Rainfall conversion mismatch: {sidecar_path}"
    )

    # Independently check rainfall's timestep correspondence
    # against the canonical PT channel. Zero-masked cells are allowed.
    for t in range(T):
        field = tensor[:, :, t, 3]
        rate = rates[t]

        is_expected = np.isclose(
            field, rate, rtol=1e-4, atol=1e-10
        )
        is_zero = np.isclose(
            field, 0.0, rtol=0.0, atol=1e-10
        )

        require(
            np.all(is_expected | is_zero),
            f"Rainfall PT/sidecar mismatch at timestep {t}: {sidecar_path}"
        )

        if rate > 1e-10:
            require(
                np.any(is_expected),
                f"Rainfall missing from PT at timestep {t}: {sidecar_path}"
            )

    return source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-sha256", action="store_true")
    args = parser.parse_args()

    with MASTER.open(newline="", encoding="utf-8-sig") as f:
        master = list(csv.DictReader(f))

    require(len(master) == 300, "Master manifest does not have 300 rows")

    derived_manifest = DATASET / "stockbridge_urnn_300_manifest.csv"
    require(derived_manifest.is_file(), "Missing derived manifest")

    with derived_manifest.open(newline="", encoding="utf-8") as f:
        derived = list(csv.DictReader(f))

    require(len(derived) == 300, "Derived manifest does not have 300 rows")

    derived_index = {
        (row["split"], row["event_id"]): row
        for row in derived
    }
    require(len(derived_index) == 300, "Duplicate derived manifest entries")

    split_counts = Counter()
    rainfall_sources = Counter()
    seen_ids = set()
    records = []
    reference_dem = None

    for index, row in enumerate(master, start=1):
        event = row["event_id"]
        split = row["split"]
        phase = row["phase"]

        require(split in COUNTS, f"Unknown split: {split}")
        require(
            re.fullmatch(r"R\d{3}_stockbridge_T000_T024", event) is not None,
            f"Invalid event ID: {event}"
        )
        require(event not in seen_ids, f"Duplicate event: {event}")
        seen_ids.add(event)

        event_number = int(event[1:4])
        expected_phase = "phase2" if event_number <= 100 else "phase3"
        require(phase == expected_phase, f"Wrong phase: {event}")

        source = Path(row["source_path"])
        require(source.is_file(), f"Missing canonical PT: {source}")
        require(
            source.stat().st_size == int(row["file_size_bytes"]),
            f"Source PT size mismatch: {event}"
        )

        if args.verify_sha256:
            require(
                digest(source) == row["sha256"],
                f"Canonical PT SHA256 mismatch: {event}"
            )

        key = (split, event)
        require(key in derived_index, f"Missing derived manifest row: {key}")
        drow = derived_index[key]

        require(
            drow["source_pt_path"] == str(source.resolve()),
            f"Source provenance mismatch: {event}"
        )
        require(
            drow["canonical_sha256"] == row["sha256"],
            f"Checksum provenance mismatch: {event}"
        )
        require(
            drow["phase"] == phase,
            f"Derived phase mismatch: {event}"
        )
        require(
            drow["canonical_view_path"] == row["view_path"],
            f"Canonical view provenance mismatch: {event}"
        )

        event_dir = DATASET / split / "flood" / "stockbridge" / event
        require(
            drow["output_event_folder"] == str(event_dir),
            f"Output provenance mismatch: {event}"
        )

        pt = torch.load(source, map_location="cpu")
        require(torch.is_tensor(pt), f"Not a tensor: {source}")
        require(
            tuple(pt.shape) == (H, W, T, 5),
            f"Wrong canonical tensor shape: {event}"
        )
        require(
            pt.dtype == torch.float32,
            f"Wrong canonical dtype: {event}"
        )

        tensor = pt.numpy()

        flood = load_npy(
            event_dir / "flood.npy",
            (T, HP, WP)
        )
        rain = load_npy(
            event_dir / "rainfall.npy",
            (T,)
        )

        expected_flood = np.transpose(
            tensor[..., 0], (2, 0, 1)
        )

        require(
            np.isfinite(expected_flood).all(),
            f"Non-finite canonical flood depth: {event}"
        )
        require(
            np.array_equal(flood[:, :H, :W], expected_flood),
            f"Flood depth mismatch: {event}"
        )
        require(
            not np.any(flood[:, H:, :]),
            f"Nonzero bottom flood padding: {event}"
        )
        require(
            not np.any(flood[:, :, W:]),
            f"Nonzero right flood padding: {event}"
        )

        geodata = DATASET / split / "geodata" / "stockbridge"

        dem = load_npy(
            geodata / "absolute_DEM.npy",
            (HP, WP)
        )
        impervious = load_npy(
            geodata / "impervious.npy",
            (HP, WP)
        )
        manhole = load_npy(
            geodata / "manhole.npy",
            (HP, WP)
        )

        expected_dem = np.pad(
            tensor[:, :, 0, 4],
            ((0, HP - H), (0, WP - W)),
            mode="edge"
        )

        require(
            np.array_equal(dem, expected_dem),
            f"DEM mismatch: {event}"
        )
        require(
            np.all(impervious == np.float32(0.05)),
            f"Impervious placeholder mismatch: {event}"
        )
        require(
            np.all(manhole == np.float32(0.0)),
            f"Manhole placeholder mismatch: {event}"
        )

        if reference_dem is None:
            reference_dem = np.array(dem, copy=True)
        else:
            require(
                np.array_equal(dem, reference_dem),
                f"Cross-split DEM mismatch: {event}"
            )

        rainfall_source = check_rainfall(
            tensor,
            source.with_suffix(".pt.json"),
            rain
        )

        rainfall_sources[rainfall_source] += 1
        split_counts[split] += 1

        records.append({
            "event_id": event,
            "phase": phase,
            "split": split,
            "rainfall_source": rainfall_source,
            "qc_status": "PASS",
        })

        if index % 25 == 0:
            print(f"QC {index:03d}/300 PASS", flush=True)

    expected_ids = {
        f"R{i:03d}_stockbridge_T000_T024"
        for i in range(1, 301)
    }
    require(seen_ids == expected_ids, "Missing/extra event IDs")

    for split, expected in COUNTS.items():
        require(
            split_counts[split] == expected,
            f"Wrong {split} count: {split_counts[split]}"
        )

        split_dir = DATASET / split / "flood" / "stockbridge"
        folders = {p.name for p in split_dir.iterdir() if p.is_dir()}
        expected_folders = {
            row["event_id"]
            for row in master
            if row["split"] == split
        }
        require(
            folders == expected_folders,
            f"Unexpected or missing event folders in {split}"
        )

        event_list = DATASET / f"stockbridge_{split}.txt"
        names = event_list.read_text(encoding="utf-8").splitlines()
        require(
            len(names) == expected and set(names) == expected_folders,
            f"Event list mismatch: {split}"
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with (REPORT_DIR / "qc_events.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "status": "PASS",
        "checked_events": len(records),
        "train": split_counts["train"],
        "valid": split_counts["valid"],
        "test": split_counts["test"],
        "rainfall_sources": dict(rainfall_sources),
        "canonical_sha256_verified": args.verify_sha256,
        "flood_values_match": True,
        "rainfall_values_match": True,
        "dem_values_match": True,
        "padding_verified": True,
        "static_placeholders_verified": True,
        "frozen_membership_verified": True,
    }

    (REPORT_DIR / "qc_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8"
    )

    print("\n===== INDEPENDENT QC PASSED =====", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print("Reports:", REPORT_DIR, flush=True)


if __name__ == "__main__":
    main()
