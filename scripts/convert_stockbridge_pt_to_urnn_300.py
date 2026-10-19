#!/usr/bin/env python3
"""Convert frozen Stockbridge Phase 2+3 PT experiment to U-RNN.

Never modifies canonical PT files or the historical Phase 2 U-RNN dataset.
Uses source_path from the authoritative 300-event master manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

import convert_stockbridge_pt_to_urnn_full as legacy


HOME = Path.home()
BASE = HOME / "flood/02_data/CityCAT_Winchester"

MASTER = (
    BASE / "pt/stockbridge_phase2plus3_300_v01/"
    "stockbridge_phase2plus3_300_manifest.csv"
)

ROOTS = {
    "phase2": BASE / "pt/stockbridge_phase2_core_v01",
    "phase3": BASE / "pt/stockbridge_phase3_core_v01",
}

VIEW_ROOT = BASE / "pt/stockbridge_phase2plus3_300_v01"

OUTPUT = (
    BASE / "urnn/stockbridge_urnn_phase2plus3_300_v01"
)

SPLITS = ("train", "valid", "test")

EXPECTED = {
    "phase2": {"train": 70, "valid": 15, "test": 15},
    "phase3": {"train": 140, "valid": 30, "test": 30},
}

REQUIRED_COLUMNS = {
    "event_id", "phase", "split",
    "source_path", "view_path",
    "link_method", "file_size_bytes", "sha256",
}

EVENT_PATTERN = re.compile(
    r"^R(\d{3})_stockbridge_T000_T024$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_master(verify_sha256: bool) -> list[dict[str, str]]:
    if not MASTER.is_file():
        raise FileNotFoundError(MASTER)

    with MASTER.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])

        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Missing manifest columns: {missing}")

        rows = list(reader)

    if len(rows) != 300:
        raise ValueError(f"Expected 300 records, found {len(rows)}")

    counts = Counter()
    event_ids = set()
    source_paths = set()
    sidecar_counts = Counter()

    for index, row in enumerate(rows, start=1):
        phase = row["phase"]
        split = row["split"]
        event_id = row["event_id"]

        if phase not in ROOTS:
            raise ValueError(f"Row {index}: invalid phase {phase}")

        if split not in SPLITS:
            raise ValueError(f"Row {index}: invalid split {split}")

        match = EVENT_PATTERN.fullmatch(event_id)
        if not match:
            raise ValueError(f"Row {index}: invalid event_id {event_id}")

        event_number = int(match.group(1))
        expected_phase = "phase2" if event_number <= 100 else "phase3"

        if phase != expected_phase:
            raise ValueError(f"Row {index}: incorrect phase for {event_id}")

        if event_id in event_ids:
            raise ValueError(f"Duplicate event_id: {event_id}")
        event_ids.add(event_id)

        source = Path(row["source_path"]).expanduser().resolve(
            strict=True
        )
        expected_source = (
            ROOTS[phase] / split / f"{event_id}.pt"
        ).resolve()

        if source != expected_source:
            raise ValueError(
                f"Unexpected canonical source for {event_id}: {source}"
            )

        if source in source_paths:
            raise ValueError(f"Duplicate source path: {source}")
        source_paths.add(source)

        expected_size = int(row["file_size_bytes"])
        if source.stat().st_size != expected_size:
            raise ValueError(f"File size mismatch: {source}")

        view = Path(row["view_path"]).expanduser().resolve(
            strict=True
        )
        expected_view = (
            VIEW_ROOT / split / f"{event_id}.pt"
        ).resolve()

        if view != expected_view:
            raise ValueError(f"Unexpected view path: {view}")

        if row["link_method"] == "hardlink":
            if not os.path.samefile(source, view):
                raise ValueError(f"Broken hardlink relation: {event_id}")

        sidecar = source.with_suffix(source.suffix + ".json")
        if sidecar.is_file():
            sidecar_counts[phase] += 1
        elif phase == "phase3":
            raise FileNotFoundError(
                f"Missing Phase 3 rainfall sidecar: {sidecar}"
            )

        expected_hash = row["sha256"].lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise ValueError(f"Invalid SHA256 field: {event_id}")

        if verify_sha256:
            actual_hash = sha256_file(source)
            if actual_hash != expected_hash:
                raise ValueError(f"SHA256 mismatch: {event_id}")

        counts[(phase, split)] += 1

    expected_ids = {
        f"R{i:03d}_stockbridge_T000_T024"
        for i in range(1, 301)
    }
    if event_ids != expected_ids:
        raise ValueError(
            f"Event ID mismatch: "
            f"missing={sorted(expected_ids - event_ids)}, "
            f"extra={sorted(event_ids - expected_ids)}"
        )

    for phase, split_counts in EXPECTED.items():
        for split, expected in split_counts.items():
            actual = counts[(phase, split)]
            if actual != expected:
                raise ValueError(
                    f"{phase}/{split}: expected {expected}, found {actual}"
                )

    print("===== PREFLIGHT PASSED =====", flush=True)
    print("Manifest:", MASTER, flush=True)
    print("Total events:", len(rows), flush=True)
    print("SHA256 verification:", verify_sha256, flush=True)

    for phase in ROOTS:
        for split in SPLITS:
            print(
                f"{phase}/{split}: {counts[(phase, split)]}",
                flush=True,
            )

    print("Sidecars:", dict(sidecar_counts), flush=True)
    return rows


def write_provenance(stage: Path, records: list[dict]) -> None:
    json_path = stage / "stockbridge_urnn_300_manifest.json"
    csv_path = stage / "stockbridge_urnn_300_manifest.csv"

    json_path.write_text(
        json.dumps(records, indent=2) + "\n",
        encoding="utf-8",
    )

    fields = list(legacy.MANIFEST_FIELDS) + [
        "event_id",
        "phase",
        "canonical_view_path",
        "canonical_sha256",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for record in records:
            flat = record.copy()
            for key in (
                "original_shape",
                "padded_shape",
                "crop_back_shape",
            ):
                flat[key] = "x".join(map(str, record[key]))
            writer.writerow(flat)

    metadata = {
        "experiment": "Stockbridge Phase 2+3 300",
        "master_manifest": str(MASTER),
        "master_manifest_sha256": sha256_file(MASTER),
        "train_events": 210,
        "valid_events": 45,
        "test_events": 45,
        "original_grid": [305, 326],
        "padded_grid": [308, 328],
        "timesteps": 25,
        "rainfall_alignment": "future",
        "rainfall_source_units": "m/s",
        "rainfall_output_units": "mm/timestep",
        "canonical_sources_modified": False,
    }

    (stage / "conversion_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-sha256", action="store_true")
    args = parser.parse_args()

    rows = read_master(args.verify_sha256)

    if OUTPUT.exists():
        raise FileExistsError(
            f"Output already exists; refusing to overwrite: {OUTPUT}"
        )

    stage = OUTPUT.with_name(OUTPUT.name + ".building")

    if stage.exists():
        raise FileExistsError(
            f"Staging directory already exists: {stage}"
        )

    print("Destination:", OUTPUT, flush=True)
    print("Staging:", stage, flush=True)

    if args.dry_run:
        print("DRY RUN PASSED: no files created.", flush=True)
        return

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    stage.mkdir()

    records = []

    try:
        for split in SPLITS:
            split_rows = [
                row for row in rows if row["split"] == split
            ]
            event_names = []

            for row in split_rows:
                source = Path(row["source_path"])

                record = legacy.convert_event(
                    split, source, stage
                )

                if record["event_name"] != row["event_id"]:
                    raise ValueError(
                        f"Converted name mismatch: {source}"
                    )

                # Store final, not temporary staging, output path.
                record["output_event_folder"] = str(
                    OUTPUT / split / "flood" / legacy.LOCATION
                    / row["event_id"]
                )

                record["event_id"] = row["event_id"]
                record["phase"] = row["phase"]
                record["canonical_view_path"] = row["view_path"]
                record["canonical_sha256"] = row["sha256"]

                records.append(record)
                event_names.append(row["event_id"])

                print(
                    f"CONVERTED {len(records):03d}/300 "
                    f"{split}/{row['event_id']}",
                    flush=True,
                )

            legacy.write_event_list(
                stage, split, event_names
            )

        if len(records) != 300:
            raise ValueError(
                f"Expected 300 conversions, got {len(records)}"
            )

        write_provenance(stage, records)

        if OUTPUT.exists():
            raise FileExistsError(
                f"Output appeared during build: {OUTPUT}"
            )

        os.rename(stage, OUTPUT)

    except Exception:
        print(
            f"CONVERSION FAILED. Partial staging retained: {stage}",
            flush=True,
        )
        raise

    print("===== CONVERSION COMPLETE =====", flush=True)
    print("Events:", len(records), flush=True)
    print("Output:", OUTPUT, flush=True)
    print("Next: independent dataset QC.", flush=True)


if __name__ == "__main__":
    main()
