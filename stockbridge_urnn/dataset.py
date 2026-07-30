"""Stockbridge split adapter for the original U-RNN dataset loader.

The upstream loader maps every split other than ``train`` to its ``test``
directory. This small adapter selects the literal train/valid/test directory
while inheriting upstream event loading, tensor preparation, and unit handling.
"""

from __future__ import annotations

from pathlib import Path

from src.lib.dataset.Dynamic2DFlood import Dynamic2DFlood


LOCATION = "stockbridge"
SUPPORTED_SPLITS = ("train", "valid", "test")


class StockbridgeDynamic2DFlood(Dynamic2DFlood):
    """Load the converted Stockbridge dataset through U-RNN conventions.

    Flood depths and DEM values are stored in metres and are converted to
    millimetres by the inherited U-RNN preparation methods. Rainfall is already
    in millimetres per timestep and remains unchanged.
    """

    def __init__(self, data_root: str | Path, split: str, duration: int = 25) -> None:
        if split not in SUPPORTED_SPLITS:
            supported = ", ".join(SUPPORTED_SPLITS)
            raise ValueError(f"Unsupported split {split!r}; expected one of: {supported}")

        root = Path(data_root).expanduser()
        data_dir = root / split
        flood_root = data_dir / "flood"
        location_root = flood_root / LOCATION
        event_list = root / f"stockbridge_{split}.txt"

        if not location_root.is_dir():
            raise FileNotFoundError(f"Missing Stockbridge flood directory: {location_root}")
        if not event_list.is_file():
            raise FileNotFoundError(f"Missing Stockbridge event list: {event_list}")

        # These attributes mirror Dynamic2DFlood.__init__, except data_dir uses
        # the literal split so validation events are not redirected to test.
        self.data_root = str(root)
        self.duration = duration
        self.data_dir = str(data_dir)
        self.geo_root = str(data_dir / "geodata")
        self.flood_root = str(flood_root)
        self.locations = [LOCATION]
        self.locations_dir = [str(location_root)]
        self.event_names = self._load_event_names(split, str(event_list))
        self.num_samples = len(self.event_names)

        print(
            f"Loaded StockbridgeDynamic2DFlood {split} with "
            f"{self.num_samples} samples"
        )
