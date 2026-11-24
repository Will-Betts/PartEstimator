"""User settings, kept in `settings.json` beside the launcher.

- the sheet metal thickness cutoff;
- the stock allowances: how much the recommended stock adds to the part
  on length, width, plate thickness and bar diameter before rounding up to
  the next 1/4 in (machining/geometry/stock.py);
- the tabbed-fixturing standard for milled plate parts: tab size and
  spacing, the channel-width table, the minimum frame, the vise's grip and
  clearance, and when a blank may stand on edge (machining/geometry/fixture.py).

Edit them from Settings... in the window, or in the file itself. A value
missing from the file, or out of range, falls back to the shipped default.
`PE_SETTINGS` names another file (the tests point it at one that does not
exist, so a developer's own settings cannot change what they check).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .classify import DEFAULT_CUTOFF_IN


def tool_dir() -> Path:
    """The PartEstimator folder (the one holding the launcher)."""
    return Path(__file__).resolve().parents[1]


SETTINGS_PATH = tool_dir() / "settings.json"


def settings_path() -> Path:
    """The file in use: PE_SETTINGS when set, else SETTINGS_PATH. Read at
    each call, so setting it after import still counts."""
    return Path(os.environ.get("PE_SETTINGS") or SETTINGS_PATH)


#: Settings field -> stock allowance name (stock.set_allowances).
ALLOWANCE_FIELDS = {
    "stock_length_pad_in": "length_in",
    "stock_width_pad_in": "width_in",
    "stock_thickness_pad_in": "thickness_in",
    "stock_diameter_pad_in": "diameter_in",
}
MAX_PAD_IN = 12.0

#: Allowed range of each number in the file; anything else falls back.
RANGES = {
    "sheet_cutoff_in": (0.0, 2.0, False),          # (low, high, low allowed)
    **{k: (0.0, MAX_PAD_IN, True) for k in ALLOWANCE_FIELDS},
    "tab_height_in": (0.0, 1.0, False),
    "tab_width_in": (0.0, 2.0, False),
    "tab_spacing_in": (0.5, 100.0, True),
    "min_tabs": (1, 50, True),
    "min_frame_in": (0.0, 6.0, True),
    "vise_grip_in": (0.0, 3.0, True),
    "vise_clearance_in": (0.0, 6.0, True),
    "vertical_min_thickness_in": (0.0, 12.0, True),
    "max_stickout_ratio": (0.0, 100.0, False),
    "channel_depth_1_in": (0.0, 12.0, False),
    "channel_width_1_in": (0.0, 2.0, False),
    "channel_depth_2_in": (0.0, 12.0, False),
    "channel_width_2_in": (0.0, 2.0, False),
    "channel_width_3_in": (0.0, 2.0, False),
}


@dataclass
class Settings:
    # Flat, uniform-thickness parts at or under this are sheet metal (inches).
    sheet_cutoff_in: float = DEFAULT_CUTOFF_IN
    # Added to the part before rounding up, for the recommended stock (inches).
    # Length and width: work-holding. Thickness (plate over the sheet break)
    # and diameter (round bar): cleanup.
    stock_length_pad_in: float = 1.0
    stock_width_pad_in: float = 1.0
    stock_thickness_pad_in: float = 0.125
    stock_diameter_pad_in: float = 0.125
    # Tabbed fixturing for milled plate parts (geometry/fixture.py).
    fixture_enabled: bool = True
    tab_height_in: float = 0.060
    tab_width_in: float = 0.250
    tab_spacing_in: float = 4.0
    min_tabs: int = 3
    # Channel width by the depth cut from one side: up to depth 1, width 1;
    # up to depth 2, width 2; deeper, width 3.
    channel_depth_1_in: float = 0.50
    channel_width_1_in: float = 0.250
    channel_depth_2_in: float = 1.00
    channel_width_2_in: float = 0.375
    channel_width_3_in: float = 0.500
    min_frame_in: float = 0.250
    vise_grip_in: float = 0.188
    vise_clearance_in: float = 0.375
    vertical_min_thickness_in: float = 0.250
    max_stickout_ratio: float = 10.0

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        path = path or settings_path()
        s = cls()
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return s
        if not isinstance(data, dict):
            return s
        for f in fields(cls):
            if f.name not in data:
                continue
            raw = data[f.name]
            if f.type in (bool, "bool"):
                if isinstance(raw, bool):
                    setattr(s, f.name, raw)
                continue
            try:
                v = float(raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(v):
                continue
            lo, hi, lo_ok = RANGES.get(f.name, (0.0, MAX_PAD_IN, True))
            if (lo <= v if lo_ok else lo < v) and v <= hi:
                setattr(s, f.name, int(v) if f.type in (int, "int") else v)
        if s.channel_depth_2_in <= s.channel_depth_1_in:
            s.channel_depth_1_in, s.channel_depth_2_in = (cls.channel_depth_1_in,
                                                          cls.channel_depth_2_in)
        return s

    def save(self, path: Path | None = None) -> None:
        path = path or settings_path()
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    def stock_allowances(self) -> dict:
        """The stock allowances, as stock.set_allowances takes them."""
        return {name: getattr(self, f) for f, name in ALLOWANCE_FIELDS.items()}

    def fixture(self):
        """The tabbed-fixturing standard, as geometry/fixture.py takes it."""
        from .machining.geometry.fixture import FixtureSettings

        return FixtureSettings(
            enabled=bool(self.fixture_enabled),
            tab_height_in=self.tab_height_in, tab_width_in=self.tab_width_in,
            tab_spacing_in=self.tab_spacing_in, min_tabs=int(self.min_tabs),
            channel_table=((self.channel_depth_1_in, self.channel_width_1_in),
                           (self.channel_depth_2_in, self.channel_width_2_in),
                           (math.inf, self.channel_width_3_in)),
            min_frame_in=self.min_frame_in, vise_grip_in=self.vise_grip_in,
            vise_clearance_in=self.vise_clearance_in,
            vertical_min_thickness_in=self.vertical_min_thickness_in,
            max_stickout_ratio=self.max_stickout_ratio)

    def apply(self) -> None:
        """Use these stock allowances and this fixturing standard in this
        process from now on (the analysis hands them on to its worker
        processes)."""
        from .machining.geometry import fixture, stock

        stock.set_allowances(**self.stock_allowances())
        fixture.set_standard(self.fixture())
