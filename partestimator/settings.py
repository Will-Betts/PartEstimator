"""User settings, kept in `settings.json` beside the launcher.

- the sheet metal thickness cutoff;
- the stock allowances: how much the recommended stock adds to the part
  on length, width, plate thickness and bar diameter before rounding up to
  the next 1/4 in (machining/geometry/stock.py).

Edit them from Settings... in the window, or in the file itself. A value
missing from the file, or out of range, falls back to the shipped default.
`PE_SETTINGS` names another file (the tests point it at one that does not
exist, so a developer's own settings cannot change what they check).
"""
from __future__ import annotations

import json
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
            try:
                v = float(data[f.name])
            except (TypeError, ValueError):
                continue
            ok = (0.0 < v < 2.0 if f.name == "sheet_cutoff_in"
                  else 0.0 <= v <= MAX_PAD_IN)
            if ok:
                setattr(s, f.name, v)
        return s

    def save(self, path: Path | None = None) -> None:
        path = path or settings_path()
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    def stock_allowances(self) -> dict:
        """The stock allowances, as stock.set_allowances takes them."""
        return {name: getattr(self, f) for f, name in ALLOWANCE_FIELDS.items()}

    def apply(self) -> None:
        """Use these stock allowances in this process from now on (the
        analysis hands them on to its worker processes)."""
        from .machining.geometry import stock

        stock.set_allowances(**self.stock_allowances())
