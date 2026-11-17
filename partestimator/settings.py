"""User settings, kept in `settings.json` beside the launcher.

Only one today: the sheet metal thickness cutoff. Edit it from
Settings... in the window, or in the file itself.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .classify import DEFAULT_CUTOFF_IN


def tool_dir() -> Path:
    """The PartEstimator folder (the one holding the launcher)."""
    return Path(__file__).resolve().parents[1]


SETTINGS_PATH = tool_dir() / "settings.json"


@dataclass
class Settings:
    # Flat, uniform-thickness parts at or under this are sheet metal (inches).
    sheet_cutoff_in: float = DEFAULT_CUTOFF_IN

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        path = path or SETTINGS_PATH
        s = cls()
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return s
        try:
            v = float(data.get("sheet_cutoff_in", s.sheet_cutoff_in))
            if 0.0 < v < 2.0:
                s.sheet_cutoff_in = v
        except (TypeError, ValueError):
            pass
        return s

    def save(self, path: Path | None = None) -> None:
        path = path or SETTINGS_PATH
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
