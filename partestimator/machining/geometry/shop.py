"""Shop profile: how this shop actually runs, as opposed to what the
tooling is theoretically capable of.

The split matters. `machining_rates.csv` holds published manufacturer
data -- surface speeds, feeds per tooth, unit power -- and those numbers
are citable and should not be edited casually. This file holds the
machine's limits and the overheads that turn a published PEAK rate into
an achieved AVERAGE one, and it is meant to be tuned against your own
measured cycle times.

The single most important entry here is `rough_open_duty`, and the
reason deserves stating plainly: a published MRR is what the tool
sustains while fully engaged. A real roughing toolpath is not fully
engaged for most of its length -- it ramps or helixes in at every level,
slows through corners to hold chip load, links between regions, retracts
between stepdowns, and finishes each level with a partial-width cleanup
pass. Charging the peak rate for the whole volume is the largest single
source of optimism in a bottom-up estimate.

Also here: thread detection. STEP files almost never carry thread data,
so tapped holes are inferred from tap-drill diameters and can be
overridden per part in the UI.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

PROFILE_PATH = Path(__file__).resolve().parent.parent / "data" / "shop_profile.csv"

# Standard tap drills, inch and metric coarse. A hole within
# TAP_DRILL_TOLERANCE_IN of one of these is a tapping candidate.
#
# This is a heuristic and it is wrong sometimes: a clearance hole can
# land on a tap-drill size by coincidence. It is used because the
# alternative -- charging nothing for tapping -- is wrong on every
# tapped part, and the count is surfaced in the UI so it can be
# corrected per part.
TAP_DRILLS_IN = {
    0.0810: "#5-40", 0.0890: "#6-32", 0.1015: "#8-32", 0.1160: "#10-24",
    0.1590: "#10-32", 0.1360: "#6-40", 0.1495: "#12-24",
    0.2010: "1/4-20", 0.2130: "1/4-28",
    0.2570: "5/16-18", 0.2720: "5/16-24",
    0.3125: "3/8-16", 0.3320: "3/8-24",
    0.3680: "7/16-14", 0.3906: "7/16-20",
    0.4219: "1/2-13", 0.4531: "1/2-20",
    0.5312: "5/8-11", 0.6562: "3/4-10",
    # metric coarse
    0.1260: "M4x0.7", 0.1654: "M5x0.8", 0.2008: "M6x1.0",
    0.2717: "M8x1.25", 0.3425: "M10x1.5", 0.4134: "M12x1.75",
}

# Threads per inch for the inch sizes, pitch in mm for the metric ones,
# used to derive tapping feed (tapping is pitch-synchronised).
TAP_PITCH_IN = {
    "#5-40": 1 / 40, "#6-32": 1 / 32, "#8-32": 1 / 32, "#10-24": 1 / 24,
    "#10-32": 1 / 32, "#6-40": 1 / 40, "#12-24": 1 / 24,
    "1/4-20": 1 / 20, "1/4-28": 1 / 28,
    "5/16-18": 1 / 18, "5/16-24": 1 / 24,
    "3/8-16": 1 / 16, "3/8-24": 1 / 24,
    "7/16-14": 1 / 14, "7/16-20": 1 / 20,
    "1/2-13": 1 / 13, "1/2-20": 1 / 20,
    "5/8-11": 1 / 11, "3/4-10": 1 / 10,
    "M4x0.7": 0.7 / 25.4, "M5x0.8": 0.8 / 25.4, "M6x1.0": 1.0 / 25.4,
    "M8x1.25": 1.25 / 25.4, "M10x1.5": 1.5 / 25.4, "M12x1.75": 1.75 / 25.4,
}

TAP_DRILL_TOLERANCE_IN = 0.004

# Keys that must lie in (0, 1]. Listed explicitly so a new multiplier
# cannot be caught by a name-prefix rule it was never meant to match.
FRACTION_KEYS = {
    "drive_efficiency",
    "rough_open_duty", "rough_pocket_duty", "pocket_engagement",
    "rest_rough_fraction", "rest_rough_rate_fraction",
    "finish_duty", "semi_finish_stepover_fraction",
    "spring_pass_fraction", "chamfer_fraction", "tap_depth_fraction",
    "rapid_fraction",
    "deburr_convex_fraction", "deburr_concave_fraction",
    "thin_wall_feed_derate",
    "turn_rough_duty", "turn_finish_duty", "turn_part_sfm_fraction",
    "turn_rapid_fraction", "turn_deburr_fraction", "turn_auto_fraction",
    "fixture_channel_mrr_fraction",
}


@dataclass
class ShopProfile:
    values: dict

    def get(self, key: str, default: float = 0.0) -> float:
        return float(self.values.get(key, default))

    def flag(self, key: str, default: bool = False) -> bool:
        return bool(round(self.get(key, 1.0 if default else 0.0)))

    def note(self, key: str) -> str:
        return self.notes.get(key, "")

    notes: dict = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = {}


def load_profile(path: str | Path | None = None,
                 overrides: dict | None = None) -> ShopProfile:
    """The shop profile, with `overrides` ({key: value}, the shop's own
    tuning -- see machining/tuning.py) laid over the file and checked by the
    same rules."""
    path = Path(path) if path is not None else PROFILE_PATH
    if not path.exists():
        raise FileNotFoundError(f"Shop profile not found at {path}")
    values: dict = {}
    notes: dict = {}
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for line, raw in enumerate(reader, start=2):
            key = (raw.get("key") or "").strip()
            if not key:
                continue
            try:
                values[key] = float(raw["value"])
            except (TypeError, ValueError):
                raise ValueError(
                    f"{path.name} line {line}: {key} value is not a number "
                    f"({raw.get('value')!r})") from None
            notes[key] = (raw.get("notes") or "").strip()
    for key, value in (overrides or {}).items():
        values[key] = float(value)
    for required in ("spindle_hp", "rough_open_duty", "rough_pocket_duty"):
        if required not in values:
            raise ValueError(f"{path.name} is missing required key {required!r}")
    # Validate by explicit name, not by prefix. Prefix matching looked
    # tidy and was wrong: `finish_ra_factor_*` are multipliers that
    # legitimately exceed 1, but the old `finish_*` rule rejected them.
    for key in FRACTION_KEYS:
        if key in values and not 0 < values[key] <= 1.0:
            raise ValueError(
                f"{path.name}: {key} is a fraction and must be in (0, 1], "
                f"got {values[key]}")
    for key, v in values.items():
        if key.startswith("calibrate_") and v <= 0:
            raise ValueError(
                f"{path.name}: {key} must be positive, got {v}")
        if key.startswith("finish_ra_factor_") and v <= 0:
            raise ValueError(
                f"{path.name}: {key} must be positive, got {v}")
    return ShopProfile(values=values, notes=notes)


def identify_tapped_holes(holes, tolerance_in: float = TAP_DRILL_TOLERANCE_IN):
    """Returns [(hole, thread_label, pitch_in)] for holes at tap-drill sizes.

    Only BLIND holes and through holes both qualify -- a tapped hole can be
    either -- so diameter is the only discriminator available.
    """
    MM_PER_IN = 25.4
    out = []
    for h in holes:
        dia_in = float(getattr(h, "nominal_diameter_mm", 0.0)) / MM_PER_IN
        if dia_in <= 0:
            continue
        best = None
        best_err = tolerance_in
        for drill_in, label in TAP_DRILLS_IN.items():
            err = abs(dia_in - drill_in)
            if err <= best_err:
                best, best_err = label, err
        if best is not None:
            out.append((h, best, TAP_PITCH_IN.get(best, 1 / 20)))
    return out
