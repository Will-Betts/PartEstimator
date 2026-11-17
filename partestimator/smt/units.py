"""Units.

THE RULE: geometry is ALWAYS millimetres internally.

The CAD kernel works in millimetres, the STEP reader is told to convert to
millimetres on import, and every number stored on a `Part` is in millimetres.
Inches exist only at the two edges of the program -- what gets displayed, and
what gets written into a DXF. Nothing in the middle knows or cares.

That rule is worth defending. A tool that carries a unit around with every
number eventually multiplies by 25.4 in one place and not another, and the
result is a blank that is 2.5% wrong -- big enough to scrap parts, small enough
to look plausible on screen.

Detecting what the file used
----------------------------
STEP declares its own units, and OpenCASCADE honours that on import, so an inch
model arrives correctly converted to millimetres either way. But knowing what
the author intended is still useful: it sets a sensible default display unit and
it explains a thickness like 1.905 mm, which is 0.075 inch and nothing standard
in metric. `sniff_step_units` reads the declaration straight out of the file
text rather than asking the kernel, which is both cheaper and version-proof.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

MM_PER_INCH = 25.4


@dataclass(frozen=True)
class Unit:
    name: str            # "mm" | "in"
    label: str
    area_label: str
    per_mm: float        # multiply a millimetre value by this to display it
    decimals: int        # sensible precision for a length
    area_decimals: int
    # DXF $INSUNITS code: 1 = inches, 4 = millimetres
    dxf_code: int
    # Force unit used for tonnage. US shops quote short tons, metric quote tonnes.
    force_label: str
    newtons_per_force_unit: float

    def from_mm(self, value: float) -> float:
        return value * self.per_mm

    def to_mm(self, value: float) -> float:
        return value / self.per_mm

    def length(self, mm: float, with_label: bool = True) -> str:
        v = self.from_mm(mm)
        s = f"{v:,.{self.decimals}f}"
        return f"{s} {self.label}" if with_label else s

    def area(self, mm2: float, with_label: bool = True) -> str:
        v = mm2 * self.per_mm ** 2
        s = f"{v:,.{self.area_decimals}f}"
        return f"{s} {self.area_label}" if with_label else s


MM = Unit(
    name="mm", label="mm", area_label="mm²", per_mm=1.0,
    decimals=2, area_decimals=0, dxf_code=4,
    force_label="t", newtons_per_force_unit=9806.65,          # metric tonne
)

INCH = Unit(
    name="in", label="in", area_label="in²", per_mm=1.0 / MM_PER_INCH,
    decimals=4, area_decimals=3, dxf_code=1,
    force_label="tons", newtons_per_force_unit=8896.44,        # US short ton
)

UNITS = {"mm": MM, "in": INCH}


def get_unit(name: str) -> Unit:
    key = (name or "mm").lower().strip()
    if key in ("inch", "inches", "in", "imperial"):
        return INCH
    return UNITS.get(key, MM)


# ---------------------------------------------------------------------------
# STEP unit sniffing
# ---------------------------------------------------------------------------

_INCH_PAT = re.compile(r"CONVERSION_BASED_UNIT\s*\(\s*'(INCH|inch)", re.I)
_INCH_NAME_PAT = re.compile(r"'(INCH|INCHES)'", re.I)
_MM_PAT = re.compile(r"SI_UNIT\s*\(\s*\.MILLI\.\s*,\s*\.METRE\.", re.I)


def sniff_step_units(path: str | Path, max_bytes: int = 400_000) -> str | None:
    """What unit does this STEP file declare? Returns 'mm', 'in', or None.

    Reads the head of the file only -- unit declarations live near the top of
    the DATA section, well inside this window even for large assemblies.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            head = fh.read(max_bytes)
    except OSError:
        return None

    if _INCH_PAT.search(head) or _INCH_NAME_PAT.search(head):
        return "in"
    if _MM_PAT.search(head):
        return "mm"
    return None


def choose_display_unit(preference: str, file_unit: str | None) -> Unit:
    """Resolve the display unit.

    "auto" follows whatever the file declared, which is almost always what the
    person who drew it wants to read back.
    """
    if preference and preference != "auto":
        return get_unit(preference)
    if file_unit:
        return get_unit(file_unit)
    return MM


# ---------------------------------------------------------------------------
# gauge and tooling tables, imperial
# ---------------------------------------------------------------------------

# Sheet steel gauge -> inches. The same physical gauges as the metric table in
# materials.py, expressed the way a US shop orders them.
GAUGE_INCHES: dict[int, float] = {
    26: 0.0179, 24: 0.0239, 22: 0.0299, 20: 0.0359, 18: 0.0478,
    16: 0.0598, 14: 0.0747, 12: 0.1046, 11: 0.1196, 10: 0.1345,
    8: 0.1644, 7: 0.1793,
}

# Common imperial stock thicknesses, in inches.
STANDARD_GAUGES_IN: tuple[float, ...] = (
    0.018, 0.024, 0.030, 0.036, 0.048, 0.060, 0.063, 0.075, 0.090,
    0.100, 0.105, 0.120, 0.125, 0.134, 0.164, 0.187, 0.250, 0.312,
    0.375, 0.500,
)

# Common imperial V-die openings, in inches.
STANDARD_V_DIES_IN: tuple[float, ...] = (
    0.25, 0.375, 0.5, 0.625, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5,
    3.0, 4.0, 5.0, 6.0, 8.0,
)


def standard_gauges_mm_for(unit: Unit) -> tuple[float, ...]:
    """Stock thicknesses in MILLIMETRES, drawn from the right tradition.

    Snapping a 0.075 inch sheet against a metric stock list produces 2.0 mm,
    which is not what it is. Matching against imperial stock gives 0.075 inch
    back, exactly.
    """
    if unit.name == "in":
        return tuple(g * MM_PER_INCH for g in STANDARD_GAUGES_IN)
    from .materials import STANDARD_GAUGES_MM

    return STANDARD_GAUGES_MM


def standard_dies_mm_for(unit: Unit) -> tuple[float, ...]:
    if unit.name == "in":
        return tuple(v * MM_PER_INCH for v in STANDARD_V_DIES_IN)
    from .materials import STANDARD_V_DIES

    return STANDARD_V_DIES


def nearest_gauge_number_in(thickness_mm: float, rel_tol: float = 0.06):
    """Best-matching US sheet gauge number, or None."""
    t_in = thickness_mm / MM_PER_INCH
    if not GAUGE_INCHES:
        return None
    n = min(GAUGE_INCHES, key=lambda k: abs(GAUGE_INCHES[k] - t_in))
    return n if abs(GAUGE_INCHES[n] - t_in) / t_in < rel_tol else None
