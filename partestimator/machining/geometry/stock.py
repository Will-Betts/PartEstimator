"""Stock size from the selected bounding box.

Reported as three columns, matching how stock is actually called out on
the floor:

    OD/ST     W          L
    ------    -------    -------
    prismatic:  thickness, width, length
    cylindrical: diameter, (blank), bar length

The allowances are imperial by definition, so everything here is computed
in inches regardless of the display unit.

  Prismatic (plate / sheet / block)
    OD/ST = the SHORTEST of the three box dimensions:
              <= 0.190 in  -> used as-is (sheet; bought to gauge, so the
                              part thickness IS the material thickness)
              >  0.190 in  -> + 0.125 in, then ceil to the next 1/4 in
    W     = middle dimension + 1.00 in, ceil to the next 1/4 in
    L     = longest dimension + 1.00 in, ceil to the next 1/4 in

  Cylindrical (barstock)
    OD/ST = diameter + 0.125 in, ceil to the next 1/4 in
    W     = blank (a round bar has no second cross dimension)
    L     = length + 1.00 in, ceil to the next 1/4 in

Why the 0.190 in break: below it the part is sheet, which is bought to a
gauge thickness and not machined on the faces at all, so adding grip
stock and rounding up would invent material and -- more importantly for
the cycle-time model -- invent a facing operation that never happens.
Above it the part comes from plate, which is bought oversize and faced
down, so it gets the 0.125 in cleanup allowance before rounding.

The 1.00 in on W and L is work-holding: enough to get into a vise or
soft jaws and still have the part clear of the cut.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

MM_PER_INCH = 25.4

# Below this thickness the part is sheet: the stock thickness equals the
# part thickness, with no cleanup allowance and no rounding.
SHEET_THICKNESS_MAX_IN = 0.190
# Plate thickness and bar diameter get this much for cleanup before
# rounding up to the next increment.
CLEANUP_ALLOWANCE_IN = 0.125
# Width and length get this much for work-holding before rounding.
GRIP_ALLOWANCE_IN = 1.00
ROUND_INCREMENT_IN = 0.25


def ceil_to_increment(value_in: float, increment_in: float = ROUND_INCREMENT_IN) -> float:
    """Rounds up to the next multiple of `increment_in`. A value already on
    a multiple stays put (it's a ceiling, not a strict 'next')."""
    if increment_in <= 0:
        return value_in
    # Nudge by a tiny epsilon so floating point noise (e.g. 0.7499999999)
    # doesn't push a value onto the next increment.
    return math.ceil(value_in / increment_in - 1e-9) * increment_in


def thickness_stock_in(part_thickness_in: float) -> float:
    """Stock thickness for a prismatic part, applying the sheet break."""
    if part_thickness_in <= SHEET_THICKNESS_MAX_IN:
        return round(float(part_thickness_in), 6)
    return ceil_to_increment(part_thickness_in + CLEANUP_ALLOWANCE_IN)


@dataclass
class StockSize:
    kind: str                  # "rectilinear" or "cylindrical"

    # The three reported columns. `width_in` is None for barstock, which
    # is what puts a blank in the W column rather than a zero -- a zero
    # would read as a real measurement of nothing.
    thickness_or_diameter_in: float = 0.0   # the OD/ST column
    width_in: float | None = None           # the W column
    length_in: float = 0.0                  # the L column

    is_sheet: bool = False     # thickness passed through without allowance
    volume_in3: float = 0.0

    @property
    def is_round(self) -> bool:
        return self.kind == "cylindrical"

    def dims_mm(self) -> tuple:
        """Stock envelope in mm, for drawing the translucent box."""
        if self.kind == "rectilinear":
            return (self.length_in * MM_PER_INCH,
                    (self.width_in or 0.0) * MM_PER_INCH,
                    self.thickness_or_diameter_in * MM_PER_INCH)
        return (self.thickness_or_diameter_in * MM_PER_INCH,
                self.length_in * MM_PER_INCH)

    def table_rows(self) -> list[tuple[str, str]]:
        """(label, value) pairs for the on-screen table."""
        w = "" if self.width_in is None else f"{self.width_in:.3f}"
        return [
            ("OD/ST", f"{self.thickness_or_diameter_in:.3f}"),
            ("W", w),
            ("L", f"{self.length_in:.3f}"),
        ]

    def tsv(self, include_header: bool = True) -> str:
        """Tab-separated, so pasting into Excel lands in three cells.

        Tabs rather than commas on purpose: Excel splits pasted text on
        tabs automatically, whereas a comma-separated line arrives as one
        cell and needs Text-to-Columns.
        """
        w = "" if self.width_in is None else f"{self.width_in:.3f}"
        values = f"{self.thickness_or_diameter_in:.3f}\t{w}\t{self.length_in:.3f}"
        if include_header:
            return "OD/ST\tW\tL\n" + values
        return values


def rectilinear_stock(extents_mm) -> StockSize:
    """`extents_mm` is the oriented box's three extents, in any order.

    Sorted so length >= width >= thickness. The two largest faces are what
    gets clamped; the smallest dimension is the thickness, which is the
    one that decides sheet vs plate.
    """
    dims_in = sorted((float(d) / MM_PER_INCH for d in extents_mm), reverse=True)
    part_thickness = dims_in[2]

    thickness = thickness_stock_in(part_thickness)
    width = ceil_to_increment(dims_in[1] + GRIP_ALLOWANCE_IN)
    length = ceil_to_increment(dims_in[0] + GRIP_ALLOWANCE_IN)

    return StockSize(
        kind="rectilinear",
        thickness_or_diameter_in=thickness,
        width_in=width,
        length_in=length,
        is_sheet=part_thickness <= SHEET_THICKNESS_MAX_IN,
        volume_in3=thickness * width * length,
    )


def cylindrical_stock(diameter_mm: float, length_mm: float) -> StockSize:
    """Barstock. The sheet break does not apply to round stock -- bar is
    bought in fractional diameters, so a small bar still rounds up."""
    diameter = ceil_to_increment(diameter_mm / MM_PER_INCH + CLEANUP_ALLOWANCE_IN)
    length = ceil_to_increment(length_mm / MM_PER_INCH + GRIP_ALLOWANCE_IN)
    return StockSize(
        kind="cylindrical",
        thickness_or_diameter_in=diameter,
        width_in=None,
        length_in=length,
        is_sheet=False,
        volume_in3=math.pi * (diameter / 2.0) ** 2 * length,
    )


def compute_stock(primary: str, rect_box, cylinder) -> StockSize | None:
    """Stock for whichever bounding box was selected."""
    if primary == "cylindrical" and cylinder is not None:
        return cylindrical_stock(2.0 * cylinder.radius_mm, cylinder.height_mm)
    if rect_box is not None:
        return rectilinear_stock(rect_box.extents_mm)
    return None
