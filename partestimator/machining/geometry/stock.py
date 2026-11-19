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

Those rules give the RECOMMENDED stock. The NC programmer may choose
otherwise -- a bar instead of plate, a size already on the shelf, or the
bare part envelope -- and that changes both the roughing time and the
stock weight. So every StockSize records its `basis`:

  recommended   the rules above
  minimum       the part's own envelope, no allowance, no rounding
  custom        dimensions typed in by the user
  extrusion     a standard profile (see geometry/extrusions.py) cut to length

`classify_basis()` names a custom size that happens to equal one of the
first two, so typing the recommended numbers back in reads as recommended.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

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


RECOMMENDED = "recommended"
MINIMUM = "minimum"
CUSTOM = "custom"
EXTRUSION = "extrusion"

#: Two stock sizes closer than this (inches, per dimension) are the same.
SAME_SIZE_TOL_IN = 5e-4


@dataclass
class StockSize:
    kind: str                  # "rectilinear", "cylindrical" or "extrusion"

    # The three reported columns. `width_in` is None for barstock, which
    # is what puts a blank in the W column rather than a zero -- a zero
    # would read as a real measurement of nothing.
    thickness_or_diameter_in: float = 0.0   # the OD/ST column
    width_in: float | None = None           # the W column
    length_in: float = 0.0                  # the L column

    is_sheet: bool = False     # thickness passed through without allowance
    volume_in3: float = 0.0
    basis: str = RECOMMENDED

    # Extrusions only. The OD/ST and W columns then hold the profile's
    # envelope (W blank for round tube). `placement` is where the viewer
    # draws it: (centre_mm, length direction, profile x, profile y), each a
    # 3-tuple.
    profile: object = None     # extrusions.ExtrusionProfile
    placement: tuple | None = None
    min_length_in: float = 0.0
    recommended_length_in: float = 0.0
    fits: bool = True          # does the part's cross-section fit the profile?
    enough_material: bool = True   # can the profile's section hold the part's?
    section_inside: float | None = None   # share of the part's vertices inside it

    @property
    def is_round(self) -> bool:
        return self.kind == "cylindrical"

    @property
    def is_extrusion(self) -> bool:
        return self.kind == "extrusion"

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


def _sorted_in(extents_mm) -> list[float]:
    """Box extents in inches, largest first."""
    return sorted((float(d) / MM_PER_INCH for d in extents_mm), reverse=True)


def rectilinear_stock(extents_mm) -> StockSize:
    """`extents_mm` is the oriented box's three extents, in any order.

    Sorted so length >= width >= thickness. The two largest faces are what
    gets clamped; the smallest dimension is the thickness, which is the
    one that decides sheet vs plate.
    """
    dims_in = _sorted_in(extents_mm)
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


def custom_rectilinear(thickness_in: float, width_in: float, length_in: float,
                       basis: str = CUSTOM) -> StockSize:
    """Plate or block of exactly these dimensions, in inches. Sorted the same
    way the rules sort them, so the order they are given in does not matter."""
    length, width, thickness = sorted((float(thickness_in), float(width_in),
                                       float(length_in)), reverse=True)
    return StockSize(
        kind="rectilinear",
        thickness_or_diameter_in=thickness,
        width_in=width,
        length_in=length,
        is_sheet=0.0 < thickness <= SHEET_THICKNESS_MAX_IN,
        volume_in3=thickness * width * length,
        basis=basis,
    )


def custom_cylindrical(diameter_in: float, length_in: float,
                       basis: str = CUSTOM) -> StockSize:
    d, length = float(diameter_in), float(length_in)
    return StockSize(
        kind="cylindrical",
        thickness_or_diameter_in=d,
        width_in=None,
        length_in=length,
        volume_in3=math.pi * (d / 2.0) ** 2 * length,
        basis=basis,
    )


def minimum_rectilinear(extents_mm) -> StockSize:
    """The part's own box: no cleanup, no grip, no rounding."""
    t_l, t_w, t_t = _sorted_in(extents_mm)
    return custom_rectilinear(t_t, t_w, t_l, basis=MINIMUM)


def minimum_cylindrical(diameter_mm: float, length_mm: float) -> StockSize:
    return custom_cylindrical(diameter_mm / MM_PER_INCH, length_mm / MM_PER_INCH,
                              basis=MINIMUM)


def compute_stock(primary: str, rect_box, cylinder) -> StockSize | None:
    """Stock for whichever bounding box was selected."""
    if primary == "cylindrical" and cylinder is not None:
        return cylindrical_stock(2.0 * cylinder.radius_mm, cylinder.height_mm)
    if rect_box is not None:
        return rectilinear_stock(rect_box.extents_mm)
    return None


def snap(kind: str, basis: str, rect_box, cylinder) -> StockSize | None:
    """Recommended or minimum stock of the given kind ("rectilinear" or
    "cylindrical"), or None when the part has no fit of that kind."""
    if kind == "cylindrical":
        if cylinder is None:
            return None
        d, h = 2.0 * cylinder.radius_mm, cylinder.height_mm
        return minimum_cylindrical(d, h) if basis == MINIMUM else cylindrical_stock(d, h)
    if rect_box is None:
        return None
    if basis == MINIMUM:
        return minimum_rectilinear(rect_box.extents_mm)
    return rectilinear_stock(rect_box.extents_mm)


def stock_features(stock: StockSize, part_volume_in3: float) -> dict:
    """The feature-vector columns that come from the chosen stock.

    Shared by geometry/derived.py and the tuning library, which re-applies a
    part's stock to its captured geometry without the geometry itself.
    """
    out = {
        "stock_is_round": 1 if stock.is_round else 0,
        "stock_is_extrusion": 1 if stock.is_extrusion else 0,
        "stock_volume_in3": float(stock.volume_in3),
        # One set of columns for both forms, matching the OD/ST, W, L table.
        # A round bar leaves W at zero because the CSV needs a number;
        # `stock_is_round` is what distinguishes "no width" from "zero width".
        "stock_od_or_thk_in": float(stock.thickness_or_diameter_in),
        "stock_width_in": float(stock.width_in or 0.0),
        "stock_length_in": float(stock.length_in),
        # Sheet parts skip the facing operation entirely, which is a real
        # fork in the cycle-time model rather than a size difference.
        "stock_is_sheet": 1 if stock.is_sheet else 0,
    }
    if stock.volume_in3 > 0:
        out["stock_removal_ratio"] = float((stock.volume_in3 - part_volume_in3)
                                           / stock.volume_in3)
        out["stock_removed_volume_in3"] = float(max(stock.volume_in3 - part_volume_in3, 0.0))
    return out


def same_size(a: StockSize | None, b: StockSize | None) -> bool:
    if a is None or b is None or a.kind != b.kind:
        return False
    pairs = ((a.thickness_or_diameter_in, b.thickness_or_diameter_in),
             (a.width_in or 0.0, b.width_in or 0.0),
             (a.length_in, b.length_in))
    return all(abs(x - y) <= SAME_SIZE_TOL_IN for x, y in pairs)


def classify_basis(stock: StockSize | None, rect_box, cylinder) -> StockSize | None:
    """`stock` with its basis set to recommended or minimum when it matches
    one of those sizes, else custom. Extrusions keep their own basis."""
    if stock is None or stock.is_extrusion:
        return stock
    for basis in (RECOMMENDED, MINIMUM):
        ref = snap(stock.kind, basis, rect_box, cylinder)
        if same_size(stock, ref):
            return replace(stock, basis=basis, is_sheet=ref.is_sheet)
    return replace(stock, basis=CUSTOM)


def fit_warnings(stock: StockSize | None, rect_box, cylinder) -> list[str]:
    """Plain-language problems with a chosen stock: too small for the part."""
    if stock is None:
        return []
    if stock.is_extrusion:
        out = []
        if not stock.fits:
            out.append("The part's cross-section does not fit inside this profile.")
        elif not stock.enough_material:
            if stock.section_inside is not None:
                out.append(f"The part does not lie inside this profile's section: "
                           f"only {stock.section_inside:.0%} of it does.")
            else:
                out.append("This profile holds less material than the part: its "
                           "section cannot contain the part's.")
        if stock.length_in + SAME_SIZE_TOL_IN < stock.min_length_in:
            out.append(f'Length {stock.length_in:.3f}" is shorter than the part '
                       f'({stock.min_length_in:.3f}").')
        return out
    ref = snap(stock.kind, MINIMUM, rect_box, cylinder)
    if ref is None:
        return []
    if stock.is_round:
        out = []
        if stock.thickness_or_diameter_in + SAME_SIZE_TOL_IN < ref.thickness_or_diameter_in:
            out.append(f'Diameter {stock.thickness_or_diameter_in:.3f}" is smaller than '
                       f'the part ({ref.thickness_or_diameter_in:.3f}").')
        if stock.length_in + SAME_SIZE_TOL_IN < ref.length_in:
            out.append(f'Length {stock.length_in:.3f}" is shorter than the part '
                       f'({ref.length_in:.3f}").')
        return out
    have = (stock.length_in, stock.width_in or 0.0, stock.thickness_or_diameter_in)
    need = (ref.length_in, ref.width_in or 0.0, ref.thickness_or_diameter_in)
    if any(h + SAME_SIZE_TOL_IN < n for h, n in zip(have, need)):
        return [f'Stock {have[2]:.3f}" x {have[1]:.3f}" x {have[0]:.3f}" does not '
                f'enclose the part ({need[2]:.3f}" x {need[1]:.3f}" x {need[0]:.3f}").']
    return []
