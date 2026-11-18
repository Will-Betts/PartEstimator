"""Standard extrusion profiles as stock: pick one, cut it to length.

The profiles live in `machining/data/extrusions.csv`, one row per size,
defined by shape and nominal dimensions (inches):

  shape        a                b                t              t2
  -----------  ---------------  ---------------  -------------  -------------
  angle        leg 1            leg 2            thickness
  channel      depth (web)      flange width     web thickness  flange thick.
  i_beam       depth            flange width     web thickness  flange thick.
  tee          flange width     overall height   stem thick.    flange thick.
  rect_tube    width            height           wall
  round_tube   OD                                wall
  flat_bar     width            thickness
  hex_bar      across flats

Cross-section area is computed from those dimensions with SHARP corners:
real extrusions have root and toe radii that add or remove a few percent.
Add rows freely; the loader rejects a row whose dimensions are impossible.

STANDARD PROFILES. `machining/data/standard_extrusions.csv` holds the
aerospace extrusion standards, one row per die (dash number), picked by part
number such as AND10134-1205. The cancelled AND101xx drawings and the SAE
AMS101xx standards that replaced them share the same dash numbers, so
AMS10134-1205 finds the same row. Columns `a_in` ... `r_in` map onto the
same fields as above, plus three shapes only the standards use:

  shape        a                b                c              t      t2      r
  -----------  ---------------  ---------------  -------------  -----  ------  -----------
  zee          web height       bottom flange    top flange     thick.
  bulb_angle   leg with bulb    plain leg                       thick.         bulb radius
  bulb_tee     flange width     overall height                  stem   flange  bulb radius

`r_in` is the root fillet on the other shapes, kept for reference only.
`area_in2` is the area the standard lists (radii included); it is the area
used for stock volume and weight when given, and the outline is drawn with
sharp corners. Every row was checked against the drawings published by
everyspec.com; AND10141's table is not posted there and its rows come from
the aircraftextrusion.com catalog.

FITTING. Which way the profile runs along the part is chosen automatically:
every axis of the part's bounding box is tried as the extrusion direction
(and, for round profiles, the axis of the enclosing cylinder), and of those
whose cross-section fits inside the profile, the shortest wins -- the
cross-section area is fixed, so the shortest cut is the least material.
The check is against the profile's ENVELOPE, exactly for bars and round
tube, but only approximately for open shapes: whether an L-bracket's own
section really sits inside an angle's legs is for the programmer to judge.

A second check catches what the envelope cannot. With the part's mesh
vertices (the app always has them), the profile is turned, flipped and slid
within the part's section box to the pose holding the most of them; it holds
the part when 98% lie inside its section, and that pose is how it is drawn.
Without vertices, a cruder test stands in. The part's mean section
(its volume over its length) has to come out of the profile material lying
inside a window the size of the part's section. A solid 0.9 in square bar
fits inside a 2 x 2 x 1/4 angle's envelope, but no 0.9 in window holds more
than 0.39 in^2 of that angle, against the 0.81 in^2 the bar needs, so the
angle is rejected; an L-bracket cut from it passes. The window is tried at
the envelope's corners, edge midpoints and centre, in both orientations.
The 5% slack covers the corner radii the sharp-corner areas leave out.

LENGTH, like the L column of ordinary stock:
  minimum      = the part's length along the chosen direction
  recommended  = minimum + 1.00 in for work-holding, up to the next 1/4 in
"""
from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import stock as stock_mod

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "extrusions.csv"
STANDARD_PATH = Path(__file__).resolve().parents[1] / "data" / "standard_extrusions.csv"
MM_PER_INCH = stock_mod.MM_PER_INCH
FIT_TOL_IN = 1e-3
#: A profile may hold this much less than the part's volume and still pass:
#: real sections carry root fillets the sharp-corner area does not count.
VOLUME_SLACK = 0.05
CYLINDER_FIT_SLACK = 0.01

#: Shape key -> label shown in the picker, in display order.
SHAPES = {
    "angle": "Angle",
    "channel": "Channel",
    "i_beam": "I-beam",
    "tee": "Tee",
    "rect_tube": "Rectangular / square tube",
    "round_tube": "Round tube",
    "flat_bar": "Flat / square bar",
    "hex_bar": "Hex bar",
    "zee": "Zee",
    "bulb_angle": "Bulb angle",
    "bulb_tee": "Bulb tee",
}
ROUND_SHAPES = ("round_tube",)
#: Shapes whose area is measured off the drawn outline (they have arcs).
BULB_SHAPES = ("bulb_angle", "bulb_tee")
CIRCLE_SEGMENTS = 48
BULB_SEGMENTS = 64     # per full circle: the drawn bulb is within 0.2% of a true one

#: Standard -> what it covers. AMS101xx (SAE) replaced AND101xx with the
#: same dash numbers.
STANDARDS = {
    "AND10133": "Angle, equal leg",
    "AND10134": "Angle, unequal leg",
    "AND10135": "Angle, bulb section",
    "AND10136": "Tee",
    "AND10137": "Channel",
    "AND10138": "Zee, equal leg",
    "AND10139": "Zee, unequal leg",
    "AND10140": "I-section",
    "AND10141": "Tee, bulb section",
}


def ams_name(spec: str) -> str:
    """AND10134 -> AMS10134, the SAE standard that superseded it."""
    return "AMS" + spec[3:] if spec.upper().startswith("AND") else spec


def _fmt(x: float) -> str:
    return f"{x:.3f}"


@dataclass(frozen=True)
class ExtrusionProfile:
    name: str
    shape: str
    a: float
    b: float = 0.0
    t: float = 0.0
    t2: float = 0.0
    notes: str = ""
    c: float = 0.0                 # zee: top flange
    r: float = 0.0                 # bulb shapes: bulb radius
    spec: str = ""                 # "AND10134" for a standard profile, else ""
    listed_area_in2: float = 0.0   # the standard's own area, radii included
    custom: bool = False           # from the user's own custom_extrusions.csv

    @property
    def shape_label(self) -> str:
        return SHAPES.get(self.shape, self.shape)

    @property
    def is_standard(self) -> bool:
        return bool(self.spec)

    @property
    def alias(self) -> str:
        """The same die under the SAE number: AND10134-1205 -> AMS10134-1205."""
        return ams_name(self.name) if self.is_standard else ""

    @property
    def size_text(self) -> str:
        """Nominal size in the standard's own terms, inches."""
        s, f = self.shape, _fmt
        if s in ("zee",):
            return f"{f(self.a)} x {f(self.b)} x {f(self.c)} x {f(self.t)}"
        if s == "bulb_angle":
            return f"{f(self.a)} x {f(self.b)} x {f(self.t)}, R{f(self.r)} bulb"
        if s == "bulb_tee":
            return (f"{f(self.a)} x {f(self.b)}, flange {f(self.t2)}, stem {f(self.t)}, "
                    f"{f(2 * self.r)} bulb")
        if s in ("round_tube",):
            return f"{f(self.a)} OD x {f(self.t)} wall"
        if s in ("flat_bar",):
            return f"{f(self.a)} x {f(self.b)}"
        if s in ("hex_bar",):
            return f"{f(self.a)} A/F"
        return f"{f(self.a)} x {f(self.b)} x {f(self.t)}"

    @property
    def area_in2(self) -> float:
        """Section area for stock volume: the standard's listed area when
        there is one, else the sharp-cornered area of the drawn section."""
        return self.listed_area_in2 or self.section_area_in2

    @property
    def section_area_in2(self) -> float:
        """Area of the section exactly as `outline_loops()` draws it."""
        a, b, t, t2 = self.a, self.b, self.t, self.t2
        s = self.shape
        if s == "zee":
            return t * (a + self.b + self.c - 2.0 * t)
        if s in BULB_SHAPES:
            loops = self.outline_loops()
            return _shoelace(loops[0]) - sum(_shoelace(lp) for lp in loops[1:])
        if s == "angle":
            return t * (a + b - t)
        if s == "channel":
            return a * t + 2.0 * (b - t) * t2
        if s == "i_beam":
            return 2.0 * b * t2 + (a - 2.0 * t2) * t
        if s == "tee":
            return a * t2 + (b - t2) * t
        if s == "rect_tube":
            return a * b - (a - 2.0 * t) * (b - 2.0 * t)
        if s == "round_tube":
            return math.pi / 4.0 * (a ** 2 - (a - 2.0 * t) ** 2)
        if s == "flat_bar":
            return a * b
        if s == "hex_bar":
            return math.sqrt(3.0) / 2.0 * a ** 2
        raise ValueError(f"unknown extrusion shape {s!r}")

    @property
    def envelope_in(self) -> tuple[float, float]:
        """Outer size along the profile's x and y (inches)."""
        if self.shape == "round_tube":
            return (self.a, self.a)
        if self.shape == "hex_bar":
            return (2.0 * self.a / math.sqrt(3.0), self.a)   # corners on x
        if self.shape == "zee":
            return (self.b + self.c - self.t, self.a)
        if self.shape == "bulb_angle":
            return (self.a, max(self.b, 2.0 * self.r))
        return (self.a, self.b)

    def describe(self) -> str:
        if self.custom:
            return f"{self.name} (custom) {self.shape_label} {self.size_text}"
        if self.is_standard:
            title = STANDARDS.get(self.spec, self.shape_label)
            return f"{self.name} ({self.alias}) {title} {self.size_text}"
        return f"{self.shape_label} {self.name}"

    # -- fitting -------------------------------------------------------

    def fits_rect(self, c1: float, c2: float, loose: bool = False) -> bool:
        """Does a c1 x c2 rectangle (inches, centred) fit inside the profile's
        envelope, in either orientation?

        `loose` asks only whether the section box could fit at all, for when
        the part's surface points will decide: a round part's box corners
        stick out of a round tube that the part itself fits perfectly.
        """
        tol = FIT_TOL_IN
        if loose and self.shape in ("round_tube", "hex_bar"):
            ex_, ey_ = self.envelope_in
            return max(c1, c2) <= max(ex_, ey_) + tol and min(c1, c2) <= min(ex_, ey_) + tol
        if self.shape == "round_tube":
            return math.hypot(c1, c2) <= self.a + tol
        if self.shape == "hex_bar":
            r = self.a / math.sqrt(3.0)
            for w, h in ((c1, c2), (c2, c1)):
                if h <= self.a + tol and w <= 2.0 * r - h / math.sqrt(3.0) + tol:
                    return True
            return False
        big, small = max(self.envelope_in), min(self.envelope_in)
        return max(c1, c2) <= big + tol and min(c1, c2) <= small + tol

    def section_capacity(self, c1: float, c2: float) -> float:
        """Most profile area (in^2) inside a c1 x c2 window over the section."""
        loops = self.outline_loops()
        ex_, ey_ = (max(abs(x) for x, _ in loops[0]), max(abs(y) for _, y in loops[0]))
        best = 0.0
        for w, h in ((c1, c2), (c2, c1)):
            for x0 in (-ex_, -w / 2.0, ex_ - w):
                for y0 in (-ey_, -h / 2.0, ey_ - h):
                    rect = (x0, y0, x0 + w, y0 + h)
                    area = _clipped_area(loops[0], rect)
                    area -= sum(_clipped_area(lp, rect) for lp in loops[1:])
                    best = max(best, area)
        return best

    def fits_circle(self, d: float) -> bool:
        # The enclosing-cylinder fit runs about 1% large on a true cylinder.
        tol = max(FIT_TOL_IN, CYLINDER_FIT_SLACK * d)
        if self.shape == "round_tube":
            return d <= self.a + tol
        return d <= min(self.envelope_in) + tol

    # -- drawing -------------------------------------------------------

    def outline_loops(self) -> list[list[tuple[float, float]]]:
        """Closed outline(s) in inches, centred on the envelope. Tubes give
        an outer and an inner loop with the same number of points."""
        a, b, t, t2 = self.a, self.b, self.t, self.t2
        ha, hb = a / 2.0, b / 2.0
        s = self.shape
        if s == "angle":
            return [[(-ha, -hb), (ha, -hb), (ha, -hb + t), (-ha + t, -hb + t),
                     (-ha + t, hb), (-ha, hb)]]
        if s == "channel":
            return [[(-ha, -hb), (ha, -hb), (ha, hb), (ha - t2, hb), (ha - t2, -hb + t),
                     (-ha + t2, -hb + t), (-ha + t2, hb), (-ha, hb)]]
        if s == "i_beam":
            ht = t / 2.0
            return [[(-ha, -hb), (-ha + t2, -hb), (-ha + t2, -ht), (ha - t2, -ht),
                     (ha - t2, -hb), (ha, -hb), (ha, hb), (ha - t2, hb), (ha - t2, ht),
                     (-ha + t2, ht), (-ha + t2, hb), (-ha, hb)]]
        if s == "tee":
            ht = t / 2.0
            return [[(-ht, -hb), (ht, -hb), (ht, hb - t2), (ha, hb - t2), (ha, hb),
                     (-ha, hb), (-ha, hb - t2), (-ht, hb - t2)]]
        if s == "rect_tube":
            return [_rect(ha, hb), _rect(ha - t, hb - t)]
        if s == "round_tube":
            return [_circle(ha), _circle(ha - t)]
        if s == "flat_bar":
            return [_rect(ha, hb)]
        if s == "hex_bar":
            r = a / math.sqrt(3.0)
            return [[(r * math.cos(k * math.pi / 3.0), r * math.sin(k * math.pi / 3.0))
                     for k in range(6)]]
        if s == "zee":
            # Web up the middle, bottom flange to +x, top flange to -x.
            c = self.c
            pts = [(0.0, 0.0), (b, 0.0), (b, t), (t, t), (t, a), (t - c, a),
                   (t - c, a - t), (0.0, a - t)]
            return [_centred(pts)]
        if s == "bulb_angle":
            return [_centred(_bulb_angle(a, b, t, self.r))]
        if s == "bulb_tee":
            return [_centred(_bulb_tee(a, b, t, t2, self.r))]
        raise ValueError(f"unknown extrusion shape {s!r}")


def _shoelace(poly) -> float:
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2)
                   in zip(poly, poly[1:] + poly[:1]))) / 2.0


def _clipped_area(poly, rect) -> float:
    """Area of `poly` inside an axis-aligned rectangle (Sutherland-Hodgman;
    correct in area for a concave polygon against a convex window)."""
    x0, y0, x1, y1 = rect
    edges = ((lambda p: p[0] >= x0, lambda a, b: _cut_x(a, b, x0)),
             (lambda p: p[0] <= x1, lambda a, b: _cut_x(a, b, x1)),
             (lambda p: p[1] >= y0, lambda a, b: _cut_y(a, b, y0)),
             (lambda p: p[1] <= y1, lambda a, b: _cut_y(a, b, y1)))
    out = list(poly)
    for inside, cut in edges:
        if not out:
            return 0.0
        src, out = out, []
        for k, cur in enumerate(src):
            prev = src[k - 1]
            if inside(cur):
                if not inside(prev):
                    out.append(cut(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(cut(prev, cur))
    return _shoelace(out) if len(out) >= 3 else 0.0


def _cut_x(a, b, x):
    t = (x - a[0]) / (b[0] - a[0])
    return (x, a[1] + t * (b[1] - a[1]))


def _cut_y(a, b, y):
    t = (y - a[1]) / (b[1] - a[1])
    return (a[0] + t * (b[0] - a[0]), y)


def _rect(hx: float, hy: float) -> list[tuple[float, float]]:
    return [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]


def _circle(r: float) -> list[tuple[float, float]]:
    return [(r * math.cos(2 * math.pi * k / CIRCLE_SEGMENTS),
             r * math.sin(2 * math.pi * k / CIRCLE_SEGMENTS)) for k in range(CIRCLE_SEGMENTS)]


def _centred(pts) -> list[tuple[float, float]]:
    """Shift an outline so its bounding box is centred on the origin."""
    xs, ys = [x for x, _ in pts], [y for _, y in pts]
    cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    return [(x - cx, y - cy) for x, y in pts]


def _arc(cx, cy, r, start, end, segments: int = BULB_SEGMENTS):
    """Points along a circle from angle `start` to `end` (radians, CCW,
    end > start), both ends included, and every quarter point between so
    the drawn bulb reaches the envelope exactly."""
    n = max(2, int(math.ceil(segments * (end - start) / (2.0 * math.pi))))
    angles = {start + (end - start) * k / n for k in range(n + 1)}
    q = math.ceil(start / (math.pi / 2.0))
    while q * math.pi / 2.0 < end:
        angles.add(q * math.pi / 2.0)
        q += 1
    return [(cx + r * math.cos(th), cy + r * math.sin(th)) for th in sorted(angles)]


def _bulb_angle(a, b, t, r):
    """AND10135: the A leg along +x with a round bulb of radius r on its
    inner face at the toe, tangent to the toe and the leg's outer face.
    Heel at the origin, B leg up +y."""
    cx, cy = a - r, r
    # The leg runs square to the toe; the bulb rises off its inner face
    # (y = t), crossing it on the toe side and again on the heel side.
    half = math.asin((r - t) / r)
    return ([(0.0, 0.0), (a, 0.0), (a, t)] + _arc(cx, cy, r, -half, math.pi + half)
            + [(t, t), (t, b), (0.0, b)])


def _bulb_tee(a, b, t, t2, r):
    """AND10141: flange across the top, stem down the middle, and a round
    bulb of radius r on the stem's tip, flush with the stem's +x face."""
    ht = t / 2.0
    cx, cy = ht - r, r
    # The stem's -x face meets the bulb on its upper side.
    start = math.acos((r - t) / r)
    return ([(ht, r), (ht, b - t2), (a / 2.0, b - t2), (a / 2.0, b), (-a / 2.0, b),
             (-a / 2.0, b - t2), (-ht, b - t2)]
            + _arc(cx, cy, r, start, 2.0 * math.pi)[:-1])


def validate(p: ExtrusionProfile) -> None:
    """Raise ValueError if the dimensions cannot describe a real section."""
    if p.shape not in SHAPES:
        raise ValueError(f"{p.name}: unknown shape {p.shape!r}")
    if p.a <= 0:
        raise ValueError(f"{p.name}: a must be positive")
    needs_b = p.shape not in ("round_tube", "hex_bar")
    if needs_b and p.b <= 0:
        raise ValueError(f"{p.name}: b must be positive")
    if p.shape in ("angle", "channel", "i_beam", "tee", "rect_tube", "round_tube",
                   "zee", "bulb_angle", "bulb_tee") and p.t <= 0:
        raise ValueError(f"{p.name}: t must be positive")
    if p.shape in ("channel", "i_beam", "tee", "bulb_tee") and p.t2 <= 0:
        raise ValueError(f"{p.name}: t2 must be positive")
    if p.shape == "zee" and p.c <= 0:
        raise ValueError(f"{p.name}: c must be positive")
    if p.shape in BULB_SHAPES and p.r <= 0:
        raise ValueError(f"{p.name}: r (bulb radius) must be positive")
    checks = {
        "angle": p.t < min(p.a, p.b),
        "channel": p.t < p.b and 2 * p.t2 < p.a,
        "i_beam": p.t < p.b and 2 * p.t2 < p.a,
        "tee": p.t < p.a and p.t2 < p.b,
        "rect_tube": 2 * p.t < min(p.a, p.b),
        "round_tube": 2 * p.t < p.a,
        "zee": p.t < min(p.b, p.c) and 2 * p.t < p.a,
        # The bulb is thicker than the leg and clear of the other leg.
        "bulb_angle": p.t < 2 * p.r and p.t < p.b and p.t + 2 * p.r < p.a,
        "bulb_tee": (p.t < 2 * p.r and p.t2 + 2 * p.r < p.b
                     and 2 * p.r - p.t / 2 <= p.a / 2),
    }
    if not checks.get(p.shape, True):
        raise ValueError(f"{p.name}: wall or flange is too thick for the section")


def _num(v) -> float:
    v = (v or "").strip()
    return float(v) if v else 0.0


@lru_cache(maxsize=4)
def load_catalog(path: str | None = None) -> tuple[ExtrusionProfile, ...]:
    """The general catalog: nominal commercial sizes, by shape."""
    p = Path(path) if path else DATA_PATH
    out = []
    with p.open("r", newline="", encoding="utf-8") as fh:
        rows = (r for r in csv.DictReader(fh) if (r.get("name") or "").strip()
                and not r["name"].lstrip().startswith("#"))
        for row in rows:
            prof = ExtrusionProfile(
                name=row["name"].strip(), shape=row["shape"].strip(),
                a=_num(row.get("a_in")), b=_num(row.get("b_in")),
                t=_num(row.get("t_in")), t2=_num(row.get("t2_in")),
                notes=(row.get("notes") or "").strip())
            validate(prof)
            out.append(prof)
    keys = [(q.shape, q.name) for q in out]
    if len(set(keys)) != len(keys):
        raise ValueError("extrusions.csv has a duplicate shape + name")
    return tuple(out)


@lru_cache(maxsize=4)
def load_standard(path: str | None = None) -> tuple[ExtrusionProfile, ...]:
    """The AND / AMS standard profiles, named by part number (AND10134-1205)."""
    p = Path(path) if path else STANDARD_PATH
    out = []
    with p.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            spec = (row.get("spec") or "").strip().upper()
            if not spec or spec.startswith("#"):
                continue
            source = (row.get("source") or "").strip()
            prof = ExtrusionProfile(
                name=f"{spec}-{row['dash'].strip()}", shape=row["shape"].strip(),
                a=_num(row.get("a_in")), b=_num(row.get("b_in")), c=_num(row.get("c_in")),
                t=_num(row.get("t_in")), t2=_num(row.get("t2_in")), r=_num(row.get("r_in")),
                listed_area_in2=_num(row.get("area_in2")), spec=spec,
                notes=f"{source}. Verify against the current SAE AMS revision.")
            validate(prof)
            out.append(prof)
    names = [q.name for q in out]
    if len(set(names)) != len(names):
        raise ValueError("standard_extrusions.csv has a duplicate part number")
    return tuple(out)


# ---------------------------------------------------------------------------
# the user's own library
# ---------------------------------------------------------------------------

#: The user's own profiles: beside the launcher, like settings.json, so
#: deleting the tool folder still removes everything the tool keeps.
CUSTOM_PATH = Path(__file__).resolve().parents[3] / "custom_extrusions.csv"
CUSTOM_COLUMNS = ("name", "shape", "a_in", "b_in", "c_in", "t_in", "t2_in", "r_in",
                  "area_in2", "notes")

CUSTOM_TEMPLATE = """\
# Custom extrusion library: your own dies and supplier sizes, offered beside
# the built-in catalog and the AND / AMS standards. One row per profile,
# dimensions in inches. Lines starting with # are ignored.
#
#   shape        a              b               c            t      t2      r
#   angle        leg 1          leg 2                        thick.
#   channel      depth (web)    flange width                 web    flange
#   i_beam       depth          flange width                 web    flange
#   tee          flange width   overall height               stem   flange
#   rect_tube    width          height                       wall
#   round_tube   OD                                          wall
#   flat_bar     width          thickness
#   hex_bar      across flats
#   zee          web height     bottom flange   top flange   thick.
#   bulb_angle   leg with bulb  plain leg                    thick.         bulb radius
#   bulb_tee     flange width   overall height               stem   flange  bulb radius
#
# area_in2 is optional: give the drawing's area (radii included) and stock
# weight uses it; leave it blank and the sharp-cornered section is used.
# A row the tool cannot read is skipped and reported under the stock panel.
# Keep the header line below; add your rows after it.
name,shape,a_in,b_in,c_in,t_in,t2_in,r_in,area_in2,notes
# Examples (remove the # to use):
# ACME-2210,angle,2.000,1.500,,0.125,,,,ACME 2 x 1-1/2 x 1/8 angle
# DIE-0457,bulb_tee,1.750,1.125,,0.060,0.070,0.109,0.205,Supplier drawing D-0457
"""


@dataclass(frozen=True)
class CustomLibrary:
    path: Path
    profiles: tuple[ExtrusionProfile, ...] = ()
    errors: tuple[str, ...] = ()      # "line 7: ...", one per row skipped

    @property
    def exists(self) -> bool:
        return self.path.is_file()


_custom: CustomLibrary | None = None


def custom_library(reload: bool = False) -> CustomLibrary:
    """The user's library, read once and kept until `reload`."""
    global _custom
    if reload or _custom is None or _custom.path != CUSTOM_PATH:
        _custom = read_custom(CUSTOM_PATH)
    return _custom


def read_custom(path: Path) -> CustomLibrary:
    """Read a custom library file. Never raises for what is in it: a row
    that cannot be used is skipped and described in `errors`, because a
    typo in a file the user edits by hand must not stop the tool."""
    path = Path(path)
    if not path.is_file():
        return CustomLibrary(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        return CustomLibrary(path, errors=(f"cannot read the file: {exc}",))
    # Keep line numbers while dropping comments and blank lines.
    lines = [(n, ln) for n, ln in enumerate(text.splitlines(), 1)
             if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        return CustomLibrary(path)
    header = next(csv.reader([lines[0][1]]))
    header = [h.strip().lower() for h in header]
    missing = [c for c in ("name", "shape", "a_in") if c not in header]
    if missing:
        return CustomLibrary(path, errors=(
            f"line {lines[0][0]}: the header needs the columns {', '.join(missing)}",))
    builtin = {(p.shape, p.name) for p in load_catalog() + load_standard()}
    seen: set[tuple[str, str]] = set()
    out, errors = [], []
    for n, ln in lines[1:]:
        row = dict(zip(header, (v.strip() for v in next(csv.reader([ln])))))
        try:
            name, shape = row.get("name", ""), row.get("shape", "").lower()
            if not name:
                raise ValueError("no name")
            prof = ExtrusionProfile(
                name=name, shape=shape,
                a=_num(row.get("a_in")), b=_num(row.get("b_in")), c=_num(row.get("c_in")),
                t=_num(row.get("t_in")), t2=_num(row.get("t2_in")), r=_num(row.get("r_in")),
                listed_area_in2=_num(row.get("area_in2")),
                notes=row.get("notes", ""), custom=True)
            validate(prof)
            if (shape, name) in builtin:
                raise ValueError(f"{name}: already a built-in {SHAPES[shape].lower()}; "
                                 f"give it another name")
            if (shape, name) in seen:
                raise ValueError(f"{name}: listed twice")
            if prof.listed_area_in2 < 0:
                raise ValueError(f"{name}: area_in2 cannot be negative")
        except ValueError as exc:
            errors.append(f"line {n}: {exc}")
            continue
        seen.add((shape, name))
        out.append(prof)
    return CustomLibrary(path, tuple(out), tuple(errors))


def ensure_custom_file(path: Path | None = None) -> Path:
    """Create the custom library from the template if there is none yet."""
    path = Path(path) if path else CUSTOM_PATH
    if not path.exists():
        path.write_text(CUSTOM_TEMPLATE, encoding="utf-8")
    return path


def load_profiles() -> tuple[ExtrusionProfile, ...]:
    """Every profile the tool offers: the catalog, the standards, then the
    user's own library."""
    return load_catalog() + load_standard() + custom_library().profiles


def find(shape: str, name: str) -> ExtrusionProfile | None:
    return next((p for p in load_profiles() if p.shape == shape and p.name == name), None)


_PART_NUMBER = re.compile(r"^\s*(AND|AMS)\s*-?\s*(\d{5})\s*-?\s*(\d{4})\s*$", re.IGNORECASE)


def find_part_number(text: str) -> ExtrusionProfile | None:
    """A standard profile by part number, written either way and loosely:
    "AND10134-1205", "AMS10134-1205", "and 10134 1205"."""
    m = _PART_NUMBER.match(text or "")
    if not m:
        return None
    name = f"AND{m.group(2)}-{m.group(3)}"
    return next((p for p in load_standard() if p.name == name), None)


# ---------------------------------------------------------------------------
# fitting a profile to a part
# ---------------------------------------------------------------------------

#: With the part's mesh vertices to hand, a profile holds the part when at
#: least this share of them lies inside its section (within INSIDE_TOL_IN of
#: the outline counts: the part's own faces sit exactly on it).
INSIDE_FRACTION = 0.98
INSIDE_TOL_IN = 0.01
MAX_POINTS = 1500
COARSE_POINTS = 128


@dataclass
class _Candidate:
    fits: bool                    # section box inside the profile's envelope
    length_in: float
    centre_mm: np.ndarray
    d: np.ndarray                 # extrusion direction
    u: np.ndarray                 # profile x
    v: np.ndarray                 # profile y
    section_in: tuple[float, float]   # part's section box along u, v
    inside: float | None = None   # share of the part's points inside the section

    @property
    def placement(self):
        return (tuple(self.centre_mm), tuple(self.d), tuple(self.u), tuple(self.v))


def _unit(v) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(a))
    return a / n if n else a


def _any_perpendicular(d: np.ndarray) -> np.ndarray:
    ref = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    return _unit(ref - float(ref @ d) * d)


def _inside_fraction(loops, xy: np.ndarray, tol: float = INSIDE_TOL_IN) -> float:
    """Share of 2D points inside the section (even-odd over all loops, so a
    tube's bore is outside), counting points within `tol` of the outline."""
    if len(xy) == 0:
        return 1.0
    x, y = xy[:, :1], xy[:, 1:2]
    inside = np.zeros(len(xy), dtype=bool)
    near = np.zeros(len(xy), dtype=bool)
    for loop in loops:
        a = np.asarray(loop, dtype=float)
        b = np.roll(a, -1, axis=0)
        ax, ay, bx, by = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
        crosses = ((ay > y) != (by > y)) & (
            x < (bx - ax) * (y - ay) / np.where(by == ay, 1e-300, by - ay) + ax)
        inside ^= (np.count_nonzero(crosses, axis=1) % 2).astype(bool)
        ex, ey = bx - ax, by - ay
        L2 = np.where(ex * ex + ey * ey == 0, 1e-300, ex * ex + ey * ey)
        t = np.clip(((x - ax) * ex + (y - ay) * ey) / L2, 0.0, 1.0)
        dist2 = (x - (ax + t * ex)) ** 2 + (y - (ay + t * ey)) ** 2
        near |= (dist2.min(axis=1) <= tol * tol)
    return float(np.count_nonzero(inside | near)) / len(xy)


def _offsets(slack: float) -> list[float]:
    return [0.0] if slack <= FIT_TOL_IN else [0.0, slack, -slack]


def _orient(profile: ExtrusionProfile, cand: _Candidate, pts_mm: np.ndarray | None,
            axis_a: np.ndarray, axis_b: np.ndarray, ca: float, cb: float) -> _Candidate:
    """Turn, flip and slide the profile within the part's section box to the
    pose holding the most of the part. Without points, just centre it."""
    ex, ey = profile.envelope_in
    pairs = [(axis_a, axis_b, ca, cb)] if ex >= ey else [(axis_b, axis_a, cb, ca)]
    if abs(ex - ey) <= FIT_TOL_IN:
        pairs.append((pairs[0][1], pairs[0][0], pairs[0][3], pairs[0][2]))
    if pts_mm is None:
        ua, va, cu, cv = pairs[0]
        cand.u, cand.v, cand.section_in = ua, va, (cu, cv)
        return cand
    loops = profile.outline_loops()
    rel = (pts_mm - cand.centre_mm) / MM_PER_INCH
    # Screen every pose on a coarse subset, then measure only the best one
    # on all the points: most poses are plainly wrong after a hundred.
    coarse = rel[::max(1, len(rel) // COARSE_POINTS)]
    best = None
    for ua, va, cu, cv in pairs:
        for su in (1.0, -1.0):
            for sv in (1.0, -1.0):
                u, v = su * ua, sv * va
                cx, cy = coarse @ u, coarse @ v
                for ou in _offsets((ex - cu) / 2.0):
                    for ov in _offsets((ey - cv) / 2.0):
                        frac = _inside_fraction(loops, np.column_stack((cx - ou, cy - ov)))
                        if best is None or frac > best[0] + 1e-9:
                            best = (frac, u, v, ou, ov, (cu, cv))
    frac, u, v, ou, ov, sec = best
    if frac >= INSIDE_FRACTION - 0.05:
        frac = _inside_fraction(loops, np.column_stack((rel @ u - ou, rel @ v - ov)))
    cand.u, cand.v, cand.section_in, cand.inside = u, v, sec, frac
    cand.centre_mm = cand.centre_mm + (ou * u + ov * v) * MM_PER_INCH
    return cand


def _candidates(profile: ExtrusionProfile, rect_box, cylinder,
                pts_mm: np.ndarray | None) -> list[_Candidate]:
    """Every way the profile could run along the part."""
    out = []
    if rect_box is not None:
        ext = [float(e) / MM_PER_INCH for e in rect_box.extents_mm]
        axes = [_unit(row) for row in rect_box.axes]
        centre = np.asarray(rect_box.center_mm, dtype=float)
        for k in range(3):
            i, j = [m for m in range(3) if m != k]
            if ext[i] < ext[j]:
                i, j = j, i
            c = _Candidate(profile.fits_rect(ext[i], ext[j], loose=pts_mm is not None),
                           ext[k], centre.copy(), axes[k], axes[i], axes[j], (ext[i], ext[j]))
            out.append(_orient(profile, c, pts_mm if c.fits else None,
                               axes[i], axes[j], ext[i], ext[j]))
    if cylinder is not None and profile.shape in ROUND_SHAPES:
        d = _unit(cylinder.axis_direction)
        u = _any_perpendicular(d)
        v = np.cross(d, u)
        dia = 2.0 * cylinder.radius_mm / MM_PER_INCH
        c = _Candidate(profile.fits_circle(dia), cylinder.height_mm / MM_PER_INCH,
                       np.asarray(cylinder.axis_point_mm, dtype=float), d, u, v, (dia, dia))
        out.append(_orient(profile, c, pts_mm if c.fits else None, u, v, dia, dia))
    return out


def surface_samples(points_mm, triangles, n: int = MAX_POINTS, seed: int = 0):
    """Points spread over the part's surface, weighted by triangle area, plus
    every vertex.

    Vertices alone are not enough: a prism's vertices all sit on its two end
    faces, so seen across its length they cluster on two lines and can pass
    for a channel's flanges. Sampling the side faces too closes that hole.
    Seeded, so the same part always gives the same answer.
    """
    if points_mm is None:
        return None
    pts = np.asarray(points_mm, dtype=float).reshape(-1, 3)
    if triangles is None or len(pts) == 0:
        return pts
    tri = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    if len(tri) == 0:
        return pts
    a, b, c = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    if area.sum() <= 0:
        return pts
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(tri), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    flip = r1 + r2 > 1.0
    r1[flip], r2[flip] = 1.0 - r1[flip], 1.0 - r2[flip]
    samples = a[pick] + r1[:, None] * (b[pick] - a[pick]) + r2[:, None] * (c[pick] - a[pick])
    return np.vstack((pts, samples))


def _sample(points_mm, limit: int = 2 * MAX_POINTS) -> np.ndarray | None:
    if points_mm is None:
        return None
    pts = np.asarray(points_mm, dtype=float).reshape(-1, 3)
    if len(pts) == 0:
        return None
    if len(pts) > limit:
        pts = pts[np.linspace(0, len(pts) - 1, limit).astype(int)]
    return pts


def _shortest_fitting_length(profile: ExtrusionProfile, rect_box, cylinder,
                             loose: bool) -> float | None:
    """The shortest way the profile's envelope could run along the part, by
    the cheap envelope checks alone; None when no way fits."""
    lengths = []
    if rect_box is not None:
        ext = [float(e) / MM_PER_INCH for e in rect_box.extents_mm]
        for k in range(3):
            i, j = [m for m in range(3) if m != k]
            if profile.fits_rect(ext[i], ext[j], loose=loose):
                lengths.append(ext[k])
    if cylinder is not None and profile.shape in ROUND_SHAPES:
        if profile.fits_circle(2.0 * cylinder.radius_mm / MM_PER_INCH):
            lengths.append(cylinder.height_mm / MM_PER_INCH)
    return min(lengths) if lengths else None


def best_profile(profiles, rect_box, cylinder=None, part_volume_in3: float | None = None,
                 points_mm=None) -> ExtrusionProfile | None:
    """The profile that holds the part with the least material (at the
    recommended length), or None.

    Hundreds of standard profiles make the full fit test too slow to run on
    every one, so each is first bounded below by its area times the shortest
    length its envelope could run; smallest first, a profile that cannot
    beat the best so far is never fitted.
    """
    pts = _sample(points_mm)
    bounded = []
    for prof in profiles:
        shortest = _shortest_fitting_length(prof, rect_box, cylinder, loose=pts is not None)
        if shortest is not None:
            floor = prof.area_in2 * stock_mod.ceil_to_increment(
                shortest + stock_mod.GRIP_ALLOWANCE_IN)
            bounded.append((floor, prof))
    bounded.sort(key=lambda fp: fp[0])
    best, best_vol = None, None
    for floor, prof in bounded:
        if best_vol is not None and floor >= best_vol - 1e-12:
            break
        s = extrusion_stock(prof, rect_box, cylinder, part_volume_in3=part_volume_in3,
                            points_mm=pts)
        if (s is not None and s.fits and s.enough_material
                and (best_vol is None or s.volume_in3 < best_vol)):
            best, best_vol = prof, s.volume_in3
    return best


def extrusion_stock(profile: ExtrusionProfile, rect_box, cylinder=None,
                    length_in: float | None = None,
                    part_volume_in3: float | None = None,
                    points_mm=None) -> stock_mod.StockSize | None:
    """`profile` cut to length for this part.

    `length_in` None means the recommended length; the minimum and
    recommended are always reported. Whether the profile can hold the part
    is judged from the part's mesh vertices when `points_mm` is given (also
    used to turn the profile the right way round), else from
    `part_volume_in3` against the section's capacity.
    """
    pts = _sample(points_mm)
    cands = _candidates(profile, rect_box, cylinder, pts)
    if not cands:
        return None

    def holds_part(c: _Candidate) -> bool:
        if c.inside is not None:
            return c.inside >= INSIDE_FRACTION
        if part_volume_in3 is None or c.length_in <= 0:
            return True
        need = (1.0 - VOLUME_SLACK) * part_volume_in3 / c.length_in   # mean section, in^2
        return profile.section_capacity(*c.section_in) >= need

    usable = [c for c in cands if c.fits and holds_part(c)]
    fitting = usable or [c for c in cands if c.fits]
    if fitting:
        best = min(fitting, key=lambda c: (c.length_in, -(c.inside or 0.0)))
    else:
        # Nothing fits: run it along the part's longest direction and say so.
        best = max(cands, key=lambda c: c.length_in)
    min_len = best.length_in
    recommended = stock_mod.ceil_to_increment(min_len + stock_mod.GRIP_ALLOWANCE_IN)
    length = recommended if length_in is None else float(length_in)
    ex, ey = profile.envelope_in
    return stock_mod.StockSize(
        kind="extrusion",
        thickness_or_diameter_in=ex,
        width_in=None if profile.shape in ROUND_SHAPES else ey,
        length_in=length,
        volume_in3=profile.area_in2 * length,
        basis=stock_mod.EXTRUSION,
        profile=profile,
        placement=best.placement,
        min_length_in=min_len,
        recommended_length_in=recommended,
        fits=best.fits,
        enough_material=bool(usable) or (best.fits and holds_part(best)),
        section_inside=best.inside,
    )
