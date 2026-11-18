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
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import stock as stock_mod

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "extrusions.csv"
MM_PER_INCH = stock_mod.MM_PER_INCH
FIT_TOL_IN = 1e-3
#: A profile may hold this much less than the part's volume and still pass:
#: real sections carry root fillets the sharp-corner area does not count.
VOLUME_SLACK = 0.05

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
}
ROUND_SHAPES = ("round_tube",)
CIRCLE_SEGMENTS = 48


@dataclass(frozen=True)
class ExtrusionProfile:
    name: str
    shape: str
    a: float
    b: float = 0.0
    t: float = 0.0
    t2: float = 0.0
    notes: str = ""

    @property
    def shape_label(self) -> str:
        return SHAPES.get(self.shape, self.shape)

    @property
    def area_in2(self) -> float:
        a, b, t, t2 = self.a, self.b, self.t, self.t2
        s = self.shape
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
        return (self.a, self.b)

    def describe(self) -> str:
        return f"{self.shape_label} {self.name}"

    # -- fitting -------------------------------------------------------

    def fits_rect(self, c1: float, c2: float) -> bool:
        """Does a c1 x c2 rectangle (inches, centred) fit inside the profile's
        envelope, in either orientation?"""
        tol = FIT_TOL_IN
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
        if self.shape == "round_tube":
            return d <= self.a + FIT_TOL_IN
        return d <= min(self.envelope_in) + FIT_TOL_IN

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


def validate(p: ExtrusionProfile) -> None:
    """Raise ValueError if the dimensions cannot describe a real section."""
    if p.shape not in SHAPES:
        raise ValueError(f"{p.name}: unknown shape {p.shape!r}")
    if p.a <= 0:
        raise ValueError(f"{p.name}: a must be positive")
    needs_b = p.shape not in ("round_tube", "hex_bar")
    if needs_b and p.b <= 0:
        raise ValueError(f"{p.name}: b must be positive")
    if p.shape in ("angle", "channel", "i_beam", "tee", "rect_tube", "round_tube") and p.t <= 0:
        raise ValueError(f"{p.name}: t must be positive")
    if p.shape in ("channel", "i_beam", "tee") and p.t2 <= 0:
        raise ValueError(f"{p.name}: t2 must be positive")
    checks = {
        "angle": p.t < min(p.a, p.b),
        "channel": p.t < p.b and 2 * p.t2 < p.a,
        "i_beam": p.t < p.b and 2 * p.t2 < p.a,
        "tee": p.t < p.a and p.t2 < p.b,
        "rect_tube": 2 * p.t < min(p.a, p.b),
        "round_tube": 2 * p.t < p.a,
    }
    if not checks.get(p.shape, True):
        raise ValueError(f"{p.name}: wall or flange is too thick for the section")


def _num(v) -> float:
    v = (v or "").strip()
    return float(v) if v else 0.0


@lru_cache(maxsize=4)
def load_profiles(path: str | None = None) -> tuple[ExtrusionProfile, ...]:
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


def find(shape: str, name: str) -> ExtrusionProfile | None:
    return next((p for p in load_profiles() if p.shape == shape and p.name == name), None)


# ---------------------------------------------------------------------------
# fitting a profile to a part
# ---------------------------------------------------------------------------

#: With the part's mesh vertices to hand, a profile holds the part when at
#: least this share of them lies inside its section (within INSIDE_TOL_IN of
#: the outline counts: the part's own faces sit exactly on it).
INSIDE_FRACTION = 0.98
INSIDE_TOL_IN = 0.01
MAX_POINTS = 1500


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
    best = None
    for ua, va, cu, cv in pairs:
        for su in (1.0, -1.0):
            for sv in (1.0, -1.0):
                u, v = su * ua, sv * va
                px, py = rel @ u, rel @ v
                for ou in _offsets((ex - cu) / 2.0):
                    for ov in _offsets((ey - cv) / 2.0):
                        frac = _inside_fraction(loops, np.column_stack((px - ou, py - ov)))
                        if best is None or frac > best[0] + 1e-9:
                            best = (frac, u, v, ou, ov, (cu, cv))
                        if frac >= 0.999:
                            break
    frac, u, v, ou, ov, sec = best
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
            c = _Candidate(profile.fits_rect(ext[i], ext[j]), ext[k], centre.copy(),
                           axes[k], axes[i], axes[j], (ext[i], ext[j]))
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


def _sample(points_mm, limit: int = MAX_POINTS) -> np.ndarray | None:
    if points_mm is None:
        return None
    pts = np.asarray(points_mm, dtype=float).reshape(-1, 3)
    if len(pts) == 0:
        return None
    if len(pts) > limit:
        pts = pts[np.linspace(0, len(pts) - 1, limit).astype(int)]
    return pts


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
