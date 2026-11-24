"""Tabbed fixturing: the part cut out of its plate with a channel round its
outline, held to the frame of left-over stock by small tabs.

A shop standard rather than a fixture per part. Two procedures:

  VERTICAL, one operation (3+2 machine)
      The blank stands on one edge in a 5-axis vise. Side 1 is faced down to
      the part, the channel is cut round the outline to just past the middle
      and the side-1 features are machined; the blank turns 180 degrees and
      side 2 gets the same. The two channel cuts meet, leaving the tabs --
      centred in the thickness -- holding the part. The clamped edge carries
      the vise's grip plus clearance for the holder.

  FLAT, two operations (thin or tall plate, which would chatter on edge)
      Op 1: the blank lies flat, the top is faced, the channel is cut down to
      the part's bottom and the top features are machined; the layer under
      the part is left as a floor. Op 2: flipped, the floor is faced off down
      to the part; the tabs, standing on the part's bottom, hold it.

Either way the part is centred in its stock and the stock is grown, where it
has to be, so a frame of at least `min_frame_in` stands outside the channel.

Everything here is numpy and scipy on the part's mesh: the outline is found
on a fine grid in the stock's frame, which is plenty for a cutter path and
for the volumes the estimate needs. The solid bodies are built with the CAD
kernel in fixture_cad.py, from the plan made here.

PLAN FRAME (all plan coordinates, in inches):
  X along the clamping edge (flat: along the stock's length)
  Y up out of the jaws      (flat: across the stock's width)
  Z out of side 1           (flat: up)
  origin at the stock's corner on the clamping edge, side-2 face (flat: the
  bottom front-left corner).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from . import stock as stock_mod

MM = stock_mod.MM_PER_INCH
VERTICAL = "vertical"
FLAT = "flat"

#: Clamping edge choices for the vertical procedure: which edge of the
#: stock goes in the vise.
CLAMP_EDGES = ("long edge", "long edge (other side)", "short edge",
               "short edge (other side)")

#: Tabs are kept this far from the ends of a straight edge (inches).
TAB_CORNER_MARGIN_IN = 0.125
#: Tabs reach this far into the part and into the frame, so they join both.
TAB_OVERLAP_IN = 0.020
#: Flat stock with less than this under the part has no floor to face off.
NO_FLOOR_IN = 0.005
#: Grid cell size limits for the outline (inches).
MIN_CELL_IN, MAX_CELL_IN = 0.004, 0.02
GRID_CELLS = 1000


@dataclass
class FixtureSettings:
    """The shop standard. Defaults are the shipped ones; Settings... holds
    the shop's own (settings.json)."""
    enabled: bool = False          # tab new parts by default
    tab_height_in: float = 0.060
    tab_width_in: float = 0.250
    tab_spacing_in: float = 4.0
    min_tabs: int = 3
    # (depth cut from one side up to, channel width), deepest last; past the
    # last depth the last width is used.
    channel_table: tuple = ((0.50, 0.250), (1.00, 0.375), (math.inf, 0.500))
    min_frame_in: float = 0.250
    vise_grip_in: float = 0.188
    vise_clearance_in: float = 0.375
    vertical_min_thickness_in: float = 0.250
    max_stickout_ratio: float = 10.0

    def channel_width(self, depth_in: float) -> float:
        for upto, width in self.channel_table:
            if depth_in <= upto + 1e-9:
                return float(width)
        return float(self.channel_table[-1][1])

    def to_dict(self) -> dict:
        d = asdict(self)
        d["channel_table"] = [[None if math.isinf(a) else a, b] for a, b in self.channel_table]
        return d

    @classmethod
    def from_dict(cls, d: dict | None) -> "FixtureSettings":
        if not d:
            return cls()
        d = dict(d)
        if "channel_table" in d:
            d["channel_table"] = tuple((math.inf if a is None else float(a), float(b))
                                       for a, b in d["channel_table"])
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


_standard = FixtureSettings()


def standard() -> FixtureSettings:
    """The fixturing standard in use in this process."""
    return _standard


def set_standard(s: FixtureSettings | dict | None) -> FixtureSettings:
    """Use this standard from now on (None: the shipped one)."""
    global _standard
    _standard = (s if isinstance(s, FixtureSettings)
                 else FixtureSettings.from_dict(s) if s else FixtureSettings())
    return _standard


@dataclass
class Tab:
    x: float            # centre on the part's outline, plan X (in)
    y: float
    nx: float           # outward normal of the outline there
    ny: float
    length: float       # from the part's surface (+ overlap), across the channel
    width: float
    z0: float
    z1: float
    on_straight: bool = True
    # How far in from the outline the part's surface is at the tab: the tab
    # reaches in to it (zero where the wall is at the outline).
    inset: float = 0.0

    def corners(self) -> np.ndarray:
        """The tab's footprint, four (x, y) corners, counter-clockwise."""
        n = np.array([self.nx, self.ny])
        t = np.array([-self.ny, self.nx])
        c = np.array([self.x, self.y]) - n * (self.inset + TAB_OVERLAP_IN)
        h = t * self.width / 2.0
        return np.array([c - h, c - h + n * self.length, c + h + n * self.length, c + h])


@dataclass
class FixturePlan:
    procedure: str                      # VERTICAL or FLAT
    why: str
    operations: int
    frame: list                         # 3 rows: plan X, Y, Z in the part's file coordinates
    origin_mm: list                     # the plan origin in the part's file coordinates
    stock_in: list                      # stock size along plan X, Y, Z
    part_lo_in: list                    # the part's box in plan coordinates
    part_hi_in: list
    channel_width_in: float
    clamp_edge: str
    flipped: bool
    outline: list                       # the part's outer outline(s), [(x, y), ...] loops
    channel: list                       # the channel's outer edge(s)
    tabs: list = field(default_factory=list)          # Tab
    footprint_area_in2: float = 0.0
    channel_area_in2: float = 0.0       # inside the channel's outer edge, part included
    perimeter_in: float = 0.0
    stock_volume_in3: float = 0.0
    frame_volume_in3: float = 0.0       # left on the frame after the last op
    facing_volume_in3: float = 0.0
    channel_volume_in3: float = 0.0     # the ring the channel cutter takes out
    removed_volume_in3: float = 0.0     # stock - part - frame
    side_hole_count: int = 0
    warnings: list = field(default_factory=list)
    settings: dict = field(default_factory=dict)

    @property
    def tab_count(self) -> int:
        return len(self.tabs)

    def to_dict(self) -> dict:
        d = asdict(self)
        return _plain(d)

    @classmethod
    def from_dict(cls, d: dict) -> "FixturePlan":
        d = dict(d)
        d["tabs"] = [t if isinstance(t, Tab) else Tab(**t) for t in d.get("tabs", [])]
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    def summary(self) -> str:
        sx, sy, sz = self.stock_in
        how = ("vertical in the vise, one operation" if self.procedure == VERTICAL
               else "flat, two operations" if self.operations > 1 else "flat, one operation")
        return (f"{how}: {self.tab_count} tabs, {self.channel_width_in:.3f}\" channel, "
                f"stock {sz:.3f} x {min(sx, sy):.3f} x {max(sx, sy):.3f}")


def _plain(v):
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return _plain(v.tolist())
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isinf(f) else f
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


# ---------------------------------------------------------------------------
# the outline, on a grid
# ---------------------------------------------------------------------------

def _surface_points(pts_xy: np.ndarray, tris: np.ndarray, step: float) -> np.ndarray:
    """Points along every triangle edge of the projected mesh, no further
    apart than `step`: they enclose the footprint, which a fill then closes."""
    a, b = pts_xy[tris[:, [0, 1, 2]]], pts_xy[tris[:, [1, 2, 0]]]
    a, b = a.reshape(-1, 2), b.reshape(-1, 2)
    # Each edge is shared by two triangles: once is enough.
    key = np.sort(np.stack([tris[:, [0, 1, 2]].ravel(), tris[:, [1, 2, 0]].ravel()], 1), 1)
    _u, first = np.unique(key, axis=0, return_index=True)
    a, b = a[first], b[first]
    n = np.maximum(1, np.ceil(np.linalg.norm(b - a, axis=1) / step).astype(int))
    idx = np.repeat(np.arange(len(a)), n + 1)
    t = _ramp(n)
    return a[idx] + (b[idx] - a[idx]) * t[:, None]


def _ramp(n: np.ndarray) -> np.ndarray:
    """concatenate(linspace(0, 1, k + 1) for k in n), vectorised."""
    counts = n + 1
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    pos = np.arange(counts.sum()) - starts
    return pos / np.repeat(n, counts)


class Grid:
    """A boolean raster over the plan XY plane."""

    def __init__(self, lo, hi, cell):
        self.cell = float(cell)
        self.lo = np.asarray(lo, dtype=float)
        shape = np.ceil((np.asarray(hi, dtype=float) - self.lo) / self.cell).astype(int) + 1
        self.shape = (int(shape[0]), int(shape[1]))

    def index(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ij = np.floor((np.asarray(xy, dtype=float) - self.lo) / self.cell).astype(int)
        return (np.clip(ij[:, 0], 0, self.shape[0] - 1), np.clip(ij[:, 1], 0, self.shape[1] - 1))

    def mask_of_points(self, xy: np.ndarray) -> np.ndarray:
        m = np.zeros(self.shape, dtype=bool)
        i, j = self.index(xy)
        m[i, j] = True
        return m

    def contains(self, mask: np.ndarray, xy) -> np.ndarray:
        i, j = self.index(np.atleast_2d(xy))
        return mask[i, j]


def footprint(pts_xy: np.ndarray, tris: np.ndarray, pad: float):
    """(grid, mask) of the part seen along plan Z, holes filled: what the
    channel goes round. `pad` leaves room round it for the channel."""
    from scipy import ndimage

    lo, hi = pts_xy.min(axis=0) - pad, pts_xy.max(axis=0) + pad
    cell = float(np.clip((hi - lo).max() / GRID_CELLS, MIN_CELL_IN, MAX_CELL_IN))
    grid = Grid(lo, hi, cell)
    edges = _surface_points(pts_xy, tris, cell * 0.5)
    mask = grid.mask_of_points(np.vstack([edges, pts_xy]))
    # Close one-cell gaps, fill, and take the closing back off.
    mask = ndimage.binary_dilation(mask, iterations=1)
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_erosion(mask, iterations=1, border_value=0)
    return grid, mask


def grow(grid: Grid, mask: np.ndarray, distance: float) -> np.ndarray:
    """Everything within `distance` of the mask, holes filled."""
    from scipy import ndimage

    dist = ndimage.distance_transform_edt(~mask) * grid.cell
    return ndimage.binary_fill_holes(mask | (dist <= distance))


def loops(grid: Grid, mask: np.ndarray, tolerance: float | None = None) -> list[np.ndarray]:
    """The mask's boundary loops in plan XY, counter-clockwise round the
    region (clockwise round a hole), simplified to within `tolerance`.

    Traced along cell edges, so a diagonal pinch is split to keep regions
    apart (always turn left), then straightened by Douglas-Peucker.
    """
    m = np.pad(mask, 1)
    # Directed edges with the region on the left. Vertex (i, j) is the
    # corner at the low x, low y of padded cell (i, j). Only cells on the
    # boundary are visited, found with array shifts.
    out: dict = {}
    inner = m[1:-1, 1:-1]
    sides = (
        (~m[1:-1, :-2], (0, 0), (1, 0)),     # nothing below: bottom edge, going +x
        (~m[2:, 1:-1], (1, 0), (1, 1)),      # nothing right: right edge, going +y
        (~m[1:-1, 2:], (1, 1), (0, 1)),      # nothing above: top edge, going -x
        (~m[:-2, 1:-1], (0, 1), (0, 0)),     # nothing left: left edge, going -y
    )
    for open_side, (a0, a1), (b0, b1) in sides:
        ii, jj = np.nonzero(inner & open_side)
        ii, jj = (ii + 1).tolist(), (jj + 1).tolist()
        for i, j in zip(ii, jj):
            out.setdefault((i + a0, j + a1), []).append((i + b0, j + b1))
    result = []
    while out:
        start = next(iter(out))
        loop = [start]
        prev, cur = None, start
        while True:
            nexts = out.get(cur)
            if not nexts:
                break
            if len(nexts) == 1 or prev is None:
                nxt = nexts.pop(0)
            else:
                d_in = (cur[0] - prev[0], cur[1] - prev[1])
                # Prefer the left turn, then straight, then right.
                def rank(p):
                    d = (p[0] - cur[0], p[1] - cur[1])
                    cross = d_in[0] * d[1] - d_in[1] * d[0]
                    return -cross
                nexts.sort(key=rank)
                nxt = nexts.pop(0)
            if not nexts:
                del out[cur]
            prev, cur = cur, nxt
            if cur == start:
                break
            loop.append(cur)
        if len(loop) >= 4:
            xy = (np.asarray(loop, dtype=float) - 1.0) * grid.cell + grid.lo
            result.append(simplify(xy, tolerance if tolerance is not None else 0.75 * grid.cell))
    return [p for p in result if len(p) >= 3]


OFFSET_TOL_IN = 0.0005


def _polyline_segments(polylines, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The B-rep edge polylines (mm, file coordinates) as plan-XY segments
    (inches, before the plan's shift): (starts, ends)."""
    a, b = [], []
    for pl in polylines or ():
        p = np.asarray(pl, dtype=float).reshape(-1, 3)
        if len(p) < 2:
            continue
        xy = (p @ frame.T / MM)[:, :2]
        a.append(xy[:-1])
        b.append(xy[1:])
    if not a:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.vstack(a), np.vstack(b)


def offset_loops(grid: Grid, fmask: np.ndarray, grown: np.ndarray, pts_xy: np.ndarray,
                 tris: np.ndarray, distance: float, tol: float = OFFSET_TOL_IN,
                 edges: tuple[np.ndarray, np.ndarray] | None = None) -> list[np.ndarray]:
    """The outer edge(s) of `grown` (the footprint grown by `distance`), true
    to the part: the projected outline offset by exactly `distance`.

    The grid only finds the loops. Each traced vertex is then put on the
    offset curve: from the nearest point q of the projected part, at
    `distance` along the line through it -- so straight edges offset to
    straight lines and corners to arcs, with no staircase -- and the result
    is simplified to within `tol`.

    The part is the projected mesh plus `edges`, the B-rep's own edges as
    plan-XY segments (starts, ends): sampled much finer than the mesh, so
    an arc is offset as an arc, not as the mesh's facets.
    """
    from scipy import ndimage
    from scipy.spatial import cKDTree

    raw = [p for p in loops(grid, grown, tolerance=0.0) if signed_area(p) > 0]
    if not raw or not len(tris):
        return raw
    # The edges near the footprint's edge or outside it (a true arc bulges
    # past the mesh's chords): the only ones a point outside can be nearest
    # to. Those deep inside are dropped.
    e = np.sort(np.stack([tris[:, [0, 1, 2]].ravel(), tris[:, [1, 2, 0]].ravel()], 1), 1)
    e = np.unique(e, axis=0)
    a, b = pts_xy[e[:, 0]], pts_xy[e[:, 1]]
    if edges is not None and len(edges[0]):
        a, b = np.vstack([a, edges[0]]), np.vstack([b, edges[1]])
    edge_zone = ndimage.distance_transform_edt(fmask) <= 5.0     # within 5 cells of outside
    keep = grid.contains(edge_zone, a) | grid.contains(edge_zone, b)
    a, b = a[keep], b[keep]
    if not len(a):
        return raw
    step = grid.cell
    n = np.maximum(1, np.ceil(np.linalg.norm(b - a, axis=1) / step).astype(int))
    seg = np.repeat(np.arange(len(a)), n + 1)
    samples = a[seg] + (b[seg] - a[seg]) * _ramp(n)[:, None]
    tree = cKDTree(samples)
    k = min(8, len(samples))

    def nearest(xy):
        """The nearest point of the projected mesh to each of `xy`, and how far."""
        _d, idx = tree.query(xy, k=k)
        idx = np.asarray(idx).reshape(len(xy), -1)
        best_q = np.zeros_like(xy)
        best_d = np.full(len(xy), np.inf)
        for col in range(idx.shape[1]):
            s = seg[idx[:, col]]
            sa, ab = a[s], b[s] - a[s]
            L2 = np.einsum("ij,ij->i", ab, ab)
            t = np.einsum("ij,ij->i", xy - sa, ab) / np.where(L2 > 0, L2, 1.0)
            q = sa + ab * np.clip(t, 0.0, 1.0)[:, None]
            d = np.linalg.norm(xy - q, axis=1)
            better = d < best_d
            best_d[better], best_q[better] = d[better], q[better]
        return best_q, best_d

    def snap(xy):
        out_ = xy.copy()
        todo = np.arange(len(xy))
        # Pushed out from its nearest edge, a point in an inside corner can
        # still be too close to the other one: push again till it is clear of
        # both, which brings it to the corner where their offsets meet.
        for push in range(8):
            q, d = nearest(out_[todo])
            if push:                         # again: only those still too close
                close = d < distance - 1e-7
                todo, q, d = todo[close], q[close], d[close]
                if not len(todo):
                    break
            ok = d > 1e-9
            out_[todo[ok]] = q[ok] + (out_[todo[ok]] - q[ok]) * (distance / d[ok])[:, None]
        return out_

    out = []
    for loop in raw:
        pts = snap(loop)
        # Between two vertices on the curve the straight edge cuts inside it
        # round an arc, and across an inside corner. Split each edge where
        # it strays most, if by more than `tol`, the new vertex put back on
        # the curve, until none does.
        ts = np.linspace(0.0, 1.0, 10)[1:-1]
        for _round in range(16):
            nxt = np.roll(pts, -1, axis=0)
            probe = pts[:, None, :] + (nxt - pts)[:, None, :] * ts[None, :, None]
            _q, d = nearest(probe.reshape(-1, 2))
            err = np.abs(d - distance).reshape(len(pts), len(ts))
            bad = err.max(axis=1) > tol
            if not bad.any():
                break
            worst = probe[bad, err[bad].argmax(axis=1)]
            pts = np.insert(pts, np.nonzero(bad)[0] + 1, snap(worst), axis=0)
        # Drop repeats (two vertices onto one point) before straightening.
        gap = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
        pts = pts[gap > 1e-7]
        if len(pts) >= 3:
            out.append(simplify(pts, tol))
    return [p for p in out if len(p) >= 3 and signed_area(p) > 0]


def simplify(loop: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker on a closed loop."""
    pts = np.asarray(loop, dtype=float)
    if len(pts) < 4:
        return pts
    # Split at the two points farthest apart so both halves are open chains.
    i0 = 0
    i1 = int(np.argmax(np.linalg.norm(pts - pts[i0], axis=1)))
    a = _dp(pts[i0:i1 + 1], tol)
    b = _dp(np.vstack([pts[i1:], pts[:1]]), tol)
    return np.vstack([a[:-1], b[:-1]])


def _dp(chain: np.ndarray, tol: float) -> np.ndarray:
    keep = np.zeros(len(chain), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(chain) - 1)]
    while stack:
        s, e = stack.pop()
        if e - s < 2:
            continue
        a, b = chain[s], chain[e]
        seg = b - a
        L = float(np.hypot(*seg))
        mid = chain[s + 1:e]
        if L == 0:
            d = np.linalg.norm(mid - a, axis=1)
        else:
            d = np.abs(seg[0] * (mid[:, 1] - a[1]) - seg[1] * (mid[:, 0] - a[0])) / L
        k = int(np.argmax(d))
        if d[k] > tol:
            keep[s + 1 + k] = True
            stack += [(s, s + 1 + k), (s + 1 + k, e)]
    return chain[keep]


def signed_area(loop: np.ndarray) -> float:
    x, y = loop[:, 0], loop[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def perimeter(loop: np.ndarray) -> float:
    return float(np.linalg.norm(np.roll(loop, -1, axis=0) - loop, axis=1).sum())


# ---------------------------------------------------------------------------
# tabs
# ---------------------------------------------------------------------------

def section_points(pts: np.ndarray, tris: np.ndarray, z: float, step: float) -> np.ndarray:
    """Points along the part's cross-section at height `z` (plan inches):
    where the plane cuts the mesh, sampled no further apart than `step`."""
    p = pts[tris]                                     # (n, 3, 3)
    d = p[:, :, 2] - z
    out = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        a, b = p[:, i], p[:, j]
        da, db = d[:, i], d[:, j]
        cross = (da * db) < 0
        t = np.where(cross, da / np.where(cross, da - db, 1.0), 0.0)
        out.append((cross, a[:, :2] + (b[:, :2] - a[:, :2]) * t[:, None]))
    # A triangle the plane crosses is cut on exactly two of its edges.
    hits = np.stack([c for c, _x in out], axis=1)
    pts2 = np.stack([x for _c, x in out], axis=1)
    two = hits.sum(axis=1) == 2
    if not two.any():
        return np.zeros((0, 2))
    first = np.argmax(hits[two], axis=1)
    second = 2 - np.argmax(hits[two][:, ::-1], axis=1)
    rows = np.nonzero(two)[0]
    a, b = pts2[rows, first], pts2[rows, second]
    n = np.maximum(1, np.ceil(np.linalg.norm(b - a, axis=1) / step).astype(int))
    idx = np.repeat(np.arange(len(a)), n + 1)
    return a[idx] + (b[idx] - a[idx]) * _ramp(n)[:, None]


#: A tab's end must sit in at least this much solid part (along the tab),
#: so it never ends in a hole, a pocket or a groove (inches).
MIN_SOLID_IN = 0.040
#: Heights at which the wall counts as reaching the outline (inches).
TOUCH_TOL_IN = 0.030
#: How far in from the outline a tab may reach to find the part's surface.
MAX_INSET_IN = 0.50
#: Weight of a tab's reach inward against its distance from the even
#: spacing round the outline: 1" of reach costs as much as this much arc.
INSET_COST = 8.0
#: Outline turns sharper than this make a corner, which tabs keep clear of.
CORNER_DEG = 30.0


class Slices:
    """The part's cross-sections at a stack of heights (plan inches), with
    rays cast in from the channel to find where a tab meets the part."""

    def __init__(self, pts: np.ndarray, tris: np.ndarray, z_lo: float, z_hi: float):
        h = max(z_hi - z_lo, 1e-6)
        n = int(np.clip(np.ceil(h / 0.03), 8, 40))
        # Mid-cells, nudged so no slice lands exactly on a vertex level.
        self.z = z_lo + (np.arange(n) + 0.5) * h / n + h * 1e-5
        self.dz = h / n
        p = pts[tris]                                        # (t, 3, 3)
        self.segments = []
        for z in self.z:
            d = p[:, :, 2] - z
            cut = []
            for i, j in ((0, 1), (1, 2), (2, 0)):
                da, db = d[:, i], d[:, j]
                cross = (da * db) < 0
                t = np.where(cross, da / np.where(cross, da - db, 1.0), 0.0)
                cut.append((cross, p[:, i, :2] + (p[:, j, :2] - p[:, i, :2]) * t[:, None]))
            hits = np.stack([c for c, _x in cut], axis=1)
            xy = np.stack([x for _c, x in cut], axis=1)
            two = hits.sum(axis=1) == 2
            first = np.argmax(hits[two], axis=1)
            second = 2 - np.argmax(hits[two][:, ::-1], axis=1)
            rows = np.nonzero(two)[0]
            self.segments.append((xy[rows, first], xy[rows, second]))

    def probe(self, origins: np.ndarray, dirs: np.ndarray, reach: float):
        """For rays from `origins` (outside the part) along `dirs`: (entry,
        solid), each (slices, rays) -- the distance to the first surface
        and how much solid follows it; inf where nothing is met."""
        k = len(origins)
        entry = np.full((len(self.z), k), np.inf)
        solid = np.zeros((len(self.z), k))
        lo, hi = origins.min(axis=0) - reach, origins.max(axis=0) + reach
        for s_i, (a, b) in enumerate(self.segments):
            if not len(a):
                continue
            keep = ((np.maximum(a, b) >= lo) & (np.minimum(a, b) <= hi)).all(axis=1)
            a, b = a[keep], b[keep]
            if not len(a):
                continue
            e = b - a                                          # (m, 2)
            denom = dirs[:, 0:1] * e[None, :, 1] - dirs[:, 1:2] * e[None, :, 0]
            w = a[None, :, :] - origins[:, None, :]            # (k, m, 2)
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (w[:, :, 0] * e[None, :, 1] - w[:, :, 1] * e[None, :, 0]) / denom
                u = (w[:, :, 0] * dirs[:, 1:2] - w[:, :, 1] * dirs[:, 0:1]) / denom
            ok = (np.abs(denom) > 1e-12) & (u >= 0) & (u < 1) & (t > 0) & (t <= reach)
            t = np.where(ok, t, np.inf)
            t.sort(axis=1)
            entry[s_i] = t[:, 0]
            exit_ = t[:, 1] if t.shape[1] > 1 else np.full(k, np.inf)
            # Nothing past the entry within reach: solid at least to the end.
            # (No entry at all: no solid; `entry` is inf there anyway.)
            hit = np.isfinite(t[:, 0])
            last = np.where(np.isfinite(exit_), exit_, reach)
            solid[s_i] = np.where(hit, last - np.where(hit, t[:, 0], 0.0), 0.0)
        return entry, solid


def _outline_spots(loops_, step: float, keep_clear: float):
    """Candidate tab spots along the outline, `step` apart: (arc position,
    point, outward normal, whether on a straight stretch), clear of sharp
    corners by `keep_clear`."""
    spots, arc0 = [], 0.0
    for loop in loops_:
        n = len(loop)
        seg = np.roll(loop, -1, axis=0) - loop
        lens = np.linalg.norm(seg, axis=1)
        dirs = seg / np.where(lens > 0, lens, 1.0)[:, None]
        turn = np.degrees(np.arccos(np.clip(
            np.einsum("ij,ij->i", dirs, np.roll(dirs, 1, axis=0)), -1.0, 1.0)))
        corner = turn > CORNER_DEG                       # at the START of segment k
        starts = arc0 + np.concatenate([[0.0], np.cumsum(lens)[:-1]])
        corners = [starts[k] for k in range(n) if corner[k]]
        total = float(lens.sum())
        for k in range(n):
            if lens[k] <= 0:
                continue
            m = max(1, int(np.floor(lens[k] / step)))
            for u in (np.arange(m) + 0.5) * lens[k] / m:
                arc = starts[k] + u
                near = min((min(abs(arc - c), total - abs(arc - c)) for c in corners),
                           default=np.inf)
                if near < keep_clear:
                    continue
                d = dirs[k]
                # On a curve, the normal of the arc around the spot rather than
                # of one short chord.
                if not (corner[k] or corner[(k + 1) % n]):
                    d = dirs[k] + 0.5 * (dirs[k - 1] + dirs[(k + 1) % n])
                    d = d / (np.linalg.norm(d) or 1.0)
                straight = bool(lens[k] >= 2 * keep_clear)
                spots.append((float(arc), loop[k] + dirs[k] * u,
                              np.array([d[1], -d[0]]), straight))
        arc0 += total
    return spots


def place_tabs(outline: list[np.ndarray], grid: Grid, channel_mask: np.ndarray,
               s: FixtureSettings, width_in: float, z_lo: float, z_hi: float,
               count: int | None = None, slices: Slices | None = None
               ) -> tuple[list[Tab], list[str]]:
    """Tabs spread evenly round the outer outline, each where it can join
    the part well.

    At every candidate spot the part is probed from the channel, at a
    stack of heights and across the tab's width: how far in its surface is
    and how much solid lies behind it. A tab goes at the middle of the
    longest stretch of height where the wall reaches the outline -- on a
    contoured edge, wherever the part is widest -- or, where it never
    quite does, at the nearest surface, reaching in to it. Its end must sit
    in solid part, never in a hole or a pocket. Spots needing less reach
    are preferred, and corners are kept clear.
    """
    notes = []
    loops_ = sorted(outline, key=lambda p: -abs(signed_area(p)))
    total = sum(perimeter(p) for p in loops_)
    if total <= 0:
        return [], ["The part has no outline to put tabs on."]
    want = count if count else max(int(s.min_tabs), int(round(total / max(s.tab_spacing_in, 0.5))))
    keep_clear = TAB_CORNER_MARGIN_IN + s.tab_width_in / 2
    spots = _outline_spots(loops_, max(s.tab_width_in / 2, total / 400.0), keep_clear)
    if not spots:
        return [], ["No room for tabs on the outline."]

    def frame_reached(p, nrm):
        end = p + nrm * (width_in + TAB_OVERLAP_IN)
        return not grid.contains(channel_mask, end)[0]

    spots = [sp for sp in spots if frame_reached(sp[1], sp[2])]
    half_h = s.tab_height_in / 2.0
    zc_default = (z_lo + z_hi) / 2.0

    # Probe every spot: three rays across the tab's width, from just out in
    # the channel, at every slice.
    placed = []          # (arc, p, nrm, straight, inset, zc)
    if slices is not None and spots:
        lat = np.array([-0.45, 0.0, 0.45]) * s.tab_width_in
        P = np.array([sp[1] for sp in spots])
        N = np.array([sp[2] for sp in spots])
        T = np.column_stack([-N[:, 1], N[:, 0]])
        start = 2 * grid.cell + 0.01
        origins = (P[:, None, :] + N[:, None, :] * start
                   + T[:, None, :] * lat[None, :, None]).reshape(-1, 2)
        dirs = np.repeat(-N, len(lat), axis=0)
        reach = start + MAX_INSET_IN + MIN_SOLID_IN + 0.1
        entry, solid = slices.probe(origins, dirs, reach)
        depth = entry - start                                   # in from the outline
        ok = np.isfinite(depth) & (solid >= MIN_SOLID_IN) & (depth <= MAX_INSET_IN)
        depth = np.where(ok, np.maximum(depth, 0.0), np.inf)
        depth = depth.reshape(len(slices.z), len(spots), len(lat)).max(axis=2)   # (z, spot)
        for i, (arc, p, nrm, straight) in enumerate(spots):
            r = depth[:, i]
            if not np.isfinite(r).any():
                continue
            touch = r <= max(TOUCH_TOL_IN, 2 * grid.cell)
            if touch.any():
                # The longest run of heights where the wall is at the outline.
                runs, k = [], 0
                while k < len(touch):
                    if touch[k]:
                        j = k
                        while j + 1 < len(touch) and touch[j + 1]:
                            j += 1
                        runs.append((k, j))
                        k = j + 1
                    else:
                        k += 1
                a, b = max(runs, key=lambda ab: ab[1] - ab[0])
            else:
                best = np.nanmin(np.where(np.isfinite(r), r, np.nan))
                near = np.nonzero(r <= best + TOUCH_TOL_IN)[0]
                a = b = int(near[len(near) // 2])
            zc = float((slices.z[a] + slices.z[b]) / 2.0)
            zc = float(np.clip(zc, z_lo + half_h, max(z_hi - half_h, z_lo + half_h)))
            inside = np.abs(slices.z - zc) <= half_h + slices.dz / 2.0
            inset = float(r[inside].max()) if inside.any() else float(r[np.argmin(np.abs(slices.z - zc))])
            if not np.isfinite(inset):
                continue
            placed.append((arc, p, nrm, straight, inset, zc))
    else:
        placed = [(arc, p, nrm, straight, 0.0, zc_default) for arc, p, nrm, straight in spots]

    chosen: list[Tab] = []
    min_sep = max(2 * s.tab_width_in, total / (want * 3.0))
    for k in range(want):
        target = (k + 0.5) * total / want
        best = None
        for arc, p, nrm, straight, inset, zc in placed:
            if any(np.hypot(t.x - p[0], t.y - p[1]) < min_sep for t in chosen):
                continue
            gap = abs(arc - target)
            gap = min(gap, total - gap)
            score = gap + INSET_COST * inset
            if best is None or score < best[0]:
                best = (score, p, nrm, straight, inset, zc)
        if best is None:
            continue
        _s, p, nrm, straight, inset, zc = best
        chosen.append(Tab(float(p[0]), float(p[1]), float(nrm[0]), float(nrm[1]),
                          width_in + inset + 2 * TAB_OVERLAP_IN, float(s.tab_width_in),
                          zc - half_h, zc + half_h, straight, inset))
    if len(chosen) < want:
        notes.append(f"Only {len(chosen)} of {want} tabs found a place where they reach the "
                     f"frame and join solid part.")
    if not chosen:
        notes.append("No tabs could be placed: set the tab count by hand or turn tabbed "
                     "fixturing off for this part.")
    return chosen, notes


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------

def _rot_z(quarter_turns: int) -> np.ndarray:
    c = [1, 0, -1, 0][quarter_turns % 4]
    s_ = [0, 1, 0, -1][quarter_turns % 4]
    return np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]], dtype=float)


def base_frame(rect_axes, rect_extents_mm) -> np.ndarray:
    """Rows: length, width, thickness of the stock box, right-handed."""
    axes = np.asarray(rect_axes, dtype=float)
    order = np.argsort(-np.asarray(rect_extents_mm, dtype=float), kind="stable")
    f = axes[order]
    if np.dot(np.cross(f[0], f[1]), f[2]) < 0:
        f[2] = -f[2]
    return f


def side_one_up(frame: np.ndarray, planes) -> bool:
    """Whether side 1 should face along +frame Z: the side carrying more flat
    area facing out of it."""
    if not planes:
        return True
    z = frame[2]
    up = down = 0.0
    for pl in planes:
        n, a = ((pl.normal, pl.area_mm2) if hasattr(pl, "normal") else pl)
        d = float(np.dot(n, z))
        if d > 0.999:
            up += a
        elif d < -0.999:
            down += a
    return up >= down


def stock_dims_for(stock, frame_extents_in) -> tuple[float, float, float]:
    """A plate stock's size along the plan's length, width and thickness
    axes. The stock reports them sorted; they go to the axes in the order of
    the part's own extents."""
    t = float(stock.thickness_or_diameter_in)
    a, b = float(stock.length_in), float(stock.width_in or 0.0)
    return (a, b, t) if frame_extents_in[0] >= frame_extents_in[1] else (b, a, t)


def plan(points_mm, triangles, rect_axes, rect_extents_mm, stock, part_volume_in3: float,
         settings: FixtureSettings | None = None, *, planes=None, holes=None,
         flipped: bool = False, clamp_edge: str = CLAMP_EDGES[0],
         channel_width_in: float | None = None, tab_count: int | None = None,
         grow_stock: bool = False, edge_polylines=None) -> FixturePlan:
    """Lay the part out in `stock` (a plate StockSize) for tabbed fixturing.

    `edge_polylines` (the B-rep edges, (n, 3) mm each, as the analysis has
    them) make the channel follow arcs exactly rather than the mesh facets.

    With `grow_stock` the stock is enlarged (to the next 1/4") wherever the
    frame would be thinner than the minimum, and the plan is for that size.
    `flipped` swaps sides 1 and 2; `clamp_edge` (vertical only) is one of
    CLAMP_EDGES; `channel_width_in` and `tab_count` override the standard.
    """
    s = settings or standard()
    pts = np.asarray(points_mm, dtype=float).reshape(-1, 3)
    tris = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    f0 = base_frame(rect_axes, rect_extents_mm)
    up = side_one_up(f0, planes)
    if flipped:
        up = not up
    if not up:
        f0 = np.array([f0[0], -f0[1], -f0[2]])
    local0 = pts @ f0.T / MM
    ext0 = local0.max(axis=0) - local0.min(axis=0)

    sx0, sy0, sz = stock_dims_for(stock, ext0)
    part_h = float(ext0[2])

    def procedure_for(sx, sy, turns):
        stick = (sx if turns % 2 else sy) - s.vise_grip_in
        if sz + 1e-9 < s.vertical_min_thickness_in:
            return FLAT, (f"stock {sz:.3f}\" thick is under the {s.vertical_min_thickness_in:.3f}\" "
                          f"minimum for standing it in the vise")
        if stick > s.max_stickout_ratio * sz + 1e-9:
            return FLAT, (f"{stick:.2f}\" above the jaws is more than "
                          f"{s.max_stickout_ratio:g}x the {sz:.3f}\" thickness")
        return VERTICAL, (f"{stick:.2f}\" above the jaws, {stick / sz:.1f}x the "
                          f"{sz:.3f}\" thickness")

    turns = CLAMP_EDGES.index(clamp_edge) if clamp_edge in CLAMP_EDGES else 0
    # A short-edge clamp turns the plan a quarter: X then runs along the
    # short edge. "Other side" is a half turn.
    quarter = {0: 0, 1: 2, 2: 1, 3: 3}[turns]
    proc, why = procedure_for(sx0, sy0, quarter)

    for _attempt in range(3):
        frame = _rot_z(quarter).T @ f0 if proc == VERTICAL else f0.copy()
        if proc == VERTICAL and quarter % 2:
            sx, sy = sy0, sx0
        else:
            sx, sy = sx0, sy0
        local = pts @ frame.T / MM
        lo, hi = local.min(axis=0), local.max(axis=0)
        depth = sz / 2.0 if proc == VERTICAL else sz
        w = float(channel_width_in) if channel_width_in else s.channel_width(depth)
        grid, fmask = footprint(local[:, :2], tris, w + 2 * MAX_CELL_IN + 0.05)
        cmask = grow(grid, fmask, w)
        cell2 = grid.cell ** 2
        # The channel's outer edge is the outline grown by its width, so its
        # box is the part's box grown by the same: exact, not on the grid.
        c_lo, c_hi = lo[:2] - w, hi[:2] + w
        grip = (s.vise_grip_in + s.vise_clearance_in) if proc == VERTICAL else 0.0
        need_x = (c_hi[0] - c_lo[0]) + 2 * s.min_frame_in
        need_y = (c_hi[1] - c_lo[1]) + 2 * s.min_frame_in + grip
        if grow_stock:
            sx = max(sx, stock_mod.ceil_to_increment(need_x))
            sy = max(sy, stock_mod.ceil_to_increment(need_y))
            if proc == VERTICAL and quarter % 2:
                sx0, sy0 = sy, sx
            else:
                sx0, sy0 = sx, sy
            again, why2 = procedure_for(sx0, sy0, quarter)
            if again != proc:
                proc, why = again, why2
                continue
            why = why2
        break

    # Centre the part: in Z in the thickness, in X across the stock, in Y in
    # the stock above the vise's grip and clearance.
    centre = (lo + hi) / 2.0
    target = np.array([sx / 2.0, grip + (sy - grip) / 2.0, sz / 2.0])
    shift = target - centre                              # plan = local + shift
    origin_local = -shift                                # the plan origin, in local inches
    origin_mm = (origin_local * MM) @ frame              # ... in the file's coordinates
    plo, phi = lo + shift, hi + shift

    # Traced along grid cells, a curved or slanted edge is a staircase one
    # cell high; straightened past that, its corners are the part's own and
    # tabs can sit on arcs and angled edges.
    outline = [p + shift[:2] for p in loops(grid, fmask, tolerance=1.5 * grid.cell)]
    outline = [p for p in outline if signed_area(p) > 0]
    # The channel's outer edge: the part's projected outline offset by the
    # channel width exactly (lines stay lines, corners become arcs).
    channel = [p + shift[:2] for p in offset_loops(
        grid, fmask, cmask, local[:, :2], tris, w,
        edges=_polyline_segments(edge_polylines, frame))]
    grid.lo = grid.lo + shift[:2]

    # Each tab at its own height, where the part's wall is at the outline:
    # the channel runs the part's full height either way (flat: down to its
    # bottom, the floor below faced off in op 2).
    slices = Slices(local + shift, tris, plo[2], phi[2])
    tabs, notes = place_tabs(outline, grid, cmask, s, w, plo[2], phi[2], count=tab_count,
                             slices=slices)

    warnings = list(notes)
    c_lo, c_hi = c_lo + shift[:2], c_hi + shift[:2]
    if (c_lo[0] < s.min_frame_in - 1e-6 or sx - c_hi[0] < s.min_frame_in - 1e-6
            or c_lo[1] - grip < s.min_frame_in - 1e-6 or sy - c_hi[1] < s.min_frame_in - 1e-6):
        warnings.append(f"The frame outside the channel is under {s.min_frame_in:.3f}\" "
                        f"somewhere: the stock is too small for tabbed fixturing "
                        f"(snap to recommended to size it).")
    if phi[2] - plo[2] > sz + 1e-6 or sx < phi[0] - plo[0] or sy < phi[1] - plo[1]:
        warnings.append("The stock does not enclose the part.")

    side_holes = 0
    for h in holes or ():
        d = getattr(h, "axis_direction", None)
        if d is not None and abs(float(np.dot(np.asarray(d, dtype=float), frame[2]))) < 0.3:
            side_holes += 1
    if side_holes:
        warnings.append(f"{side_holes} hole(s) run along the plate, into the part's edges: "
                        f"the frame round the part blocks them; they need another setup.")

    foot_area = float(fmask.sum()) * cell2
    chan_area = float(cmask.sum()) * cell2
    stock_vol = sx * sy * sz
    frame_vol = max(sx * sy - chan_area, 0.0) * part_h
    facing = sx * sy * max(sz - part_h, 0.0)
    ring = max(chan_area - foot_area, 0.0) * part_h
    removed = max(stock_vol - float(part_volume_in3 or 0.0) - frame_vol, 0.0)
    ops = 1 if proc == VERTICAL else 2
    if proc == FLAT and sz - part_h <= NO_FLOOR_IN:
        # Sheet, bought to the part's thickness: there is no floor for a
        # second op to face off. The channel goes straight through in op 1.
        ops = 1
        why += "; sheet at the part's thickness, so one operation with the channel cut through"
        warnings.append("The channel cuts through the bottom of the sheet: run it on a "
                        "spoilboard or sacrificial plate.")
    return FixturePlan(
        procedure=proc, why=why, operations=ops,
        frame=frame.tolist(), origin_mm=list(map(float, origin_mm)),
        stock_in=[float(sx), float(sy), float(sz)],
        part_lo_in=list(map(float, plo)), part_hi_in=list(map(float, phi)),
        channel_width_in=float(w), clamp_edge=clamp_edge if proc == VERTICAL else "",
        flipped=bool(flipped),
        outline=[p.tolist() for p in outline], channel=[p.tolist() for p in channel],
        tabs=tabs, footprint_area_in2=foot_area, channel_area_in2=chan_area,
        perimeter_in=float(sum(perimeter(p) for p in outline)),
        stock_volume_in3=stock_vol, frame_volume_in3=frame_vol, facing_volume_in3=facing,
        channel_volume_in3=ring, removed_volume_in3=removed, side_hole_count=side_holes,
        warnings=warnings, settings=s.to_dict(),
    )


def grown_stock(plan_: FixturePlan, basis: str = stock_mod.RECOMMENDED):
    """The plan's stock as a StockSize."""
    sx, sy, sz = plan_.stock_in
    return stock_mod.custom_rectilinear(sz, min(sx, sy), max(sx, sy), basis=basis)


def applies(stock, removal_type: str, settings: FixtureSettings | None = None) -> bool:
    """Whether a part is tabbed: the standard is on, it is milled, and from
    plate (not bar, extrusion or nothing)."""
    s = settings or standard()
    return (bool(s.enabled) and removal_type == "mill" and stock is not None
            and getattr(stock, "kind", "") == "rectilinear")


def plan_for_results(results: dict, stock, settings: FixtureSettings | None = None,
                     **options) -> FixturePlan | None:
    """plan() for an analysis result (pipeline.compute_analysis, or the
    Machining window's copy of it) and a plate stock; None without a mesh."""
    rect = results.get("rect_box")
    pts, tris = results.get("render_points"), results.get("render_triangles")
    if rect is None or pts is None or tris is None or not len(pts) or stock is None:
        return None
    sp = results.get("shape_props")
    vol = sp.volume_mm3 / MM ** 3 if sp is not None else 0.0
    return plan(pts, tris, rect.axes, rect.extents_mm, stock, vol, settings,
                planes=results.get("planes"), holes=results.get("holes"),
                edge_polylines=results.get("outline_edges"), **options)


def matches_stock(plan_: FixturePlan | dict | None, stock) -> bool:
    """Whether a saved plan was made for this stock size."""
    if plan_ is None or stock is None or getattr(stock, "kind", "") != "rectilinear":
        return False
    p = plan_ if isinstance(plan_, FixturePlan) else FixturePlan.from_dict(plan_)
    have = sorted([float(stock.thickness_or_diameter_in), float(stock.width_in or 0.0),
                   float(stock.length_in)])
    return all(abs(a - b) <= stock_mod.SAME_SIZE_TOL_IN
               for a, b in zip(have, sorted(p.stock_in)))


def overlay(plan_: FixturePlan) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """For drawing on the part where the file put it: the channel's outer
    edge at the part's top and bottom (loops of (n, 3) points, mm) and each
    tab's eight corners (mm), in the part's file coordinates."""
    frame = np.asarray(plan_.frame, dtype=float)
    origin = np.asarray(plan_.origin_mm, dtype=float)

    def world(xyz_in):
        return origin + (np.asarray(xyz_in, dtype=float) * MM) @ frame

    zb, zt = plan_.part_lo_in[2], plan_.part_hi_in[2]
    loops_ = []
    for loop in plan_.channel:
        xy = np.asarray(loop, dtype=float)
        for z in (zb, zt):
            loops_.append(world(np.column_stack([xy, np.full(len(xy), z)])))
    tabs = []
    for t in plan_.tabs:
        c = t.corners()
        tabs.append(world(np.vstack([np.column_stack([c, np.full(4, t.z0)]),
                                     np.column_stack([c, np.full(4, t.z1)])])))
    return loops_, tabs
