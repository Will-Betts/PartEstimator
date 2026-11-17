"""Place the flat pattern on the sheet the way you would actually cut it.

THE PROBLEM THIS SOLVES
-----------------------
Nothing in the unfold decides how the blank is ORIENTED. The solver holds one
flange fixed and swings the rest into its plane, so the flat pattern inherits
whatever direction that flange happened to face in the model -- which is an
accident of how the part was drawn, not a decision anybody made.

That is fine for the geometry and wrong for every number that comes off it. A
blank sitting at 30 degrees to the axes reports a bounding box far larger than
the material it needs, and the bounding box is exactly what a material cost
estimate is built on. A 400 x 100 blank drawn on the diagonal measures roughly
396 x 296 -- three times the area, for a part that has not changed at all.

So the flat pattern is rotated until its minimum-area rectangle is square to the
axes, longer side along X. The rotation is rigid, so nothing about the part
changes: the same area, the same cut length, the same holes in the same places
relative to each other. Only the reported envelope, and the way it lands in the
DXF, get better.

WHERE IT IS APPLIED, AND WHY THAT MATTERS
-----------------------------------------
The angle is folded into the FLAT FRAME -- the single transform that carries the
part from its bent position into the plane -- before anything is built from it.
Everything downstream therefore inherits it for free and cannot disagree: the
preview, the DXF, the extruded solid and the reported dimensions are all derived
from that one matrix.

The alternative, rotating each output separately, is how the DXF and the preview
came to disagree in v3 and it took a screenshot from the shop floor to notice.

GRAIN
-----
Free rotation is only available when the grain direction is unconstrained. Sheet
is anisotropic -- it bends more readily across the rolling direction than along
it, and a tight bend with the grain will crack material that would have survived
across it. If a grain direction is set, the blank may only be placed square to
it, so the choice collapses to 0 or 90 degrees and the minimum-area rectangle
does not get a vote. That is a real constraint from the material, not a
limitation of the code, and the tool should not quietly optimise past it.
"""

from __future__ import annotations

import math

import numpy as np

from . import geom

#: How the blank is turned before it is measured and exported.
MODES = ("min_area", "as_modelled", "grain")


def rotation_2d(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=float)


def min_area_angle(pts_2d, prefer_long_x: bool = True) -> float:
    """The angle to ROTATE points BY so their tightest rectangle is axis-square.

    `geom.min_area_rect` reports the direction the rectangle's first axis
    points, so squaring the part to the axes means rotating by the negative of
    that. Two refinements follow, IN THIS ORDER, and the order is the whole
    subtlety:

    1. Fold the rotation into a quarter turn. A rectangle has four-fold
       symmetry, so all four right-angle placements give the same envelope;
       taking the smallest keeps the blank looking like the part you drew
       instead of arriving unexpectedly on its side.

    2. THEN decide whether it needs one more quarter turn to put the long side
       along X, which is how every sheet size and every nesting program is
       quoted.

    Doing these the other way round silently undoes the second one -- the fold
    in step 1 throws away exactly the quarter turn step 2 just added. That is
    why the decision is made against the ACTUAL rotated envelope rather than
    against the rectangle's own reported width and height.
    """
    pts = np.asarray(pts_2d, dtype=float).reshape(-1, 2)
    if len(pts) < 3:
        return 0.0

    _w, _h, theta = geom.min_area_rect(pts)
    angle = _wrap_quarter(-theta)

    if prefer_long_x:
        ew, eh = envelope(apply(pts, angle))
        if eh > ew:
            angle += math.pi / 2.0

    return angle


def _wrap_quarter(angle: float) -> float:
    """Fold into (-45, 45] degrees."""
    return math.remainder(angle, math.pi / 2.0)


def grain_angle(pts_2d, grain_deg: float) -> float:
    """Square the blank to the rolling direction, then pick the better of 0/90.

    The grain direction is a property of the SHEET, so the blank has to be
    placed square to it. Both of the two legal placements have the same
    envelope with its sides swapped, so the only free choice left is which of
    them puts the long side along X.
    """
    base = -math.radians(grain_deg)
    pts = np.asarray(pts_2d, dtype=float).reshape(-1, 2)
    if len(pts) < 2:
        return base

    turned = pts @ rotation_2d(base).T
    span = turned.max(axis=0) - turned.min(axis=0)
    if span[1] > span[0]:
        base += math.pi / 2.0
    return base


def choose_angle(
    pts_2d, mode: str = "min_area", grain_deg: float | None = None,
) -> float:
    """The rotation to apply to the flat pattern, in radians.

    A grain constraint always wins: it comes from the material, and no envelope
    saving is worth a cracked bend.
    """
    if grain_deg is not None:
        return grain_angle(pts_2d, grain_deg)
    if mode == "as_modelled":
        return 0.0
    if mode == "grain":
        # Asked for grain alignment without saying which way. Nothing to align
        # to, so leave it as modelled rather than guess a rolling direction.
        return 0.0
    return min_area_angle(pts_2d)


def apply(pts_2d, angle: float) -> np.ndarray:
    pts = np.asarray(pts_2d, dtype=float).reshape(-1, 2)
    if not angle:
        return pts.copy()
    return pts @ rotation_2d(angle).T


def envelope(pts_2d) -> tuple[float, float]:
    """Axis-aligned width and height of a set of points."""
    pts = np.asarray(pts_2d, dtype=float).reshape(-1, 2)
    if len(pts) == 0:
        return (0.0, 0.0)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return (float(hi[0] - lo[0]), float(hi[1] - lo[1]))


def saving(pts_2d, angle: float) -> float:
    """Fraction of bounding-box area the rotation saves, 0 if it saves none.

    Reported rather than assumed. On a part that was already square to the axes
    this is zero and the tool should say so, not imply it did something clever.
    """
    before_w, before_h = envelope(pts_2d)
    after_w, after_h = envelope(apply(pts_2d, angle))
    before = before_w * before_h
    after = after_w * after_h
    if before <= 0:
        return 0.0
    return max(0.0, (before - after) / before)


def frame_rotation_3d(angle: float) -> np.ndarray:
    """The 4x4 that applies `angle` about Z, for composing into the flat frame.

    The flat frame already carries the part into the z = 0 plane, so turning the
    blank is a rotation about Z and nothing else. Composing here rather than
    rotating each output separately is what keeps the preview, the DXF and the
    solid from ever disagreeing.
    """
    M = np.eye(4)
    M[:2, :2] = rotation_2d(angle)
    return M


def translate_to_origin(pts_2d) -> np.ndarray:
    """Move the blank so its lower-left corner sits at (0, 0).

    Cut files are easier to read and to nest when the part is in the positive
    quadrant, and it makes two DXFs of the same part directly comparable.
    """
    pts = np.asarray(pts_2d, dtype=float).reshape(-1, 2)
    if len(pts) == 0:
        return pts
    return pts - pts.min(axis=0)
