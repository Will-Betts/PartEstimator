"""Pure-numpy 2D computational-geometry primitives used to build the
oriented (minimum-volume) rectilinear bounding box and the minimum
enclosing cylinder. Kept dependency-free (no trimesh) and unit tested
independently of any CAD library -- see the project README for the
verification notes.

Performance note: these are the hot inner loops of the whole tool (they
run once per candidate box orientation / cylinder axis direction), so the
rotating-calipers step is fully vectorized over hull edges, and the
minimum-enclosing-circle routine reduces to the 2D hull before running
Welzl's algorithm. Both changes are exactness-preserving -- the MEC of a
point set is determined entirely by its hull vertices, so hulling first
changes only the runtime, not the answer.
"""
from __future__ import annotations

import random

import numpy as np


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Monotone-chain convex hull. `points`: (N, 2) array.
    Returns hull vertices in counter-clockwise order, no duplicate closing
    point. Degenerate inputs (<=2 unique points) are returned as-is.
    """
    pts = np.unique(np.asarray(points, dtype=float), axis=0)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    if len(pts) <= 2:
        return pts

    def build(seq):
        chain = []
        for p in seq:
            while len(chain) >= 2:
                a, b = chain[-2], chain[-1]
                if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) <= 0:
                    chain.pop()
                else:
                    break
            chain.append(p)
        return chain

    lower = build(pts)
    upper = build(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def min_area_rectangle_2d(points: np.ndarray, already_hulled: bool = False):
    """Rotating-calipers minimum-area bounding rectangle of a 2D point set.

    Returns (area, center(2,), extents(w, h), rotation_angle_rad) where
    rotation_angle_rad is the angle of the rectangle's "width" edge.

    Vectorized: all hull-edge orientations are evaluated in one pair of
    matrix multiplies rather than a Python loop over edges. Pass
    already_hulled=True to skip the hull step when the caller has one.
    """
    hull = np.asarray(points, dtype=float) if already_hulled else convex_hull_2d(points)
    n = len(hull)
    if n == 0:
        return 0.0, np.zeros(2), (0.0, 0.0), 0.0
    if n == 1:
        return 0.0, hull[0], (0.0, 0.0), 0.0
    if n == 2:
        d = hull[1] - hull[0]
        length = float(np.linalg.norm(d))
        angle = float(np.arctan2(d[1], d[0]))
        center = (hull[0] + hull[1]) / 2
        return 0.0, center, (length, 0.0), angle

    edges = np.roll(hull, -1, axis=0) - hull          # (n,2)
    lengths = np.linalg.norm(edges, axis=1)
    keep = lengths > 1e-12
    if not np.any(keep):
        return 0.0, hull.mean(axis=0), (0.0, 0.0), 0.0

    ux = edges[keep] / lengths[keep][:, None]          # (m,2) edge directions
    uy = np.stack([-ux[:, 1], ux[:, 0]], axis=1)       # (m,2) perpendiculars

    proj_x = hull @ ux.T                               # (n,m)
    proj_y = hull @ uy.T
    x_min, x_max = proj_x.min(axis=0), proj_x.max(axis=0)
    y_min, y_max = proj_y.min(axis=0), proj_y.max(axis=0)
    widths = x_max - x_min
    heights = y_max - y_min
    areas = widths * heights

    i = int(np.argmin(areas))
    cx = (x_max[i] + x_min[i]) / 2
    cy = (y_max[i] + y_min[i]) / 2
    center = cx * ux[i] + cy * uy[i]
    angle = float(np.arctan2(ux[i, 1], ux[i, 0]))
    return float(areas[i]), center, (float(widths[i]), float(heights[i])), angle


def _circle_from_2(a: np.ndarray, b: np.ndarray):
    center = (a + b) / 2
    r = float(np.linalg.norm(a - b) / 2)
    return center, r


def _circle_from_3(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        # Collinear points: the min enclosing circle is defined by the two
        # farthest-apart points.
        pts = [a, b, c]
        best = None
        for i in range(3):
            for j in range(i + 1, 3):
                dist = np.linalg.norm(pts[i] - pts[j])
                if best is None or dist > best[0]:
                    best = (dist, pts[i], pts[j])
        return _circle_from_2(best[1], best[2])
    ux = ((ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (cy - ay)
          + (cx ** 2 + cy ** 2) * (ay - by)) / d
    uy = ((ax ** 2 + ay ** 2) * (cx - bx) + (bx ** 2 + by ** 2) * (ax - cx)
          + (cx ** 2 + cy ** 2) * (bx - ax)) / d
    center = np.array([ux, uy])
    r = float(np.linalg.norm(center - a))
    return center, r


def _in_circle(p: np.ndarray, center: np.ndarray, r: float, eps: float = 1e-7) -> bool:
    return np.linalg.norm(p - center) <= r + eps


def _welzl(pts: list) -> tuple:
    random.Random(42).shuffle(pts)  # fixed seed -> deterministic, reproducible results
    center, r = pts[0], 0.0
    for i in range(1, len(pts)):
        if _in_circle(pts[i], center, r):
            continue
        center, r = pts[i], 0.0
        for j in range(i):
            if _in_circle(pts[j], center, r):
                continue
            center, r = _circle_from_2(pts[i], pts[j])
            for k in range(j):
                if _in_circle(pts[k], center, r):
                    continue
                center, r = _circle_from_3(pts[i], pts[j], pts[k])
    return center, r


def min_enclosing_circle(points: np.ndarray, already_hulled: bool = False):
    """Exact minimum enclosing circle. Returns (center(2,), radius).

    Reduces to the convex hull before running Welzl's algorithm. This is
    exactness-preserving -- the minimum enclosing circle is determined
    entirely by hull vertices, since any interior point is by definition
    already inside any circle containing the hull -- but it's a large
    speedup, because hulling a tessellated projection typically cuts the
    point count by 1-2 orders of magnitude and Welzl's Python-level
    constant factor is high.

    Pass already_hulled=True if the caller has already hulled the input.
    """
    arr = np.asarray(points, dtype=float)
    if len(arr) == 0:
        raise ValueError("min_enclosing_circle requires at least one point")

    hull = arr if already_hulled else convex_hull_2d(arr)
    if len(hull) == 1:
        return hull[0], 0.0
    if len(hull) == 2:
        return _circle_from_2(hull[0], hull[1])
    return _welzl([np.asarray(p, dtype=float) for p in hull])
