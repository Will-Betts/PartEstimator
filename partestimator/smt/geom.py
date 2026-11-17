"""Pure-numpy geometry primitives.

No OpenCASCADE dependency lives in this module by design -- everything here is
unit-testable without a CAD kernel installed. All transforms are 4x4 homogeneous
matrices representing rigid motions (rotation + translation, no scale).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def unit(v) -> np.ndarray:
    """Normalise a vector, raising on degenerate input."""
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < EPS:
        raise ValueError("cannot normalise a zero-length vector")
    return v / n


def identity() -> np.ndarray:
    return np.eye(4)


def translation(v) -> np.ndarray:
    M = np.eye(4)
    M[:3, 3] = np.asarray(v, dtype=float)
    return M


def rotation_about_axis(point, direction, angle: float) -> np.ndarray:
    """Rigid rotation of `angle` radians about the line through `point` along `direction`.

    Right-handed Rodrigues convention.
    """
    d = unit(direction)
    p = np.asarray(point, dtype=float)

    K = np.array(
        [
            [0.0, -d[2], d[1]],
            [d[2], 0.0, -d[0]],
            [-d[1], d[0], 0.0],
        ]
    )
    R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)

    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = p - R @ p
    return M


def apply_point(M: np.ndarray, p) -> np.ndarray:
    """Transform a position (translation applies)."""
    return M[:3, :3] @ np.asarray(p, dtype=float) + M[:3, 3]


def apply_vector(M: np.ndarray, v) -> np.ndarray:
    """Transform a direction (translation does not apply)."""
    return M[:3, :3] @ np.asarray(v, dtype=float)


def compose(*mats: np.ndarray) -> np.ndarray:
    """compose(A, B, C) applies C first, then B, then A."""
    out = np.eye(4)
    for m in mats:
        out = out @ m
    return out


def invert_rigid(M: np.ndarray) -> np.ndarray:
    """Inverse of a rigid transform, exploiting R^-1 == R^T."""
    R = M[:3, :3]
    t = M[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


# --------------------------------------------------------------------------
# predicates
# --------------------------------------------------------------------------


def parallel(d1, d2, tol: float = 1e-6) -> bool:
    """True if two directions are parallel OR antiparallel."""
    return abs(abs(float(np.dot(unit(d1), unit(d2)))) - 1.0) < tol


def codirectional(d1, d2, tol: float = 1e-6) -> bool:
    return abs(float(np.dot(unit(d1), unit(d2))) - 1.0) < tol


def antiparallel(d1, d2, tol: float = 1e-6) -> bool:
    return abs(float(np.dot(unit(d1), unit(d2))) + 1.0) < tol


def perpendicular(d1, d2, tol: float = 1e-6) -> bool:
    return abs(float(np.dot(unit(d1), unit(d2)))) < tol


def point_line_distance(p, line_point, line_dir) -> float:
    d = unit(line_dir)
    v = np.asarray(p, dtype=float) - np.asarray(line_point, dtype=float)
    return float(np.linalg.norm(v - np.dot(v, d) * d))


def same_axis(p1, d1, p2, d2, ang_tol: float = 1e-6, lin_tol: float = 1e-6) -> bool:
    """True if two (point, direction) pairs describe the same infinite line."""
    if not parallel(d1, d2, ang_tol):
        return False
    return point_line_distance(p2, p1, d1) < lin_tol


def project_to_plane(v, normal) -> np.ndarray:
    """Remove the component of `v` along `normal`."""
    n = unit(normal)
    v = np.asarray(v, dtype=float)
    return v - np.dot(v, n) * n


def signed_plane_distance(p, plane_point, plane_normal) -> float:
    n = unit(plane_normal)
    return float(
        np.dot(np.asarray(p, dtype=float) - np.asarray(plane_point, dtype=float), n)
    )


def cylinder_tangent_to_plane(
    axis_point,
    axis_dir,
    radius: float,
    plane_point,
    plane_normal,
    ang_tol: float = 1e-6,
    lin_tol: float = 1e-4,
) -> bool:
    """Analytic tangency test between a cylinder and a plane.

    A cylinder is tangent to a plane iff its axis is parallel to the plane
    (equivalently, perpendicular to the plane normal) AND the axis stands off
    the plane by exactly the radius.

    Used instead of sampling surface normals along the shared edge: it is exact,
    cheap, and immune to parametrisation quirks in imported STEP.
    """
    if not perpendicular(axis_dir, plane_normal, ang_tol):
        return False
    d = abs(signed_plane_distance(axis_point, plane_point, plane_normal))
    return abs(d - radius) < lin_tol


def cylinder_uv(points, axis_point, axis_dir, x_dir, u_reference: float = 0.0):
    """3D points on a cylinder -> (u, v) surface parameters.

    u is the angle about the axis measured from `x_dir`, v the distance along
    the axis. This matches OpenCASCADE's own cylindrical parametrisation, so the
    values are directly comparable with a face's UV bounds.

    u is unwrapped around `u_reference` -- pass the middle of the face's sweep
    and every point lands within +/- pi of it, which covers any sweep short of a
    full turn. Without that, a bend straddling the atan2 branch cut comes back
    with half its points 2*pi away and develops into nonsense.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    d = unit(axis_dir)
    x = unit(np.asarray(x_dir, dtype=float) - np.dot(x_dir, d) * d)
    y = np.cross(d, x)

    rel = pts - np.asarray(axis_point, dtype=float)
    v = rel @ d
    w = rel - np.outer(v, d)

    u = np.arctan2(w @ y, w @ x)
    u = u_reference + (u - u_reference + np.pi) % (2.0 * np.pi) - np.pi
    return u, v


def develop_cylinder(
    u, v, rho: float, u_tangent: float, sign: float,
    origin, along_axis, across,
):
    """Isometric development of a cylindrical patch onto the plane.

    THIS IS WHAT THE K-FACTOR ACTUALLY MEANS.

    A cylinder is a developable surface: it can be unrolled onto a plane without
    stretching anything. The neutral surface -- radius `rho` = R + K*t -- is by
    definition the one layer of the material that neither stretches nor
    compresses through the bend. Unrolling THAT surface is what turns a bend
    into flat material, and the arc length it contributes,

        rho * theta

    is exactly the bend allowance. The formula is not an empirical fudge bolted
    onto a rectangle; it is the arc length of this development.

    Working this way rather than pasting in a rectangle of width BA means the
    real trimmed boundary of the bend comes through: a cutout crossing the bend,
    a flange that does not meet the bend line squarely, a bend that runs off the
    edge of the part. All of them develop correctly because we are unrolling the
    actual face instead of assuming its shape.

    It also generalises. A cone is developable too, so rolled and tapered
    sections are the same idea with a different mapping.

        origin       the parent tangent line's start, in flat coordinates
        along_axis   unit bend-axis direction, in flat coordinates
        across       unit direction from the tangent line toward the child
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    o = np.asarray(origin, dtype=float)
    a = np.asarray(along_axis, dtype=float)
    c = np.asarray(across, dtype=float)

    arc = rho * sign * (u - u_tangent)
    return o[None, :] + np.outer(v, a) + np.outer(arc, c)


def chain_point_runs(runs, tol: float = 1e-6):
    """Order point runs end to end, flipping any that fit better reversed.

    Pure geometry: no orientation flags, no topology, no kernel. Used as the
    last-resort way to trace a boundary when the kernel's wire explorer returns
    something that does not close. Points either join up or they do not, and
    that is a question numpy can answer.

    Greedy nearest-endpoint chaining. On a well-formed boundary every step has
    an exact match at distance ~0, so the greediness never has to make a
    judgement call; on a malformed one it still produces the most plausible
    single path rather than giving up.
    """
    remaining = [np.asarray(r, dtype=float) for r in runs if len(r) >= 2]
    if not remaining:
        return []

    ordered = [remaining.pop(0)]
    while remaining:
        tail = ordered[-1][-1]
        best_i, best_rev, best_d = 0, False, float("inf")
        for i, run in enumerate(remaining):
            d_start = float(np.linalg.norm(run[0] - tail))
            d_end = float(np.linalg.norm(run[-1] - tail))
            if d_start < best_d:
                best_i, best_rev, best_d = i, False, d_start
            if d_end < best_d:
                best_i, best_rev, best_d = i, True, d_end
        run = remaining.pop(best_i)
        ordered.append(run[::-1] if best_rev else run)
    return ordered


def join_point_runs(runs, tol: float = 1e-6) -> np.ndarray:
    """Concatenate ordered runs, dropping duplicated shared endpoints."""
    pts: list[np.ndarray] = []
    for run in runs:
        run = np.asarray(run, dtype=float)
        if len(run) == 0:
            continue
        if pts and float(np.linalg.norm(run[0] - pts[-1])) < tol:
            run = run[1:]
        pts.extend(run)
    if not pts:
        return np.zeros((0, 3))
    return np.array(pts, dtype=float)


def loop_is_closed(pts, tol: float = 1e-3) -> bool:
    pts = np.asarray(pts, dtype=float)
    return len(pts) >= 3 and float(np.linalg.norm(pts[0] - pts[-1])) < tol


def group_by_line(items, get_axis, ang_tol: float = 1e-6, lin_tol: float = 1e-5):
    """Group items that lie on the same infinite line.

    Grouping is by EXACT `same_axis` test against each existing group's
    representative -- deliberately not by rounding coordinates into buckets.

    Rounding would be faster, but two axes that differ by less than the
    tolerance can straddle a rounding boundary and land in different buckets,
    silently failing to merge. On this codebase that would mean a bend whose
    face was split by a hole reading as two bends, which is precisely the
    failure the merging exists to prevent. A speedup that can quietly produce a
    wrong flat pattern is not worth having.

    Complexity is O(n * groups). The worst case matches a naive pairwise scan;
    the common case, where a part has far fewer distinct axes than cylindrical
    faces, is close to linear.
    """
    groups: list[list] = []
    reps: list[tuple] = []

    for it in items:
        p, d = get_axis(it)
        for gi, (rp, rd) in enumerate(reps):
            if same_axis(rp, rd, p, d, ang_tol=ang_tol, lin_tol=lin_tol):
                groups[gi].append(it)
                break
        else:
            groups.append([it])
            reps.append((np.asarray(p, dtype=float), np.asarray(d, dtype=float)))
    return groups


def bbox_of_points(pts) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    return pts.min(axis=0), pts.max(axis=0)


def min_area_rect(pts_2d) -> tuple[float, float, float]:
    """Rotating-calipers minimum-area bounding rectangle of 2D points.

    Returns (width, height, angle_radians). Used for nesting when the part is
    not axis-aligned and grain direction is unconstrained.
    """
    pts = np.asarray(pts_2d, dtype=float).reshape(-1, 2)
    if len(pts) < 3:
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        return float(hi[0] - lo[0]), float(hi[1] - lo[1]), 0.0

    hull = _convex_hull(pts)
    best = None
    n = len(hull)
    for i in range(n):
        edge = hull[(i + 1) % n] - hull[i]
        norm = np.linalg.norm(edge)
        if norm < 1e-12:
            continue
        e = edge / norm
        R = np.array([[e[0], e[1]], [-e[1], e[0]]])
        proj = hull @ R.T
        lo, hi = proj.min(axis=0), proj.max(axis=0)
        w, h = hi - lo
        area = w * h
        if best is None or area < best[0]:
            best = (area, float(w), float(h), float(np.arctan2(e[1], e[0])))
    if best is None:
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        return float(hi[0] - lo[0]), float(hi[1] - lo[1]), 0.0
    return best[1], best[2], best[3]


def _convex_hull(pts: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain."""
    pts = np.unique(pts, axis=0)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def bend_cut_plane(axis_point, axis_dir, plane_normal, plane_point):
    """Where to cut a solid to separate a bend from the flange it leaves.

    A bend's arc meets the flat flange along a TANGENT LINE. Because the bend
    axis sits exactly one radius off the flange's plane, dropping it
    perpendicular onto that plane lands precisely on the tangency -- no radius
    is needed, and using one would be a way to get it wrong when inner and outer
    radii are confused.

    The cutting plane contains that tangent line, runs along the axis, and
    stands square to the sheet. Its normal is therefore `axis x flange_normal`.

    Returns (point_on_plane, unit_normal). Raises if the axis is parallel to the
    flange normal, which cannot describe a bend -- a bend axis lies IN the plane
    of the sheet it bends.
    """
    n = unit(plane_normal)
    axis = unit(axis_dir)

    cross = np.cross(axis, n)
    mag = float(np.linalg.norm(cross))
    if mag < 1e-9:
        raise ValueError(
            "the bend axis is parallel to the flange normal, so no cutting "
            "plane exists; this is not a bend"
        )

    d = signed_plane_distance(axis_point, plane_point, n)
    tangent_point = np.asarray(axis_point, dtype=float) - d * n
    return tangent_point, cross / mag
