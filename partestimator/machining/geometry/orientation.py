"""How the part sits in its stock, when the automatic fit is not how it will
be machined.

The automatic box is the minimum-volume one, which is usually right and
sometimes not: a part with a large sloped face can come out tilted in its
billet when the shop would clamp it square to its base. Two ways to say
otherwise, both working on the stock FRAME (three orthonormal axes):

  align_axis_to_face()  make one face of the stock parallel to a planar
                        face of the part, the rest fitted as tight as
                        that allows
  rotated()             turn the part about the stock's own length,
                        width and thickness axes, in degrees

and `box_in_frame()` measures the part in any frame, giving the same
`OrientedBox` the automatic fit gives, so stock, features and the 3D view
all follow without knowing an override happened.

Conventions match bounding_box.OrientedBox: the frame's ROWS are the axes,
in world coordinates; a point's coordinates in the frame are `axes @ p`.
"""
from __future__ import annotations

import numpy as np

from .bounding_box import OrientedBox, _box_corners, _orthonormal_basis
from .hull2d import convex_hull_2d, min_area_rectangle_2d

#: Triangles of one face whose normals all agree within this are a plane.
PLANAR_TOLERANCE_DEG = 0.5

#: Labels for the frame's rows once sorted longest first.
AXIS_NAMES = ("X (length)", "Y (width)", "Z (thickness)")
STOCK_FACE_NAMES = ("end face (W x T)", "side face (L x T)", "top face (L x W)")


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("zero-length direction")
    return v / n


def right_handed(axes) -> np.ndarray:
    """`axes` with the third row flipped if need be, so X x Y = Z."""
    a = np.array(axes, dtype=float)
    if np.dot(np.cross(a[0], a[1]), a[2]) < 0:
        a[2] = -a[2]
    return a


def box_in_frame(points, axes) -> OrientedBox:
    """The tightest box round `points` whose edges run along `axes`."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    a = np.asarray(axes, dtype=float)
    proj = pts @ a.T
    lo, hi = proj.min(axis=0), proj.max(axis=0)
    extents = hi - lo
    centre = ((lo + hi) / 2.0) @ a
    return OrientedBox(kind="rectilinear", center_mm=centre, axes=a, extents_mm=extents,
                       volume_mm3=float(np.prod(extents)),
                       corners_mm=_box_corners(centre, a, extents))


def sorted_frame(box: OrientedBox) -> np.ndarray:
    """The box's axes reordered longest first: rows are length, width,
    thickness, the order the stock rules name them in."""
    order = np.argsort(-np.asarray(box.extents_mm, dtype=float), kind="stable")
    return right_handed(np.asarray(box.axes, dtype=float)[order])


def rotation_matrix(axis_index: int, degrees: float) -> np.ndarray:
    """Right-hand rotation about frame axis 0, 1 or 2."""
    t = np.radians(degrees)
    c, s = np.cos(t), np.sin(t)
    i, j = [(1, 2), (2, 0), (0, 1)][axis_index]
    r = np.eye(3)
    r[i, i], r[i, j], r[j, i], r[j, j] = c, -s, s, c
    return r


def rotated(base_axes, angles_deg) -> np.ndarray:
    """The frame after turning the PART by `angles_deg` = (about X, about Y,
    about Z) of `base_axes`, applied in that order.

    The part stays where the file put it, so it is the frame that turns, the
    opposite way: a point's coordinates in the new frame are its old ones
    rotated by R.
    """
    rx, ry, rz = (float(a) for a in angles_deg)
    r = rotation_matrix(2, rz) @ rotation_matrix(1, ry) @ rotation_matrix(0, rx)
    return r @ np.asarray(base_axes, dtype=float)


def tightest_frame_about(normal, points) -> np.ndarray:
    """A frame whose first axis is `normal`, the other two turned in its
    plane to give the smallest rectangle round the points."""
    n = _unit(normal)
    u, v = _orthonormal_basis(n)
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    hull = convex_hull_2d(np.column_stack((pts @ u, pts @ v)))
    if len(hull) >= 3:
        _area, _c, _wh, angle = min_area_rectangle_2d(hull, already_hulled=True)
        ca, sa = np.cos(angle), np.sin(angle)
        u, v = ca * u + sa * v, -sa * u + ca * v
    return np.array([n, u, v])


def align_axis_to_face(axes, axis_index: int, normal, points) -> np.ndarray:
    """`axes` turned so its row `axis_index` is parallel to `normal`: that
    stock face then lies flat on the part's face.

    The other two axes are refitted in the face's plane for the smallest
    stock, then put back in the rows nearest where they were, so the axis
    you did not touch keeps its name where it can.
    """
    old = np.asarray(axes, dtype=float)
    n = _unit(normal)
    if np.dot(n, old[axis_index]) < 0:
        n = -n                                  # parallel either way; keep its sense
    fit = tightest_frame_about(n, points)
    in_plane = [fit[1], fit[2]]
    others = [m for m in range(3) if m != axis_index]
    # Match the two refitted axes to the two old ones they are closest to.
    straight = abs(np.dot(in_plane[0], old[others[0]])) + abs(np.dot(in_plane[1], old[others[1]]))
    crossed = abs(np.dot(in_plane[1], old[others[0]])) + abs(np.dot(in_plane[0], old[others[1]]))
    if crossed > straight:
        in_plane.reverse()
    new = np.empty((3, 3))
    new[axis_index] = n
    for m, vec in zip(others, in_plane):
        new[m] = vec if np.dot(vec, old[m]) >= 0 else -vec
    return _fix_handedness_keeping(new, axis_index)


def _fix_handedness_keeping(axes, keep: int) -> np.ndarray:
    """Make `axes` right-handed by flipping a row other than `keep`."""
    a = np.array(axes, dtype=float)
    if np.dot(np.cross(a[0], a[1]), a[2]) < 0:
        flip = 0 if keep != 0 else 1
        a[flip] = -a[flip]
    return a


# ---------------------------------------------------------------------------
# faces of the tessellated part
# ---------------------------------------------------------------------------


def face_labels(triangles, n_points: int) -> np.ndarray:
    """Which CAD face each triangle came from.

    The mesh is extracted face by face and faces share no vertices, so the
    triangles joined through shared vertices are exactly one face each.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    tri = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    if len(tri) == 0:
        return np.zeros(0, dtype=np.int64)
    rows = np.concatenate((tri[:, 0], tri[:, 1]))
    cols = np.concatenate((tri[:, 1], tri[:, 2]))
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_points, n_points))
    _n, vertex_label = connected_components(graph, directed=False)
    return vertex_label[tri[:, 0]]


def planar_face(points, triangles, labels, triangle_index: int,
                tolerance_deg: float = PLANAR_TOLERANCE_DEG):
    """(unit normal, area mm^2, centroid) of the face holding this triangle,
    or None when that face is not flat."""
    pts = np.asarray(points, dtype=float)
    tri = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    if not 0 <= triangle_index < len(tri):
        return None
    mine = tri[labels == labels[triangle_index]]
    a, b, c = pts[mine[:, 0]], pts[mine[:, 1]], pts[mine[:, 2]]
    cross = np.cross(b - a, c - a)
    area2 = np.linalg.norm(cross, axis=1)
    keep = area2 > 0
    if not keep.any():
        return None
    total = cross[keep].sum(axis=0)
    if np.linalg.norm(total) == 0:
        return None
    n = total / np.linalg.norm(total)
    normals = cross[keep] / area2[keep, None]
    if np.min(normals @ n) < np.cos(np.radians(tolerance_deg)):
        return None
    w = area2[keep]
    centroid = (((a + b + c)[keep] / 3.0) * w[:, None]).sum(axis=0) / w.sum()
    return n, float(w.sum() / 2.0), centroid


def ray_box_face(origin, direction, box: OrientedBox, extents=None):
    """Which face of `box` a ray enters by: (axis index, +1 or -1), or None.

    `extents` overrides the box's own, for a stock drawn larger than the
    part but on the same centre and axes.
    """
    a = np.asarray(box.axes, dtype=float)
    half = np.asarray(box.extents_mm if extents is None else extents, dtype=float) / 2.0
    o = a @ (np.asarray(origin, dtype=float) - np.asarray(box.center_mm, dtype=float))
    d = a @ _unit(direction)
    t_enter, face = -np.inf, None
    t_exit = np.inf
    for k in range(3):
        if abs(d[k]) < 1e-12:
            if abs(o[k]) > half[k]:
                return None
            continue
        t1, t2 = (-half[k] - o[k]) / d[k], (half[k] - o[k]) / d[k]
        near, sign = (t1, -1) if t1 < t2 else (t2, 1)
        far = max(t1, t2)
        if near > t_enter:
            t_enter, face = near, (k, sign)
        t_exit = min(t_exit, far)
    if face is None or t_enter > t_exit or t_exit < 0:
        return None
    return face


def box_face_corners(center, axes, extents, axis_index: int, sign: int) -> np.ndarray:
    """The four corners of one face of a box, in order round the face."""
    a = np.asarray(axes, dtype=float)
    half = np.asarray(extents, dtype=float) / 2.0
    i, j = [m for m in range(3) if m != axis_index]
    c = np.asarray(center, dtype=float) + sign * half[axis_index] * a[axis_index]
    return np.array([c + si * half[i] * a[i] + sj * half[j] * a[j]
                     for si, sj in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def angle_between_deg(u, v) -> float:
    """Angle between two directions, ignoring which way they point."""
    c = abs(float(np.dot(_unit(u), _unit(v))))
    return float(np.degrees(np.arccos(min(1.0, c))))
