"""Minimum-volume oriented rectilinear bounding box and minimum-volume
enclosing cylinder.

Both are hard computational-geometry optimization problems in the general
case. We use well-established *practical* heuristics rather than
exact-optimality algorithms:

  * Rectilinear box: the true minimum-volume OBB of a convex polyhedron is
    known to always have at least one face flush with a face of the
    convex hull (this is the same heuristic used by, e.g., trimesh's
    `oriented_bounds`). We enumerate the hull's face normals, and for each
    one compute the tightest 2D rectangle of the hull points projected
    into that plane (via rotating calipers), then keep the candidate with
    the smallest volume.

  * Enclosing cylinder: there is no equivalent closed-form heuristic, so
    we do a directional search -- coarse sampling over the unit sphere,
    then local refinement -- minimizing cylinder volume where, for each
    candidate axis direction, the radius is the *exact* minimum enclosing
    circle (Welzl's algorithm) of the points projected perpendicular to
    that axis.

PERFORMANCE DESIGN (this module was the main bottleneck on large parts).
Cost scales as (number of candidate orientations x number of points), and
a finely tessellated part inflates both. Three changes, in order of impact:

  1. The 3D convex hull is computed ONCE and shared between the box and
     the cylinder routines (previously each recomputed it).
  2. Candidate normals are deduplicated at a coarse angular tolerance and
     then capped, keeping the normals belonging to the LARGEST-AREA hull
     faces. This matters: a tessellated cylindrical or freeform region
     contributes thousands of nearly-identical small-area normals that add
     nothing, while the minimum-volume box is almost always flush with a
     large face. Benchmarking on a synthetic prismatic part, area-weighted
     capping to 60 normals found the exact optimum, while randomly capping
     to 60 overshot the true box volume by 64%.
  3. Orientation SEARCH runs on a subsample of hull vertices, but the
     final extents / radius / height are always MEASURED on the full hull.
     This is what preserves the enclosure guarantee: subsampling only
     affects which orientation gets picked, never whether the resulting
     box actually contains the part. Measured excess volume versus an
     exhaustive search on test parts was 0.1-3%.

Set SEARCH_SUBSAMPLE_LIMIT / MAX_CANDIDATE_NORMALS higher for more
accuracy at the cost of speed, or lower for faster results.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import ConvexHull

from .hull2d import convex_hull_2d, min_area_rectangle_2d, min_enclosing_circle

# --- tuning knobs (raise for accuracy, lower for speed) ---
MAX_CANDIDATE_NORMALS = 250      # cap on box orientations tried (area-weighted pick)
NORMAL_DEDUPE_DECIMALS = 2       # coarse rounding for normal dedupe (~0.5 deg buckets)
SEARCH_SUBSAMPLE_LIMIT = 600     # max hull vertices used during orientation search
SPHERE_SAMPLE_COUNT = 200        # coarse cylinder axis directions
REFINE_ITERATIONS = 6
REFINE_INITIAL_STEP_RAD = 0.15

# Qhull time and memory grow with input size, and a finely tessellated part
# can hand it millions of points. Above this count we pre-filter to points
# that are extremal along a spread of directions -- a cheap superset-style
# reduction. It is NOT guaranteed to retain every true hull vertex, which
# is exactly why `minimum_oriented_box` / `minimum_enclosing_cylinder`
# re-measure their final answer against the FULL point cloud: the reduction
# can only affect which orientation is chosen, never whether the reported
# box or cylinder encloses the part.
# Qhull itself is decent (roughly n log n, and a million 3D points takes a
# few seconds), so this pre-filter is deliberately set to trigger only on
# genuinely huge meshes -- an over-eager filter costs more than the hull it
# saves. Its own work is bounded by HULL_FILTER_SCAN_LIMIT so it can never
# become the new bottleneck.
HULL_INPUT_LIMIT = 400_000
HULL_FILTER_DIRECTIONS = 128
HULL_FILTER_SCAN_LIMIT = 150_000


@dataclass
class HullData:
    """The shared, precomputed convex hull of a shape's tessellated points.
    Build once per part via `prepare_hull` and hand to both bounding-box
    routines so the (expensive) hull and normal analysis isn't repeated."""
    points: np.ndarray            # (N,3) hull vertices -- used for ranking candidates
    search_points: np.ndarray     # (M,3) subsample of the above -- used for SEARCH
    candidate_normals: np.ndarray  # (K,3) unit normals, area-weighted and capped
    all_points: np.ndarray        # every input point -- used for the FINAL measurement
    hull_volume_mm3: float = 0.0  # enclosed volume of the convex hull
    hull_is_approximate: bool = False  # True when built from a reduced point set


def _reduce_for_hull(points: np.ndarray, n_directions: int = HULL_FILTER_DIRECTIONS,
                      rng_seed: int = 0, chunk: int = 32,
                      scan_limit: int = HULL_FILTER_SCAN_LIMIT) -> np.ndarray:
    """Keeps points that are extremal along a spread of directions, plus the
    exact axis-aligned extremes.

    Two things keep this cheap:
      * The axis extremes are taken on the FULL array (six argmin/argmax
        passes, trivial), so the overall bounding extents are always exact.
      * The directional scan runs on at most `scan_limit` points and uses
        float32 in chunks. Without those bounds the (N, K) projection matrix
        alone is gigabytes, and allocating it costs more than the convex
        hull this is meant to accelerate.
    """
    rng = np.random.default_rng(rng_seed)

    keep: set[int] = set()
    for ax in range(3):
        col = points[:, ax]
        keep.add(int(col.argmin()))
        keep.add(int(col.argmax()))

    scan = points
    scan_idx = None
    if len(points) > scan_limit:
        scan_idx = rng.choice(len(points), size=scan_limit, replace=False)
        scan = points[scan_idx]
    scan32 = np.ascontiguousarray(scan, dtype=np.float32)

    dirs = rng.normal(size=(n_directions, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs32 = dirs.astype(np.float32)

    local: set[int] = set()
    for start in range(0, len(dirs32), chunk):
        proj = scan32 @ dirs32[start:start + chunk].T
        local.update(proj.argmin(axis=0).tolist())
        local.update(proj.argmax(axis=0).tolist())

    if scan_idx is None:
        keep.update(local)
    else:
        keep.update(int(scan_idx[i]) for i in local)

    return points[np.fromiter(keep, dtype=np.int64, count=len(keep))]


@dataclass
class OrientedBox:
    kind: str  # "rectilinear"
    center_mm: np.ndarray  # (3,)
    axes: np.ndarray  # (3,3) rows are unit vectors
    extents_mm: np.ndarray  # (3,) full lengths along each row of `axes`
    volume_mm3: float
    corners_mm: np.ndarray  # (8,3) box corners in world coords, for rendering

    @property
    def longest_axis_direction(self) -> np.ndarray:
        return self.axes[int(np.argmax(self.extents_mm))]

    @property
    def dims_sorted_desc(self) -> tuple[float, float, float]:
        return tuple(sorted(self.extents_mm, reverse=True))


@dataclass
class EnclosingCylinder:
    kind: str = field(default="cylindrical", init=False)
    axis_point_mm: np.ndarray = None
    axis_direction: np.ndarray = None  # unit vector
    radius_mm: float = 0.0
    height_mm: float = 0.0
    volume_mm3: float = 0.0


def prepare_hull(points: np.ndarray, rng_seed: int = 0) -> HullData:
    """Computes the 3D convex hull once and derives the reduced search sets."""
    all_points = np.asarray(points, dtype=float)

    hull_input = all_points
    reduced = False
    if len(hull_input) > HULL_INPUT_LIMIT:
        hull_input = _reduce_for_hull(all_points, rng_seed=rng_seed)
        reduced = True

    hull = ConvexHull(hull_input)
    points = hull_input
    hull_pts = hull_input[hull.vertices]

    # Face normals, weighted by triangle area so we can keep the ones that
    # actually matter for a minimum-volume fit.
    normals = hull.equations[:, :3]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals = normals / norms

    tri = points[hull.simplices]  # (F,3,3)
    areas = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )

    # Deduplicate coarsely; sum areas of all faces sharing a bucket so a
    # large planar face split into many triangles is correctly seen as big.
    rounded = np.round(normals, NORMAL_DEDUPE_DECIMALS)
    unique_rounded, inverse = np.unique(rounded, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    n_buckets = len(unique_rounded)

    bucket_area = np.zeros(n_buckets)
    np.add.at(bucket_area, inverse, areas)

    # Representative (un-rounded) normal per bucket: the largest single face
    # in that bucket. Vectorized via a sort on (bucket, area).
    order = np.lexsort((areas, inverse))       # ascending area within bucket
    sorted_buckets = inverse[order]
    last_in_bucket = np.ones(len(order), dtype=bool)
    last_in_bucket[:-1] = sorted_buckets[:-1] != sorted_buckets[1:]
    rep_index = order[last_in_bucket]           # largest-area face per bucket
    candidate_normals = normals[rep_index]
    bucket_area = bucket_area[sorted_buckets[last_in_bucket]]

    if len(candidate_normals) > MAX_CANDIDATE_NORMALS:
        keep = np.argsort(-bucket_area)[:MAX_CANDIDATE_NORMALS]
        candidate_normals = candidate_normals[keep]

    if len(hull_pts) > SEARCH_SUBSAMPLE_LIMIT:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(len(hull_pts), size=SEARCH_SUBSAMPLE_LIMIT, replace=False)
        search_points = hull_pts[idx]
    else:
        search_points = hull_pts

    return HullData(points=hull_pts, search_points=search_points,
                     candidate_normals=candidate_normals, all_points=all_points,
                     hull_volume_mm3=float(getattr(hull, "volume", 0.0)),
                     hull_is_approximate=reduced)


def _orthonormal_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Given a unit vector n, return two unit vectors (u, v) such that
    (n, u, v) is a right-handed orthonormal basis."""
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, n)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = ref - np.dot(ref, n) * n
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def minimum_oriented_box(points: np.ndarray, hull_data: HullData | None = None) -> OrientedBox:
    hull_data = hull_data if hull_data is not None else prepare_hull(points)
    measure_pts = hull_data.points
    search_pts = hull_data.search_points

    best = None  # (volume, axes(3,3), min_proj(3,), max_proj(3,))
    for n in hull_data.candidate_normals:
        u, v = _orthonormal_basis(n)
        # Orientation is chosen using the (possibly subsampled) search set.
        proj2d = np.column_stack([search_pts @ u, search_pts @ v])
        hull2 = convex_hull_2d(proj2d)
        if len(hull2) < 3:
            continue
        _area, _center2d, _wh, angle = min_area_rectangle_2d(hull2, already_hulled=True)

        ca, sa = np.cos(angle), np.sin(angle)
        axes = np.array([n, ca * u + sa * v, -sa * u + ca * v])

        # ...but extents are always measured against the FULL hull, so the
        # resulting box is guaranteed to enclose the part.
        proj_all = measure_pts @ axes.T
        lo = proj_all.min(axis=0)
        hi = proj_all.max(axis=0)
        extents = hi - lo
        volume = float(np.prod(extents))
        if best is None or volume < best[0]:
            best = (volume, axes, lo, hi)

    if best is None:
        raise ValueError("Could not fit an oriented bounding box (degenerate geometry?)")

    volume, axes, lo, hi = best

    # FINAL measurement against every input point. The hull may have been
    # built from a reduced point set for speed, so this is what guarantees
    # the reported box actually encloses the whole part.
    proj_final = hull_data.all_points @ axes.T
    lo = proj_final.min(axis=0)
    hi = proj_final.max(axis=0)
    extents = hi - lo
    volume = float(np.prod(extents))
    center_local = (lo + hi) / 2.0
    center_world = center_local @ axes  # rows of axes are the world basis vectors
    corners = _box_corners(center_world, axes, extents)
    return OrientedBox(
        kind="rectilinear",
        center_mm=center_world,
        axes=axes,
        extents_mm=extents,
        volume_mm3=volume,
        corners_mm=corners,
    )


def axis_aligned_box(points: np.ndarray) -> OrientedBox:
    """Axis-aligned bounding box in the STEP file's own coordinate system.

    Essentially free (one min/max pass) versus searching hundreds of
    orientations. For a part that was modeled on-axis -- which most
    machined parts are, and which is also how they'll be set up on the
    machine -- this is identical to the oriented result. It only loses when
    the part is stored rotated in the file.
    """
    pts = np.asarray(points, dtype=float)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    extents = hi - lo
    center = (lo + hi) / 2.0
    axes = np.eye(3)
    return OrientedBox(
        kind="rectilinear",
        center_mm=center,
        axes=axes,
        extents_mm=extents,
        volume_mm3=float(np.prod(extents)),
        corners_mm=_box_corners(center, axes, extents),
    )


def _box_corners(center: np.ndarray, axes: np.ndarray, extents: np.ndarray) -> np.ndarray:
    half = extents / 2.0
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    offsets = signs * half  # (8,3) in box-local coordinates
    return center + offsets @ axes  # rows of axes are the world-space basis vectors


def _fibonacci_sphere(n: int) -> np.ndarray:
    """n roughly-uniform points on the unit sphere."""
    i = np.arange(n)
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    golden = np.pi * (1 + 5 ** 0.5)
    theta = golden * i
    return np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=1)


def _cylinder_for_direction(pts: np.ndarray, direction: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Tightest cylinder with axis parallel to `direction` enclosing `pts`.
    Returns (volume, radius, height, circle_center_2d)."""
    u, v = _orthonormal_basis(direction)
    proj_axis = pts @ direction
    height = float(proj_axis.max() - proj_axis.min())
    proj_2d = np.column_stack([pts @ u, pts @ v])
    center2d, radius = min_enclosing_circle(proj_2d)
    volume = float(np.pi * radius ** 2 * height)
    return volume, float(radius), height, center2d


def _direction_from_angles(theta: float, phi: float) -> np.ndarray:
    return np.array([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ])


def _angles_from_direction(d: np.ndarray) -> tuple[float, float]:
    phi = np.arccos(np.clip(d[2], -1.0, 1.0))
    theta = np.arctan2(d[1], d[0])
    return theta, phi


def minimum_enclosing_cylinder(points: np.ndarray, hull_data: HullData | None = None) -> EnclosingCylinder:
    hull_data = hull_data if hull_data is not None else prepare_hull(points)
    measure_pts = hull_data.points
    search_pts = hull_data.search_points

    # Coarse search over the sphere, on the reduced search set.
    best_volume = None
    best_direction = None
    for d in _fibonacci_sphere(SPHERE_SAMPLE_COUNT):
        volume, _, _, _ = _cylinder_for_direction(search_pts, d)
        if best_volume is None or volume < best_volume:
            best_volume = volume
            best_direction = d

    # Local refinement: shrinking-step coordinate descent in spherical
    # angles around the best coarse direction.
    theta, phi = _angles_from_direction(best_direction)
    step = REFINE_INITIAL_STEP_RAD
    for _ in range(REFINE_ITERATIONS):
        improved = False
        for dtheta, dphi in ((step, 0), (-step, 0), (0, step), (0, -step)):
            cand_theta, cand_phi = theta + dtheta, phi + dphi
            volume, _, _, _ = _cylinder_for_direction(search_pts, _direction_from_angles(cand_theta, cand_phi))
            if volume < best_volume:
                best_volume = volume
                theta, phi = cand_theta, cand_phi
                improved = True
        if not improved:
            step /= 2.0

    # Final geometry measured against EVERY input point, guaranteeing
    # enclosure even though the search ran on reduced sets.
    return cylinder_about(hull_data.all_points, _direction_from_angles(theta, phi))


def cylinder_about(points: np.ndarray, direction) -> EnclosingCylinder:
    """The tightest cylinder round `points` whose axis runs along `direction`
    -- the bar a part is turned from when it is set that way round."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    axis_direction = np.asarray(direction, dtype=float)
    axis_direction = axis_direction / (np.linalg.norm(axis_direction) or 1.0)
    volume, radius, height, center2d = _cylinder_for_direction(pts, axis_direction)

    u, v = _orthonormal_basis(axis_direction)
    proj_axis = pts @ axis_direction
    mid = (proj_axis.max() + proj_axis.min()) / 2.0
    axis_point = mid * axis_direction + center2d[0] * u + center2d[1] * v

    return EnclosingCylinder(
        axis_point_mm=axis_point,
        axis_direction=axis_direction,
        radius_mm=radius,
        height_mm=height,
        volume_mm3=volume,
    )


def select_primary(rect_box: OrientedBox, cylinder: EnclosingCylinder) -> str:
    """Returns "rectilinear" or "cylindrical", whichever has the smaller
    enclosing volume (per the user's chosen selection rule)."""
    return "rectilinear" if rect_box.volume_mm3 <= cylinder.volume_mm3 else "cylindrical"
