"""Hole-axis orientation statistics.

On a prismatic part, hole axes fall into a handful of discrete directions
and a cluster count is the natural measure -- it equals the number of
machine orientations. On a CONTOURED part where holes are normal to a
curved surface, the axes form a *continuum*: neighbouring holes differ by
a fraction of a degree, and there is no natural gap to cluster on. Two
consequences shaped this module.

  * Transitive clustering is useless there. If A is within tolerance of B
    and B of C, chaining merges the whole continuum into one cluster no
    matter how fine the tolerance. Clustering here is therefore GREEDY and
    non-transitive: each seed claims only the axes within tolerance of
    itself. Seeds are visited in a canonical (lexicographic) order so the
    result is deterministic and not an artifact of face ordering.

  * A single count at a single tolerance is fragile. So counts are
    reported at several tolerances, alongside angular SPREAD statistics
    (max, mean and 90th-percentile deviation from the dominant axis),
    which describe a continuum robustly and do not depend on a clustering
    rule at all.

Performance: axes are canonicalised onto one hemisphere and indexed with a
scipy cKDTree, so clustering is O(n log n) rather than the O(n^2) that a
naive pairwise comparison would cost -- which matters at several thousand
holes. Angular tolerance maps to a chord distance on the unit sphere:
d = 2*sin(theta/2). Both +d and -d are queried, because a direction and
its opposite describe the same axis and canonicalisation can otherwise
split axes that straddle the hemisphere boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:  # noqa: BLE001 - fall back to brute force if scipy is absent
    cKDTree = None

# Counts are reported at each of these tolerances. The fine end measures
# how many genuinely distinct tool orientations the part demands; the
# coarse end approximates how many setups (or 3+2 orientations) it needs.
AXIS_TOLERANCES_DEG = (0.1, 0.5, 1.0, 5.0)
EPS = 1e-9


def canonical_axes(directions) -> np.ndarray:
    """Unit vectors flipped onto a single hemisphere, so that an axis and
    its reverse compare equal."""
    d = np.asarray([x for x in directions if x is not None], dtype=float)
    if d.size == 0:
        return np.zeros((0, 3))
    d = np.atleast_2d(d)
    norms = np.linalg.norm(d, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    d = d / norms

    # Flip on the first significant component, in a fixed order, so the
    # choice is stable rather than depending on floating-point sign noise.
    flip = np.zeros(len(d), dtype=bool)
    flip |= d[:, 2] < -EPS
    tie_z = np.abs(d[:, 2]) <= EPS
    flip |= tie_z & (d[:, 1] < -EPS)
    tie_zy = tie_z & (np.abs(d[:, 1]) <= EPS)
    flip |= tie_zy & (d[:, 0] < 0)
    d[flip] *= -1.0
    return d


def _chord(tolerance_deg: float) -> float:
    return 2.0 * np.sin(np.radians(tolerance_deg) / 2.0)


def cluster_axes(axes: np.ndarray, tolerance_deg: float):
    """Greedy, non-transitive, order-independent clustering.

    Returns (labels, cluster_sizes, cluster_directions).
    """
    n = len(axes)
    if n == 0:
        return np.zeros(0, dtype=int), [], np.zeros((0, 3))

    radius = _chord(tolerance_deg)
    order = np.lexsort((axes[:, 2], axes[:, 1], axes[:, 0]))  # deterministic seeds
    labels = np.full(n, -1, dtype=int)
    sizes: list[int] = []
    reps: list[np.ndarray] = []

    tree = cKDTree(axes) if cKDTree is not None else None
    for i in order:
        if labels[i] >= 0:
            continue
        if tree is not None:
            near = set(tree.query_ball_point(axes[i], radius))
            near |= set(tree.query_ball_point(-axes[i], radius))
        else:
            dots = np.abs(axes @ axes[i])
            near = set(np.nonzero(dots >= np.cos(np.radians(tolerance_deg)))[0].tolist())
        members = [j for j in near if labels[j] < 0]
        label = len(sizes)
        for j in members:
            labels[j] = label
        sizes.append(len(members))
        reps.append(axes[i])

    return labels, sizes, np.asarray(reps)


def angular_deviations_deg(axes: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Angle of each axis from `reference`, treating an axis and its
    reverse as identical (so results lie in 0..90 degrees)."""
    if len(axes) == 0:
        return np.zeros(0)
    dots = np.clip(np.abs(axes @ np.asarray(reference, dtype=float)), 0.0, 1.0)
    return np.degrees(np.arccos(dots))


@dataclass
class AxisStats:
    count: int = 0
    counts_by_tolerance: dict = field(default_factory=dict)
    dominant_direction: np.ndarray = None
    frac_on_dominant: float = 0.0
    max_deviation_deg: float = 0.0
    mean_deviation_deg: float = 0.0
    p90_deviation_deg: float = 0.0
    frac_within_1deg: float = 0.0
    frac_within_5deg: float = 0.0


def compute_axis_stats(directions, tolerances_deg=AXIS_TOLERANCES_DEG) -> AxisStats:
    axes = canonical_axes(directions)
    stats = AxisStats(count=len(axes))
    if len(axes) == 0:
        stats.counts_by_tolerance = {t: 0 for t in tolerances_deg}
        return stats

    for tol in tolerances_deg:
        _labels, sizes, reps = cluster_axes(axes, tol)
        stats.counts_by_tolerance[tol] = len(sizes)
        if tol == max(tolerances_deg):
            biggest = int(np.argmax(sizes))
            stats.dominant_direction = reps[biggest]
            stats.frac_on_dominant = float(sizes[biggest]) / len(axes)

    # Spread is measured from the dominant direction of the COARSEST
    # clustering, which is the closest thing to "the part's main tool axis".
    deviations = angular_deviations_deg(axes, stats.dominant_direction)
    stats.max_deviation_deg = float(deviations.max())
    stats.mean_deviation_deg = float(deviations.mean())
    stats.p90_deviation_deg = float(np.percentile(deviations, 90))
    stats.frac_within_1deg = float((deviations <= 1.0).mean())
    stats.frac_within_5deg = float((deviations <= 5.0).mean())
    return stats
