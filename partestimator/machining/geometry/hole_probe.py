"""Ray-cast checks on recognised holes: is it real, and can a tool reach it?

Two questions that pure surface classification can't answer, both resolved
by casting rays against the solid.

1. VALIDATION -- is there material around the bore?
   A concave cylindrical face that sweeps a full circle looks like a hole,
   but so does the inside of an open arc, a scallop, or the relieved side
   of a slot. The test: step a hair radially OUTWARD from the bore wall and
   ask whether that point is inside solid material. For a genuine bore it
   always is, all the way round. Sampling several angles and requiring a
   majority rejects features that are only partly enclosed.

   Inside/outside is decided by ray parity -- cast a ray from the probe
   point and count how many times it crosses the boundary; an odd count
   means it started inside. This uses IntCurvesFace_ShapeIntersector, which
   is spatially indexed, rather than BRepClass3d_SolidClassifier, whose
   per-call cost is proportional to the whole shape (that was the original
   performance disaster in this codebase).

2. ACCESSIBILITY -- can a straight tool get to the mouth?
   From each open mouth, fire rays outward ALONG THE HOLE AXIS: one down
   the centre and a ring at the bore radius, approximating the tool
   envelope. Anything they hit is material between the mouth and open
   space, so the hole is obstructed from that side -- "captured" if every
   mouth is blocked. Through holes are probed from both mouths, blind
   holes only from the entry.

   A blocked approach doesn't mean the part is unmachinable; it means the
   hole can't be reached straight down its own axis from that side, which
   implies a different setup, an angled head, or extended tooling.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCP.TopoDS import TopoDS_Shape
from OCP.gp import gp_Dir, gp_Lin, gp_Pnt

# Angular samples around the bore for the material check.
MATERIAL_SAMPLE_COUNT = 4
# Fraction of samples that must find material for the hole to be accepted.
MATERIAL_PASS_FRACTION = 0.6
# A bore whose faces cover at least this much of the circle is already
# known to be fully enclosed -- nothing partial can sweep a whole circle --
# so the material check is skipped for it. That is the large majority of
# holes on a real part, and skipping them removes most of the ray casting
# in this stage. Partial sweeps (the ambiguous cases) are still checked.
VALIDATION_SKIP_SWEEP_DEG = 359.0
# How far outside the wall to place the probe point, as a fraction of the
# bore radius (clamped to an absolute floor so tiny holes still work).
PROBE_OFFSET_FRACTION = 0.04
PROBE_OFFSET_MIN_MM = 0.01

# Rays fired outward from each mouth: one centre ray plus this many on a
# ring at the mouth radius.
APPROACH_RING_COUNT = 4
APPROACH_RING_FRACTION = 0.85   # ring radius as a fraction of the mouth radius
# Ignore hits closer than this -- the mouth face itself sits at zero.
APPROACH_START_EPS_MM = 0.05

# How far to look for an obstruction above a mouth. This used to run the
# full 2.5x part diagonal, which was both the dominant cost in this stage
# and the wrong question: a wall 300 mm up the axis does not obstruct a
# drill, and every ray still had to traverse the entire spatial index to
# find it. Obstruction is only meaningful within tool-approach reach, so
# the rays are cut to a few diameters with an absolute floor of one inch.
# On a part with hundreds of holes this is the single biggest saving in
# the stage, and it makes the answer more correct rather than less.
APPROACH_CLEARANCE_DIAMETERS = 6.0
APPROACH_CLEARANCE_MIN_MM = 25.4

# Wall-clock budget for the whole stage. A part with thousands of holes
# can otherwise run for many minutes with the progress bar crawling, which
# is indistinguishable from a hang. On expiry the remaining holes are KEPT
# and marked unprobed rather than dropped: an unprobed hole is a hole we
# know nothing about, and silently deleting it would corrupt the hole
# count -- by far the worst outcome for a feature meant to predict
# drilling time. The row records how many were skipped so it can be
# filtered or re-run.
PROBE_TIME_BUDGET_S = 45.0
# Progress and cancellation are polled this often. The old code checked
# cancellation every 128 holes, so on a slow part Cancel appeared dead for
# a minute at a time.
PROBE_POLL_INTERVAL = 8

RAY_TOLERANCE = 1e-6


@dataclass
class ProbeStats:
    """How much of the probing actually happened -- provenance for the row."""
    holes_in: int = 0
    holes_probed: int = 0
    holes_unprobed: int = 0
    holes_rejected: int = 0
    timed_out: bool = False
    cancelled: bool = False
    elapsed_s: float = 0.0


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _perpendicular_frame(direction):
    d = _unit(direction)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(ref, d))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = ref - np.dot(ref, d) * d
    u /= np.linalg.norm(u)
    return u, np.cross(d, u)


def _count_hits(intersector, origin, direction, max_distance, start=RAY_TOLERANCE) -> int:
    line = gp_Lin(gp_Pnt(*[float(x) for x in origin]),
                  gp_Dir(*[float(x) for x in _unit(direction)]))
    intersector.Perform(line, start, max_distance)
    if not intersector.IsDone():
        return -1
    return int(intersector.NbPnt())


def _nearest_hit(intersector, origin, direction, max_distance, start) -> float | None:
    line = gp_Lin(gp_Pnt(*[float(x) for x in origin]),
                  gp_Dir(*[float(x) for x in _unit(direction)]))
    intersector.PerformNearest(line, start, max_distance)
    if intersector.IsDone() and intersector.NbPnt() > 0:
        return float(intersector.WParameter(1))
    return None


def _swept_degrees(hole) -> float:
    """Largest angular sweep among the hole's cylindrical steps. Assembled
    holes don't carry it directly, so fall back to assuming a full circle
    (which only means the cheap path is taken)."""
    sweep = getattr(hole, "angular_sweep_deg", None)
    if sweep is not None:
        return float(sweep)
    steps = getattr(hole, "steps", None) or []
    values = [getattr(s, "angular_sweep_deg", None) for s in steps]
    values = [v for v in values if v is not None]
    return float(max(values)) if values else 360.0


def probe_holes(shape: TopoDS_Shape, holes, diagonal_mm: float,
                 validate: bool = True, accessibility: bool = True,
                 should_cancel=None, progress=None,
                 time_budget_s: float = PROBE_TIME_BUDGET_S):
    """Fills in validation and accessibility fields on each hole.

    Returns `(kept_holes, ProbeStats)`. Holes that fail validation are
    dropped; holes left unprobed when the time budget expires are kept and
    flagged, because dropping them would understate the hole count.
    """
    stats = ProbeStats(holes_in=len(holes))
    if not holes:
        return [], stats

    started = time.monotonic()
    deadline = started + time_budget_s if time_budget_s and time_budget_s > 0 else None

    intersector = IntCurvesFace_ShapeIntersector()
    intersector.Load(shape, RAY_TOLERANCE)
    far = max(diagonal_mm * 2.5, 1.0)
    total = len(holes)

    kept = []
    stop_at = None          # index from which everything is left unprobed
    for index, h in enumerate(holes):
        if index % PROBE_POLL_INTERVAL == 0:
            if should_cancel is not None and should_cancel():
                stats.cancelled = True
                stop_at = index
                break
            if deadline is not None and time.monotonic() > deadline:
                stats.timed_out = True
                stop_at = index
                break
            if progress is not None:
                progress(index + 1, total)

        if h.axis_direction is None or h.axis_point is None:
            kept.append(h)
            continue

        axis_dir = _unit(h.axis_direction)
        u, v = _perpendicular_frame(axis_dir)
        radius = max(h.nominal_diameter_mm / 2.0, 1e-6)

        lo = min(s.axial_start for s in h.steps) if h.steps else 0.0
        hi = max(s.axial_end for s in h.steps) if h.steps else 0.0
        mid = 0.5 * (lo + hi)

        # Full-circle bores can't be open arcs, so skip their material check.
        sweep_deg = _swept_degrees(h)
        needs_validation = validate and sweep_deg < VALIDATION_SKIP_SWEEP_DEG
        if needs_validation:
            ok, thickness = _check_material_around(
                intersector, h.axis_point, axis_dir, u, v, radius, mid, far)
            h.validated = ok
            h.wall_thickness_mm = thickness
            if not ok:
                h.reject_reason = "not enclosed by material (open arc, groove or scallop)"
                stats.holes_rejected += 1
                continue

        if accessibility:
            _probe_mouths(intersector, h, axis_dir, u, v, lo, hi, far)

        h.probe_skipped = False
        stats.holes_probed += 1
        kept.append(h)

    # Everything past the stop point is retained unprobed. Its accessibility
    # fields stay at their defaults, which read as "not obstructed" -- the
    # neutral answer, not a claim.
    if stop_at is not None:
        for h in holes[stop_at:]:
            h.probe_skipped = True
            stats.holes_unprobed += 1
            kept.append(h)

    if progress is not None:
        progress(total, total)
    stats.elapsed_s = time.monotonic() - started
    return kept, stats


def _check_material_around(intersector, axis_point, axis_dir, u, v,
                            radius, mid_axial, far):
    """Majority vote over angular samples: is the bore wall backed by solid?
    Also returns the thinnest wall found, which is a useful signal in its
    own right (a hole close to breaking out)."""
    offset = max(radius * PROBE_OFFSET_FRACTION, PROBE_OFFSET_MIN_MM)
    centre = np.asarray(axis_point, dtype=float) + axis_dir * mid_axial

    inside = 0
    tested = 0
    thinnest = None
    for k in range(MATERIAL_SAMPLE_COUNT):
        angle = 2.0 * np.pi * k / MATERIAL_SAMPLE_COUNT
        radial = np.cos(angle) * u + np.sin(angle) * v
        probe = centre + radial * (radius + offset)

        hits = _count_hits(intersector, probe, radial, far)
        if hits < 0:
            continue
        tested += 1
        if hits % 2 == 1:          # odd crossings -> started inside material
            inside += 1
            distance = _nearest_hit(intersector, probe, radial, far, RAY_TOLERANCE)
            if distance is not None:
                thinnest = distance if thinnest is None else min(thinnest, distance)

    if tested == 0:
        return True, None          # couldn't tell; don't discard the hole
    return (inside / tested) >= MATERIAL_PASS_FRACTION, thinnest


def _probe_mouths(intersector, hole, axis_dir, u, v, lo, hi, far):
    """Fires the tool-approach rays outward from each open mouth."""
    # The entry mouth is the end the classifier identified as open; the
    # outward direction is away from the hole. Probing a blind hole's capped
    # end would report a guaranteed (and meaningless) obstruction.
    entry_axial = hi if hole.entry_at_high else lo
    entry_dir = axis_dir if hole.entry_at_high else -axis_dir
    mouths = [(entry_axial, entry_dir, "entry")]
    if hole.termination == "through":
        mouths.append((lo if hole.entry_at_high else hi,
                       -axis_dir if hole.entry_at_high else axis_dir,
                       "exit"))

    radius = max(hole.major_diameter_mm / 2.0, hole.nominal_diameter_mm / 2.0, 1e-6)
    ring_radius = radius * APPROACH_RING_FRACTION

    # Only look as far as a tool would need. Beyond this there is nothing
    # to obstruct: the drill is already clear of the part.
    reach = min(far, max(APPROACH_CLEARANCE_DIAMETERS * 2.0 * radius,
                         APPROACH_CLEARANCE_MIN_MM))

    for axial, direction, which in mouths:
        origin = np.asarray(hole.axis_point, dtype=float) + axis_dir * axial
        obstructed = False
        clearance = None

        rays = [origin]
        for k in range(APPROACH_RING_COUNT):
            angle = 2.0 * np.pi * k / APPROACH_RING_COUNT
            rays.append(origin + ring_radius * (np.cos(angle) * u + np.sin(angle) * v))

        for start_point in rays:
            distance = _nearest_hit(intersector, start_point, direction, reach,
                                     APPROACH_START_EPS_MM)
            if distance is not None:
                obstructed = True
                clearance = distance if clearance is None else min(clearance, distance)

        if which == "entry":
            hole.entry_obstructed = obstructed
            hole.entry_clearance_mm = clearance
        else:
            hole.exit_obstructed = obstructed
            hole.exit_clearance_mm = clearance
