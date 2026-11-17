"""Hole recognition: assembles co-axial features into whole holes.

The earlier approach treated every concave cylindrical face as its own
hole. That is wrong for anything but the simplest drilled hole:

  * A counterbored hole is TWO co-axial cylinders of different diameter.
    Counting them separately reports two holes and two "distinct
    diameters", inflating both the hole count and the tool-change count.
  * A countersink is a cone, so it was invisible as a hole feature and its
    bore was reported as if it were plain.
  * A blind hole's drill point is a cone; without it, blind holes were
    indistinguishable from through holes.

So a hole is modelled here as a STACK of co-axial features along one axis
line -- cylindrical steps, plus any cones and any capping plane -- which
is then classified end to end:

    entry        plain | countersink | counterbore | counterbore+countersink
    termination  through | blind_flat | blind_tapered

Conventions
-----------
* The **nominal diameter** is the SMALLEST cylindrical step, because that
  is the functional hole -- a counterbored hole is quoted by its through
  diameter, with the counterbore called out separately.
* The **major diameter** is the largest step (the counterbore/countersink
  face diameter), which is what the tool has to cut at entry.
* **Depth** is the full axial extent of the stack.
* The **entry end** is the end opposite the termination. For a through
  hole with a feature at one end only, that featured end is taken as entry.

All of this is geometric inference, not manufacturing intent. In
particular a tapped hole looks exactly like a plain drilled hole -- STEP
almost never carries thread data -- so nothing here reports threads.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Two features share an axis line within these tolerances.
AXIS_ANGLE_TOLERANCE_RAD = np.radians(1.0)
AXIS_OFFSET_TOLERANCE_MM = 0.05
# Axial gap allowed between two steps that should be treated as adjacent.
AXIAL_GAP_TOLERANCE_MM = 0.25
# Diameters within this fraction count as "the same step".
DIAMETER_MATCH_FRACTION = 0.02
# A cone whose small end is below this fraction of the bore radius is a
# drill point rather than a countersink.
DRILL_POINT_RADIUS_FRACTION = 0.30
# A capping plane must sit within this multiple of the bore radius of the
# axis, and its area must be no more than this multiple of the bore disc.
CAP_CENTROID_TOLERANCE_FRACTION = 0.60
CAP_AREA_MAX_FRACTION = 2.5


@dataclass(eq=False)
class HoleStep:
    """One cylindrical section of a hole."""
    diameter_mm: float
    depth_mm: float
    axial_start: float
    axial_end: float
    area_mm2: float


@dataclass(eq=False)
class HoleFeature:
    axis_point: np.ndarray
    axis_direction: np.ndarray
    nominal_diameter_mm: float     # smallest step -- the functional hole
    major_diameter_mm: float       # largest step -- what's cut at entry
    total_depth_mm: float
    entry_type: str                # plain|countersink|counterbore|counterbore+countersink
    termination: str               # through|blind_flat|blind_tapered
    point_angle_deg: float | None
    area_mm2: float
    face_count: int
    steps: list = field(default_factory=list)
    internal_groove_count: int = 0  # buried wider steps: snap-ring grooves, not counterbores
    # Which end of the axial span the tool enters from. Needed by the
    # accessibility probe, which must fire outward from the OPEN mouth of a
    # blind hole -- firing from the capped end would report a false block.
    entry_at_high: bool = True
    axial_lo: float = 0.0
    axial_hi: float = 0.0
    # Largest angular coverage among the hole's bores. A full circle proves
    # the feature is enclosed, which lets the ray-cast validation skip it.
    angular_sweep_deg: float = 360.0

    # --- filled in by hole_probe.probe_holes (needs the shape) ---
    validated: bool = True          # False -> rejected as not a real enclosed bore
    reject_reason: str = ""
    wall_thickness_mm: float | None = None   # min material radially outside the bore
    entry_obstructed: bool = False
    entry_clearance_mm: float | None = None
    exit_obstructed: bool | None = None      # None when there is no second mouth
    exit_clearance_mm: float | None = None
    # True when the probe stage ran out of time before reaching this hole.
    # The hole is still counted -- we know it exists and how big it is --
    # but its accessibility fields are defaults, not measurements.
    probe_skipped: bool = False

    @property
    def is_captured(self) -> bool:
        """True when every open mouth is blocked along the hole axis, so a
        straight tool cannot reach it from either side."""
        if self.exit_obstructed is None:
            return self.entry_obstructed
        return self.entry_obstructed and self.exit_obstructed

    @property
    def is_obstructed(self) -> bool:
        """True when at least one mouth is blocked."""
        return bool(self.entry_obstructed or self.exit_obstructed)

    # Compatibility with the previous HoleFace shape, so existing grouping
    # and feature code keeps working.
    @property
    def radius_mm(self) -> float:
        return self.nominal_diameter_mm / 2.0

    @property
    def diameter_mm(self) -> float:
        return self.nominal_diameter_mm

    @property
    def depth_mm(self) -> float:
        return self.total_depth_mm

    @property
    def depth_to_diameter_ratio(self) -> float:
        return (self.total_depth_mm / self.nominal_diameter_mm
                if self.nominal_diameter_mm > 0 else float("inf"))

    @property
    def is_through(self) -> bool:
        return self.termination == "through"

    @property
    def is_counterbored(self) -> bool:
        return "counterbore" in self.entry_type

    @property
    def is_countersunk(self) -> bool:
        return "countersink" in self.entry_type


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _same_axis(p1, d1, p2, d2) -> bool:
    d1 = _unit(d1)
    d2 = _unit(d2)
    if float(np.linalg.norm(np.cross(d1, d2))) > np.sin(AXIS_ANGLE_TOLERANCE_RAD):
        return False
    delta = np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float)
    perp = delta - np.dot(delta, d1) * d1
    return float(np.linalg.norm(perp)) <= AXIS_OFFSET_TOLERANCE_MM


def _axial_of(point, axis_point, axis_direction) -> float:
    """Coordinate of `point` along the axis, measured from axis_point."""
    return float(np.dot(np.asarray(point, dtype=float) - np.asarray(axis_point, dtype=float),
                        _unit(axis_direction)))


def _project_axial_range(feature, ref_point, ref_dir):
    """Re-express a feature's own axial range in the stack's reference
    frame. Co-axial features can be anchored at different points along the
    same line and can point in opposite directions, so their raw v-ranges
    aren't directly comparable."""
    d_ref = _unit(ref_dir)
    d_own = _unit(feature.axis_direction)
    sign = 1.0 if float(np.dot(d_ref, d_own)) >= 0 else -1.0
    base = _axial_of(feature.axis_point, ref_point, d_ref)
    a = base + sign * feature.axial_start
    b = base + sign * feature.axial_end
    return (min(a, b), max(a, b))


def build_hole_features(bores, cones, planes) -> list[HoleFeature]:
    """Groups already-merged bores (one per axis+diameter) into holes.

    `bores` are round_surfaces.HoleFace records -- each is a single
    diameter on a single axis, already assembled across seam-split faces.

    Sharing an axis is NOT sufficient to be one hole. Two holes drilled
    through two separate flanges can be perfectly co-axial while having
    solid material (or open air) between them; merging those reports one
    absurdly deep hole and badly overstates drilling time on large parts.
    So each axis group is additionally split wherever consecutive features
    are separated by an axial GAP. Features that touch or overlap -- which
    is what a counterbore does to its own through bore -- stay together.
    """
    n = len(bores)
    used = [False] * n
    features: list[HoleFeature] = []

    for i in range(n):
        if used[i]:
            continue
        seed = bores[i]
        if seed.axis_direction is None:
            used[i] = True
            continue
        used[i] = True
        group = [seed]
        for j in range(i + 1, n):
            if used[j] or bores[j].axis_direction is None:
                continue
            if _same_axis(seed.axis_point, seed.axis_direction,
                          bores[j].axis_point, bores[j].axis_direction):
                group.append(bores[j])
                used[j] = True

        for run in _split_axial_runs(group, seed.axis_point, seed.axis_direction):
            features.append(_classify_stack(run, cones, planes))

    return features


def _split_axial_runs(group, ref_point, ref_dir):
    """Splits co-axial features into runs of axially connected features.

    Two features belong to the same hole only if their axial spans touch or
    overlap (within AXIAL_GAP_TOLERANCE_MM). A real gap means there is
    something -- material or free space -- between them, so they are
    separate holes that happen to share a centreline.
    """
    if len(group) <= 1:
        return [group]

    spans = []
    for b in group:
        lo, hi = _project_axial_range(b, ref_point, ref_dir)
        spans.append((lo, hi, b))
    spans.sort(key=lambda s: s[0])

    runs = [[spans[0]]]
    current_max_end = spans[0][1]
    for lo, hi, b in spans[1:]:
        if lo <= current_max_end + AXIAL_GAP_TOLERANCE_MM:
            runs[-1].append((lo, hi, b))
            current_max_end = max(current_max_end, hi)
        else:
            runs.append([(lo, hi, b)])
            current_max_end = hi

    return [[b for _lo, _hi, b in run] for run in runs]


def _classify_stack(group, cones, planes) -> HoleFeature:
    ref_point = group[0].axis_point
    ref_dir = _unit(group[0].axis_direction)

    steps: list[HoleStep] = []
    for b in group:
        lo, hi = _project_axial_range(b, ref_point, ref_dir)
        steps.append(HoleStep(
            diameter_mm=b.diameter_mm,
            depth_mm=b.depth_mm,
            axial_start=lo,
            axial_end=hi,
            area_mm2=b.area_mm2,
        ))
    steps.sort(key=lambda s: s.axial_start)

    stack_lo = min(s.axial_start for s in steps)
    stack_hi = max(s.axial_end for s in steps)
    nominal = min(s.diameter_mm for s in steps)
    nominal_radius = nominal / 2.0

    # A wider step that touches NEITHER mouth is buried inside the bore --
    # that's an internal groove (snap-ring, O-ring, relief), not a
    # counterbore. Counting it as one would invent a machining operation
    # and overstate the major diameter.
    tol = nominal * DIAMETER_MATCH_FRACTION
    groove_count = 0
    open_steps = []
    for s in steps:
        if s.diameter_mm <= nominal + tol:
            open_steps.append(s)
            continue
        if _touches(s.axial_start, s.axial_end, stack_lo) or \
           _touches(s.axial_start, s.axial_end, stack_hi):
            open_steps.append(s)
        else:
            groove_count += 1
    major = max((s.diameter_mm for s in open_steps), default=nominal)

    # --- co-axial cones ------------------------------------------------
    my_cones = []
    for c in cones:
        if _same_axis(ref_point, ref_dir, c.axis_point, c.axis_direction):
            lo, hi = _project_axial_range(c, ref_point, ref_dir)
            my_cones.append((lo, hi, c))

    # --- termination ---------------------------------------------------
    termination = "through"
    point_angle = None

    drill_point = None
    for lo, hi, c in my_cones:
        # A drill point tapers to (near) nothing and sits at one end.
        if c.radius_min_mm <= nominal_radius * DRILL_POINT_RADIUS_FRACTION:
            near_low = abs(hi - stack_lo) <= AXIAL_GAP_TOLERANCE_MM or lo <= stack_lo
            near_high = abs(lo - stack_hi) <= AXIAL_GAP_TOLERANCE_MM or hi >= stack_hi
            if near_low or near_high:
                drill_point = (lo, hi, c, "low" if near_low else "high")
                break

    cap_end = None
    if drill_point is not None:
        termination = "blind_tapered"
        point_angle = float(np.degrees(2.0 * drill_point[2].half_angle_rad))
        cap_end = drill_point[3]
    else:
        flat = _find_flat_bottom(planes, ref_point, ref_dir, stack_lo, stack_hi, nominal_radius)
        if flat is not None:
            termination = "blind_flat"
            cap_end = flat

    # --- entry ---------------------------------------------------------
    # Entry is the end opposite the termination; for a through hole, take
    # whichever end actually carries an enlarged feature.
    entry_at_high = (cap_end == "low")
    if termination == "through":
        entry_at_high = _entry_end_for_through(steps, my_cones, nominal, stack_lo, stack_hi)

    entry_type = _classify_entry(steps, my_cones, nominal, stack_lo, stack_hi,
                                  entry_at_high, nominal_radius)

    total_depth = stack_hi - stack_lo
    if drill_point is not None:
        lo, hi, _c, _end = drill_point
        total_depth = max(stack_hi, hi) - min(stack_lo, lo)

    area = sum(s.area_mm2 for s in steps) + sum(c.area_mm2 for _lo, _hi, c in my_cones)

    return HoleFeature(
        axis_point=np.asarray(ref_point, dtype=float),
        axis_direction=ref_dir,
        nominal_diameter_mm=nominal,
        major_diameter_mm=major,
        total_depth_mm=total_depth,
        entry_type=entry_type,
        termination=termination,
        point_angle_deg=point_angle,
        area_mm2=area,
        face_count=sum(getattr(b, "face_count", 1) for b in group),
        steps=steps,
        internal_groove_count=groove_count,
        entry_at_high=entry_at_high,
        axial_lo=stack_lo,
        axial_hi=stack_hi,
        angular_sweep_deg=float(max(
            (getattr(b, "angular_sweep_deg", 360.0) for b in group), default=360.0)),
    )


def _find_flat_bottom(planes, ref_point, ref_dir, stack_lo, stack_hi, nominal_radius):
    """A flat-bottomed blind hole is capped by a small disc perpendicular
    to the axis. Requires the plane's normal to be parallel to the axis,
    its centroid to sit on the axis, and its area to be no larger than a
    couple of bore discs -- otherwise any large face at the same height
    would qualify."""
    disc_area = np.pi * nominal_radius ** 2
    for p in planes:
        if p.centroid is None:
            continue
        if float(np.linalg.norm(np.cross(_unit(p.normal), ref_dir))) > np.sin(AXIS_ANGLE_TOLERANCE_RAD):
            continue
        offset = np.asarray(p.centroid, dtype=float) - np.asarray(ref_point, dtype=float)
        radial = offset - np.dot(offset, ref_dir) * ref_dir
        if float(np.linalg.norm(radial)) > nominal_radius * CAP_CENTROID_TOLERANCE_FRACTION:
            continue
        if disc_area > 0 and p.area_mm2 > disc_area * CAP_AREA_MAX_FRACTION:
            continue
        axial = float(np.dot(offset, ref_dir))
        if abs(axial - stack_lo) <= AXIAL_GAP_TOLERANCE_MM:
            return "low"
        if abs(axial - stack_hi) <= AXIAL_GAP_TOLERANCE_MM:
            return "high"
    return None


def _entry_end_for_through(steps, cones, nominal, stack_lo, stack_hi) -> bool:
    """For a through hole, entry is the end carrying an enlarged feature
    (counterbore or countersink). Returns True if that's the high end."""
    tol = nominal * DIAMETER_MATCH_FRACTION
    low_step = min(steps, key=lambda s: s.axial_start)
    high_step = max(steps, key=lambda s: s.axial_end)
    if high_step.diameter_mm > nominal + tol and low_step.diameter_mm <= nominal + tol:
        return True
    if low_step.diameter_mm > nominal + tol and high_step.diameter_mm <= nominal + tol:
        return False
    for lo, hi, _c in cones:
        if _touches(lo, hi, stack_hi):
            return True
        if _touches(lo, hi, stack_lo):
            return False
    return True


def _touches(lo: float, hi: float, target: float) -> bool:
    """Is either end of an axial span at `target`?

    Both ends have to be tested, because a countersink (or counterbore) can
    legitimately be authored either INSIDE the bore's parametric range --
    ending flush with the face -- or extending OUTSIDE it, starting at the
    face and running outward. Checking only the end that faces 'outward'
    catches one convention and silently misses the other.
    """
    return (abs(lo - target) <= AXIAL_GAP_TOLERANCE_MM
            or abs(hi - target) <= AXIAL_GAP_TOLERANCE_MM)


def _classify_entry(steps, cones, nominal, stack_lo, stack_hi,
                     entry_at_high: bool, nominal_radius: float) -> str:
    tol = nominal * DIAMETER_MATCH_FRACTION
    entry_axial = stack_hi if entry_at_high else stack_lo

    has_counterbore = False
    for s in steps:
        if s.diameter_mm <= nominal + tol:
            continue
        if _touches(s.axial_start, s.axial_end, entry_axial):
            has_counterbore = True

    has_countersink = False
    for lo, hi, c in cones:
        # A countersink flares outward from the bore rather than closing in.
        if c.radius_min_mm <= nominal_radius * DRILL_POINT_RADIUS_FRACTION:
            continue
        if c.radius_max_mm <= nominal_radius * 1.05:
            continue
        if _touches(lo, hi, entry_axial):
            has_countersink = True

    if has_counterbore and has_countersink:
        return "counterbore+countersink"
    if has_counterbore:
        return "counterbore"
    if has_countersink:
        return "countersink"
    return "plain"


def _dia_of(h) -> float:
    """Nominal diameter, accepting either an assembled HoleFeature or a raw
    round_surfaces.HoleFace bore (which the reference code path still
    produces)."""
    return float(getattr(h, "nominal_diameter_mm", None) or h.diameter_mm)


def _depth_of(h) -> float:
    return float(getattr(h, "total_depth_mm", None) or h.depth_mm)


def group_holes_by_diameter(holes, tolerance_mm: float = 0.01) -> list[dict]:
    """Groups holes by nominal diameter for the report table."""
    ordered = sorted(holes, key=_dia_of)
    groups: list[dict] = []
    for h in ordered:
        dia = _dia_of(h)
        entry = getattr(h, "entry_type", "plain")
        term = getattr(h, "termination", "through")
        if groups and abs(dia - groups[-1]["_dia_sum"] / groups[-1]["_count"]) <= 2 * tolerance_mm:
            g = groups[-1]
            g["_dia_sum"] += dia
            g["_count"] += 1
            g["hole_count"] += 1
            g["area_mm2"] += h.area_mm2
            g["depths_mm"].append(_depth_of(h))
            g["entry_types"].append(entry)
            g["terminations"].append(term)
        else:
            groups.append({
                "_dia_sum": dia,
                "_count": 1,
                "hole_count": 1,
                "area_mm2": h.area_mm2,
                "depths_mm": [_depth_of(h)],
                "entry_types": [entry],
                "terminations": [term],
            })

    result = []
    for g in groups:
        mean_dia = g["_dia_sum"] / g["_count"]
        depths = g["depths_mm"]
        result.append({
            "nominal_diameter_mm": mean_dia,
            "nominal_radius_mm": mean_dia / 2.0,
            "hole_count": g["hole_count"],
            "area_mm2": g["area_mm2"],
            "max_depth_mm": max(depths),
            "avg_depth_mm": float(sum(depths) / len(depths)),
            "entry_types": g["entry_types"],
            "terminations": g["terminations"],
        })
    return result
