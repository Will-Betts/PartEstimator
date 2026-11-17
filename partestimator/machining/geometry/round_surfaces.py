"""Concave cylindrical ("hole") face detection, grouped by radius, plus
concave-round ("internal corner"/fillet) radius scanning.

Two things look alike in a STEP B-rep and have to be told apart:

  1. A drilled/bored hole wall: a cylindrical face that curves toward its
     own axis (material surrounds a void) and sweeps close to the full
     360 degrees around that axis.
  2. An edge fillet / internal-corner round: geometrically also often a
     cylindrical face (a constant-radius blend along a straight edge is a
     cylinder whose axis runs along that edge) or a toroidal face (a blend
     around a curved/corner edge), but it only sweeps a partial arc
     (typically <= ~180 degrees, e.g. a 90-degree corner blend).

Concavity test: rather than computing a differential surface normal (which
depends on getting face-orientation flags and OCP's GeomLProp bindings
exactly right -- fragile, and the thing that broke once already), we use
BRepClass3d_SolidClassifier, OCCT's standard point-in-solid membership
test. For a sample point on the face, step a small distance *toward* the
cylinder's axis (or the torus's tube centerline) and ask whether that
probe point landed inside solid material or in a void:
  - Hole/bore or internal fillet (concave): the void is toward the axis/
    tube-center, so the probe lands OUTSIDE the solid.
  - Boss/shaft OD or external round-over (convex): material continues
    toward the axis/tube-center, so the probe lands INSIDE the solid.
This is purely topological (in three inside/outside checks per face) and
doesn't depend on differential geometry at all, so it's the more robust
choice here.

Critically, a single physical hole is very often NOT a single face in a
STEP file. Exporters routinely split a full 360-degree cylinder at a
parametric seam into two ~180-degree half-cylinder faces (sometimes more,
e.g. 4 x 90 degrees, or irregular splits where other features intersect
the bore). Classifying each face in isolation against a "must sweep
~360 degrees" rule therefore misfiles every seam-split hole as a fillet --
which is exactly the bug this module previously had.

So instead of judging faces individually, we first **merge co-axial
concave cylindrical faces into bores**: faces are grouped by shared
(axis line, radius), their angular sweeps are summed, and only then is
the group classified:
  - merged sweep >= ~300 degrees  -> a hole/bore (its area is the sum of
    all its constituent faces' areas)
  - merged sweep <  ~300 degrees  -> fillet(s)/partial round
Grouping by axis line (direction *and* position, direction-insensitive so
that opposite-pointing axis representations of the same line match) is
what makes two half-cylinders of one bore land in the same group while
keeping genuinely separate holes apart.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepGProp import BRepGProp
from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Torus
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_FACE, TopAbs_OUT
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Shape
from OCP.gp import gp_Pnt, gp_Vec

RADIUS_GROUP_TOLERANCE_MM = 0.01  # radii within this tolerance are treated as one nominal size
FULL_CIRCLE_RAD = 2 * np.pi
HOLE_MIN_ANGULAR_SWEEP_RAD = np.radians(300)  # merged sweep >= this -> "hole", else -> "fillet"
CLASSIFIER_TOLERANCE_MM = 1e-6

# Tolerances for deciding that two cylindrical faces lie on the SAME axis
# line (and are therefore parts of one split bore).
AXIS_DIRECTION_TOLERANCE = 1e-4   # max sin-angle between axis directions
AXIS_POSITION_TOLERANCE_MM = 1e-3  # max perpendicular distance between axis lines


@dataclass(eq=False)
class HoleFace:
    """One physical hole/bore, possibly assembled from several co-axial
    STEP faces (see module docstring on seam splitting).

    eq=False for the same reason as _ConcaveCylFace below: this now carries
    numpy array fields, and the auto-generated field-wise __eq__ would
    compare arrays and raise "truth value of an array is ambiguous" from
    anything that tests equality (list.remove, `in`, ==).
    """
    radius_mm: float
    diameter_mm: float
    depth_mm: float
    depth_to_diameter_ratio: float
    area_mm2: float
    angular_sweep_deg: float
    face_count: int = 1  # how many STEP faces were merged to form this hole
    # Axis geometry, retained so downstream feature extraction can cluster
    # drilling directions and spot co-axial counterbores/countersinks.
    axis_point: np.ndarray = None      # a point on the axis line
    axis_direction: np.ndarray = None  # unit vector
    axial_start: float = 0.0           # bore extent along the axis, measured
    axial_end: float = 0.0             # from axis_point (i.e. the v-range)


@dataclass
class FilletFace:
    radius_mm: float  # cylinder radius, or torus minor radius
    area_mm2: float
    surface_type: str  # "cylinder" or "torus"


def _face_area(face) -> float:
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    return props.Mass()


def _point_is_outside_solid(classifier: BRepClass3d_SolidClassifier, point: gp_Pnt) -> bool:
    classifier.Perform(point, CLASSIFIER_TOLERANCE_MM)
    return classifier.State() == TopAbs_OUT


def _is_concave_cylinder(adaptor: BRepAdaptor_Surface, classifier: BRepClass3d_SolidClassifier) -> bool:
    """True if stepping from the face toward its own axis lands in a void
    (bore/hole case) rather than staying inside material (boss/shaft OD)."""
    cyl = adaptor.Cylinder()
    radius = cyl.Radius()
    axis = cyl.Axis()
    u_mid = 0.5 * (adaptor.FirstUParameter() + adaptor.LastUParameter())
    v_mid = 0.5 * (adaptor.FirstVParameter() + adaptor.LastVParameter())
    point = adaptor.Value(u_mid, v_mid)

    loc = axis.Location()
    dirv = axis.Direction()
    w = gp_Vec(loc, point)
    t = w.Dot(gp_Vec(dirv.XYZ()))
    foot = gp_Pnt(loc.X() + dirv.X() * t, loc.Y() + dirv.Y() * t, loc.Z() + dirv.Z() * t)

    radial = np.array([point.X() - foot.X(), point.Y() - foot.Y(), point.Z() - foot.Z()])
    radial_len = np.linalg.norm(radial)
    if radial_len < 1e-9:
        return False
    radial_unit = radial / radial_len
    step = max(radius * 0.02, 1e-3)
    probe = gp_Pnt(
        point.X() - radial_unit[0] * step,
        point.Y() - radial_unit[1] * step,
        point.Z() - radial_unit[2] * step,
    )
    return _point_is_outside_solid(classifier, probe)


def _is_concave_torus(adaptor: BRepAdaptor_Surface, classifier: BRepClass3d_SolidClassifier) -> bool:
    """True if stepping from the face toward the torus's tube centerline
    lands in a void (internal fillet blend) rather than staying inside
    material (external round-over)."""
    torus = adaptor.Torus()
    minor_r = torus.MinorRadius()
    major_r = torus.MajorRadius()
    pos = torus.Position()

    u_mid = 0.5 * (adaptor.FirstUParameter() + adaptor.LastUParameter())
    v_mid = 0.5 * (adaptor.FirstVParameter() + adaptor.LastVParameter())
    point = adaptor.Value(u_mid, v_mid)

    loc = pos.Location()
    xdir = pos.XDirection()
    ydir = pos.YDirection()
    tube_center = gp_Pnt(
        loc.X() + major_r * (xdir.X() * np.cos(u_mid) + ydir.X() * np.sin(u_mid)),
        loc.Y() + major_r * (xdir.Y() * np.cos(u_mid) + ydir.Y() * np.sin(u_mid)),
        loc.Z() + major_r * (xdir.Z() * np.cos(u_mid) + ydir.Z() * np.sin(u_mid)),
    )
    toward = np.array([tube_center.X() - point.X(), tube_center.Y() - point.Y(), tube_center.Z() - point.Z()])
    toward_len = np.linalg.norm(toward)
    if toward_len < 1e-9:
        return False
    toward_unit = toward / toward_len
    step = max(minor_r * 0.2, 1e-3)
    probe = gp_Pnt(
        point.X() + toward_unit[0] * step,
        point.Y() + toward_unit[1] * step,
        point.Z() + toward_unit[2] * step,
    )
    return _point_is_outside_solid(classifier, probe)


@dataclass(eq=False)
class _ConcaveCylFace:
    """Intermediate record for one concave cylindrical face, before
    co-axial faces get merged into whole bores.

    eq=False is REQUIRED, not stylistic. The auto-generated __eq__ compares
    fields as a tuple, and two of these fields are numpy arrays -- comparing
    those yields an array, whose truth value raises
    "The truth value of an array with more than one element is ambiguous".
    Any operation that tests equality (list.remove, `in`, ==) would blow up
    on a part with two or more bores. Identity comparison is also the
    correct semantics here: these records represent specific faces, so two
    distinct faces that happen to share a radius are not interchangeable.
    """
    radius_mm: float
    sweep_rad: float
    depth_mm: float
    area_mm2: float
    axis_point: np.ndarray  # any point on the axis line
    axis_direction: np.ndarray  # unit vector
    axial_start: float = 0.0  # v-parameter range of this face along the axis
    axial_end: float = 0.0
    # Angular sector: where the face starts around the axis, and the frame
    # that angle is measured in. Needed to UNION sectors rather than sum
    # them; co-axial faces can carry different reference directions, so the
    # raw u-parameter is not comparable between faces without x_dir.
    sweep_start: float = 0.0
    x_dir: np.ndarray = None


def _same_axis_line(a: _ConcaveCylFace, b: _ConcaveCylFace) -> bool:
    """True if both faces' axes describe the same infinite line, ignoring
    direction sign and which point along the line was used to anchor it."""
    cross = np.cross(a.axis_direction, b.axis_direction)
    if np.linalg.norm(cross) > AXIS_DIRECTION_TOLERANCE:
        return False
    # Parallel: check they're not merely parallel but truly collinear, by
    # measuring the perpendicular offset between the two anchor points.
    delta = b.axis_point - a.axis_point
    perpendicular = delta - np.dot(delta, a.axis_direction) * a.axis_direction
    return float(np.linalg.norm(perpendicular)) <= AXIS_POSITION_TOLERANCE_MM


TWO_PI = 2.0 * np.pi
AXIAL_RUN_GAP_TOLERANCE_MM = 0.25


def _axial_span(face, ref) -> tuple[float, float]:
    """A face's axial range expressed in the reference face's frame.
    Co-axial faces can be anchored at different points and point in
    opposite directions, so raw v-ranges aren't directly comparable."""
    d_ref = ref.axis_direction / np.linalg.norm(ref.axis_direction)
    d_own = face.axis_direction / np.linalg.norm(face.axis_direction)
    sign = 1.0 if float(np.dot(d_ref, d_own)) >= 0 else -1.0
    base = float(np.dot(np.asarray(face.axis_point, dtype=float)
                        - np.asarray(ref.axis_point, dtype=float), d_ref))
    a = base + sign * face.axial_start
    b = base + sign * face.axial_end
    return (min(a, b), max(a, b))


def _split_axial_runs(group: list[_ConcaveCylFace]) -> list[list[_ConcaveCylFace]]:
    """Splits a co-axial, equal-radius group into axially connected runs.

    Without this, two holes of the same diameter drilled on one centreline
    through two flanges merge into a single bore spanning both -- hugely
    overstating depth. Faces that touch or overlap stay together (that's a
    seam-split bore); a real gap starts a new hole.
    """
    if len(group) <= 1:
        return [group]
    ref = group[0]
    spans = sorted(((*_axial_span(f, ref), f) for f in group), key=lambda s: s[0])

    runs = [[spans[0][2]]]
    current_end = spans[0][1]
    for lo, hi, face in spans[1:]:
        if lo <= current_end + AXIAL_RUN_GAP_TOLERANCE_MM:
            runs[-1].append(face)
            current_end = max(current_end, hi)
        else:
            runs.append([face])
            current_end = hi
    return runs


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _sector_in_ref_frame(face, ref) -> tuple[float, float]:
    """(start_angle, sweep) for `face`, measured in `ref`'s angular frame.

    Each cylindrical surface carries its own XDirection, so raw u-values
    from different faces are not comparable. This re-expresses the face's
    sector as a real 3D direction and re-measures it against the reference
    frame. If the two axes point opposite ways, increasing u rotates the
    other way round, so the interval runs backwards from its start.
    """
    d_ref = _unit(ref.axis_direction)
    x_ref = _unit(np.asarray(ref.x_dir, dtype=float) - np.dot(ref.x_dir, d_ref) * d_ref)
    y_ref = np.cross(d_ref, x_ref)

    d_own = _unit(face.axis_direction)
    x_own = _unit(np.asarray(face.x_dir, dtype=float) - np.dot(face.x_dir, d_own) * d_own)
    y_own = np.cross(d_own, x_own)

    u0 = float(face.sweep_start)
    start_vec = np.cos(u0) * x_own + np.sin(u0) * y_own
    angle = float(np.arctan2(np.dot(start_vec, y_ref), np.dot(start_vec, x_ref)))

    sweep = float(face.sweep_rad)
    if float(np.dot(d_ref, d_own)) < 0:
        angle -= sweep          # u advances the other way in the ref frame
    return angle % TWO_PI, sweep


def _angular_union(group: list[_ConcaveCylFace]) -> float:
    """Total angle actually covered around the axis, merging overlaps.

    Faces covering the same sector must not have their sweeps added -- see
    the note at the call site about stacked corner fillets.
    """
    ref = group[0]
    if ref.x_dir is None:
        # No frame recorded (older records): fall back to summing, capped
        # at a full turn so it can't exceed a circle.
        return min(sum(float(f.sweep_rad) for f in group), TWO_PI)

    segments: list[tuple[float, float]] = []
    for f in group:
        if f.x_dir is None:
            return min(sum(float(x.sweep_rad) for x in group), TWO_PI)
        if float(f.sweep_rad) >= TWO_PI - 1e-9:
            return TWO_PI
        if float(f.sweep_rad) <= 0:
            continue
        start, sweep = _sector_in_ref_frame(f, ref)
        end = start + sweep
        if end <= TWO_PI:
            segments.append((start, end))
        else:                      # wraps past 2pi -- split in two
            segments.append((start, TWO_PI))
            segments.append((0.0, end - TWO_PI))

    if not segments:
        return 0.0
    segments.sort()
    total = 0.0
    cur_start, cur_end = segments[0]
    for start, end in segments[1:]:
        if start <= cur_end + 1e-9:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    total += cur_end - cur_start
    return min(total, TWO_PI)


def _merge_coaxial_bores(cyl_faces: list[_ConcaveCylFace]) -> tuple[list[HoleFace], list[FilletFace]]:
    """Groups concave cylindrical faces sharing an axis line and radius,
    then classifies each merged group as a hole or as fillet(s).

    Grouping is done purely by INDEX with a `used` flag array. The obvious
    implementation -- keeping a list of pending faces and calling
    `unassigned.remove(candidate)` -- is a trap: list.remove() locates the
    item by `==`, which invokes the dataclass's field-wise equality, which
    compares numpy arrays and raises "truth value of an array ... is
    ambiguous". Working with indices sidesteps equality entirely, and also
    avoids remove()'s O(n) rescan per removal.
    """
    n = len(cyl_faces)
    used = [False] * n
    groups: list[list[_ConcaveCylFace]] = []

    for i in range(n):
        if used[i]:
            continue
        used[i] = True
        seed = cyl_faces[i]
        group = [seed]
        # Repeatedly sweep the remaining faces, since a bore split into
        # >2 pieces may only match transitively through an intermediate.
        changed = True
        while changed:
            changed = False
            for j in range(n):
                if used[j]:
                    continue
                candidate = cyl_faces[j]
                if abs(candidate.radius_mm - seed.radius_mm) > RADIUS_GROUP_TOLERANCE_MM:
                    continue
                if any(_same_axis_line(member, candidate) for member in group):
                    group.append(candidate)
                    used[j] = True
                    changed = True
        groups.append(group)

    holes: list[HoleFace] = []
    fillets: list[FilletFace] = []
    # Split each (axis, radius) group into axially connected runs, so two
    # separate holes of the SAME diameter on one centreline stay separate.
    runs: list[list[_ConcaveCylFace]] = []
    for group in groups:
        runs.extend(_split_axial_runs(group))

    for group in runs:
        # ANGULAR UNION, not sum. Summing is wrong for co-axial faces that
        # cover the same angular sector: a vertical corner filleted at
        # several heights yields e.g. four 90-degree faces on one axis at
        # one radius, and summing calls that a 360-degree hole. They all
        # occupy the SAME sector, so the union is still 90 degrees -- a
        # fillet. A genuinely seam-split bore has DISJOINT halves, so its
        # union is still the full circle and it is still a hole.
        total_sweep = _angular_union(group)
        total_area = sum(f.area_mm2 for f in group)
        mean_radius = float(np.mean([f.radius_mm for f in group]))
        # Depth is the axial extent, which each split face shares -- take
        # the max rather than summing (summing would double-count a bore
        # split lengthwise down its seam).
        depth = max(f.depth_mm for f in group)

        if total_sweep >= HOLE_MIN_ANGULAR_SWEEP_RAD:
            diameter = 2 * mean_radius
            # Axial extent spans every face in the group (a bore split
            # lengthwise contributes several partial ranges).
            axial_start = min(f.axial_start for f in group)
            axial_end = max(f.axial_end for f in group)
            holes.append(HoleFace(
                radius_mm=mean_radius,
                diameter_mm=diameter,
                depth_mm=depth,
                depth_to_diameter_ratio=(depth / diameter) if diameter > 0 else float("inf"),
                area_mm2=total_area,
                angular_sweep_deg=float(np.degrees(total_sweep)),
                face_count=len(group),
                axis_point=group[0].axis_point,
                axis_direction=group[0].axis_direction,
                axial_start=axial_start,
                axial_end=axial_end,
            ))
        else:
            for f in group:
                fillets.append(FilletFace(radius_mm=f.radius_mm, area_mm2=f.area_mm2, surface_type="cylinder"))
    return holes, fillets


def find_hole_and_fillet_faces(shape: TopoDS_Shape) -> tuple[list[HoleFace], list[FilletFace]]:
    """Returns (holes, fillets). Concave cylindrical faces are collected
    first, then merged by shared axis line + radius so that seam-split
    bores are recognized as single holes (see module docstring)."""
    cyl_faces: list[_ConcaveCylFace] = []
    fillets: list[FilletFace] = []
    classifier = BRepClass3d_SolidClassifier(shape)

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        adaptor = BRepAdaptor_Surface(face, True)
        surf_type = adaptor.GetType()

        if surf_type == GeomAbs_Cylinder:
            if _is_concave_cylinder(adaptor, classifier):
                cyl = adaptor.Cylinder()
                axis = cyl.Axis()
                loc = axis.Location()
                dirv = axis.Direction()
                direction = np.array([dirv.X(), dirv.Y(), dirv.Z()], dtype=float)
                direction /= np.linalg.norm(direction)
                cyl_faces.append(_ConcaveCylFace(
                    radius_mm=cyl.Radius(),
                    sweep_rad=adaptor.LastUParameter() - adaptor.FirstUParameter(),
                    depth_mm=abs(adaptor.LastVParameter() - adaptor.FirstVParameter()),
                    area_mm2=_face_area(face),
                    axis_point=np.array([loc.X(), loc.Y(), loc.Z()], dtype=float),
                    axis_direction=direction,
                ))
        elif surf_type == GeomAbs_Torus:
            if _is_concave_torus(adaptor, classifier):
                minor_r = adaptor.Torus().MinorRadius()
                area = _face_area(face)
                fillets.append(FilletFace(radius_mm=minor_r, area_mm2=area, surface_type="torus"))

        explorer.Next()

    holes, cyl_fillets = _merge_coaxial_bores(cyl_faces)
    fillets.extend(cyl_fillets)
    return holes, fillets


def group_hole_area_by_radius(holes: list[HoleFace], tolerance_mm: float = RADIUS_GROUP_TOLERANCE_MM) -> list[dict]:
    """Groups holes whose radii are within `tolerance_mm` of each other and
    sums their area. Returns a list of dicts sorted by nominal diameter,
    ascending. `hole_count` is the number of distinct physical holes in the
    group (already merged across seam-split faces upstream)."""
    ordered = sorted(holes, key=lambda h: h.radius_mm)
    groups: list[dict] = []
    for h in ordered:
        if groups and abs(h.radius_mm - groups[-1]["_radius_sum"] / groups[-1]["_count"]) <= tolerance_mm:
            g = groups[-1]
            g["_radius_sum"] += h.radius_mm
            g["_count"] += 1
            g["area_mm2"] += h.area_mm2
            g["hole_count"] += 1
            g["depths_mm"].append(h.depth_mm)
        else:
            groups.append({
                "_radius_sum": h.radius_mm,
                "_count": 1,
                "area_mm2": h.area_mm2,
                "hole_count": 1,
                "depths_mm": [h.depth_mm],
            })

    result = []
    for g in groups:
        mean_radius = g["_radius_sum"] / g["_count"]
        depths = g["depths_mm"]
        result.append({
            "nominal_radius_mm": mean_radius,
            "nominal_diameter_mm": 2 * mean_radius,
            "avg_depth_mm": float(sum(depths) / len(depths)) if depths else 0.0,
            "area_mm2": g["area_mm2"],
            "hole_count": g["hole_count"],
            "max_depth_mm": max(g["depths_mm"]),
        })
    return result


def smallest_internal_radius_mm(fillets: list[FilletFace]) -> float | None:
    if not fillets:
        return None
    return min(f.radius_mm for f in fillets)
