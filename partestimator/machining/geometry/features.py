"""Additional machining-difficulty indicators pulled directly from STEP
B-rep topology/geometry -- no ML, just face/edge queries against
OpenCASCADE. Every function here is a *heuristic signal*, not a certified
manufacturability check; see the per-function docstrings for the specific
limitations of each one.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepGProp import BRepGProp
from OCP.GeomAbs import (
    GeomAbs_BezierSurface,
    GeomAbs_BSplineSurface,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_OffsetSurface,
    GeomAbs_OtherSurface,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_SurfaceOfExtrusion,
    GeomAbs_SurfaceOfRevolution,
    GeomAbs_Torus,
)
from OCP.GProp import GProp_GProps
from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Shape
from OCP.gp import gp_Dir, gp_Lin, gp_Pnt

from .normals import sampled_outward_normal
from .round_surfaces import FilletFace

# ---------------------------------------------------------------------------
# Freeform vs. analytic surface split
# ---------------------------------------------------------------------------

_TYPE_NAMES = {
    GeomAbs_Plane: "Plane",
    GeomAbs_Cylinder: "Cylinder",
    GeomAbs_Cone: "Cone",
    GeomAbs_Sphere: "Sphere",
    GeomAbs_Torus: "Torus",
    GeomAbs_BSplineSurface: "BSpline (freeform)",
    GeomAbs_BezierSurface: "Bezier (freeform)",
    GeomAbs_SurfaceOfRevolution: "Surface of Revolution",
    GeomAbs_SurfaceOfExtrusion: "Surface of Extrusion",
    GeomAbs_OffsetSurface: "Offset Surface",
    GeomAbs_OtherSurface: "Other",
}
_ANALYTIC_TYPES = {GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus}
_FREEFORM_TYPES = {GeomAbs_BSplineSurface, GeomAbs_BezierSurface}


def _face_area(face) -> float:
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    return props.Mass()


def surface_area_by_type(shape: TopoDS_Shape) -> dict:
    """Splits total surface area into analytic (planar/cylindrical/conical/
    spherical/toroidal -- fast to machine with standard tool moves) vs.
    freeform (BSpline/Bezier -- typically needs 3-5 axis surfacing passes)
    vs. other (surfaces of revolution/extrusion/offset -- geometrically
    simple but not "primitive"; bucketed separately rather than guessed
    into either category)."""
    totals: dict[int, float] = defaultdict(float)
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        adaptor = BRepAdaptor_Surface(face, True)
        totals[adaptor.GetType()] += _face_area(face)
        explorer.Next()

    analytic = sum(a for t, a in totals.items() if t in _ANALYTIC_TYPES)
    freeform = sum(a for t, a in totals.items() if t in _FREEFORM_TYPES)
    other = sum(a for t, a in totals.items() if t not in _ANALYTIC_TYPES and t not in _FREEFORM_TYPES)

    return {
        "analytic_area_mm2": analytic,
        "freeform_area_mm2": freeform,
        "other_area_mm2": other,
        "by_type_mm2": {_TYPE_NAMES.get(t, f"Surface type {t}"): a for t, a in totals.items()},
    }


# ---------------------------------------------------------------------------
# Small internal corner radii (reported from round_surfaces.FilletFace list)
# ---------------------------------------------------------------------------

def smallest_fillet_radii_mm(fillets: list[FilletFace], top_n: int = 5) -> list[float]:
    """The `top_n` smallest internal-fillet/corner radii found -- a lower
    bound on required end-mill diameter for those internal corners."""
    return sorted(f.radius_mm for f in fillets)[:top_n]


# ---------------------------------------------------------------------------
# Deep-hole flagging (reuses HoleFace.depth_to_diameter_ratio)
# ---------------------------------------------------------------------------

DEEP_HOLE_RATIO_THRESHOLD = 4.0  # rule-of-thumb: beyond this, peck drilling / reduced feed is typical


def flag_deep_holes(holes, ratio_threshold: float = DEEP_HOLE_RATIO_THRESHOLD):
    return [h for h in holes if h.depth_to_diameter_ratio >= ratio_threshold]


# ---------------------------------------------------------------------------
# Undercut / non-3-axis-accessible face heuristic
# ---------------------------------------------------------------------------

PRINCIPAL_DIRECTIONS = np.array([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1],
], dtype=float)

UNDERCUT_GRID = 2  # NxN interior parametric samples per face (4 samples; was 3 -> 9)
UNDERCUT_NORMAL_EPSILON = 1e-6


@dataclass
class UndercutResult:
    flagged_face_count: int = 0
    flagged_area_mm2: float = 0.0
    total_face_count: int = 0
    directions_tested: int = 6


def detect_undercut_faces(shape: TopoDS_Shape, directions: np.ndarray = PRINCIPAL_DIRECTIONS,
                           grid: int = UNDERCUT_GRID) -> UndercutResult:
    """Flags faces that are not fully "front-facing" (normal pointing
    outward with positive component) from ANY of the candidate directions.

    Limitation: this only checks each face's own normal coverage. It does
    NOT perform a true visibility/collision check against the rest of the
    part -- a face can pass this test (its normal faces the tool) yet still
    be physically blocked by other features from that direction, and a
    face can fail this test due to local curvature even though the part
    could tilt slightly to reach it on a real 5-axis or well-fixtured
    3-axis setup. Treat this as a coarse first-pass complexity flag, not a
    manufacturability verdict.

    Performance: planar faces (by far the most common face type on machined
    parts) have a single constant normal, so they're short-circuited with
    one sample instead of grid^2. Remaining faces stop sampling as soon as
    no candidate direction can still succeed, which prunes most of the
    differential-geometry evaluations -- these dominated the cost of this
    pass on large parts.
    """
    result = UndercutResult(directions_tested=len(directions))
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        result.total_face_count += 1
        adaptor = BRepAdaptor_Surface(face, True)
        u0, u1 = adaptor.FirstUParameter(), adaptor.LastUParameter()
        v0, v1 = adaptor.FirstVParameter(), adaptor.LastVParameter()

        # A plane's normal doesn't vary over the face -- one sample decides it.
        effective_grid = 1 if adaptor.GetType() == GeomAbs_Plane else grid
        us = np.linspace(u0, u1, effective_grid + 2)[1:-1]
        vs = np.linspace(v0, v1, effective_grid + 2)[1:-1]

        # `alive` tracks which tool directions are still viable for this
        # face; once all are eliminated we can stop sampling entirely.
        alive = np.ones(len(directions), dtype=bool)
        any_normal = False
        for u in us:
            for v in vs:
                n = sampled_outward_normal(face, adaptor, u, v)
                if n is None:
                    continue
                any_normal = True
                alive &= (directions @ n) > UNDERCUT_NORMAL_EPSILON
                if not alive.any():
                    break
            if not alive.any():
                break

        if any_normal and not alive.any():
            result.flagged_face_count += 1
            result.flagged_area_mm2 += _face_area(face)

        explorer.Next()
    return result


# ---------------------------------------------------------------------------
# Thin-wall / thin-feature heuristic (ray casting via IntCurvesFace)
# ---------------------------------------------------------------------------

# 1/16" (0.0625" = 1.5875 mm). Below roughly this, a wall starts to deflect
# and chatter under normal milling loads, so it needs reduced stepover,
# lighter depth of cut and often a spring pass -- i.e. it machines slower.
# It's a recognisable stock fraction rather than an arbitrary metric value,
# and it sits close to the usual "minimum practical wall" rules of thumb
# for aluminium and mild steel. Thin-walled work in practice depends as
# much on the wall's height-to-thickness ratio as on thickness alone, so
# treat this as a flag for "look here", not a machinability verdict.
THIN_WALL_DEFAULT_THRESHOLD_MM = 1.5875
THIN_WALL_GRID = 3  # NxN interior parametric samples per face (thorough mode)
THIN_WALL_QUICK_GRID = 1  # one sample per face -- the default, cheap pass
# Faces below this fraction of the mean face area are skipped in quick mode;
# slivers contribute little and cost the same as a big face to probe.
THIN_WALL_QUICK_MIN_AREA_FRACTION = 0.05
# Hard ceiling on rays so a pathological part can't stall the load.
THIN_WALL_QUICK_MAX_SAMPLES = 20000
THIN_WALL_RAY_START_EPS_MM = 1e-3  # skip this much before testing for a hit, to avoid re-hitting the origin face

# --------------------------------------------------------------------
# Sample rejection. The raw minimum of this scan is not trustworthy, and
# the reason is geometric rather than numerical: a ray fired inward from a
# point near a fillet, chamfer or concave corner hits the ADJACENT face of
# the same wall almost immediately. That distance is real, but it is not a
# wall thickness -- there is no thin wall there at all. One such sample is
# enough to drag the reported minimum to a value that does not exist
# anywhere on the part.
#
# The discriminator is the angle at which the ray lands. A genuine
# opposing wall faces back along the ray, so its outward normal is close
# to antiparallel to the ray direction. A neighbouring face at a corner is
# hit obliquely. Requiring the hit face to be within 60 degrees of
# head-on removes the corner artifacts while keeping every real
# through-wall measurement, including tapered and lightly curved walls.
OPPOSING_FACE_MIN_COS = 0.5          # cos(60 deg)
# Hits closer than this are the origin face found again through numerical
# noise, not a wall.
THIN_WALL_MIN_VALID_MM = 5e-3
# How much of the measured area must be thin before the part is called
# thin-walled. This exists because the percentiles alone are the wrong
# detector: a genuine thin web covering 6% of a part's surface leaves p10
# untouched (by definition -- only 6% of the area is below it), so a flag
# driven off p10 would miss it. Area fraction is the sensitive measure and
# the percentiles are the magnitude. One percent is low enough to catch a
# real web and high enough that a single stray sample cannot reach it.
MIN_THIN_AREA_FRACTION = 0.01


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Area-weighted percentile (q in 0..1).

    Weighting matters here. In quick mode the scan takes one sample per
    face, so an unweighted percentile treats a 4 mm chamfer face and a
    200 mm wall as equally important -- and a part carrying hundreds of
    small blend faces would report its blends rather than its walls.
    Weighting each sample by its face area asks the question that actually
    matters: how thin is the material over a meaningful AREA of the part.
    """
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    total = float(w.sum())
    if total <= 0:
        return float(np.percentile(v, q * 100.0))
    cumulative = np.cumsum(w)
    idx = int(np.searchsorted(cumulative, q * total))
    return float(v[min(idx, len(v) - 1)])


@dataclass
class ThinWallResult:
    samples_checked: int = 0          # samples that produced a usable measurement
    samples_rejected: int = 0         # hits discarded as corner/grazing artifacts
    thin_sample_count: int = 0
    thin_area_mm2: float = 0.0        # face area represented by thin samples
    measured_area_mm2: float = 0.0    # face area represented by all usable samples
    # Distribution, not a single number. p10 is the headline figure: it is
    # the thickness that 10% of the measured AREA falls below, so it
    # describes a region rather than a lucky ray.
    min_thickness_mm: float | None = None
    p10_thickness_mm: float | None = None
    p25_thickness_mm: float | None = None
    median_thickness_mm: float | None = None
    threshold_mm: float = THIN_WALL_DEFAULT_THRESHOLD_MM
    worst_points_mm: list = field(default_factory=list)  # (point, thickness) pairs, sorted thinnest-first,
    # for both the text summary and highlighting thin regions in the 3D viewer

    @property
    def thin_area_fraction(self) -> float:
        if self.measured_area_mm2 <= 0:
            return 0.0
        return float(self.thin_area_mm2 / self.measured_area_mm2)


def detect_thin_walls(shape: TopoDS_Shape, max_search_distance_mm: float,
                       threshold_mm: float = THIN_WALL_DEFAULT_THRESHOLD_MM,
                       grid: int = THIN_WALL_GRID, max_reported: int = 2000,
                       quick: bool = False, face_areas=None,
                       should_cancel=None) -> ThinWallResult:
    """Approximate local wall-thickness scan: from a sample grid of points on
    every face, casts a ray inward (along the negative outward normal) and
    measures the distance to the nearest opposite surface using
    IntCurvesFace_ShapeIntersector. Flags samples below `threshold_mm`.

    `quick=True` is the default pass run on every load: one sample per face
    instead of grid^2, sliver faces skipped, and a hard ceiling on total
    rays. That's roughly one ray per face -- comparable to the hole probe --
    and enough to answer "does this part have thin walls at all". The
    thorough grid remains available from the button for a closer look.

    Limitation: this is a sparse-sampling heuristic, not an exhaustive
    thickness field. It can miss very localized thin spots between grid
    samples and can be noisy very close to edges/fillets. Use it to triage
    suspect regions, not as a certified thickness report.
    """
    intersector = IntCurvesFace_ShapeIntersector()
    intersector.Load(shape, 1e-6)

    result = ThinWallResult(threshold_mm=threshold_mm)
    thicknesses: list[float] = []
    sample_weights: list[float] = []
    candidates_for_report = []

    if quick:
        grid = THIN_WALL_QUICK_GRID
    min_area = 0.0
    if quick and face_areas:
        valid = [a for a in face_areas if a]
        if valid:
            min_area = (sum(valid) / len(valid)) * THIN_WALL_QUICK_MIN_AREA_FRACTION

    face_index = 0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        if quick and result.samples_checked >= THIN_WALL_QUICK_MAX_SAMPLES:
            break
        if should_cancel is not None and (face_index % 256 == 0) and should_cancel():
            break
        face = TopoDS.Face_s(explorer.Current())
        this_area = None
        if face_areas is not None and face_index < len(face_areas):
            this_area = face_areas[face_index]
        if min_area > 0 and this_area is not None and this_area < min_area:
            face_index += 1
            explorer.Next()
            continue
        face_index += 1
        # Each sample stands in for its share of the face it was taken on,
        # which is what makes the percentiles area-weighted.
        sample_weight = float(this_area) / max(grid * grid, 1) if this_area else 1.0
        adaptor = BRepAdaptor_Surface(face, True)
        u0, u1 = adaptor.FirstUParameter(), adaptor.LastUParameter()
        v0, v1 = adaptor.FirstVParameter(), adaptor.LastVParameter()
        us = np.linspace(u0, u1, grid + 2)[1:-1]
        vs = np.linspace(v0, v1, grid + 2)[1:-1]

        for u in us:
            for v in vs:
                normal = sampled_outward_normal(face, adaptor, u, v)
                if normal is None:
                    continue
                p = adaptor.Value(u, v)
                inward = -normal
                norm_len = np.linalg.norm(inward)
                if norm_len < 1e-9:
                    continue
                inward = inward / norm_len
                line = gp_Lin(gp_Pnt(p.X(), p.Y(), p.Z()), gp_Dir(inward[0], inward[1], inward[2]))
                intersector.PerformNearest(line, THIN_WALL_RAY_START_EPS_MM, max_search_distance_mm)
                if not (intersector.IsDone() and intersector.NbPnt() > 0):
                    continue
                dist = float(intersector.WParameter(1))
                if dist < THIN_WALL_MIN_VALID_MM:
                    # The origin face found again through numerical noise.
                    result.samples_rejected += 1
                    continue
                if not _is_opposing_hit(intersector, inward):
                    # Hit a neighbouring face at a corner or fillet, not a
                    # wall. Counting this would invent a thin wall that
                    # does not exist on the part.
                    result.samples_rejected += 1
                    continue

                thicknesses.append(dist)
                sample_weights.append(sample_weight)
                result.samples_checked += 1
                result.measured_area_mm2 += sample_weight
                if dist < threshold_mm:
                    result.thin_sample_count += 1
                    result.thin_area_mm2 += sample_weight
                    candidates_for_report.append((np.array([p.X(), p.Y(), p.Z()]), dist))

        explorer.Next()

    if thicknesses:
        values = np.asarray(thicknesses, dtype=float)
        weights = np.asarray(sample_weights, dtype=float)
        result.min_thickness_mm = float(values.min())
        result.p10_thickness_mm = _weighted_percentile(values, weights, 0.10)
        result.p25_thickness_mm = _weighted_percentile(values, weights, 0.25)
        result.median_thickness_mm = _weighted_percentile(values, weights, 0.50)

    candidates_for_report.sort(key=lambda pair: pair[1])
    result.worst_points_mm = candidates_for_report[:max_reported]
    return result


def _is_opposing_hit(intersector, ray_direction) -> bool:
    """Did the ray land on a face that genuinely faces back along it?

    A true opposing wall has an outward normal close to PARALLEL to the
    (inward-travelling) ray. A face hit obliquely -- the far side of a fillet, the flank of a
    chamfer, the neighbouring wall of a concave corner -- does not, and the
    distance to it is not a wall thickness.
    """
    try:
        hit_face = intersector.Face(1)
        hit_u = intersector.UParameter(1)
        hit_v = intersector.VParameter(1)
        hit_adaptor = BRepAdaptor_Surface(hit_face, True)
        hit_normal = sampled_outward_normal(hit_face, hit_adaptor, hit_u, hit_v)
    except Exception:  # noqa: BLE001 - a face we cannot evaluate is not evidence
        return True
    if hit_normal is None:
        return True
    # The ray travels INSIDE the material and leaves through the far wall, so
    # that wall's OUTWARD normal points the same way as the ray (top face +z,
    # ray -z, bottom face outward normal -z). v2.5 tested for antiparallel,
    # which rejected every genuine wall and left the scan with no samples.
    return float(np.dot(hit_normal, ray_direction)) >= OPPOSING_FACE_MIN_COS
