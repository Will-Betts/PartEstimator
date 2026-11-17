"""Face interrogation, classification, and coaxial cylinder merging."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import geom
from ..config import Tolerance
from ..diag import LOGGER

# --- OCP imports ----------------------------------------------------------
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepGProp import BRepGProp
from OCP.BRepTools import BRepTools, BRepTools_WireExplorer
from OCP.GCPnts import GCPnts_QuasiUniformDeflection
from OCP.GeomAbs import (
    GeomAbs_BSplineSurface,
    GeomAbs_BezierSurface,
    GeomAbs_Circle,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_Line,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_Torus,
)
from OCP.GProp import GProp_GProps
from OCP.TopAbs import (
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_REVERSED,
    TopAbs_VERTEX,
    TopAbs_WIRE,
)
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Face, TopoDS_Shape
from OCP.TopTools import (
    TopTools_IndexedDataMapOfShapeListOfShape,
    TopTools_IndexedMapOfShape,
)


def _v(p) -> np.ndarray:
    """gp_Pnt / gp_Dir / gp_Vec -> numpy."""
    return np.array([p.X(), p.Y(), p.Z()], dtype=float)


SURFACE_NAMES = {
    GeomAbs_Plane: "plane",
    GeomAbs_Cylinder: "cylinder",
    GeomAbs_Cone: "cone",
    GeomAbs_Sphere: "sphere",
    GeomAbs_Torus: "torus",
    GeomAbs_BSplineSurface: "bspline",
    GeomAbs_BezierSurface: "bezier",
}


@dataclass
class FaceInfo:
    """Everything the detector needs about one face, extracted once."""

    index: int
    face: TopoDS_Face
    kind: str                       # plane | cylinder | cone | ...
    area: float
    centroid: np.ndarray

    # planes
    normal: np.ndarray | None = None      # outward, orientation-corrected
    point: np.ndarray | None = None

    # cylinders / cones
    axis_point: np.ndarray | None = None
    axis_dir: np.ndarray | None = None
    radius: float = 0.0
    x_dir: np.ndarray | None = None       # the surface's U parameter origin
    half_angle: float = 0.0               # cone only
    u_min: float = 0.0
    u_max: float = 0.0
    v_min: float = 0.0
    v_max: float = 0.0

    #: Which side of a curved surface the material is on. True when the solid
    #: lies INSIDE the surface (a boss, the outer face of a bend); False when it
    #: lies outside (a hole, the inner face of a bend). None for planes, which
    #: have no inside.
    #:
    #: This costs one orientation flag to capture and separates a slot end from
    #: a corner radius, and a countersink from a conical chamfer, which nothing
    #: else does cheaply. See `smt.features`.
    convex: bool | None = None

    def to_desc(self):
        """The kernel-free view of this face, for `smt.features`.

        The classifier runs on plain numpy so its rules can be tested without a
        CAD kernel installed. Given how much of this tool's bug history is
        things that only appeared once OCP was present, that boundary is worth
        the small cost of a conversion.
        """
        from ..features import FaceDesc

        return FaceDesc(
            index=self.index,
            kind=self.kind,
            area=self.area,
            centroid=self.centroid,
            normal=self.normal,
            point=self.point,
            axis_point=self.axis_point,
            axis_dir=self.axis_dir,
            radius=self.radius,
            half_angle=self.half_angle,
            sweep=self.sweep,
            axial_length=self.axial_length,
            convex=self.convex,
        )

    @property
    def sweep(self) -> float:
        return self.u_max - self.u_min

    @property
    def axial_length(self) -> float:
        return self.v_max - self.v_min

    @property
    def material_length(self) -> float:
        """True material along the bend line, net of cutouts.

        area = radius * sweep * material_length for a cylindrical band, so this
        inverts cleanly and accounts exactly for holes crossing the bend. It is
        the right length for a TONNAGE calculation, where the envelope length
        would overstate the force required.
        """
        denom = self.radius * self.sweep
        return self.area / denom if denom > 1e-12 else 0.0


# ---------------------------------------------------------------------------


def face_index_map(shape: TopoDS_Shape) -> TopTools_IndexedMapOfShape:
    """Indexed map of every unique face, 1-based.

    Used instead of hashing shapes ourselves. `TopoDS_Shape.HashCode()` was
    REMOVED in OCCT 7.8, which current build123d ships -- calling it raises
    AttributeError on a modern install. The indexed map does the same job using
    the kernel's own shape hashing and has a stable API across versions.
    """
    m = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, m)
    return m


def unique_faces(shape: TopoDS_Shape) -> list[TopoDS_Face]:
    m = face_index_map(shape)
    return [TopoDS.Face_s(m.FindKey(i)) for i in range(1, m.Extent() + 1)]


def uv_bounds(face: TopoDS_Face, adaptor: BRepAdaptor_Surface):
    """(umin, umax, vmin, vmax), tolerating OCP signature differences.

    Some OCP builds return the tuple; others expect output parameters. The
    adaptor's parameter range is an equivalent fallback for a trimmed face.
    """
    try:
        res = BRepTools.UVBounds_s(face)
        if res is not None and len(res) == 4:
            return tuple(float(x) for x in res)
    except (TypeError, ValueError, RuntimeError):
        pass
    return (
        float(adaptor.FirstUParameter()), float(adaptor.LastUParameter()),
        float(adaptor.FirstVParameter()), float(adaptor.LastVParameter()),
    )


def edge_face_map(shape: TopoDS_Shape) -> TopTools_IndexedDataMapOfShapeListOfShape:
    m = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, m)
    return m


def surface_area(face: TopoDS_Face) -> tuple[float, np.ndarray]:
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    return props.Mass(), _v(props.CentreOfMass())


def solid_volume(shape: TopoDS_Shape) -> float:
    """Volume, or 0.0 if the kernel will not give one.

    Only used for cross-checks and reporting, so a shape whose volume cannot be
    computed -- an open shell, a self-intersecting translation -- should cost
    the cross-check, not the whole run.
    """
    try:
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape, props)
        return float(props.Mass())
    except Exception:  # noqa: BLE001
        LOGGER.warning("volume could not be computed for this shape")
        return 0.0


def edge_length(edge) -> float:
    props = GProp_GProps()
    BRepGProp.LinearProperties_s(edge, props)
    return props.Mass()


def describe_face(face: TopoDS_Face, index: int) -> FaceInfo:
    """Pull every parameter we might need off a face in one pass."""
    ad = BRepAdaptor_Surface(face)
    stype = ad.GetType()
    kind = SURFACE_NAMES.get(stype, "other")
    area, centroid = surface_area(face)

    info = FaceInfo(index=index, face=face, kind=kind, area=area, centroid=centroid)

    umin, umax, vmin, vmax = uv_bounds(face, ad)
    info.u_min, info.u_max, info.v_min, info.v_max = umin, umax, vmin, vmax

    if stype == GeomAbs_Plane:
        pln = ad.Plane()
        n = _v(pln.Axis().Direction())
        # BRepAdaptor reports the underlying surface, ignoring how the face uses
        # it. A REVERSED face has its outward normal flipped.
        if face.Orientation() == TopAbs_REVERSED:
            n = -n
        info.normal = geom.unit(n)
        info.point = _v(pln.Location())

    elif stype == GeomAbs_Cylinder:
        cyl = ad.Cylinder()
        ax = cyl.Axis()
        info.axis_point = _v(ax.Location())
        info.axis_dir = geom.unit(_v(ax.Direction()))
        info.radius = cyl.Radius()
        info.x_dir = geom.unit(_v(cyl.Position().XDirection()))
        # A cylinder's own surface normal points radially OUTWARD. A REVERSED
        # face flips the solid's outward normal to point inward, which means the
        # material is outside the surface -- a hole rather than a boss.
        info.convex = face.Orientation() != TopAbs_REVERSED

    elif stype == GeomAbs_Cone:
        con = ad.Cone()
        ax = con.Axis()
        info.axis_point = _v(ax.Location())
        info.axis_dir = geom.unit(_v(ax.Direction()))
        info.radius = con.RefRadius()
        info.x_dir = geom.unit(_v(con.Position().XDirection()))
        info.half_angle = float(con.SemiAngle())
        info.convex = face.Orientation() != TopAbs_REVERSED

    return info


def describe_all(shape: TopoDS_Shape) -> list[FaceInfo]:
    """Read every face, leaving a breadcrumb before each one.

    The breadcrumb is the whole point. Reading a face means asking the kernel
    about its surface, and a malformed face can take the process down without
    raising anything Python can see. Logging the index first -- at DEBUG, so it
    goes to the file -- means the last line of `smt.log` names the face that
    did it, instead of the log simply stopping.

    A face that raises is skipped with a warning rather than aborting the part.
    A part missing one sliver still gives a usable answer; a part that refuses
    to open gives none.
    """
    out: list[FaceInfo] = []
    faces = unique_faces(shape)
    LOGGER.debug("reading %d faces", len(faces))
    for i, f in enumerate(faces):
        LOGGER.debug("reading face %d/%d", i, len(faces))
        try:
            out.append(describe_face(f, i))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("face %d could not be read (%s); skipped", i, exc)
    return out


# ---------------------------------------------------------------------------
# coaxial cylinder merging
# ---------------------------------------------------------------------------


@dataclass
class MergedCylinder:
    """One logical cylindrical band, possibly assembled from several faces.

    THIS IS THE ONE THAT BITES. A hole or cutout crossing a bend splits the
    cylindrical face into several topologically separate faces that share an
    axis and a radius. Without merging, a single bend reads as three bends and
    every downstream number is wrong.
    """

    faces: list[FaceInfo]
    axis_point: np.ndarray
    axis_dir: np.ndarray
    radius: float
    u_min: float
    u_max: float
    v_min: float
    v_max: float

    @property
    def sweep(self) -> float:
        return self.u_max - self.u_min

    @property
    def area(self) -> float:
        return sum(f.area for f in self.faces)

    @property
    def length_envelope(self) -> float:
        return self.v_max - self.v_min

    @property
    def length_material(self) -> float:
        denom = self.radius * self.sweep
        return self.area / denom if denom > 1e-12 else 0.0

    @property
    def centroid(self) -> np.ndarray:
        tot = self.area
        if tot <= 0:
            return self.faces[0].centroid
        return sum(f.centroid * f.area for f in self.faces) / tot


def _angle_offset(ref_x: np.ndarray, other_x: np.ndarray, axis: np.ndarray) -> float:
    """Signed angle about `axis` from ref_x to other_x.

    Two faces cut from the same cylinder normally share a parametrisation, but
    faces that came from different surfaces (or survived a translation round
    trip) may not. Normalising U by this offset before merging keeps the swept
    angle honest instead of silently mixing two parameter origins.
    """
    c = float(np.dot(ref_x, other_x))
    s = float(np.dot(np.cross(ref_x, other_x), axis))
    return math.atan2(s, c)


def merge_coaxial(
    cylinders: list[FaceInfo], tol: Tolerance
) -> list[MergedCylinder]:
    """Group cylindrical faces that are the same logical band."""
    groups: list[list[FaceInfo]] = []

    # Group by axis line first (exact tests, no rounding -- see geom.group_by_line),
    # then split each line's cylinders by radius. Two faces on different axes can
    # never be the same band, so this avoids the pointless comparisons without
    # weakening any test.
    for line in geom.group_by_line(
        cylinders, lambda f: (f.axis_point, f.axis_dir),
        ang_tol=tol.angular, lin_tol=tol.coaxial,
    ):
        by_radius: list[list[FaceInfo]] = []
        for f in line:
            for g in by_radius:
                if tol.radius_close(f.radius, g[0].radius):
                    g.append(f)
                    break
            else:
                by_radius.append([f])
        groups.extend(by_radius)

    merged: list[MergedCylinder] = []
    for g in groups:
        ref = g[0]
        axis = ref.axis_dir
        u_lo, u_hi = math.inf, -math.inf
        v_lo, v_hi = math.inf, -math.inf

        for f in g:
            flip = float(np.dot(f.axis_dir, axis)) < 0
            off = _angle_offset(ref.x_dir, f.x_dir, axis)

            umin, umax = f.u_min + off, f.u_max + off
            if flip:
                umin, umax = -umax, -umin
            u_lo, u_hi = min(u_lo, umin), max(u_hi, umax)

            # V is a distance along the axis, measured from each face's own
            # location point. Re-base onto the group's reference point.
            base = float(np.dot(f.axis_point - ref.axis_point, axis))
            vmin, vmax = f.v_min, f.v_max
            if flip:
                vmin, vmax = -vmax, -vmin
            v_lo = min(v_lo, base + vmin)
            v_hi = max(v_hi, base + vmax)

        merged.append(
            MergedCylinder(
                faces=g,
                axis_point=ref.axis_point,
                axis_dir=axis,
                radius=ref.radius,
                u_min=u_lo,
                u_max=u_hi,
                v_min=v_lo,
                v_max=v_hi,
            )
        )
    return merged


# ---------------------------------------------------------------------------
# adjacency
# ---------------------------------------------------------------------------


def adjacent_faces(
    shape: TopoDS_Shape, infos: list[FaceInfo]
) -> dict[int, set[int]]:
    """Face index -> indices of faces sharing at least one edge.

    Lookups go through the kernel's indexed map rather than any hashing of our
    own. `FindIndex` returns the 1-based position or 0 if absent, which is O(1)
    and -- unlike the removed `HashCode` -- present in every OCCT version.
    """
    fmap = face_index_map(shape)
    m = edge_face_map(shape)
    adj: dict[int, set[int]] = {f.index: set() for f in infos}
    n = len(infos)

    for i in range(1, m.Extent() + 1):
        idxs = []
        for f in m.FindFromIndex(i):
            pos = fmap.FindIndex(f)
            if 1 <= pos <= n:
                idxs.append(pos - 1)
        for a in idxs:
            for b in idxs:
                if a != b:
                    adj[a].add(b)
    return adj


def edge_has_curve(edge) -> bool:
    """Is there a 3D curve on this edge at all?

    THIS IS THE GUARD THAT WAS IN THE WRONG FILE.

    v7 added exactly this check, and put it only in `smt/occ/tess.py` -- the
    display code. But `discretise_edge` below is called from `face_loops`, which
    detection uses on every flange and every bend face, and from the DXF export.
    Neither goes anywhere near the viewer. So the crash reported against v7 and
    v8 reproduced with `--diagnose`, which loads and analyses a file with no Qt,
    no 3D view and no tessellation whatsoever -- and that is what proved the
    fix had been applied to the wrong module.

    An edge can have nothing to sample in TWO ways, and only the first is
    obvious:

      * it is DEGENERATE -- a cone apex or a sphere pole, zero length;
      * it carries only a PCURVE, a curve in some face's parameter space, with
        no 3D geometry of its own. Seam edges are like this, and so are edges
        from certain translators -- which is precisely the "not super clean"
        STEP files this tool exists to handle.

    Handing either to `BRepAdaptor_Curve` is undefined behaviour. Undefined
    behaviour in a native library is an access violation, not a Python
    exception, which is why wrapping the call in `try/except` never helped: the
    process is killed before any handler runs and all that reaches you is exit
    code -1073741819.
    """
    try:
        if BRep_Tool.Degenerated_s(edge):
            return False
    except Exception:  # noqa: BLE001 -- unreadable means unusable
        return False

    # OCP's binding for this has varied across OCCT versions: some builds want
    # the two output parameters passed in, others return them. Try both rather
    # than assume, because guessing wrong here is the difference between a
    # guard and a crash.
    for args in ((edge, 0.0, 0.0), (edge,)):
        try:
            res = BRep_Tool.Curve_s(*args)
        except (TypeError, RuntimeError, NotImplementedError):
            continue
        except Exception:  # noqa: BLE001
            return False
        curve = res[0] if isinstance(res, tuple) else res
        return curve is not None

    return False


def discretise_edge(edge, deflection: float = 0.01) -> list[np.ndarray]:
    """Sample an edge into points. Returns [] for an edge with no 3D curve.

    Callers must cope with an empty result. Returning nothing is always
    survivable -- a missing segment is repaired geometrically by
    `geom.chain_point_runs`, or shows up as an open loop and triggers the
    fallback path. Crashing the process is not survivable.
    """
    if not edge_has_curve(edge):
        LOGGER.debug("edge has no 3D curve; skipped")
        return []

    try:
        curve = BRepAdaptor_Curve(edge)
        d = GCPnts_QuasiUniformDeflection(curve, deflection)
        if not d.IsDone():
            return [_v(curve.Value(curve.FirstParameter())),
                    _v(curve.Value(curve.LastParameter()))]
        return [_v(d.Value(i)) for i in range(1, d.NbPoints() + 1)]
    except Exception:  # noqa: BLE001 -- one bad edge must not stop the part
        LOGGER.debug("edge could not be sampled; skipped")
        return []


def _walk_with_explorer(wire, face, deflection):
    """BRepTools_WireExplorer: edges in connection order, orientation applied."""
    attempts = []
    if face is not None:
        attempts.append((wire, face))
    attempts.append((wire,))

    for args in attempts:
        try:
            exp = BRepTools_WireExplorer(*args)
        except Exception:  # noqa: BLE001 -- try the next form
            continue
        segments = []
        try:
            while exp.More():
                edge = exp.Current()
                seg = discretise_edge(edge, deflection)
                if len(seg) < 2:
                    exp.Next()
                    continue
                # An edge stores points in its own curve direction, which is
                # independent of the direction the WIRE traverses it. A
                # REVERSED edge sampled as-is appends a run that runs backwards.
                if edge.Orientation() == TopAbs_REVERSED:
                    seg = seg[::-1]
                segments.append(seg)
                exp.Next()
        except Exception:  # noqa: BLE001
            continue
        if segments:
            return segments
    return []


def _chain_edges(wire, deflection):
    """Order edge point-runs by geometry alone, ignoring topology entirely.

    The last-resort path, and the one that cannot be defeated by an unusual
    wire: sample every edge, then greedily chain the runs end to end, flipping
    any run that fits better reversed. It knows nothing about orientation flags
    or seam handling because it does not need to -- points either join up or
    they do not.
    """
    segments: list[np.ndarray] = []
    exp = TopExp_Explorer(wire, TopAbs_EDGE)
    while exp.More():
        seg = discretise_edge(TopoDS.Edge_s(exp.Current()), deflection)
        if len(seg) >= 2:
            segments.append(np.asarray(seg, dtype=float))
        exp.Next()

    if not segments:
        return []

    return geom.chain_point_runs(segments)


def wire_points(
    wire, face: TopoDS_Face | None = None, deflection: float = 0.05,
    close_tol: float = 1e-3,
) -> np.ndarray:
    """Points around a wire, in connection order, correctly oriented.

    TWO THINGS HAVE TO BE RIGHT, AND A THIRD HAS TO BE CHECKED.

    1. ORDER. `TopExp_Explorer` over a wire yields its edges in whatever order
       the topology happens to store them -- NOT the order they connect. Walking
       a wire that way makes the polyline jump between unrelated parts of the
       boundary, which draws a recognisable outline with stray sweeps across it.

    2. ORIENTATION. An edge stores points in its own curve's parameter
       direction, independent of the direction the wire traverses it. Sampling a
       REVERSED edge without flipping appends a run that runs backwards.

    3. IT HAS TO ACTUALLY CLOSE. `BRepTools_WireExplorer` is the right tool for
       1 and 2, but it returns nothing at all on wires it cannot interpret --
       and an empty result previously meant the flange was silently skipped and
       simply never drawn. So the result is now VALIDATED: a boundary loop must
       come back closed. If it does not, fall through to `_chain_edges`, which
       orders the runs by geometry alone and cannot be defeated by unusual
       topology.

    Trusting a kernel walker without checking its output is what turned a
    traversal problem into missing flanges.
    """
    segments = _walk_with_explorer(wire, face, deflection)
    if segments:
        pts = geom.join_point_runs(segments)
        if geom.loop_is_closed(pts, close_tol):
            return pts

    chained = _chain_edges(wire, deflection)
    if chained:
        return geom.join_point_runs(chained)

    return geom.join_point_runs(segments) if segments else np.zeros((0, 3))


def face_loops_safe(face: TopoDS_Face, index: int,
                    deflection: float = 0.05,
                    cache: dict | None = None) -> list[np.ndarray]:
    """`face_loops` with a breadcrumb, so a native crash names its face.

    Nothing in Python can catch an access violation -- the process is simply
    gone. What CAN be done is to record what was about to be touched, flushed
    to disk, so the last line in `smt.log` names the face that killed it. That
    turns "it crashed" into "it crashed on face 47", which is the difference
    between guessing and knowing.

    `cache` is a plain dict shared between detection and the 3D view. Tracing a
    face's boundary means walking its wires and sampling every edge, which is
    among the most expensive things this tool does -- and both passes wanted the
    same outlines for the same faces. Doing it once is a straight halving of the
    work on any part with many planar faces, which is every sheet metal part.
    """
    if cache is not None and index in cache:
        return cache[index]

    LOGGER.debug("outline: face %d", index)
    try:
        loops = face_loops(face, deflection)
    except Exception:  # noqa: BLE001
        LOGGER.warning("face %d: outline could not be extracted", index)
        loops = []

    if cache is not None:
        cache[index] = loops
    return loops


def face_loops(face: TopoDS_Face, deflection: float = 0.05) -> list[np.ndarray]:
    """Discretised boundary loops of a face, outer wire first.

    Captured once at detection time and cached on the model, so the GUI can
    redraw the flat pattern at any K-factor with nothing but numpy. See
    `smt.model2d` for why that matters.
    """
    outer = BRepTools.OuterWire_s(face)

    wires = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(face, TopAbs_WIRE, wires)

    loops: list[np.ndarray] = []
    outer_loop: np.ndarray | None = None

    for i in range(1, wires.Extent() + 1):
        w = TopoDS.Wire_s(wires.FindKey(i))
        arr = wire_points(w, face, deflection)
        if len(arr) < 3:
            continue
        if w.IsSame(outer):
            outer_loop = arr
        else:
            loops.append(arr)

    return ([outer_loop] if outer_loop is not None else []) + loops


def edge_curve_kind(edge) -> str:
    if not edge_has_curve(edge):
        return "none"
    c = BRepAdaptor_Curve(edge)
    t = c.GetType()
    if t == GeomAbs_Line:
        return "line"
    if t == GeomAbs_Circle:
        return "circle"
    return "other"
