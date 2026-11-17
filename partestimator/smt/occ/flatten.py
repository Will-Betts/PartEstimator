"""Build the flat pattern, then extrude it into the unfolded solid.

WHY THIS DOES NOT UNFOLD THE SOLID DIRECTLY
-------------------------------------------
Genuinely unfolding a B-rep -- cutting the bend zones out, lofting flat strips
between the cut edges, boolean-fusing the result and keeping it watertight
across arbitrary parts -- is where the overwhelming majority of the development
effort in a tool like this goes, and it fails in interesting ways on real parts.

We get the same answer for a fraction of the work:

  1. Transform each flange face into the flat using the solver's rigid motions.
     Because the faces are already planar and end up parallel to the base plane,
     landing them exactly in that plane is a pure translation -- no projection,
     no distortion, no approximation.
  2. Fill each bend gap with a strip exactly BA wide.
  3. Fuse the 2D footprints.
  4. Extrude by the thickness.

For a purely bent part the extruded result is geometrically identical to a
"true" unfold, and it is valid by construction.
"""

from __future__ import annotations

import math

import numpy as np

from .. import geom
from ..model import FlatPattern, Part
from ..unfold import UnfoldResult
from .topo import discretise_edge, edge_length, wire_points

from OCP.BRep import BRep_Tool
from OCP.BOPAlgo import BOPAlgo_Builder
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_Transform,
)
from OCP.BRepGProp import BRepGProp
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCP.GProp import GProp_GProps
from OCP.gp import gp_Ax3, gp_Dir, gp_Pln, gp_Pnt, gp_Trsf, gp_Vec
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_WIRE
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.TopTools import TopTools_IndexedMapOfShape
from OCP.BRepTools import BRepTools


class FlattenError(Exception):
    pass


def _to_trsf(M: np.ndarray) -> gp_Trsf:
    t = gp_Trsf()
    t.SetValues(
        float(M[0, 0]), float(M[0, 1]), float(M[0, 2]), float(M[0, 3]),
        float(M[1, 0]), float(M[1, 1]), float(M[1, 2]), float(M[1, 3]),
        float(M[2, 0]), float(M[2, 1]), float(M[2, 2]), float(M[2, 3]),
    )
    return t


def _apply(shape, M: np.ndarray):
    return BRepBuilderAPI_Transform(shape, _to_trsf(M), True).Shape()


def flat_frame(origin: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """4x4 mapping the flat plane onto the global XY plane.

    Rows of the rotation are the flat frame's basis vectors, so applying it
    sends `normal` to +Z and `origin` to the origin.
    """
    n = geom.unit(normal)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(seed, n))) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    ex = geom.unit(np.cross(seed, n))
    ey = np.cross(n, ex)

    M = np.eye(4)
    M[:3, :3] = np.vstack([ex, ey, n])
    M[:3, 3] = -M[:3, :3] @ np.asarray(origin, dtype=float)
    return M


def _drop_to_xy(face):
    """Translate an already-XY-parallel face onto z = 0 exactly."""
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    z = props.CentreOfMass().Z()
    if abs(z) < 1e-12:
        return face
    M = geom.translation([0.0, 0.0, -z])
    return _apply(face, M)


def _polygon_face(pts_2d):
    """A planar face in z = 0 from an ordered 2D boundary."""
    poly = BRepBuilderAPI_MakePolygon()
    for x, y in pts_2d:
        poly.Add(gp_Pnt(float(x), float(y), 0.0))
    poly.Close()
    wire = poly.Wire()
    mk = BRepBuilderAPI_MakeFace(wire)
    if not mk.IsDone():
        raise FlattenError("could not build a face from the developed boundary")
    return mk.Face()


def _bend_faces(part, fb, M_frame: np.ndarray):
    """Flat faces covering one bend zone.

    Prefers the true development of the trimmed cylindrical face (see
    `model2d._develop_bend` and `geom.develop_cylinder`) and falls back to a
    rectangle only when the face was not captured. The exported DXF and the
    on-screen preview therefore come from the same construction -- if they ever
    disagree, one of them is wrong and you would never know which.
    """
    from ..model2d import _develop_bend

    p0 = geom.apply_point(M_frame, fb.line_start)
    p1 = geom.apply_point(M_frame, fb.line_end)
    u = geom.apply_vector(M_frame, fb.step_dir)
    u[2] = 0.0
    n = float(np.linalg.norm(u))
    if n < 1e-12:
        raise FlattenError(f"bend {fb.bend_id}: degenerate unfold direction")
    u_hat = u[:2] / n

    bend = part.bends.get(fb.bend_id)
    developed = _develop_bend(bend, fb, M_frame, p0[:2], u_hat)

    if developed:
        faces = []
        for loop in developed:
            try:
                faces.append(_polygon_face(loop))
            except Exception:  # noqa: BLE001 -- skip a degenerate sub-loop
                continue
        if faces:
            return faces

    w = u_hat * fb.width
    return [_polygon_face([
        (p0[0], p0[1]),
        (p1[0], p1[1]),
        (p1[0] + w[0], p1[1] + w[1]),
        (p0[0] + w[0], p0[1] + w[1]),
    ])]


def _fuse_all(pieces):
    """Fuse every piece in ONE operation rather than N sequential ones.

    The pieces of a flat pattern abut exactly edge-to-edge, which is the
    hardest case for a boolean: the inputs are degenerate precisely where they
    meet. Chaining `BRepAlgoAPI_Fuse` one piece at a time compounds that -- each
    result becomes the input to the next, so a single marginal operation
    corrupts everything downstream.

    `BOPAlgo_Builder` takes all the arguments at once and solves the whole
    arrangement together, which is both more robust and considerably faster.
    Sequential fusing is kept only as a fallback.
    """
    if len(pieces) == 1:
        return pieces[0]

    try:
        builder = BOPAlgo_Builder()
        for piece in pieces:
            builder.AddArgument(piece)
        # Flat-pattern pieces abut exactly, and "exactly" in floating point
        # means within a rounding error either way. A small fuzzy value lets
        # the builder treat coincident edges as coincident and actually merge
        # the regions instead of leaving them as neighbours.
        builder.SetFuzzyValue(1e-5)
        builder.Perform()
        if not builder.HasErrors():
            return builder.Shape()
    except Exception:  # noqa: BLE001 -- fall back to pairwise fusing
        pass

    fused = pieces[0]
    for piece in pieces[1:]:
        op = BRepAlgoAPI_Fuse(fused, piece)
        op.Build()
        if not op.IsDone():
            raise FlattenError(
                "boolean fuse failed while assembling the flat pattern. This "
                "usually means two footprints overlap -- check for a bend angle "
                "that folds material back onto itself."
            )
        fused = op.Shape()
    return fused


def build_flat(part: Part, result: UnfoldResult, opts=None) -> FlatPattern:
    """Assemble the 2D profile and the extruded flat solid.

    The frame comes from `model2d.oriented_flat_frame`, which is the SAME
    matrix the on-screen preview is drawn from -- including the rotation that
    turns the blank to its tightest bounding box. Sharing it is what guarantees
    the DXF and the preview cannot disagree about which way round the part is.
    """
    from ..model2d import oriented_flat_frame

    M_frame, angle, saved = oriented_flat_frame(part, result, opts)
    part.flat_rotation = angle
    part.flat_rotation_saving = saved

    pieces = []
    for fid, M in result.transforms.items():
        _n, _o, face, side = result.sides[fid]
        flange = part.flanges.get(fid)

        # EVERY face on this side, not just the referenced one. A flange can be
        # several disconnected coplanar regions -- two tabs sharing a bend line
        # -- and transforming only the largest is how the others went missing
        # from the DXF and from the extruded solid.
        faces = flange.faces_for_side(side) if flange is not None else []
        if not faces:
            faces = [face] if face is not None else []
        if not faces:
            raise FlattenError(f"flange {fid} has no kernel face handle")

        for f in faces:
            pieces.append(_drop_to_xy(_apply(f, M_frame @ M)))

    for fb in result.flat_bends:
        pieces.extend(_bend_faces(part, fb, M_frame))

    fused = _fuse_all(pieces)

    unify = ShapeUpgrade_UnifySameDomain(fused, True, True, True)
    unify.Build()
    fused = unify.Shape()

    face_map = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(fused, TopAbs_FACE, face_map)

    # KEEP EVERY FACE. The previous version took FindKey(1) and discarded the
    # rest, so whenever the pieces merely touched instead of merging -- which is
    # the normal outcome when they abut exactly -- the DXF and the STEP solid
    # silently lost whole flanges that the preview was quite correctly showing.
    all_faces = [
        TopoDS.Face_s(face_map.FindKey(i))
        for i in range(1, face_map.Extent() + 1)
    ]
    if not all_faces:
        raise FlattenError("the flat pattern fused to nothing")

    # Largest first, so `flat.face` means something to callers that want one.
    def _area(f):
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(f, props)
        return props.Mass()

    all_faces.sort(key=_area, reverse=True)

    if len(all_faces) > 1:
        part.note(
            f"the flat pattern is {len(all_faces)} separate regions. All of them "
            f"are exported; regions usually mean the pieces touched without "
            f"merging, which is harmless, but several LARGE regions can mean a "
            f"bend was missed."
        )

    flat = FlatPattern(faces=all_faces)
    for f in all_faces:
        _measure(flat, f)

    for fb in result.flat_bends:
        a = geom.apply_point(M_frame, fb.line_start)
        b = geom.apply_point(M_frame, fb.line_end)
        u = geom.apply_vector(M_frame, fb.step_dir)
        u2 = np.array([u[0], u[1]])
        n = float(np.linalg.norm(u2))
        if n > 1e-12:
            u2 = u2 / n * (fb.width / 2.0)
        flat.bend_lines.append(
            (a[:2] + u2, b[:2] + u2, fb.bend_id)
        )

    solids = [
        BRepPrimAPI_MakePrism(
            f, gp_Vec(0.0, 0.0, float(part.thickness))
        ).Shape()
        for f in all_faces
    ]
    flat.solid = solids[0] if len(solids) == 1 else _fuse_all(solids)

    return flat


def _measure(flat: FlatPattern, profile) -> None:
    """Accumulate area, cut length and bounds for one region of the profile.

    Called once per region and adds to what is already there, so a flat pattern
    that came out as several regions reports the total rather than whichever one
    happened to be measured last.
    """
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(profile, props)
    flat.area += props.Mass()

    outer = BRepTools.OuterWire_s(profile)

    wires = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(profile, TopAbs_WIRE, wires)

    edges = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(profile, TopAbs_EDGE, edges)
    for i in range(1, edges.Extent() + 1):
        flat.perimeter += edge_length(TopoDS.Edge_s(edges.FindKey(i)))

    all_pts: list[np.ndarray] = []
    for i in range(1, wires.Extent() + 1):
        w = TopoDS.Wire_s(wires.FindKey(i))
        # Ordered and orientation-corrected -- see topo.wire_points for why
        # walking a wire with TopExp_Explorer produces a scrambled outline.
        pts3 = wire_points(w, profile)
        arr = pts3[:, :2] if len(pts3) else np.zeros((0, 2))
        if w.IsSame(outer):
            flat.outline.append(arr)
        else:
            flat.holes.append(arr)
        all_pts.extend(list(pts3))

    if all_pts:
        arr = np.array(all_pts)[:, :2]
        flat._pts = np.vstack([getattr(flat, "_pts", np.zeros((0, 2))), arr])
        lo, hi = flat._pts.min(axis=0), flat._pts.max(axis=0)
        flat.bbox = (float(hi[0] - lo[0]), float(hi[1] - lo[1]))
        flat.min_rect = geom.min_area_rect(flat._pts)


def refold_check(part: Part, result: UnfoldResult) -> float:
    """Volume agreement between the flat solid and the original.

    A pure bend conserves volume exactly, so any discrepancy beyond rounding
    means the flat pattern has the wrong area -- a missed bend, a double-counted
    bend zone, or a formed feature that was silently developed as if it were a
    bend. This is the cheapest end-to-end correctness check available.
    """
    if part.flat is None or part.volume <= 0:
        return float("nan")
    expected = part.flat.area * part.thickness
    return abs(expected - part.volume) / part.volume
