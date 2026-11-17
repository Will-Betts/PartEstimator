"""Unfold the actual solid, so every feature survives into the flat.

THE DIFFERENCE FROM `flatten.build_flat`
----------------------------------------
`build_flat` develops the OUTLINE: it transforms each flange's face into the
plane, fills the bend gaps with strips, fuses the footprints and extrudes the
result to thickness. That is exactly right for a blank size, a DXF and a nest,
and it is what every number in the report comes from.

What it cannot give you is a flat part you can look at, because extruding a
footprint throws away everything that was not part of the outline. A countersink
is not part of the outline. Neither is a chamfered edge, a counterbore, or an
edge break. They vanish, and the flattened part arrives looking like a
gingerbread man.

This module takes the other route: instead of redrawing the part flat, it MOVES
THE MATERIAL. Cut the solid at each bend, take the whole three-dimensional chunk
belonging to each flange, and apply that flange's rigid transform -- the same
one the solver already computed. Every feature comes along because it is
physically attached to the metal being moved. A countersink arrives as a
countersink, at the right place, at the right angle, without anything having to
recognise it first.

That last point is the one that matters. The alternative -- extrude the blank
and then re-cut the recognised features -- looks nearly identical for everything
the classifier understood, and silently loses anything it did not. Moving the
material has no such failure mode: geometry the tool cannot name still arrives,
because it was never asked to name it.

HOW THE CUT IS MADE
-------------------
For each bend, the flange it leaves is bounded by the TANGENT LINE where the
flat surface meets the arc. Project the bend axis onto the flange's plane and
you have that line. The cutting plane contains it, contains the axis direction,
and stands perpendicular to the flange -- so its normal is the cross product of
the axis direction with the flange normal.

Cut with every such plane at once (`BRepAlgoAPI_Splitter`), and the solid falls
into flange chunks and bend chunks. Each chunk is then matched to a flange by
looking for one of that flange's own planar faces inside it, which is a far more
reliable test than comparing centroids. Bend chunks match nothing, and are
replaced by flat strips exactly one bend allowance wide -- because the curved
material has to become straight material, and that is the whole point of a bend
allowance.

WHAT HAPPENS WHEN IT FAILS
--------------------------
Booleans on real translated geometry fail sometimes, and this has never been run
against a real part -- see SYSTEM_SPECS. So every stage is guarded and the whole
thing returns `None` rather than raising. The caller falls back to the extruded
profile, which is the code that has been working since v1, and says on screen
which one you are looking at.

Shipping untested kernel work behind a fallback that announces itself is the
only honest way to ship untested kernel work.
"""

from __future__ import annotations

import math

import numpy as np

from .. import geom
from ..diag import LOGGER, Stage, null_stage

from OCP.BRepAlgoAPI import BRepAlgoAPI_Splitter
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_Transform
from OCP.BRepGProp import BRepGProp
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
from OCP.TopExp import TopExp
from OCP.TopoDS import TopoDS
from OCP.TopTools import TopTools_IndexedMapOfShape, TopTools_ListOfShape
from OCP.gp import gp_Ax3, gp_Dir, gp_Pln, gp_Pnt, gp_Trsf, gp_Vec

from .topo import describe_face


class SolidUnfoldError(Exception):
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


def _volume_and_centre(shape):
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    c = props.CentreOfMass()
    return props.Mass(), np.array([c.X(), c.Y(), c.Z()])


def cut_plane_for(bend, normal: np.ndarray, origin: np.ndarray):
    """(point, normal) of the plane separating a bend from one of its flanges.

    The geometry lives in `smt.geom.bend_cut_plane` so it can be tested without
    a CAD kernel present -- which matters here more than anywhere, because this
    is the calculation that decides where the solid gets cut, and a wrong answer
    produces a plausible-looking part with a flange in the wrong place.
    """
    try:
        return geom.bend_cut_plane(
            bend.axis_point, bend.axis_dir, normal, origin
        )
    except ValueError as exc:
        raise SolidUnfoldError(f"bend {bend.id}: {exc}") from exc


def _tool_face(point: np.ndarray, normal: np.ndarray, reach: float):
    """A bounded planar face big enough to cut the whole part.

    Bounded rather than infinite on purpose: OCC's boolean algorithms are
    markedly better behaved with finite tools, and `reach` is taken from the
    part's own bounding box so 'big enough' is not a guess.
    """
    pln = gp_Pln(
        gp_Pnt(*(float(v) for v in point)),
        gp_Dir(*(float(v) for v in normal)),
    )
    mk = BRepBuilderAPI_MakeFace(pln, -reach, reach, -reach, reach)
    if not mk.IsDone():
        raise SolidUnfoldError("could not build a cutting plane")
    return mk.Face()


def _split(solid, tools):
    """Cut the solid with every tool at once."""
    args = TopTools_ListOfShape()
    args.Append(solid)
    tool_list = TopTools_ListOfShape()
    for t in tools:
        tool_list.Append(t)

    splitter = BRepAlgoAPI_Splitter()
    splitter.SetArguments(args)
    splitter.SetTools(tool_list)
    # The tangent planes touch the solid exactly along a line, which is the
    # hardest case for a boolean: the inputs are degenerate precisely where they
    # meet. A small fuzzy value lets the algorithm treat 'exactly touching' as
    # touching instead of as an ambiguous near-miss.
    splitter.SetFuzzyValue(1e-5)
    splitter.Build()

    # NOTE: no HasErrors() here. OCP 7.9.3.1 does not bind it on
    # BRepAlgoAPI_Splitter, so calling it raised AttributeError and EVERY solid
    # unfold silently fell back to the extruded blank. IsDone() is bound and
    # is false whenever the algorithm reported an error.
    if not splitter.IsDone():
        raise SolidUnfoldError("the splitter could not cut the solid")

    pieces = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(splitter.Shape(), TopAbs_SOLID, pieces)
    return [
        TopoDS.Solid_s(pieces.FindKey(i)) for i in range(1, pieces.Extent() + 1)
    ]


def _planar_faces(piece):
    fmap = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(piece, TopAbs_FACE, fmap)
    out = []
    for i in range(1, fmap.Extent() + 1):
        try:
            info = describe_face(TopoDS.Face_s(fmap.FindKey(i)), i - 1)
        except Exception:  # noqa: BLE001 -- skip a face we cannot read
            continue
        if info.kind == "plane" and info.normal is not None:
            out.append(info)
    return out


def _assign(piece, part, result, thickness: float, tol) -> int | None:
    """Which flange does this chunk belong to?

    Matched by finding one of the flange's OWN surfaces inside the chunk --
    a planar face parallel to the flange's reference plane and lying either on
    it or one thickness behind it. Centroid proximity was the obvious
    alternative and is much weaker: on a tray, the base and a folded-over
    flange have centroids a few millimetres apart and the wrong one wins.

    Returns None for a bend zone, whose faces are all either curved or standing
    through the thickness, so none of them matches anything.
    """
    best_id, best_area = None, 0.0

    for info in _planar_faces(piece):
        if info.area <= tol.min_face_area:
            continue
        for fid in result.transforms:
            flange = part.flanges.get(fid)
            if flange is None:
                continue
            n = flange.normal
            if not (
                geom.codirectional(info.normal, n, tol.tangent_angular)
                or geom.antiparallel(info.normal, n, tol.tangent_angular)
            ):
                continue
            d = abs(geom.signed_plane_distance(info.point, flange.origin, n))
            if d < tol.tangent or tol.radius_close(d, thickness):
                if info.area > best_area:
                    best_id, best_area = fid, info.area

    return best_id


def unfold_solid(part, result, opts, M_frame: np.ndarray,
                 stage: Stage | None = None):
    """The flat part as a real solid, features and all. None if it cannot.

    `M_frame` is the same oriented flat frame the outline and the DXF are built
    from, so the solid lands in the identical position and orientation as
    everything else. Sharing it is what stops the 3D view and the DXF quietly
    disagreeing about which way round the part is.
    """
    stage = stage or null_stage()

    if not part.bends:
        # A flat blank is already flat. Nothing to cut, nothing to move -- and
        # returning the original solid keeps every feature by definition.
        stage.detail("flat blank: the part is already its own flat pattern")
        return _apply(part.solid_bent, M_frame)

    tol = opts.tol

    try:
        from .tess import bounding_box

        lo, hi = bounding_box(part.solid_bent)
        reach = float(np.linalg.norm(hi - lo)) or 1.0

        stage("Cutting the solid at each bend...")
        tools = []
        for fb in result.flat_bends:
            bend = part.bends.get(fb.bend_id)
            if bend is None:
                continue
            for fid in (bend.flange_a, bend.flange_b):
                side = result.sides.get(fid)
                if side is None:
                    continue
                normal, origin = side[0], side[1]
                try:
                    point, cut_normal = cut_plane_for(bend, normal, origin)
                    tools.append(_tool_face(point, cut_normal, reach))
                except SolidUnfoldError as exc:
                    LOGGER.warning("solid unfold: %s", exc)

        if not tools:
            raise SolidUnfoldError("no usable cutting planes could be built")

        pieces = _split(part.solid_bent, tools)
        stage.done(f"{len(pieces)} piece(s)")

        stage.check_cancelled()
        stage("Swinging each flange flat...")
        moved = []
        assigned = 0
        for piece in pieces:
            volume, _centre = _volume_and_centre(piece)
            if volume <= 1e-9:
                continue
            fid = _assign(piece, part, result, part.thickness, tol)
            if fid is None:
                continue                    # a bend zone; replaced below
            moved.append(_apply(piece, M_frame @ result.transforms[fid]))
            assigned += 1

        if not moved:
            raise SolidUnfoldError(
                "no piece of the cut solid could be matched to a flange"
            )
        stage.done(f"{assigned} flange chunk(s) placed")

        stage.check_cancelled()
        stage("Filling the bend zones...")
        from .flatten import _bend_faces

        strips = []
        for fb in result.flat_bends:
            try:
                for face in _bend_faces(part, fb, M_frame):
                    strips.append(
                        BRepPrimAPI_MakePrism(
                            face, gp_Vec(0.0, 0.0, float(part.thickness))
                        ).Shape()
                    )
            except Exception as exc:  # noqa: BLE001 -- one strip is not fatal
                LOGGER.warning("bend %s: flat strip failed (%s)", fb.bend_id, exc)
        stage.done(f"{len(strips)} strip(s)")

        stage.check_cancelled()
        stage("Fusing the flat solid...")
        from .flatten import _fuse_all

        solid = _fuse_all(moved + strips) if (moved + strips) else None
        if solid is None:
            raise SolidUnfoldError("nothing to fuse")
        stage.done()

        _sanity_check(part, solid)
        _strict_gate(part, solid)
        return solid

    except Exception as exc:  # noqa: BLE001 -- the fallback is the whole point
        LOGGER.warning(
            "solid unfold failed (%s); falling back to the extruded profile",
            exc,
        )
        part.warn(
            f"the flat part could not be built by moving the real material "
            f"({exc}). Falling back to an extruded blank, which is "
            f"dimensionally correct but does not carry countersinks, chamfers "
            f"or other machined detail. Every reported dimension is unaffected."
        )
        return None


def _strict_gate(part, solid) -> None:
    """Refuse a solid that is not unmistakably a flat blank.

    Added when the HasErrors() crash was fixed and this module ran for the first
    time on a real kernel: on a plain L-bracket it produced six loose solids,
    two of them standing proud of the sheet, at +1.3% volume. The old 5%
    volume check let that through. A flat pattern must be ONE solid exactly one
    thickness deep; anything else falls back to the extruded blank, which is
    what every export used before this fix anyway.
    """
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    from .tess import bounding_box

    count = 0
    exp = TopExp_Explorer(solid, TopAbs_SOLID)
    while exp.More():
        count += 1
        exp.Next()
    if count != 1:
        raise SolidUnfoldError(f"unfold produced {count} separate solids, not one")

    lo, hi = bounding_box(solid)
    depth = float(hi[2] - lo[2])
    t = float(part.thickness)
    if t > 0 and abs(depth - t) > max(0.02 * t, 0.01):
        raise SolidUnfoldError(
            f"unfold is {depth:.3f} mm deep, not one {t:.3f} mm sheet"
        )

    if part.volume > 0:
        volume, _c = _volume_and_centre(solid)
        if abs(volume - part.volume) / part.volume > 0.01:
            raise SolidUnfoldError(
                f"unfold volume {volume:.1f} vs part {part.volume:.1f} mm^3 (>1%)"
            )


def _sanity_check(part, solid) -> None:
    """A rigid motion conserves volume, so the flat must weigh the same.

    Cheap and decisive. If the split dropped a chunk, or a piece was assigned to
    two flanges and placed twice, the volume moves and this says so -- rather
    than handing back a plausible-looking solid with a flange missing.
    """
    if part.volume <= 0:
        return
    volume, _c = _volume_and_centre(solid)
    if volume <= 0:
        raise SolidUnfoldError("the fused flat solid has no volume")

    # Bend zones are replaced by strips one bend allowance wide, so the volume
    # is not identical -- the allowance is by construction the length that makes
    # the flat and the arc contain the same material, but discretisation of the
    # developed boundary moves it slightly.
    error = abs(volume - part.volume) / part.volume
    LOGGER.info(
        "solid unfold: volume %.1f vs %.1f (%.2f%%)",
        volume, part.volume, error * 100.0,
    )
    if error > 0.05:
        part.warn(
            f"the unfolded solid's volume differs from the original by "
            f"{error * 100:.1f}%. A rigid unfold should conserve it, so a piece "
            f"was probably dropped or placed twice. Trust the reported "
            f"dimensions over the 3D shape."
        )
