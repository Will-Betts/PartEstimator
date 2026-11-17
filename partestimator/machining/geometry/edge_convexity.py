"""Edge convexity: how much of the part's edge length is INTERNAL corners.

Total edge length is a poor complexity measure -- it's dominated by tangent
edges and by however the modeller happened to split surfaces. Concave edge
length is much better aligned with machining effort, because internal
corners are exactly what force small tools, slow contouring passes and
extra finishing work. An outside corner costs nothing; an inside corner
costs a tool change and a feed reduction.

Convexity test
--------------
For an edge shared by faces F1 and F2, with OUTWARD normals n1 and n2 at a
point on the edge, and t the edge tangent oriented along F1's wire
direction (interior of F1 on the left):

    dot(n1 x n2, t) > 0  ->  CONVEX   (outside corner)
    dot(n1 x n2, t) < 0  ->  CONCAVE  (inside corner)

Derived on two cases with known answers -- the outside corner of a box, and
the inside corner where a wall meets a slab -- and asserted in selftest.py
against a real box (which must have zero concave edge length) and a plate
with a bore (whose two circular edges must be concave).

Orientation bookkeeping: the tangent from the curve is flipped if the edge
is REVERSED within F1, and flipped again if F1 itself is REVERSED, since
both invert the effective wire direction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.GCPnts import GCPnts_AbscissaPoint
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Shape
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

from .normals import sampled_outward_normal

# Below this angle between the two face normals the edge is "smooth"
# (tangent-continuous) and neither convex nor concave in any meaningful
# sense -- e.g. where a fillet blends into a wall. Counting those would
# reintroduce exactly the noise this measure exists to avoid.
SMOOTH_EDGE_ANGLE_RAD = np.radians(5.0)


@dataclass
class EdgeConvexityResult:
    concave_length_mm: float = 0.0
    convex_length_mm: float = 0.0
    smooth_length_mm: float = 0.0
    total_length_mm: float = 0.0
    concave_edge_count: int = 0
    convex_edge_count: int = 0

    @property
    def concave_fraction(self) -> float:
        classified = self.concave_length_mm + self.convex_length_mm
        return (self.concave_length_mm / classified) if classified > 0 else 0.0


def _uv_on_face(edge, face, param):
    """Parameter-space point (u, v) on `face` for a point at curve parameter
    `param` on `edge`, via the edge's pcurve. Returns None when the edge has
    no pcurve on that face."""
    result = BRep_Tool.CurveOnSurface_s(edge, face, 0.0, 0.0)
    if result is None:
        return None
    curve2d = result[0] if isinstance(result, (tuple, list)) else result
    if curve2d is None:
        return None
    pnt2d = curve2d.Value(param)
    return pnt2d.X(), pnt2d.Y()


def compute_edge_convexity(shape: TopoDS_Shape, should_cancel=None) -> EdgeConvexityResult:
    """Classifies every two-face edge as concave, convex or smooth and sums
    their lengths."""
    result = EdgeConvexityResult()

    edge_to_faces = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, edge_to_faces)

    for i in range(1, edge_to_faces.Extent() + 1):
        if should_cancel is not None and (i % 512 == 0) and should_cancel():
            break

        edge = TopoDS.Edge_s(edge_to_faces.FindKey(i))
        if BRep_Tool.Degenerated_s(edge):
            continue
        faces = edge_to_faces.FindFromIndex(i)
        if faces.Extent() != 2:
            continue  # boundary edge of an open shell, or a non-manifold join

        try:
            curve = BRepAdaptor_Curve(edge)
            length = float(GCPnts_AbscissaPoint.Length_s(curve))
            if length <= 0:
                continue
            mid = 0.5 * (curve.FirstParameter() + curve.LastParameter())

            face_list = [TopoDS.Face_s(shp) for shp in faces]
            if len(face_list) != 2:
                continue
            f1, f2 = face_list

            normals = []
            ok = True
            for f in (f1, f2):
                uv = _uv_on_face(edge, f, mid)
                if uv is None:
                    ok = False
                    break
                adaptor = BRepAdaptor_Surface(f, True)
                n = sampled_outward_normal(f, adaptor, uv[0], uv[1])
                if n is None:
                    ok = False
                    break
                normals.append(n)
            if not ok:
                continue
            n1, n2 = normals

            cross = np.cross(n1, n2)
            sin_angle = float(np.linalg.norm(cross))
            result.total_length_mm += length
            if sin_angle < np.sin(SMOOTH_EDGE_ANGLE_RAD):
                result.smooth_length_mm += length
                continue

            # Tangent, oriented along f1's wire direction.
            d1 = curve.DN(mid, 1)
            tangent = np.array([d1.X(), d1.Y(), d1.Z()], dtype=float)
            norm = np.linalg.norm(tangent)
            if norm < 1e-12:
                continue
            tangent /= norm

            edge_orientation_in_f1 = _edge_orientation_in_face(edge, f1)
            if edge_orientation_in_f1 == TopAbs_REVERSED:
                tangent = -tangent
            # NOTE: no separate flip for f1 being REVERSED. TopExp_Explorer
            # composes orientations as it descends, so the edge orientation
            # read from inside f1 already includes f1's own reversal.
            # Flipping again (as v2.5 did) classified half of a plain box's
            # edges as concave.

            if float(np.dot(cross, tangent)) < 0:
                result.concave_length_mm += length
                result.concave_edge_count += 1
            else:
                result.convex_length_mm += length
                result.convex_edge_count += 1
        except Exception:  # noqa: BLE001 - one bad edge shouldn't lose the metric
            continue

    return result


def _edge_orientation_in_face(edge, face):
    """The edge's orientation as recorded within `face` (which can differ
    from the edge's standalone orientation)."""
    explorer = TopExp_Explorer(face, TopAbs_EDGE)
    while explorer.More():
        candidate = explorer.Current()
        if candidate.IsSame(edge):
            return candidate.Orientation()
        explorer.Next()
    return edge.Orientation()
