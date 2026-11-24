"""B-rep edge extraction for CAD-style display.

The part is rendered as a triangle mesh (that's how any GPU draws a curved
surface), but it does NOT have to *look* like a mesh. A STEP file carries
real topological edges -- the boundary curves where two faces meet -- and
those are what a CAD viewer draws as crisp black lines. Tessellation
triangle edges are a rendering artifact by comparison.

So this module pulls the actual TopoDS_Edge curves out of the shape and
discretizes each into a polyline, which the viewer overlays on a smoothly
shaded surface. The result reads like the STEP file rather than like a mesh.

Three details worth noting:

  * Edges are collected through a TopTools_IndexedMapOfShape rather than a
    plain TopExp_Explorer traversal. An edge shared by two faces is visited
    once per face by the explorer, so the naive approach draws every
    interior edge twice -- harmless visually but double the geometry.
  * Degenerate edges (e.g. the collapsed "edge" at the apex of a cone, or
    at a sphere's pole) are skipped: they have no length and discretizing
    them produces junk.
  * SEAM edges are skipped. Any closed surface -- every cylindrical bore,
    every cone -- carries a parametric seam where its u-parameter wraps
    around. That seam is a bookkeeping artifact of the surface's
    parametrisation, not a feature of the part: nothing is machined there
    and no CAD viewer draws it. Left in, it appears as a stray line running
    the length of every hole, which reads convincingly (and misleadingly)
    like the hole's axis. A seam is detected by its two adjacent faces
    being the SAME face.
"""
from __future__ import annotations

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.GCPnts import GCPnts_AbscissaPoint, GCPnts_QuasiUniformDeflection
from OCP.GeomAbs import GeomAbs_Line
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCP.TopExp import TopExp
from OCP.TopoDS import TopoDS, TopoDS_Shape
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

# Chordal tolerance for turning a curved edge into a polyline, as a fraction
# of THAT EDGE's own length -- not of the whole part.
#
# This matters for visibility, not just looks. A polyline approximating a
# curve is a series of chords, and a chord always cuts to the *inside* of
# the curve. Around a bore or pocket that inside is solid material, so a
# chord that's too coarse literally sinks beneath the surface and the edge
# disappears -- which is why edges on concave features were dropping out.
# Scaling by part size made this worst exactly where it hurt most: a 5 mm
# hole in a 400 mm plate got a 0.2 mm chord error, far more than the
# feature could tolerate. Scaling by the edge's own length keeps small
# features crisp while not over-refining long ones.
EDGE_RELATIVE_DEFLECTION = 0.002
MIN_EDGE_DEFLECTION_MM = 0.002
MAX_EDGE_DEFLECTION_MM = 0.25


def _is_seam(face_list) -> bool:
    """A seam edge has the SAME face on both sides -- the surface wraps
    around and meets itself there. Every closed cylinder and cone has one,
    and drawing it puts a spurious line down the length of every hole."""
    faces = list(face_list)
    if len(faces) != 2:
        return False
    return bool(faces[0].IsSame(faces[1]))


# The curved edges again, much finer, for the tabbed-fixturing channel: it is
# offset from the part's outline, and an arc must come out an arc, not the
# display's chords (up to MAX_EDGE_DEFLECTION_MM off). About 0.0005".
OUTLINE_DEFLECTION_MM = 0.0125


def extract_outline_edges(shape: TopoDS_Shape) -> list[np.ndarray]:
    """The shape's curved edges as fine polylines (mm): see
    OUTLINE_DEFLECTION_MM. Straight edges are left out -- the mesh has
    those exactly."""
    return extract_edge_polylines(shape, OUTLINE_DEFLECTION_MM, curved_only=True)


def extract_edge_polylines(shape: TopoDS_Shape, deflection_mm: float | None = None,
                            diagonal_mm: float | None = None,
                            curved_only: bool = False) -> list[np.ndarray]:
    """Returns a list of (P_i, 3) arrays, each an ordered polyline tracing
    one topological edge of the shape, in world coordinates (mm).

    `deflection_mm` overrides the automatic size-relative tolerance (with
    `curved_only`, it is used as given for every edge, and straight edges
    are skipped). `diagonal_mm` lets the caller pass an already-computed
    bounding diagonal to avoid recomputing it.
    """
    if deflection_mm is None:
        if diagonal_mm is None:
            from .mesh_utils import shape_diagonal_mm
            diagonal_mm = shape_diagonal_mm(shape)
        if diagonal_mm and diagonal_mm > 0:
            deflection_mm = float(np.clip(
                diagonal_mm * EDGE_RELATIVE_DEFLECTION,
                MIN_EDGE_DEFLECTION_MM,
                MAX_EDGE_DEFLECTION_MM,
            ))
        else:
            deflection_mm = MIN_EDGE_DEFLECTION_MM

    # Ancestors are needed to spot seam edges (both sides are the same face).
    edge_map = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, edge_map)

    polylines: list[np.ndarray] = []
    for i in range(1, edge_map.Extent() + 1):
        edge = TopoDS.Edge_s(edge_map.FindKey(i))
        if BRep_Tool.Degenerated_s(edge):
            continue
        if _is_seam(edge_map.FindFromIndex(i)):
            continue
        try:
            curve = BRepAdaptor_Curve(edge)
            if curved_only and curve.GetType() == GeomAbs_Line:
                continue

            # Per-edge tolerance, scaled to this edge's own length so small
            # features stay crisp (see EDGE_RELATIVE_DEFLECTION above).
            edge_deflection = deflection_mm
            try:
                length = GCPnts_AbscissaPoint.Length_s(curve) if not curved_only else 0.0
                if length > 0:
                    edge_deflection = float(np.clip(
                        length * EDGE_RELATIVE_DEFLECTION,
                        MIN_EDGE_DEFLECTION_MM,
                        MAX_EDGE_DEFLECTION_MM,
                    ))
            except Exception:  # noqa: BLE001
                pass  # fall back to the part-relative value

            sampler = GCPnts_QuasiUniformDeflection(curve, edge_deflection)
            if not sampler.IsDone():
                continue
            n_pts = sampler.NbPoints()
            if n_pts < 2:
                continue
            pts = np.empty((n_pts, 3), dtype=float)
            for k in range(1, n_pts + 1):
                p = sampler.Value(k)
                pts[k - 1] = (p.X(), p.Y(), p.Z())
            polylines.append(pts)
        except Exception:  # noqa: BLE001
            # A single pathological edge shouldn't cost us the whole display.
            continue

    return polylines


def polylines_to_line_cells(polylines: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Packs polylines into the (points, line_cells) pair that VTK/PyVista
    expects, where line_cells is a flat array of
    [n, i0, i1, ... i(n-1), m, j0, j1, ...].

    Returns (empty (0,3), empty (0,)) for empty input so callers can pass
    the result straight through without special-casing.
    """
    if not polylines:
        return np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=np.int64)

    points = np.vstack(polylines)
    cells: list[int] = []
    offset = 0
    for pl in polylines:
        n = len(pl)
        cells.append(n)
        cells.extend(range(offset, offset + n))
        offset += n
    return points, np.asarray(cells, dtype=np.int64)
