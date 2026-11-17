"""Tessellation helpers.

Several downstream calculations (the oriented bounding box, the enclosing
cylinder, the thin-wall ray casting, and the 3D viewer) need either a
discretized point cloud of the shape or a full triangulated mesh. We
tessellate once via BRepMesh_IncrementalMesh and reuse the triangulation.

Note on OCP vs. pythonocc-core: OCP's binding generator (pywrap) suffixes
bound C++ static methods with "_s" to avoid colliding with same-named
instance methods -- e.g. `BRep_Tool.Triangulation_s(face, loc)` here,
versus pythonocc-core's plain `BRep_Tool.Triangulation(face, loc)`, and
`TopoDS.Face_s(shape)` versus pythonocc-core's lowercase `topods.Face(...)`
convenience object. Both bindings otherwise mirror the same underlying
OpenCASCADE C++ API.
"""
from __future__ import annotations

import numpy as np
from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Tool
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Shape
from OCP.TopTools import TopTools_IndexedMapOfShape

# Linear deflection is derived from the part's own size rather than being a
# fixed absolute value. A hardcoded 0.1 mm deflection is reasonable for a
# 25 mm part but wildly over-tessellates a 600 mm one -- and mesh size
# drives almost everything downstream (OCCT meshing time, convex hull cost,
# and viewer render cost), so this was the single biggest performance
# problem on large parts. 0.2% of the bounding diagonal keeps the display
# and the bounding-box fit visually/numerically indistinguishable while
# cutting triangle counts dramatically.
DEFAULT_RELATIVE_DEFLECTION = 0.002
MIN_DEFLECTION_MM = 0.02   # don't go finer than this on tiny parts
MAX_DEFLECTION_MM = 2.0    # don't go coarser than this on huge parts
DEFAULT_ANGULAR_DEFLECTION_RAD = 0.5


def shape_diagonal_mm(shape: TopoDS_Shape) -> float:
    """Bounding-box diagonal length of the shape, used to scale tolerances."""
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box, True)
    if box.IsVoid():
        return 0.0
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return float(np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin]))


def tessellate(shape: TopoDS_Shape, linear_deflection_mm: float | None = None,
                angular_deflection_rad: float = DEFAULT_ANGULAR_DEFLECTION_RAD) -> float:
    """Mutates `shape`'s BRep triangulation data in place (standard OCCT
    pattern: BRep_Tool.Triangulation_s reads whatever mesh was last computed
    by BRepMesh_IncrementalMesh). Returns the deflection actually used.

    If `linear_deflection_mm` is None (the default), it's derived from the
    part's bounding-box diagonal -- see DEFAULT_RELATIVE_DEFLECTION.
    """
    if linear_deflection_mm is None:
        diagonal = shape_diagonal_mm(shape)
        if diagonal > 0:
            linear_deflection_mm = float(
                np.clip(diagonal * DEFAULT_RELATIVE_DEFLECTION, MIN_DEFLECTION_MM, MAX_DEFLECTION_MM)
            )
        else:
            linear_deflection_mm = MIN_DEFLECTION_MM

    mesh = BRepMesh_IncrementalMesh(shape, linear_deflection_mm, False, angular_deflection_rad, True)
    mesh.Perform()
    return linear_deflection_mm


def _location_matrix(loc: TopLoc_Location):
    """Returns (R (3,3), t (3,)) for a TopLoc_Location, or None if it's the
    identity (the common case, which we then skip entirely)."""
    if loc.IsIdentity():
        return None
    trsf = loc.Transformation()
    R = np.array([[trsf.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)], dtype=float)
    t = np.array([trsf.Value(i, 4) for i in (1, 2, 3)], dtype=float)
    return R, t


def _face_nodes(triangulation, n_nodes: int) -> np.ndarray:
    """Reads a face's triangulation nodes into an (n,3) array, using the
    cheapest accessor the installed OCP build offers.

    This is a hot path: the naive version calls Node(i).Transformed(trsf)
    then .X()/.Y()/.Z(), which is ~6 pybind11 boundary crossings and two
    gp_Pnt allocations *per vertex*. On a part with a million mesh vertices
    that alone dominated load time. We avoid the per-point Transformed()
    (the location transform is applied to the whole array in numpy by the
    caller) and prefer Coord(), which returns all three components in a
    single call.
    """
    out = np.empty((n_nodes, 3), dtype=float)
    node = triangulation.Node
    try:
        # Fast path: one call per vertex, returns all three coordinates.
        for i in range(n_nodes):
            out[i] = node(i + 1).Coord()
        return out
    except (AttributeError, TypeError):
        pass
    for i in range(n_nodes):
        p = node(i + 1)
        out[i] = (p.X(), p.Y(), p.Z())
    return out


def _face_triangle_indices(triangulation, n_tri: int) -> np.ndarray:
    """Reads a face's triangle node indices into an (m,3) 1-based array."""
    out = np.empty((n_tri, 3), dtype=np.int64)
    triangle = triangulation.Triangle
    try:
        for i in range(n_tri):
            out[i] = triangle(i + 1).Get()
        return out
    except (AttributeError, TypeError):
        pass
    for i in range(n_tri):
        tri = triangle(i + 1)
        out[i] = (tri.Value(1), tri.Value(2), tri.Value(3))
    return out


def face_map_of(shape: TopoDS_Shape) -> TopTools_IndexedMapOfShape:
    """Stable, indexed list of the shape's faces. Sharing one map between
    the mesh pass and the face scan is what lets per-face results computed
    in one be looked up in the other."""
    face_map = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, face_map)
    return face_map


def extract_mesh(shape: TopoDS_Shape, face_map=None, with_face_stats: bool = False):
    """Single-pass extraction of (points (N,3), triangles (M,3)).

    Replaces the old pair of functions that each walked every face and
    every vertex separately -- the traversal now happens once and the
    location transform is applied per face as one numpy matrix multiply
    instead of per vertex through the bindings.

    With `with_face_stats=True` it also returns a per-face
    (area, centroid) table aligned to `face_map`. That matters for speed:
    the face scan needs an area for every face and a centroid for every
    planar one, and computing them here costs almost nothing because the
    nodes are already in hand. Reading the triangulation a second time in
    the scan -- which is what happened before -- doubled the single most
    expensive Python-level loop in the program.

    Triangle winding is corrected for TopAbs_REVERSED faces so surface
    normals (and therefore shading) come out consistent.
    """
    if face_map is None:
        face_map = face_map_of(shape)

    point_blocks: list[np.ndarray] = []
    triangle_blocks: list[np.ndarray] = []
    face_stats: list[tuple] = []
    offset = 0

    for i in range(1, face_map.Extent() + 1):
        face = TopoDS.Face_s(face_map.FindKey(i))
        loc = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, loc)
        if triangulation is None:
            face_stats.append((None, None))
            continue

        n_nodes = triangulation.NbNodes()
        n_tri = triangulation.NbTriangles()
        if n_nodes <= 0:
            face_stats.append((None, None))
            continue

        pts = _face_nodes(triangulation, n_nodes)
        mat = _location_matrix(loc)
        if mat is not None:
            R, t = mat
            pts = pts @ R.T + t
        point_blocks.append(pts)

        area = 0.0
        centroid = None
        if n_tri > 0:
            raw = _face_triangle_indices(triangulation, n_tri) - 1
            if with_face_stats:
                a = pts[raw[:, 0]]
                b = pts[raw[:, 1]]
                c = pts[raw[:, 2]]
                areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
                area = float(areas.sum())
                if area > 0:
                    centroid = (((a + b + c) / 3.0) * areas[:, None]).sum(axis=0) / area

            tris = raw + offset
            if face.Orientation() == TopAbs_REVERSED:
                tris = tris[:, [0, 2, 1]]
            triangle_blocks.append(tris)

        face_stats.append((area, centroid))
        offset += n_nodes

    if not point_blocks:
        raise ValueError("Shape produced no triangulation; tessellate() may not have been called.")

    points = np.vstack(point_blocks)
    triangles = (np.vstack(triangle_blocks) if triangle_blocks
                 else np.zeros((0, 3), dtype=np.int64))
    if with_face_stats:
        return points, triangles, face_stats
    return points, triangles


def face_area_from_triangulation(face) -> float | None:
    """Area of one face, summed from its existing triangulation.

    Why this exists: `BRepGProp.SurfaceProperties_s` computes an exact area
    by numerically integrating over the face's parametric domain. That is
    accurate but comparatively slow, and calling it once per face was a
    major cost on parts with many thousands of faces. The triangulation has
    already been built for display, so summing triangle areas is nearly
    free.

    Accuracy: a chordal mesh sits slightly inside a convex surface, so this
    very slightly UNDER-estimates curved faces (planar faces are exact).
    That's fine for bulk statistics like the analytic-vs-freeform split or
    the undercut flagged area. It is NOT used for hole wall areas, which
    are a headline number and stay on the exact integrator -- see
    `face_scan.analyze_faces`.

    Returns None if the face has no triangulation.
    """
    loc = TopLoc_Location()
    triangulation = BRep_Tool.Triangulation_s(face, loc)
    if triangulation is None:
        return None
    n_nodes = triangulation.NbNodes()
    n_tri = triangulation.NbTriangles()
    if n_nodes == 0 or n_tri == 0:
        return 0.0

    pts = _face_nodes(triangulation, n_nodes)
    # A TopLoc_Location is a rigid motion, which preserves areas, so the
    # transform can be skipped entirely here.
    idx = _face_triangle_indices(triangulation, n_tri) - 1
    a = pts[idx[:, 0]]
    b = pts[idx[:, 1]]
    c = pts[idx[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())


def face_area_and_centroid_from_triangulation(face):
    """(area, centroid) for one face from its existing triangulation.

    The centroid is the area-weighted mean of the triangle centroids. It's
    needed to tell, for example, a hole's flat bottom from an unrelated
    planar face that merely happens to sit at the same height -- a plane
    equation alone can't distinguish them.

    Returns (None, None) if the face has no triangulation.
    """
    loc = TopLoc_Location()
    triangulation = BRep_Tool.Triangulation_s(face, loc)
    if triangulation is None:
        return None, None
    n_nodes = triangulation.NbNodes()
    n_tri = triangulation.NbTriangles()
    if n_nodes == 0 or n_tri == 0:
        return 0.0, None

    pts = _face_nodes(triangulation, n_nodes)
    mat = _location_matrix(loc)
    if mat is not None:
        R, t = mat
        pts = pts @ R.T + t

    idx = _face_triangle_indices(triangulation, n_tri) - 1
    a = pts[idx[:, 0]]
    b = pts[idx[:, 1]]
    c = pts[idx[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = float(areas.sum())
    if total <= 0:
        return 0.0, None
    centroids = (a + b + c) / 3.0
    centroid = (centroids * areas[:, None]).sum(axis=0) / total
    return total, centroid


def get_mesh_vertices(shape: TopoDS_Shape) -> np.ndarray:
    """Backwards-compatible wrapper; prefer `extract_mesh` to avoid walking
    the shape twice."""
    return extract_mesh(shape)[0]


def get_mesh_triangles(shape: TopoDS_Shape) -> tuple[np.ndarray, np.ndarray]:
    """Backwards-compatible wrapper; prefer `extract_mesh`."""
    return extract_mesh(shape)
