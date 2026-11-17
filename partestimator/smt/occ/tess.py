"""Prepare the solid for display: real face boundaries, not a triangle soup.

WHAT CHANGED, AND WHY THE OLD ONE LOOKED LIKE A MESH
----------------------------------------------------
v6 tessellated the whole solid at a deflection of roughly one four-hundredth of
the part's diagonal, with a 20 degree angular tolerance, and painted the
triangles. On a 300 mm bracket that is a 0.9 mm chord error and eighteen facets
around a full hole -- so it looked exactly like what it was, a coarse mesh, and
painting several thousand individual triangles is what made it slow.

Both problems have the same root: the renderer was working in triangles when
the part is made of FACES, and a sheet metal part's faces are almost entirely
planes and cylinders.

So this module now hands the viewer what a CAD display actually needs:

  * for every PLANAR face, its real boundary loops, sampled off the B-rep
    curves. A plane needs no triangles at all -- it is one filled polygon with
    holes, and its outline is as smooth as the curves it came from.
  * for every CURVED face, a fine triangulation used only for silhouette and
    culling. It is shaded by a gradient derived from the analytic surface, so a
    cylinder reads as round no matter how coarsely it happens to be meshed.
  * the real B-rep edges, each tagged with the faces it borders, so edges
    belonging only to faces turned away from you can be dropped.

The result is a few dozen paths per frame instead of several thousand, and no
facet is visible anywhere.

STABILITY, AND THE ACCESS VIOLATION
-----------------------------------
v6 crashed with 0xC0000005 on opening a file. Three things in the old version
of this file could produce that, and all three are gone:

  1. `isInParallel=True` on the mesher. OCCT's parallel meshing spins up its own
     worker pool, and this is already being called from a Qt worker thread in a
     Python process. Parallelism was buying nothing here -- a sheet metal part
     meshes in milliseconds -- and it is the single most likely cause.
  2. `BRepAdaptor_Curve` on an edge with no 3D curve. Degenerate edges were
     checked for, but an edge carrying only a pcurve is a different thing and
     also has no curve to sample. Now checked explicitly, before the adaptor is
     constructed rather than after.
  3. Re-meshing the same shape up to four times to hit a triangle budget, while
     holding results from earlier passes. The budget is gone: with per-face
     drawing the triangle count no longer drives frame time.

Every per-face operation is also wrapped, so one unusual face degrades that
face rather than taking down the process.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..diag import LOGGER

from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.Bnd import Bnd_Box
from OCP.GCPnts import GCPnts_QuasiUniformDeflection
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.TopTools import (
    TopTools_IndexedDataMapOfShapeListOfShape,
    TopTools_IndexedMapOfShape,
)

from .topo import edge_has_curve, face_index_map, face_loops_safe


class TessellationError(Exception):
    pass


#: Chord error as a fraction of the part diagonal. Only affects curved faces --
#: planar outlines come from the real curves and are exact regardless, and on a
#: sheet metal part that is most of the model.
LINEAR_QUALITY = 1500.0

#: Angular deflection, radians. About 8 degrees, so a full hole comes out with
#: 45 segments.
#:
#: v11 used 4 degrees and 90 segments per hole, which is finer than anything is
#: visible at and cost roughly double the triangles on every curved face. The
#: silhouette of a curve is what the eye judges, and 45 segments is already past
#: the point where a corner can be seen on a hole a few millimetres across.
ANGULAR_DEFLECTION = 0.14


@dataclass
class FaceRender:
    """One B-rep face, in whatever form draws it best."""

    index: int
    kind: str = "other"
    area: float = 0.0

    #: Planes: outward normal and the real boundary loops, outer wire first.
    normal: np.ndarray | None = None
    loops: list[np.ndarray] = field(default_factory=list)

    #: Cylinders and cones: the analytic surface, for gradient shading.
    axis_point: np.ndarray | None = None
    axis_dir: np.ndarray | None = None
    radius: float = 0.0
    convex: bool | None = None

    #: Triangles for this face, as row indices into `Scene.triangles`.
    tri_lo: int = 0
    tri_hi: int = 0

    @property
    def n_triangles(self) -> int:
        return self.tri_hi - self.tri_lo

    @property
    def drawable_as_polygon(self) -> bool:
        """A plane with a usable outline needs no triangles to draw."""
        return (
            self.kind == "plane"
            and self.normal is not None
            and bool(self.loops)
            and len(self.loops[0]) >= 3
        )


@dataclass
class Scene:
    """Everything the 3D view draws, in millimetres, in model coordinates."""

    vertices: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    triangles: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.int64)
    )
    faces: list[FaceRender] = field(default_factory=list)

    #: (polyline, owning face indices) for every real edge of the solid.
    edges: list[tuple[np.ndarray, tuple[int, ...]]] = field(default_factory=list)

    deflection: float = 0.0

    def __len__(self) -> int:
        return len(self.faces)

    @property
    def is_empty(self) -> bool:
        return not self.faces

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        chunks = [self.vertices] if len(self.vertices) else []
        chunks += [loop for f in self.faces for loop in f.loops if len(loop)]
        if not chunks:
            return np.zeros(3), np.zeros(3)
        allpts = np.vstack(chunks)
        return allpts.min(axis=0), allpts.max(axis=0)

    def size(self) -> np.ndarray:
        lo, hi = self.bounds()
        return hi - lo

    def centre(self) -> np.ndarray:
        lo, hi = self.bounds()
        return 0.5 * (lo + hi)

    def diagonal(self) -> float:
        return float(np.linalg.norm(self.size())) or 1.0

    def triangle_normals(self) -> np.ndarray:
        if len(self.triangles) == 0:
            return np.zeros((0, 3))
        v = self.vertices[self.triangles]
        n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        mag = np.linalg.norm(n, axis=1, keepdims=True)
        return np.divide(n, mag, out=np.zeros_like(n), where=mag > 1e-15)


def bounding_box(shape) -> tuple[np.ndarray, np.ndarray]:
    """The solid's axis-aligned bounds, from the kernel rather than the mesh."""
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box, True)
    if box.IsVoid():
        return np.zeros(3), np.zeros(3)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return (
        np.array([xmin, ymin, zmin], dtype=float),
        np.array([xmax, ymax, zmax], dtype=float),
    )


def _mesh_shape(shape, linear: float, angular: float) -> None:
    """Tessellate in place. Serial, deliberately.

    `isInParallel` is FALSE and must stay false. OCCT's parallel mesher starts
    its own thread pool, and this runs on a Qt worker thread inside a Python
    process that also holds the GIL and a Qt event loop. That combination is the
    most likely source of the 0xC0000005 access violation reported against v6.

    It buys nothing anyway: a sheet metal part is a few dozen faces and meshes
    in milliseconds. Parallelism here was cost with no benefit.
    """
    mesher = BRepMesh_IncrementalMesh(shape, linear, False, angular, False)
    mesher.Perform()
    if not mesher.IsDone():
        LOGGER.warning("tessellation did not complete cleanly")


def _face_triangles(face, base: int):
    """(vertices, triangles) for one face, wound to face outward."""
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(face, loc)
    if tri is None:
        return None, None

    trsf = loc.Transformation()
    n_nodes = tri.NbNodes()
    if n_nodes < 3:
        return None, None

    pts = np.empty((n_nodes, 3))
    for k in range(1, n_nodes + 1):
        p = tri.Node(k).Transformed(trsf)
        pts[k - 1] = (p.X(), p.Y(), p.Z())

    # A REVERSED face uses its surface with the opposite sense, so its triangles
    # wind the wrong way. Flipping here means every normal computed later points
    # out of the solid and backface culling needs no special case.
    flip = face.Orientation() == TopAbs_REVERSED
    rows = []
    for k in range(1, tri.NbTriangles() + 1):
        a, b, c = tri.Triangle(k).Get()
        if flip:
            a, c = c, a
        rows.append([base + a - 1, base + b - 1, base + c - 1])

    if not rows:
        return None, None
    return pts, np.array(rows, dtype=np.int64)


def _edge_owner_map(shape) -> TopTools_IndexedDataMapOfShapeListOfShape:
    m = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, m)
    return m


def edge_polylines(shape, deflection: float):
    """Every real edge, sampled, tagged with the faces it borders.

    The face tags are what make a cheap hidden-line approximation possible: an
    edge whose every owning face is turned away from the camera is behind the
    part and can simply not be drawn. That is not true hidden-line removal --
    it will not hide an edge behind an unrelated part of the same solid -- but
    it removes the great majority of the wrong lines for almost no work, and it
    is the difference between reading as a CAD model and reading as a wireframe.
    """
    fmap = face_index_map(shape)
    owners = _edge_owner_map(shape)

    out: list[tuple[np.ndarray, tuple[int, ...]]] = []
    for i in range(1, owners.Extent() + 1):
        edge = TopoDS.Edge_s(owners.FindKey(i))
        if not edge_has_curve(edge):
            continue

        try:
            curve = BRepAdaptor_Curve(edge)
            sampler = GCPnts_QuasiUniformDeflection(curve, deflection)
            if sampler.IsDone() and sampler.NbPoints() >= 2:
                pts = np.array([
                    [sampler.Value(k).X(), sampler.Value(k).Y(), sampler.Value(k).Z()]
                    for k in range(1, sampler.NbPoints() + 1)
                ])
            else:
                a, b = curve.FirstParameter(), curve.LastParameter()
                pts = np.array([
                    [curve.Value(a).X(), curve.Value(a).Y(), curve.Value(a).Z()],
                    [curve.Value(b).X(), curve.Value(b).Y(), curve.Value(b).Z()],
                ])
        except Exception:  # noqa: BLE001 -- one unusable edge is not fatal
            continue

        if len(pts) < 2:
            continue

        tags = []
        for f in owners.FindFromIndex(i):
            pos = fmap.FindIndex(f)
            if pos >= 1:
                tags.append(pos - 1)
        out.append((pts, tuple(tags)))

    return out


def build_scene(shape, infos=None, with_edges: bool = True,
                loop_cache: dict | None = None) -> Scene:
    """Everything the viewer needs, extracted once.

    `infos` is the detector's `describe_all` output if it has already been
    computed -- it carries the analytic surface data this needs, and there is no
    reason to interrogate every face twice.
    """
    lo, hi = bounding_box(shape)
    diag = float(np.linalg.norm(hi - lo)) or 1.0
    linear = max(diag / LINEAR_QUALITY, 1e-4)

    _mesh_shape(shape, linear, ANGULAR_DEFLECTION)

    if infos is None:
        from .topo import describe_all

        infos = describe_all(shape)
    by_index = {f.index: f for f in infos}

    fmap = face_index_map(shape)
    verts: list[np.ndarray] = []
    tris: list[np.ndarray] = []
    faces: list[FaceRender] = []
    base = 0
    tri_count = 0

    for i in range(1, fmap.Extent() + 1):
        idx = i - 1
        face = TopoDS.Face_s(fmap.FindKey(i))
        info = by_index.get(idx)

        fr = FaceRender(
            index=idx,
            kind=info.kind if info is not None else "other",
            area=info.area if info is not None else 0.0,
            normal=info.normal if info is not None else None,
            axis_point=info.axis_point if info is not None else None,
            axis_dir=info.axis_dir if info is not None else None,
            radius=info.radius if info is not None else 0.0,
            convex=info.convex if info is not None else None,
        )

        # Planes get their real outline and no triangles. This is most of a
        # sheet metal part, and it is where the CAD look comes from: the
        # boundary is drawn from the actual curves rather than from a mesh
        # approximation of them.
        if fr.kind == "plane":
            fr.loops = face_loops_safe(face, idx, deflection=linear,
                                       cache=loop_cache)

        # Every face is triangulated, planes included: the VTK view (shared
        # with the machining tab) draws triangles only. Plane loops are still
        # kept for the exact outline and the face-picking test.
        if True:
            try:
                pts, rows = _face_triangles(face, base)
            except Exception:  # noqa: BLE001 -- degrade this face, not the run
                LOGGER.debug("face %d: could not be tessellated", idx)
                pts, rows = None, None
            if pts is not None and rows is not None:
                verts.append(pts)
                tris.append(rows)
                fr.tri_lo = tri_count
                tri_count += len(rows)
                fr.tri_hi = tri_count
                base += len(pts)

        faces.append(fr)

    scene = Scene(
        vertices=np.vstack(verts) if verts else np.zeros((0, 3)),
        triangles=np.vstack(tris) if tris else np.zeros((0, 3), dtype=np.int64),
        faces=faces,
        deflection=linear,
    )

    if with_edges:
        try:
            scene.edges = edge_polylines(shape, linear)
        except Exception:  # noqa: BLE001 -- a view without edges still works
            LOGGER.warning("edge extraction failed; drawing shaded surfaces only")
            scene.edges = []

    polygons = sum(1 for f in faces if f.drawable_as_polygon)
    LOGGER.info(
        "display: %d faces (%d drawn as exact polygons), %d triangles, "
        "%d edges, chord %.4f mm",
        len(faces), polygons, len(scene.triangles), len(scene.edges), linear,
    )
    if scene.is_empty:
        raise TessellationError("the shape produced nothing to display")
    return scene
