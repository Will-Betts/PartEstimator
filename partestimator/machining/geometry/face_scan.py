"""Single-pass face analysis, built around strictly LOCAL queries.

Why this was rewritten
----------------------
The first version decided whether a cylindrical face was a bore or a boss
by stepping a probe point toward the axis and asking
`BRepClass3d_SolidClassifier` whether that point was inside the solid.
That is correct, but the classifier is a *global* query: it ray-casts
against the entire shape. Called once per round face it costs roughly

    O(round_faces x total_faces)

so a part with 5,000 bores among 20,000 faces did on the order of a
hundred million face-intersection tests. That is what made the face scan
dominate load time and, on big assemblies, appear to hang outright.

The fix is to stop asking a global question. Concavity is a purely local
property of the surface plus the face's orientation flag:

  * A `gp_Cylinder`'s natural parametric normal points radially OUTWARD
    from its axis (verified: dP/du x dP/dv = R * radial_unit).
  * A `gp_Torus`'s natural normal points away from the tube centreline.
  * In OCCT a FORWARD face's normal points OUT of the material.

Therefore:

    FORWARD  cylinder -> material inside  -> shaft / boss   (convex)
    REVERSED cylinder -> material outside -> bore / hole    (CONCAVE)

which is one enum comparison per face instead of a ray cast. Same logic
for tori (external round-over vs. internal fillet).

Everything else in the pass was made local too:

  * `BRepAdaptor_Surface(face, True)` computes the trimmed restriction by
    analysing the face's wires. We only need the surface type and the UV
    box, so the adaptor is built with restriction OFF and the bounds come
    from `BRepTools.UVBounds_s`, which is cheap.
  * Face areas come from the existing triangulation. Cylindrical faces
    additionally get an analytic cross-check (area = R * du * dv, exact
    for an untrimmed parametric rectangle), so bore areas stay accurate
    without invoking the numerical integrator at all.

The old classifier-based routines remain in `round_surfaces` as the
reference implementation, and `selftest.py` asserts the fast path agrees
with them. `robust=True` restores the classifier if you ever hit geometry
where the orientation convention doesn't hold (non-solid shells, badly
exported compounds).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepGProp import BRepGProp
from OCP.BRepTools import BRepTools
from OCP.GeomAbs import GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Torus
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp
from OCP.TopoDS import TopoDS, TopoDS_Shape
from OCP.TopTools import TopTools_IndexedMapOfShape

from . import features as _features
from . import holes as _holes
from . import mesh_utils
from . import round_surfaces as _round
from .normals import sampled_outward_normal

# A trimmed cylindrical face's true area can be far below the analytic
# R*du*dv of its parameter box. If the two disagree by more than this, the
# face is genuinely trimmed and we trust the triangulated measurement.
ANALYTIC_AREA_AGREEMENT = 0.10


class ScanCancelled(RuntimeError):
    """Raised when a caller's cancel callback asks the scan to stop."""


@dataclass(eq=False)
class ConeFace:
    """A conical face. Cones co-axial with a bore are how drill points
    (blind-hole bottoms) and countersinks show up in a STEP model."""
    axis_point: np.ndarray
    axis_direction: np.ndarray
    radius_min_mm: float
    radius_max_mm: float
    axial_start: float
    axial_end: float
    half_angle_rad: float
    area_mm2: float


@dataclass(eq=False)
class PlaneFace:
    """A planar face: plane equation, area, and centroid.

    The centroid matters for hole recognition -- a flat-bottomed blind hole
    is capped by a small disc, and the plane equation alone can't tell that
    disc apart from an unrelated face lying at the same height."""
    normal: np.ndarray
    offset_mm: float          # signed distance from origin along `normal`
    area_mm2: float
    centroid: np.ndarray = None


@dataclass
class FaceScanResult:
    holes: list
    fillets: list
    surface_split: dict
    undercut: _features.UndercutResult
    face_count: int = 0
    cones: list = None
    planes: list = None

    def __post_init__(self):
        if self.cones is None:
            self.cones = []
        if self.planes is None:
            self.planes = []


def _is_concave(face) -> bool:
    """Bore/internal-fillet (True) vs. boss/external-round (False), from the
    face orientation flag alone. See the module docstring for why this is
    equivalent to the old point-in-solid probe."""
    return face.Orientation() == TopAbs_REVERSED


def analyze_faces(shape: TopoDS_Shape,
                   undercut_directions: np.ndarray = _features.PRINCIPAL_DIRECTIONS,
                   undercut_grid: int = _features.UNDERCUT_GRID,
                   do_undercut: bool = False,
                   exact_areas: bool = False,
                   robust: bool = False,
                   progress=None,
                   should_cancel=None,
                   face_map=None,
                   face_stats=None) -> FaceScanResult:
    """One traversal producing holes/fillets, the surface-type split and,
    optionally, the undercut check.

    progress(done, total) is called periodically so long scans are visible.
    should_cancel() returning True aborts with ScanCancelled.
    """
    # Faces are collected up front so we know the total for progress
    # reporting (and so a stalled scan is obviously a scan, not a freeze).
    if face_map is None:
        face_map = TopTools_IndexedMapOfShape()
        TopExp.MapShapes_s(shape, TopAbs_FACE, face_map)
    total_faces = face_map.Extent()
    # Per-face (area, centroid) already computed during the mesh pass. When
    # absent we fall back to reading the triangulation here, which costs a
    # second full walk of every node -- avoid it on the hot path.
    have_stats = face_stats is not None and len(face_stats) >= total_faces

    classifier = BRepClass3d_SolidClassifier(shape) if robust else None
    props = GProp_GProps()

    def exact_area(face) -> float:
        BRepGProp.SurfaceProperties_s(face, props)
        return props.Mass()

    cyl_faces: list = []
    fillets: list = []
    cones: list = []
    planes: list = []
    area_by_type: dict[int, float] = defaultdict(float)
    undercut = _features.UndercutResult(directions_tested=len(undercut_directions))
    report_every = max(1, total_faces // 50)

    for i in range(1, total_faces + 1):
        if should_cancel is not None and (i % 256 == 0) and should_cancel():
            raise ScanCancelled(f"cancelled after {i:,} of {total_faces:,} faces")
        if progress is not None and (i % report_every == 0 or i == total_faces):
            progress(i, total_faces)

        face = TopoDS.Face_s(face_map.FindKey(i))
        # restriction=False: we don't need the adaptor to analyse the face's
        # wires, only to tell us the underlying surface type.
        adaptor = BRepAdaptor_Surface(face, False)
        surf_type = adaptor.GetType()
        undercut.total_face_count += 1

        # Cheap parametric box of the trimmed face.
        umin, umax, vmin, vmax = BRepTools.UVBounds_s(face)

        tri_area = None
        tri_centroid = None
        if not exact_areas:
            if have_stats:
                tri_area, tri_centroid = face_stats[i - 1]
            else:
                tri_area = mesh_utils.face_area_from_triangulation(face)

        if surf_type == GeomAbs_Cylinder:
            cyl = adaptor.Cylinder()
            radius = cyl.Radius()
            sweep = umax - umin
            depth = abs(vmax - vmin)

            if exact_areas:
                area = exact_area(face)
            else:
                # Analytic area of the parameter rectangle; exact when the
                # face isn't trimmed, which is the common case for a bore.
                analytic = radius * sweep * depth
                if tri_area is None:
                    area = analytic
                elif analytic > 0 and abs(tri_area - analytic) / analytic <= ANALYTIC_AREA_AGREEMENT:
                    area = analytic
                else:
                    area = tri_area

            concave = (_round._is_concave_cylinder(BRepAdaptor_Surface(face, True), classifier)
                       if robust else _is_concave(face))
            if concave:
                axis = cyl.Axis()
                loc = axis.Location()
                dirv = axis.Direction()
                direction = np.array([dirv.X(), dirv.Y(), dirv.Z()], dtype=float)
                direction /= np.linalg.norm(direction)
                xdir = cyl.Position().XDirection()
                cyl_faces.append(_round._ConcaveCylFace(
                    radius_mm=radius,
                    sweep_rad=sweep,
                    depth_mm=depth,
                    area_mm2=area,
                    axis_point=np.array([loc.X(), loc.Y(), loc.Z()], dtype=float),
                    axis_direction=direction,
                    axial_start=float(min(vmin, vmax)),
                    axial_end=float(max(vmin, vmax)),
                    sweep_start=float(min(umin, umax)),
                    x_dir=np.array([xdir.X(), xdir.Y(), xdir.Z()], dtype=float),
                ))
        elif surf_type == GeomAbs_Torus:
            area = exact_area(face) if (exact_areas or tri_area is None) else tri_area
            concave = (_round._is_concave_torus(BRepAdaptor_Surface(face, True), classifier)
                       if robust else _is_concave(face))
            if concave:
                fillets.append(_round.FilletFace(
                    radius_mm=adaptor.Torus().MinorRadius(),
                    area_mm2=area,
                    surface_type="torus",
                ))
        elif surf_type == GeomAbs_Cone:
            area = exact_area(face) if (exact_areas or tri_area is None) else tri_area
            cone = adaptor.Cone()
            axis = cone.Axis()
            loc = axis.Location()
            dirv = axis.Direction()
            direction = np.array([dirv.X(), dirv.Y(), dirv.Z()], dtype=float)
            direction /= np.linalg.norm(direction)
            # gp_Cone parametrisation (OCCT):
            #   P(u,v) = Loc + (R + v*sin(Ang))*(cos(u)*X + sin(u)*Y)
            #                + v*cos(Ang)*Z
            # so radius(v) = RefRadius + v*sin(Ang) and the axial coordinate
            # is v*cos(Ang) -- NOT v itself, and not tan(Ang).
            half_angle = cone.SemiAngle()
            ref_r = cone.RefRadius()
            sin_a = np.sin(half_angle)
            cos_a = np.cos(half_angle)
            r_a = ref_r + vmin * sin_a
            r_b = ref_r + vmax * sin_a
            z_a = vmin * cos_a
            z_b = vmax * cos_a
            cones.append(ConeFace(
                axis_point=np.array([loc.X(), loc.Y(), loc.Z()], dtype=float),
                axis_direction=direction,
                radius_min_mm=float(min(abs(r_a), abs(r_b))),
                radius_max_mm=float(max(abs(r_a), abs(r_b))),
                axial_start=float(min(z_a, z_b)),
                axial_end=float(max(z_a, z_b)),
                half_angle_rad=float(half_angle),
                area_mm2=area,
            ))
        elif surf_type == GeomAbs_Plane:
            if have_stats and not exact_areas:
                plane_area, centroid = tri_area, tri_centroid
            else:
                plane_area, centroid = mesh_utils.face_area_and_centroid_from_triangulation(face)
            if exact_areas or plane_area is None:
                area = exact_area(face)
            else:
                area = plane_area
            plane = adaptor.Plane()
            ax = plane.Axis()
            loc = ax.Location()
            dirv = ax.Direction()
            normal = np.array([dirv.X(), dirv.Y(), dirv.Z()], dtype=float)
            normal /= np.linalg.norm(normal)
            origin = np.array([loc.X(), loc.Y(), loc.Z()], dtype=float)
            planes.append(PlaneFace(
                normal=normal,
                offset_mm=float(np.dot(normal, origin)),
                area_mm2=area,
                centroid=centroid,
            ))
        else:
            area = exact_area(face) if (exact_areas or tri_area is None) else tri_area

        area_by_type[surf_type] += area

        # --- undercut / tool-access check (optional: costly, low value) ---
        if not do_undercut:
            continue
        u_adaptor = BRepAdaptor_Surface(face, True)
        effective_grid = 1 if surf_type == GeomAbs_Plane else undercut_grid
        us = np.linspace(umin, umax, effective_grid + 2)[1:-1]
        vs = np.linspace(vmin, vmax, effective_grid + 2)[1:-1]

        alive = np.ones(len(undercut_directions), dtype=bool)
        any_normal = False
        for u in us:
            for v in vs:
                n = sampled_outward_normal(face, u_adaptor, u, v)
                if n is None:
                    continue
                any_normal = True
                alive &= (undercut_directions @ n) > _features.UNDERCUT_NORMAL_EPSILON
                if not alive.any():
                    break
            if not alive.any():
                break
        if any_normal and not alive.any():
            undercut.flagged_face_count += 1
            undercut.flagged_area_mm2 += area

    # Two stages: merge seam-split faces into per-diameter bores, then
    # assemble co-axial bores + cones + capping planes into whole holes
    # (counterbores, countersinks, blind terminations). See geometry.holes.
    bores, cyl_fillets = _round._merge_coaxial_bores(cyl_faces)
    fillets.extend(cyl_fillets)
    holes = _holes.build_hole_features(bores, cones, planes)

    analytic = sum(a for t, a in area_by_type.items() if t in _features._ANALYTIC_TYPES)
    freeform = sum(a for t, a in area_by_type.items() if t in _features._FREEFORM_TYPES)
    other = sum(a for t, a in area_by_type.items()
                if t not in _features._ANALYTIC_TYPES and t not in _features._FREEFORM_TYPES)
    surface_split = {
        "analytic_area_mm2": analytic,
        "freeform_area_mm2": freeform,
        "other_area_mm2": other,
        "by_type_mm2": {_features._TYPE_NAMES.get(t, f"Surface type {t}"): a
                        for t, a in area_by_type.items()},
    }

    return FaceScanResult(
        holes=holes,
        fillets=fillets,
        surface_split=surface_split,
        undercut=undercut,
        face_count=total_faces,
        cones=cones,
        planes=planes,
    )
