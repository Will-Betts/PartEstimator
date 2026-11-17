"""Turning geometry: what a lathe sees when it looks at the part.

A lathe does not care about pockets and walls. It cares about the PROFILE
the part sweeps as it spins -- the largest radius at every station along
the spindle axis -- and about which faces it can actually generate by
turning. Everything else (flats, cross holes, slots, off-axis holes) is
live-tool work on a turn-mill.

Measured here, all in millimetres:

  * the turning axis -- the axis of the minimum enclosing cylinder, which
    the stock model already fits;
  * the TURNED ENVELOPE: the solid of revolution of the part's maximum
    radius at each axial station. Stock minus envelope is OD roughing;
    envelope minus part is what the turning tools cannot reach (bores are
    split out separately, the rest is live-tool milling);
  * turned AREA, split OD / ID / face: faces that are surfaces of
    revolution about the turning axis (coaxial cylinders, cones, spheres,
    tori, surfaces of revolution) and planes square to it. These are
    finished by turning; the rest of the area is finished by live tools;
  * AXIAL holes: holes on the turning axis are drilled (and bored) by the
    turret with the part spinning; every other hole is a live-tool hole.

The envelope comes from the tessellated mesh, densely resampled -- a long
cylinder is often two triangles spanning its whole length, so the triangle
vertices alone would leave the middle of the profile empty.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

AXIS_ANGLE_TOL_RAD = math.radians(1.0)
PROFILE_BINS = 240


@dataclass
class TurningGeometry:
    axis_point_mm: np.ndarray
    axis_direction: np.ndarray
    length_mm: float
    max_diameter_mm: float
    envelope_volume_mm3: float
    # Mean diameter at which OD stock is removed, weighted by the volume
    # removed at each station -- the diameter the roughing rpm is set at.
    mean_od_removal_diameter_mm: float
    turned_od_area_mm2: float = 0.0
    turned_id_area_mm2: float = 0.0
    turned_face_area_mm2: float = 0.0
    other_area_mm2: float = 0.0
    total_area_mm2: float = 0.0
    # (diameter_mm, depth_mm) for every hole on the turning axis.
    axial_holes: list = field(default_factory=list)
    axial_hole_volume_mm3: float = 0.0
    off_axis_hole_count: int = 0
    # The radial profile, kept for display and tests: (z_mm, r_max_mm).
    profile_z_mm: np.ndarray | None = None
    profile_r_mm: np.ndarray | None = None

    @property
    def turned_area_mm2(self) -> float:
        return self.turned_od_area_mm2 + self.turned_id_area_mm2 + self.turned_face_area_mm2

    @property
    def turned_fraction(self) -> float:
        """Share of the surface a lathe generates.

        Zero when there is no turned OD at all: a plate's big faces are square
        to its enclosing-cylinder axis and COULD be faced, but a part with no
        round outside is not lathe work.
        """
        if self.total_area_mm2 <= 0 or self.turned_od_area_mm2 <= 0:
            return 0.0
        return self.turned_area_mm2 / self.total_area_mm2


def analyze_turning(shape, cylinder, points_mm, triangles, holes=None) -> TurningGeometry | None:
    """Turning view of the part, or None when no enclosing cylinder exists."""
    if cylinder is None or cylinder.axis_direction is None:
        return None
    axis = np.asarray(cylinder.axis_direction, float)
    axis = axis / (np.linalg.norm(axis) or 1.0)
    origin = np.asarray(cylinder.axis_point_mm, float)
    # The enclosing-cylinder fit is a numerical search and lands a fraction
    # of a degree off. Snap to the real turning axis: the axis line carrying
    # the most surface-of-revolution area, among those near the fit.
    snapped = _dominant_axis(shape, origin, axis, float(cylinder.radius_mm))
    if snapped is not None:
        origin, axis = snapped

    z, r = _sample_profile(points_mm, triangles, origin, axis)
    if z.size == 0:
        return None
    z0, z1 = float(z.min()), float(z.max())
    length = max(z1 - z0, 1e-9)
    edges = np.linspace(z0, z1, PROFILE_BINS + 1)
    idx = np.clip(np.searchsorted(edges, z, side="right") - 1, 0, PROFILE_BINS - 1)
    r_max = np.full(PROFILE_BINS, -1.0)
    np.maximum.at(r_max, idx, r)
    r_max = _fill_gaps(r_max)
    dz = length / PROFILE_BINS
    envelope = float(np.sum(math.pi * r_max ** 2) * dz)

    r_stock = float(r_max.max())
    removed = np.maximum(r_stock ** 2 - r_max ** 2, 0.0)
    if removed.sum() > 0:
        mean_d = float(np.sum(removed * (r_stock + r_max)) / removed.sum())
    else:
        mean_d = 2.0 * r_stock

    geo = TurningGeometry(
        axis_point_mm=origin, axis_direction=axis, length_mm=length,
        max_diameter_mm=2.0 * r_stock, envelope_volume_mm3=envelope,
        mean_od_removal_diameter_mm=mean_d,
        profile_z_mm=(edges[:-1] + edges[1:]) / 2.0, profile_r_mm=r_max,
    )
    _classify_faces(shape, origin, axis, r_stock, geo)
    _split_holes(holes or [], origin, axis, r_stock, geo)
    return geo


# ---------------------------------------------------------------------------


def _sample_profile(points, triangles, origin, axis):
    """(axial position, radius) for a dense resampling of the mesh edges."""
    pts = np.asarray(points, float)
    tris = np.asarray(triangles, int)
    if pts.size == 0 or tris.size == 0:
        return np.zeros(0), np.zeros(0)
    rel = pts - origin
    zall = rel @ axis
    span = float(zall.max() - zall.min()) or 1.0
    target = span / PROFILE_BINS          # one sample per bin along any edge

    # Sample along triangle EDGES, not across faces. On a cylinder or cone
    # the tessellation's long edges are generator lines lying exactly on the
    # surface, while a point inside a triangle sits on a chord, inside the
    # true radius -- which made the envelope come out smaller than the part.
    samples = [pts]
    tri_edges = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    tri_edges = np.unique(np.sort(tri_edges, axis=1), axis=0)
    a = pts[tri_edges[:, 0]]
    b = pts[tri_edges[:, 1]]
    n_sub = np.clip(np.ceil(np.linalg.norm(b - a, axis=1) / target).astype(int), 1, 400)
    for n in np.unique(n_sub):
        if n <= 1:
            continue
        sel = n_sub == n
        t = np.linspace(0.0, 1.0, n + 1)[1:-1]
        seg = a[sel][:, None, :] + t[None, :, None] * (b[sel] - a[sel])[:, None, :]
        samples.append(seg.reshape(-1, 3))
    p = np.vstack(samples) - origin
    z = p @ axis
    radial = p - np.outer(z, axis)
    return z, np.linalg.norm(radial, axis=1)


def _fill_gaps(r):
    """Empty bins take the larger of their nearest filled neighbours."""
    r = r.copy()
    filled = np.where(r >= 0)[0]
    if filled.size == 0:
        return np.zeros_like(r)
    for i in np.where(r < 0)[0]:
        left = filled[filled < i]
        right = filled[filled > i]
        cands = []
        if left.size:
            cands.append(r[left[-1]])
        if right.size:
            cands.append(r[right[0]])
        r[i] = max(cands)
    return r


def _dominant_axis(shape, origin, axis, radius_mm):
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepGProp import BRepGProp
    from OCP.GeomAbs import GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Torus
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    lines = []   # [direction, point, area]
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        exp.Next()
        ad = BRepAdaptor_Surface(face, True)
        kind = ad.GetType()
        if kind == GeomAbs_Cylinder:
            ax = ad.Cylinder().Axis()
        elif kind == GeomAbs_Cone:
            ax = ad.Cone().Axis()
        elif kind == GeomAbs_Torus:
            ax = ad.Torus().Axis()
        else:
            continue
        d = np.array([ax.Direction().X(), ax.Direction().Y(), ax.Direction().Z()])
        if abs(float(d @ axis)) < math.cos(math.radians(10.0)):
            continue
        if d @ axis < 0:
            d = -d
        p = np.array([ax.Location().X(), ax.Location().Y(), ax.Location().Z()])
        rel = p - origin
        if np.linalg.norm(rel - (rel @ axis) * axis) > 0.2 * max(radius_mm, 1.0):
            continue
        g = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, g)
        area = abs(g.Mass())
        for entry in lines:
            if _coaxial(d, p, entry[1], entry[0], 0.05 + 1e-3 * radius_mm):
                entry[2] += area
                break
        else:
            lines.append([d, p, area])
    if not lines:
        return None
    best = max(lines, key=lambda e: e[2])
    return best[1], best[0] / np.linalg.norm(best[0])


def _coaxial(direction, location, origin, axis, tol_mm) -> bool:
    d = np.asarray(direction, float)
    d = d / (np.linalg.norm(d) or 1.0)
    if np.linalg.norm(np.cross(d, axis)) > math.sin(AXIS_ANGLE_TOL_RAD):
        return False
    rel = np.asarray(location, float) - origin
    off = rel - (rel @ axis) * axis
    return float(np.linalg.norm(off)) <= tol_mm


def _classify_faces(shape, origin, axis, r_stock, geo: TurningGeometry) -> None:
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepGProp import BRepGProp
    from OCP.BRepLProp import BRepLProp_SLProps
    from OCP.GeomAbs import (GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Plane,
                             GeomAbs_Sphere, GeomAbs_SurfaceOfRevolution,
                             GeomAbs_Torus)
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    tol = 0.05 + 1e-3 * r_stock

    def xyz(p):
        return np.array([p.X(), p.Y(), p.Z()])

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        exp.Next()
        g = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, g)
        area = abs(g.Mass())
        geo.total_area_mm2 += area
        ad = BRepAdaptor_Surface(face, True)
        kind = ad.GetType()

        turned = False
        if kind == GeomAbs_Plane:
            n = ad.Plane().Axis().Direction()
            if np.linalg.norm(np.cross(xyz(n), axis)) <= math.sin(AXIS_ANGLE_TOL_RAD):
                geo.turned_face_area_mm2 += area
                continue
        elif kind in (GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Torus,
                      GeomAbs_SurfaceOfRevolution):
            if kind == GeomAbs_Cylinder:
                ax = ad.Cylinder().Axis()
            elif kind == GeomAbs_Cone:
                ax = ad.Cone().Axis()
            elif kind == GeomAbs_Torus:
                ax = ad.Torus().Axis()
            else:
                ax = ad.AxeOfRevolution()
            turned = _coaxial(xyz(ax.Direction()), xyz(ax.Location()), origin, axis, tol)
        elif kind == GeomAbs_Sphere:
            turned = _coaxial(axis, xyz(ad.Sphere().Location()), origin, axis, tol)

        if not turned:
            geo.other_area_mm2 += area
            continue

        # OD or ID: does the outward normal point away from the axis?
        u = 0.5 * (ad.FirstUParameter() + ad.LastUParameter())
        v = 0.5 * (ad.FirstVParameter() + ad.LastVParameter())
        props = BRepLProp_SLProps(ad, u, v, 1, 1e-6)
        outward = 1.0
        if props.IsNormalDefined():
            nn = xyz(props.Normal())
            if face.Orientation() == TopAbs_REVERSED:
                nn = -nn
            rel = xyz(ad.Value(u, v)) - origin
            radial = rel - (rel @ axis) * axis
            outward = float(nn @ radial)
        if outward >= 0:
            geo.turned_od_area_mm2 += area
        else:
            geo.turned_id_area_mm2 += area


def _split_holes(holes, origin, axis, r_stock, geo: TurningGeometry) -> None:
    tol = 0.05 + 1e-3 * r_stock
    for h in holes:
        d = getattr(h, "axis_direction", None)
        p = getattr(h, "axis_point", None)
        if d is None or p is None:
            continue
        if _coaxial(d, p, origin, axis, tol):
            dia = float(getattr(h, "major_diameter_mm", 0.0) or h.nominal_diameter_mm)
            depth = float(getattr(h, "total_depth_mm", 0.0))
            geo.axial_holes.append((float(h.nominal_diameter_mm), dia, depth))
            geo.axial_hole_volume_mm3 += math.pi * (dia / 2.0) ** 2 * depth
        else:
            geo.off_axis_hole_count += 1
