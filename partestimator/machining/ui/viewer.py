"""3D viewer wrapper: displays the loaded part CAD-style (smoothly shaded
surface with crisp topological edges), a grey dotted wireframe of whichever
bounding box has the smaller volume, a center-of-mass marker, and
highlighted thin-wall sample points.

Built on PyVista + pyvistaqt rather than OpenCASCADE's own AIS viewer.
pythonocc-core ships a ready-made Qt/AIS viewer widget
(`OCC.Display.qtDisplay.qtViewer3d`), but that's a pythonocc-core-specific
convenience -- OCP (the pip-installable `cadquery-ocp` binding used here)
has no equivalent, and hand-rolling a native OCCT viewer inside Qt requires
platform-specific window-handle plumbing that's brittle and hard to verify
without a real Windows target to test against. Since we already tessellate
every shape into a triangle mesh for the bounding-box math
(`geometry.mesh_utils`), rendering that same mesh with PyVista is simpler,
has a stable/well-documented public API, and needs zero OpenCASCADE-viewer
code at all.

CAD-look note: any GPU draws a curved surface as triangles, so the surface
geometry is unavoidably a mesh -- but it doesn't have to *look* like one.
Rather than outlining every triangle (the old behavior, which made parts
read as faceted mesh models), the surface is drawn with no triangle edges
at all and the real B-rep edges from the STEP file are overlaid as lines
(see `geometry.edges`). Shading uses split-sharp-edges so curved regions
look smooth while genuine corners stay crisp instead of being rounded off
by vertex-normal averaging.
"""
from __future__ import annotations

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor

PART_COLOR = (0.72, 0.74, 0.78)
EDGE_COLOR = (0.10, 0.10, 0.12)   # near-black, CAD-style feature lines
MESH_EDGE_COLOR = (0.55, 0.55, 0.60)  # only used by the optional mesh overlay
BOX_COLOR = (0.45, 0.45, 0.45)  # grey, for both bounding-box types
STOCK_LINE_COLOR = (0.12, 0.40, 0.90)   # blue dashed stock outline
STOCK_FILL_COLOR = (0.55, 0.72, 0.96)   # translucent light-blue stock body
STOCK_FILL_OPACITY = 0.18
THIN_WALL_COLOR = (0.95, 0.05, 0.75)
COM_COLOR = (0.90, 0.55, 0.05)  # amber center-of-mass marker
PICK_COLOR = (0.10, 0.75, 0.35)  # green: a face picked to set the orientation
CLICK_TOLERANCE_PX = 4           # further than this is a drag, not a click
CAPTURED_AXIS_COLOR = (0.90, 0.08, 0.08)  # red axis through captured holes
CAPTURED_AXIS_WIDTH = 3.0
# How far the axis line pokes out past each end of the hole, as a fraction
# of the hole depth -- enough to be seen emerging from the part.
CAPTURED_AXIS_OVERHANG_FRACTION = 0.8
CAPTURED_AXIS_MIN_OVERHANG_MM = 2.0

EDGE_LINE_WIDTH = 2.0
BOX_LINE_WIDTH = 2.0
# Angle above which an edge counts as "sharp" for shading purposes. 30 deg
# keeps cylinder walls smooth while preserving box corners.
SHADING_FEATURE_ANGLE = 30.0
# Center-of-mass marker radius, as a fraction of the part's bounding diagonal.
COM_MARKER_RELATIVE_RADIUS = 0.012

# Dash geometry is generated manually as many short line segments rather
# than via a VTK line-stipple pattern: stipple support varies by VTK
# build/backend, whereas explicit segments render identically everywhere.
DASH_TARGET_LENGTH_FRACTION = 0.02  # dash+gap period as a fraction of the box's diagonal
DASH_DUTY = 0.55  # portion of each period that's drawn (rest is the gap)


def _line_polydata(points: np.ndarray, cells: np.ndarray) -> pv.PolyData:
    """Builds a lines-only PolyData.

    THE point of this helper: `pv.PolyData(points)` does not just store the
    points, it also creates one *vertex cell* per point. Assigning `.lines`
    afterwards adds the line cells but leaves those vertex cells in place,
    and VTK dutifully renders both -- drawing a dot at every single point of
    every line. That's the speckling along model edges and bounding-box
    dashes. Turning off the actor's vertex-visibility property does NOT fix
    it, because these are real cells in the dataset rather than the
    property-level vertex overlay. The cells have to be cleared explicitly.
    """
    poly = pv.PolyData(points)
    poly.lines = cells
    poly.verts = np.empty(0, dtype=np.int64)  # <- removes the point dots
    return poly


def _style_line_actor(actor, depth_offset: bool = False):
    """Belt-and-braces for line actors: also disable the property-level
    vertex overlay, and optionally nudge the line toward the camera in
    depth.

    The depth nudge matters because a line lying exactly on a surface loses
    the depth test against that surface roughly half the time, which is why
    edges bounding concave regions (pocket floors, bore walls, fillet
    boundaries) drop out -- the surface curves toward the camera and wins
    the z-fight. Offsetting slightly toward the viewer makes the line draw
    in front of its own surface without punching through geometry that is
    genuinely in front of it.
    """
    try:
        prop = actor.GetProperty()
        prop.SetVertexVisibility(False)
        prop.SetRenderPointsAsSpheres(False)
    except Exception:  # noqa: BLE001
        pass

    if depth_offset:
        try:
            mapper = actor.GetMapper()
            mapper.SetResolveCoincidentTopologyToPolygonOffset()
            # Negative units pull the line toward the camera in depth.
            mapper.SetRelativeCoincidentTopologyLineOffsetParameters(0.0, -6.0)
        except Exception:  # noqa: BLE001
            pass
    return actor


def stock_box_extents(stock, rect_box) -> np.ndarray:
    """Plate stock's three sizes in mm, one per row of `rect_box.axes`.

    The stock sits on the box's own axes and centre, just larger. Its sizes
    are sorted the way stock.rectilinear_stock sorted them, so each lands on
    the right part axis.
    """
    extents = np.asarray(rect_box.extents_mm, dtype=float)
    order = np.argsort(extents)[::-1]          # largest -> smallest
    stock_sorted = np.array([stock.length_in, stock.width_in or 0.0,
                             stock.thickness_or_diameter_in]) * 25.4
    out = np.empty(3, dtype=float)
    out[order] = stock_sorted
    return out


def _cube_quad_faces() -> list[list[int]]:
    """Six quads (4 corner indices each) for the 8-corner ordering produced
    by bounding_box._box_corners. Each quad walks its corners in a
    non-self-intersecting order so it renders as a flat face, not a bowtie."""
    signs = [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    index_of = {s: i for i, s in enumerate(signs)}
    gray_order = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
    faces = []
    for axis in range(3):
        other_axes = [i for i in range(3) if i != axis]
        for fixed in (-1, 1):
            quad = []
            for a, b in gray_order:
                coord = [0, 0, 0]
                coord[axis] = fixed
                coord[other_axes[0]] = a
                coord[other_axes[1]] = b
                quad.append(index_of[tuple(coord)])
            faces.append(quad)
    return faces


def _box_corners(center: np.ndarray, axes: np.ndarray, extents: np.ndarray) -> np.ndarray:
    """Same corner ordering as bounding_box._box_corners, so the edge-pair
    and quad-face tables above apply to it too."""
    half = np.asarray(extents, dtype=float) / 2.0
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    return np.asarray(center, dtype=float) + (signs * half) @ np.asarray(axes, dtype=float)


def _cube_edge_pairs() -> list[tuple[int, int]]:
    """The 12 edges, as index pairs into the 8-corner array produced by
    bounding_box._box_corners (itertools.product-style over sx in (-1,1),
    sy in (-1,1), sz in (-1,1)). Two corners share an edge exactly when
    their sign vectors differ in a single component, so this is derived
    generically rather than hardcoded."""
    signs = [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    pairs = []
    for a in range(8):
        for b in range(a + 1, 8):
            if sum(1 for x, y in zip(signs[a], signs[b]) if x != y) == 1:
                pairs.append((a, b))
    return pairs


def _dash_segments(p1: np.ndarray, p2: np.ndarray, period: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Splits the segment p1->p2 into dashes of the given period, returning
    a list of (start, end) point pairs covering only the "on" portions."""
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    total = float(np.linalg.norm(p2 - p1))
    if total <= 0 or period <= 0:
        return [(p1, p2)]
    direction = (p2 - p1) / total
    dash_len = period * DASH_DUTY
    segments = []
    pos = 0.0
    while pos < total:
        end = min(pos + dash_len, total)
        segments.append((p1 + direction * pos, p1 + direction * end))
        pos += period
    return segments


def _dashed_polydata(segments: list[tuple[np.ndarray, np.ndarray]]) -> pv.PolyData:
    """Packs (start, end) point pairs into a single PolyData of 2-point
    lines, so all dashes draw as one actor."""
    points = []
    lines = []
    for start, end in segments:
        i = len(points)
        points.extend([start, end])
        lines.extend([2, i, i + 1])
    return _line_polydata(np.asarray(points, dtype=float), np.array(lines, dtype=np.int64))


class PartViewer:
    def __init__(self, parent=None):
        self.plotter = QtInteractor(parent)
        self.plotter.set_background("white")
        # True isometric CAD-style view: orthographic (no perspective
        # vanishing point), camera locked to the standard iso angle.
        self.plotter.enable_parallel_projection()
        self._overlay_actors: list = []
        self._part_actor = None
        self._edge_actor = None
        self._mesh_overlay_actor = None
        # Retained so the display toggles can rebuild actors without a reload.
        self._mesh = None
        self._edge_polylines: list[np.ndarray] = []
        self._diagonal_mm = 0.0
        self._show_edges = True
        self._show_mesh = False

    @property
    def widget(self):
        return self.plotter

    def clear_all(self):
        self.plotter.clear()
        self._overlay_actors = []
        self._part_actor = None
        self._edge_actor = None
        self._mesh_overlay_actor = None
        self._mesh = None
        self._edge_polylines = []
        self._highlight_actors = []

    def clear_overlays(self):
        for actor in self._overlay_actors:
            self.plotter.remove_actor(actor)
        self._overlay_actors = []

    def display_part(self, points_mm: np.ndarray, triangles: np.ndarray,
                      edge_polylines: list[np.ndarray] | None = None):
        """Draws the part CAD-style: a smoothly shaded surface with no
        triangle outlines, overlaid with the shape's real B-rep edges."""
        self.clear_all()
        points_mm = np.asarray(points_mm, dtype=float)
        n_tri = len(triangles)
        faces = np.hstack([np.full((n_tri, 1), 3, dtype=np.int64), triangles]).flatten()
        self._mesh = pv.PolyData(points_mm, faces)
        self._edge_polylines = list(edge_polylines) if edge_polylines else []

        if len(points_mm):
            self._diagonal_mm = float(np.linalg.norm(points_mm.max(axis=0) - points_mm.min(axis=0)))

        self._add_surface_actor()
        self._rebuild_edge_actor()
        self._rebuild_mesh_overlay()

    def _add_surface_actor(self):
        if self._mesh is None:
            return
        # split_sharp_edges keeps real corners crisp under smooth shading;
        # without it, vertex-normal averaging visibly rounds off box edges.
        # It's guarded because it needs a reasonably recent PyVista.
        try:
            self._part_actor = self.plotter.add_mesh(
                self._mesh, color=PART_COLOR, show_edges=False, smooth_shading=True,
                split_sharp_edges=True, feature_angle=SHADING_FEATURE_ANGLE,
                specular=0.25, specular_power=12,
            )
        except TypeError:
            self._part_actor = self.plotter.add_mesh(
                self._mesh, color=PART_COLOR, show_edges=False, smooth_shading=True,
            )

    def _rebuild_edge_actor(self):
        if self._edge_actor is not None:
            self.plotter.remove_actor(self._edge_actor)
            self._edge_actor = None
        if not (self._show_edges and self._edge_polylines):
            return
        from partestimator.machining.geometry.edges import polylines_to_line_cells
        pts, cells = polylines_to_line_cells(self._edge_polylines)
        if len(pts) == 0:
            return
        poly = _line_polydata(pts, cells)
        self._edge_actor = _style_line_actor(
            self.plotter.add_mesh(
                poly, color=EDGE_COLOR, line_width=EDGE_LINE_WIDTH, lighting=False,
            ),
            # Model edges sit exactly on the surface, so they need the depth
            # nudge or they vanish wherever the surface is concave.
            depth_offset=True,
        )

    def _rebuild_mesh_overlay(self):
        """Optional wireframe of the actual tessellation -- off by default,
        but useful for sanity-checking how finely a part got meshed."""
        if self._mesh_overlay_actor is not None:
            self.plotter.remove_actor(self._mesh_overlay_actor)
            self._mesh_overlay_actor = None
        if not (self._show_mesh and self._mesh is not None):
            return
        self._mesh_overlay_actor = _style_line_actor(
            self.plotter.add_mesh(
                self._mesh, style="wireframe", color=MESH_EDGE_COLOR, line_width=1, lighting=False,
            ),
            depth_offset=True,
        )

    def set_show_edges(self, show: bool):
        self._show_edges = bool(show)
        self._rebuild_edge_actor()

    def set_show_mesh(self, show: bool):
        self._show_mesh = bool(show)
        self._rebuild_mesh_overlay()

    def show_center_of_mass(self, point_mm: np.ndarray):
        """Marks the part's center of mass.

        The centroid is normally *inside* solid material, so a plain marker
        there would be hidden by the surface. Alongside the sphere we draw
        axis-aligned crosshair lines long enough to emerge from the part --
        the standard CAD way of indicating a datum point, and more reliable
        than trying to defeat depth testing. The sphere is still drawn for
        when the centroid does sit in open space (C-shapes, thin brackets,
        rings, and similar)."""
        center = np.asarray(point_mm, dtype=float)
        radius = max(self._diagonal_mm * COM_MARKER_RELATIVE_RADIUS, 1e-6)

        sphere = pv.Sphere(radius=radius, center=center, theta_resolution=24, phi_resolution=24)
        actor = self.plotter.add_mesh(sphere, color=COM_COLOR, specular=0.4)
        self._overlay_actors.append(actor)

        reach = max(self._diagonal_mm * 0.62, radius * 4)
        for axis in np.eye(3):
            line = pv.Line(center - axis * reach, center + axis * reach)
            line_actor = _style_line_actor(self.plotter.add_mesh(
                line, color=COM_COLOR, line_width=1.5, lighting=False,
            ))
            self._overlay_actors.append(line_actor)

    def show_primary_bounding_box(self, primary: str, rect_box, cylinder):
        """Displays only whichever bounding box (rectilinear or
        cylindrical) has the smaller volume, per the selection already
        computed in geometry.bounding_box.select_primary -- drawn as a grey
        dotted wireframe so it reads as a reference envelope and never
        obscures the part."""
        if primary == "rectilinear":
            self._show_rectilinear_box(rect_box.corners_mm)
        else:
            self._show_cylinder(cylinder.axis_point_mm, cylinder.axis_direction, cylinder.radius_mm, cylinder.height_mm)

    def _add_dashed(self, segments: list[tuple[np.ndarray, np.ndarray]]):
        if not segments:
            return
        poly = _dashed_polydata(segments)
        actor = _style_line_actor(
            self.plotter.add_mesh(poly, color=BOX_COLOR, line_width=BOX_LINE_WIDTH, lighting=False)
        )
        self._overlay_actors.append(actor)

    def _show_rectilinear_box(self, corners_mm: np.ndarray):
        corners = np.asarray(corners_mm, dtype=float)
        diagonal = float(np.linalg.norm(corners.max(axis=0) - corners.min(axis=0)))
        period = max(diagonal * DASH_TARGET_LENGTH_FRACTION, 1e-6)
        segments = []
        for a, b in _cube_edge_pairs():
            segments.extend(_dash_segments(corners[a], corners[b], period))
        self._add_dashed(segments)

    def _show_cylinder(self, axis_point_mm: np.ndarray, axis_direction, radius_mm: float, height_mm: float,
                        n_rings: int = 2, n_ribs: int = 16, ring_resolution: int = 96):
        """Dotted-wireframe cylinder: the two end rings plus a set of
        axial ribs around the circumference, all dashed."""
        d = np.asarray(axis_direction, dtype=float)
        d = d / np.linalg.norm(d)
        center = np.asarray(axis_point_mm, dtype=float)

        # Build an orthonormal frame (d, u, v) to lay out the circles in.
        ref = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(ref, d))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        u = ref - np.dot(ref, d) * d
        u /= np.linalg.norm(u)
        v = np.cross(d, u)

        half_h = height_mm / 2.0
        diagonal = float(np.hypot(2 * radius_mm, height_mm))
        period = max(diagonal * DASH_TARGET_LENGTH_FRACTION, 1e-6)

        segments = []
        # End rings (and optional intermediate rings).
        for i in range(n_rings):
            offset = -half_h + (height_mm * i / (n_rings - 1)) if n_rings > 1 else 0.0
            ring_center = center + d * offset
            angles = np.linspace(0, 2 * np.pi, ring_resolution + 1)
            pts = ring_center + radius_mm * (np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v)
            for j in range(ring_resolution):
                segments.extend(_dash_segments(pts[j], pts[j + 1], period))
        # Axial ribs.
        for k in range(n_ribs):
            ang = 2 * np.pi * k / n_ribs
            radial = radius_mm * (np.cos(ang) * u + np.sin(ang) * v)
            p1 = center + radial - d * half_h
            p2 = center + radial + d * half_h
            segments.extend(_dash_segments(p1, p2, period))

        self._add_dashed(segments)

    def show_stock(self, primary: str, stock, rect_box, cylinder):
        """Draws the stock envelope: translucent light-blue body with a blue
        dashed outline.

        The stock is drawn CONCENTRIC with the bounding box -- the allowance
        is split equally across opposing faces rather than piled onto one
        side -- so it reads the way the part actually sits in the billet.
        """
        if stock is None:
            return
        mm_per_in = 25.4
        if stock.kind == "extrusion" and stock.placement is not None:
            self._add_stock_prism(stock)
        elif stock.kind == "rectilinear" and rect_box is not None:
            corners = _box_corners(rect_box.center_mm, rect_box.axes,
                                   stock_box_extents(stock, rect_box))
            self._add_stock_box(corners)
        elif cylinder is not None:
            self._add_stock_cylinder(
                cylinder.axis_point_mm, cylinder.axis_direction,
                stock.thickness_or_diameter_in * mm_per_in / 2.0,
                stock.length_in * mm_per_in,
            )

    def _add_stock_prism(self, stock, max_ribs: int = 12):
        """An extrusion: its real section, swept along the chosen direction.

        Same styling as the box and the bar -- translucent body, dashed
        outline -- with the section outlined at both ends and a dashed rib
        along each corner (every few points round a tube).
        """
        mm = 25.4
        centre, d, u, v = (np.asarray(a, dtype=float) for a in stock.placement)
        half = stock.length_in * mm / 2.0
        loops = [np.asarray(lp, dtype=float) * mm for lp in stock.profile.outline_loops()]

        def ring(loop, z):
            return centre + loop[:, :1] * u + loop[:, 1:2] * v + z * d

        points, faces = [], []
        starts = []
        for loop in loops:
            n = len(loop)
            base = sum(len(p) for p in points)
            starts.append(base)
            points.append(ring(loop, -half))
            points.append(ring(loop, half))
            for k in range(n):                       # side walls
                k2 = (k + 1) % n
                faces.append([4, base + k, base + k2, base + n + k2, base + n + k])
        if len(loops) == 1:                          # solid section: two caps
            n = len(loops[0])
            faces.append([n] + list(range(n)))
            faces.append([n] + list(range(n, 2 * n)))
        else:                                        # tube: ring between the loops
            n = len(loops[0])
            o, i = starts[0], starts[1]
            for end in (0, n):
                for k in range(n):
                    k2 = (k + 1) % n
                    faces.append([4, o + end + k, o + end + k2, i + end + k2, i + end + k])
        pts = np.vstack(points)
        body = pv.PolyData(pts, np.hstack(faces).astype(np.int64))
        actor = self.plotter.add_mesh(
            body, color=STOCK_FILL_COLOR, opacity=STOCK_FILL_OPACITY, lighting=False,
        )
        self._overlay_actors.append(actor)

        diagonal = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        period = max(diagonal * DASH_TARGET_LENGTH_FRACTION, 1e-6)
        segments = []
        for loop in loops:
            a, b = ring(loop, -half), ring(loop, half)
            n = len(loop)
            for end in (a, b):
                for k in range(n):
                    segments.extend(_dash_segments(end[k], end[(k + 1) % n], period))
            step = max(1, n // max_ribs)
            for k in range(0, n, step):
                segments.extend(_dash_segments(a[k], b[k], period))
        if segments:
            line_actor = _style_line_actor(self.plotter.add_mesh(
                _dashed_polydata(segments), color=STOCK_LINE_COLOR,
                line_width=BOX_LINE_WIDTH, lighting=False,
            ))
            self._overlay_actors.append(line_actor)

    def _add_stock_box(self, corners_mm: np.ndarray):
        corners = np.asarray(corners_mm, dtype=float)
        quads = _cube_quad_faces()
        faces = np.hstack([[4] + q for q in quads]).astype(np.int64)
        body = pv.PolyData(corners, faces)
        actor = self.plotter.add_mesh(
            body, color=STOCK_FILL_COLOR, opacity=STOCK_FILL_OPACITY, lighting=False,
        )
        self._overlay_actors.append(actor)

        diagonal = float(np.linalg.norm(corners.max(axis=0) - corners.min(axis=0)))
        period = max(diagonal * DASH_TARGET_LENGTH_FRACTION, 1e-6)
        segments = []
        for a, b in _cube_edge_pairs():
            segments.extend(_dash_segments(corners[a], corners[b], period))
        if segments:
            line_actor = _style_line_actor(self.plotter.add_mesh(
                _dashed_polydata(segments), color=STOCK_LINE_COLOR,
                line_width=BOX_LINE_WIDTH, lighting=False,
            ))
            self._overlay_actors.append(line_actor)

    def _add_stock_cylinder(self, axis_point_mm, axis_direction, radius_mm, height_mm,
                             n_ribs: int = 16, ring_resolution: int = 96):
        d = np.asarray(axis_direction, dtype=float)
        d = d / np.linalg.norm(d)
        center = np.asarray(axis_point_mm, dtype=float)

        body = pv.Cylinder(center=center, direction=d, radius=radius_mm,
                            height=height_mm, resolution=64, capping=True)
        actor = self.plotter.add_mesh(
            body, color=STOCK_FILL_COLOR, opacity=STOCK_FILL_OPACITY, lighting=False,
        )
        self._overlay_actors.append(actor)

        ref = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(ref, d))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        u = ref - np.dot(ref, d) * d
        u /= np.linalg.norm(u)
        v = np.cross(d, u)

        half_h = height_mm / 2.0
        period = max(float(np.hypot(2 * radius_mm, height_mm)) * DASH_TARGET_LENGTH_FRACTION, 1e-6)
        segments = []
        for offset in (-half_h, half_h):
            ring_center = center + d * offset
            angles = np.linspace(0, 2 * np.pi, ring_resolution + 1)
            pts = ring_center + radius_mm * (np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v)
            for j in range(ring_resolution):
                segments.extend(_dash_segments(pts[j], pts[j + 1], period))
        for k in range(n_ribs):
            ang = 2 * np.pi * k / n_ribs
            radial = radius_mm * (np.cos(ang) * u + np.sin(ang) * v)
            segments.extend(_dash_segments(center + radial - d * half_h,
                                            center + radial + d * half_h, period))
        if segments:
            line_actor = _style_line_actor(self.plotter.add_mesh(
                _dashed_polydata(segments), color=STOCK_LINE_COLOR,
                line_width=BOX_LINE_WIDTH, lighting=False,
            ))
            self._overlay_actors.append(line_actor)

    def show_captured_hole_axes(self, holes):
        """Draws a red axis line through every hole flagged as captured --
        i.e. one whose every open mouth is blocked along its own axis, so no
        straight tool reaches it from either side.

        The line runs past both ends of the hole so it emerges from the
        part and can actually be seen; a line confined to the bore would be
        buried in material. Drawn as one actor for all such holes, so a part
        with none of them costs nothing.
        """
        if not holes:
            return
        segments = []
        for h in holes:
            if not getattr(h, "is_captured", False):
                continue
            axis_point = getattr(h, "axis_point", None)
            axis_dir = getattr(h, "axis_direction", None)
            if axis_point is None or axis_dir is None:
                continue
            d = np.asarray(axis_dir, dtype=float)
            norm = np.linalg.norm(d)
            if norm < 1e-12:
                continue
            d = d / norm
            base = np.asarray(axis_point, dtype=float)
            lo = float(getattr(h, "axial_lo", 0.0))
            hi = float(getattr(h, "axial_hi", 0.0))
            depth = max(hi - lo, 1e-6)
            overhang = max(depth * CAPTURED_AXIS_OVERHANG_FRACTION,
                            CAPTURED_AXIS_MIN_OVERHANG_MM)
            segments.append((base + d * (lo - overhang), base + d * (hi + overhang)))

        if not segments:
            return
        points = []
        lines = []
        for start, end in segments:
            i = len(points)
            points.extend([start, end])
            lines.extend([2, i, i + 1])
        poly = _line_polydata(np.asarray(points, dtype=float),
                              np.array(lines, dtype=np.int64))
        actor = _style_line_actor(self.plotter.add_mesh(
            poly, color=CAPTURED_AXIS_COLOR, line_width=CAPTURED_AXIS_WIDTH,
            lighting=False,
        ))
        self._overlay_actors.append(actor)

    def show_thin_wall_points(self, points_mm: np.ndarray):
        """Highlights the sample points flagged by the thin-wall scan as
        below the thickness threshold."""
        if len(points_mm) == 0:
            return
        cloud = pv.PolyData(np.asarray(points_mm, dtype=float))
        actor = self.plotter.add_points(
            cloud, color=THIN_WALL_COLOR, point_size=10, render_points_as_spheres=True,
        )
        self._overlay_actors.append(actor)

    # ------------------------------------------------------------ picking
    def set_click_handler(self, handler) -> None:
        """Call `handler(x, y)` with display coordinates on a plain left
        click -- one that did not drag to turn the view. None stops it."""
        self._click_handler = handler
        if getattr(self, "_click_observers", False):
            return
        try:
            self.plotter.iren.add_observer("LeftButtonPressEvent", self._on_press)
            self.plotter.iren.add_observer("LeftButtonReleaseEvent", self._on_release)
            self._click_observers = True
        except Exception:  # noqa: BLE001 -- no interactor (e.g. a test stub)
            pass

    def _on_press(self, *_args) -> None:
        try:
            self._press = tuple(self.plotter.iren.get_event_position())
        except Exception:  # noqa: BLE001
            self._press = None

    def _on_release(self, *_args) -> None:
        handler = getattr(self, "_click_handler", None)
        press = getattr(self, "_press", None)
        if handler is None or press is None:
            return
        try:
            x, y = self.plotter.iren.get_event_position()
        except Exception:  # noqa: BLE001
            return
        if abs(x - press[0]) > CLICK_TOLERANCE_PX or abs(y - press[1]) > CLICK_TOLERANCE_PX:
            return
        handler(x, y)

    def pick_part_triangle(self, x, y):
        """Index into the part's triangles of the one under (x, y), or None."""
        import vtk

        if self._mesh is None or self._part_actor is None:
            return None
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.0005)
        picker.PickFromListOn()
        picker.AddPickList(self._part_actor)
        if not picker.Pick(x, y, 0, self.plotter.renderer):
            return None
        # The actor draws a copy of the mesh split at sharp edges, so map
        # back through the picked point rather than trusting its cell id.
        cid = self._mesh.find_closest_cell(np.asarray(picker.GetPickPosition(), dtype=float))
        return int(cid) if cid is not None and cid >= 0 else None

    def pick_ray(self, x, y):
        """(origin, direction) in world space of the line of sight at (x, y)."""
        ren = self.plotter.renderer
        ends = []
        for z in (0.0, 1.0):
            ren.SetDisplayPoint(float(x), float(y), z)
            ren.DisplayToWorld()
            w = ren.GetWorldPoint()
            ends.append(np.array(w[:3], dtype=float) / (w[3] or 1.0))
        return ends[0], ends[1] - ends[0]

    def set_picking_cursor(self, on: bool) -> None:
        from PySide6.QtCore import Qt

        if on:
            self.plotter.setCursor(Qt.CrossCursor)
        else:
            self.plotter.unsetCursor()

    def clear_highlights(self, tag: str | None = None) -> None:
        """Remove the picked-face colouring: one kind ("part", "stock") or all."""
        kept = []
        for t, actor in getattr(self, "_highlight_actors", []):
            if tag is None or t == tag:
                self.plotter.remove_actor(actor)
            else:
                kept.append((t, actor))
        self._highlight_actors = kept
        self.plotter.render()

    def highlight_triangles(self, triangle_indices, tag: str = "part") -> None:
        """Colour a picked face of the part (replacing any earlier one)."""
        self.clear_highlights(tag)
        if self._mesh is None:
            return
        idx = np.asarray(triangle_indices, dtype=np.int64)
        if len(idx) == 0:
            return
        face = self._mesh.extract_cells(idx)
        actor = self.plotter.add_mesh(face, color=PICK_COLOR, lighting=False,
                                      opacity=0.85)
        try:
            # Drawn on the face itself, so pull it toward the camera to win
            # the depth test against the surface underneath.
            mapper = actor.GetMapper()
            mapper.SetResolveCoincidentTopologyToPolygonOffset()
            mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-2.0, -4.0)
        except Exception:  # noqa: BLE001
            pass
        self._highlight_actors = getattr(self, "_highlight_actors", []) + [(tag, actor)]
        self.plotter.render()

    def highlight_quad(self, corners_mm, tag: str = "stock") -> None:
        """Colour one face of the stock box: four corners in drawing order."""
        self.clear_highlights(tag)
        quad = pv.PolyData(np.asarray(corners_mm, dtype=float), faces=[4, 0, 1, 2, 3])
        actor = self.plotter.add_mesh(quad, color=PICK_COLOR, opacity=0.45, lighting=False)
        self._highlight_actors = getattr(self, "_highlight_actors", []) + [(tag, actor)]
        self.plotter.render()

    def fit_all(self):
        """Fits the camera to the current scene and locks it to a true
        (orthographic) isometric view -- the standard CAD convention,
        rather than whatever angle the camera happened to be at."""
        self.plotter.view_isometric()
        self.plotter.reset_camera()

    def close(self):
        """Release the VTK render window (called on app shutdown)."""
        self.plotter.close()
