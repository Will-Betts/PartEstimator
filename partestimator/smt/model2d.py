"""Flat pattern geometry in pure numpy, with no CAD kernel involved.

WHY THIS EXISTS
---------------
The K-factor slider has to redraw at interactive rates. Re-running the kernel's
boolean fuse on every frame would not manage that -- a fuse over a dozen faces
is tens to hundreds of milliseconds, and it allocates heavily.

The saving grace is that **detection does not depend on K at all**. Which faces
are bends, what their radii and sweeps are, which flanges they connect -- all of
that is fixed the moment the STEP file is read. Only the allowances and the
resulting unfold transforms move when K changes.

So the split is:

    load + detect        once, in a worker thread, needs the kernel
    K / material change  a numpy transform of cached boundary points, instant
    export               the exact kernel path, once, on demand

`Flange.loops_a/loops_b` hold the discretised face boundaries captured during
detection. Everything here just moves those points around.

ACCURACY NOTE
-------------
These loops are discretised, so the preview's curved edges are polylines and its
area is a polygon approximation. Straight edges, the bounding box and every bend
dimension are exact, because those come from the transforms rather than from the
discretisation. The exported DXF and STEP always take the exact kernel path --
this module never feeds an output file. The test suite pins the preview's
bounding box against the same hand calculation the kernel path is checked
against, so the two cannot silently drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import geom
from .diag import LOGGER
from .model import Part
from .unfold import UnfoldResult


@dataclass
class Flat2D:
    """Everything needed to draw and measure the flat pattern."""

    # Closed polylines, (N, 2). Flange footprints and bend strips separately,
    # so the view can render internal tangent lines differently from the
    # silhouette if it wants to.
    flange_loops: list[np.ndarray] = field(default_factory=list)
    hole_loops: list[np.ndarray] = field(default_factory=list)
    strips: list[np.ndarray] = field(default_factory=list)

    # (start_xy, end_xy, bend_id, direction)
    bend_lines: list[tuple[np.ndarray, np.ndarray, int, int]] = field(
        default_factory=list
    )

    # Flanges whose outline could not be captured. Anything in here is a hole
    # in the drawing, not an empty region of the part.
    missing_outlines: list[int] = field(default_factory=list)

    bbox_min: np.ndarray = field(default_factory=lambda: np.zeros(2))
    bbox_max: np.ndarray = field(default_factory=lambda: np.zeros(2))

    @property
    def width(self) -> float:
        """Literal X extent in the flat frame."""
        return float(self.bbox_max[0] - self.bbox_min[0])

    @property
    def height(self) -> float:
        """Literal Y extent in the flat frame."""
        return float(self.bbox_max[1] - self.bbox_min[1])

    # The flat frame's in-plane axes are chosen from an arbitrary seed vector,
    # so whether the blank's long side lands on X or Y depends on how the part
    # happened to be oriented in the STEP file. Nobody wants a blank reported as
    # "50 x 153" one day and "153 x 50" the next, so report sorted dimensions.
    @property
    def blank_length(self) -> float:
        return max(self.width, self.height)

    @property
    def blank_width(self) -> float:
        return min(self.width, self.height)

    @property
    def bbox_area(self) -> float:
        return self.width * self.height

    def min_area_rect(self) -> tuple[float, float, float]:
        """Tightest rectangle at any angle -- the real nesting question."""
        pts = self.all_points()
        if len(pts) < 3:
            return self.width, self.height, 0.0
        return geom.min_area_rect(pts)

    def all_points(self) -> np.ndarray:
        chunks = self.flange_loops + self.strips
        if not chunks:
            return np.zeros((0, 2))
        return np.vstack(chunks)

    def polygon_area(self) -> float:
        """Approximate profile area, net of holes.

        Flange footprints and strips abut without overlapping, so summing their
        shoelace areas is valid. Discretised curves make this an approximation;
        the exact figure comes from the kernel at export time.
        """
        total = sum(_shoelace(p) for p in self.flange_loops)
        total += sum(_shoelace(p) for p in self.strips)
        total -= sum(_shoelace(h) for h in self.hole_loops)
        return abs(total)


def _shoelace(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def flat_frame(origin, normal) -> np.ndarray:
    """4x4 mapping the flat plane onto the global XY plane.

    Kept in step with `occ.flatten.flat_frame` -- both must choose the same
    in-plane axes or the preview and the exported DXF would come out rotated
    relative to each other.
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


def _project(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply a 4x4 to (N,3) points and drop Z.

    Dropping Z is exact rather than a projection: after unfolding, every flange
    lies parallel to the base plane, so the frame transform leaves it at a
    constant Z that carries no shape information.
    """
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    out = pts @ M[:3, :3].T + M[:3, 3]
    return out[:, :2]


def raw_flat_points(part: Part, result: UnfoldResult, M_frame) -> np.ndarray:
    """Every 2D point the finished blank will contain, before it is turned.

    Needed to decide the orientation, which is a chicken-and-egg problem: the
    tightest bounding rectangle cannot be found until the part is flat, and the
    part should not be built flat until the orientation is known. Resolved by
    doing this pass in pure numpy off the CACHED boundary loops -- no kernel, no
    booleans, microseconds -- and only then building the real thing once.
    """
    chunks: list[np.ndarray] = []

    for fid, M in result.transforms.items():
        flange = part.flanges.get(fid)
        if flange is None:
            continue
        _n, _o, _face, side = result.sides[fid]
        full = M_frame @ M
        for loop in flange.loops_for_side(side):
            pts = _project(loop, full)
            if len(pts):
                chunks.append(pts)

    # Bend strips sit between flanges so they rarely drive the envelope, but on
    # a single-bend part the strip IS part of the outline and leaving it out
    # would turn the blank to fit a shape that is not the whole blank.
    for fb in result.flat_bends:
        a = geom.apply_point(M_frame, fb.line_start)[:2]
        b = geom.apply_point(M_frame, fb.line_end)[:2]
        u = geom.apply_vector(M_frame, fb.step_dir)[:2]
        n = float(np.linalg.norm(u))
        if n < 1e-12:
            continue
        u_hat = u / n
        chunks.append(
            np.array([a, b, b + u_hat * fb.width, a + u_hat * fb.width])
        )

    if not chunks:
        return np.zeros((0, 2))
    return np.vstack(chunks)


def oriented_flat_frame(part: Part, result: UnfoldResult, opts=None):
    """The flat frame with the blank already turned to its best orientation.

    THE SINGLE SOURCE OF TRUTH FOR HOW THE BLANK IS PLACED. The preview, the
    DXF, the extruded solid and every reported dimension are built from this one
    matrix, so they cannot disagree about which way round the part is. Rotating
    each output separately is how the DXF and the preview came to disagree in an
    earlier version, and it took a photograph from the shop floor to notice.

    Returns (M_frame, angle, saving).
    """
    from . import orient

    base_normal, base_origin, _f, _s = result.sides[result.base]
    M_frame = flat_frame(base_origin, base_normal)

    mode = getattr(opts, "orient", "min_area") if opts is not None else "min_area"
    grain = getattr(opts, "grain_deg", None) if opts is not None else None
    to_origin = getattr(opts, "orient_to_origin", True) if opts is not None else True

    pts = raw_flat_points(part, result, M_frame)
    if len(pts) < 3 or mode == "as_modelled":
        return M_frame, 0.0, 0.0

    angle = orient.choose_angle(pts, mode=mode, grain_deg=grain)
    saved = orient.saving(pts, angle)
    M_frame = orient.frame_rotation_3d(angle) @ M_frame

    if to_origin:
        turned = orient.apply(pts, angle)
        shift = -turned.min(axis=0)
        M_frame = geom.translation([float(shift[0]), float(shift[1]), 0.0]) @ M_frame

    return M_frame, float(angle), float(saved)


def build(part: Part, result: UnfoldResult, opts=None) -> Flat2D:
    """Assemble the flat pattern in 2D from cached loops and solved transforms."""
    M_frame, angle, saved = oriented_flat_frame(part, result, opts)
    part.flat_rotation = angle
    part.flat_rotation_saving = saved

    flat = Flat2D()
    missing: list[int] = []

    for fid, M in result.transforms.items():
        flange = part.flanges.get(fid)
        if flange is None:
            continue
        _n, _o, _face, side = result.sides[fid]
        regions = flange.regions_for_side(side)
        if not regions:
            # A flange with no captured outline used to be skipped silently,
            # which is exactly how a whole flange goes missing from the drawing
            # without anything saying so.
            missing.append(fid)
            continue

        full = M_frame @ M
        # Per REGION, so each region's own outer wire is an outline and only
        # its own inner wires are holes. Iterating a flattened list instead
        # would turn a second tab into a cutout in the first.
        for region in regions:
            for i, loop in enumerate(region):
                pts = _project(loop, full)
                if len(pts) < 3:
                    continue
                if i == 0:
                    flat.flange_loops.append(pts)
                else:
                    flat.hole_loops.append(pts)

    for fb in result.flat_bends:
        a = geom.apply_point(M_frame, fb.line_start)[:2]
        b = geom.apply_point(M_frame, fb.line_end)[:2]
        u = geom.apply_vector(M_frame, fb.step_dir)[:2]

        n = float(np.linalg.norm(u))
        if n < 1e-12:
            continue
        u_hat = u / n

        bend = part.bends.get(fb.bend_id)
        developed = _develop_bend(bend, fb, M_frame, a, u_hat)

        if developed:
            flat.strips.extend(developed)
        else:
            # Fallback: a plain rectangle spanning the bend line. Correct only
            # when the bend zone really is rectangular, which is why the
            # development above is preferred wherever the face is available.
            flat.strips.append(
                np.array([a, b, b + u_hat * fb.width, a + u_hat * fb.width])
            )

        # The bend line is drawn down the middle of the developed zone.
        half = u_hat * (fb.width / 2.0)
        flat.bend_lines.append((a + half, b + half, fb.bend_id, fb.direction))

    flat.missing_outlines = missing
    if missing:
        LOGGER.warning(
            "flange(s) %s have no captured outline and are missing from the "
            "flat pattern", missing,
        )

    pts = flat.all_points()
    if len(pts):
        flat.bbox_min = pts.min(axis=0)
        flat.bbox_max = pts.max(axis=0)

    return flat


def _develop_bend(bend, fb, M_frame, origin_2d, across_2d):
    """Unroll the real bend face onto the flat, using K as the offset surface.

    The bend zone is only a rectangle when the flanges meet it squarely along
    its whole length and nothing is cut out of it. On real parts it is routinely
    not: a slot crosses the bend, a flange stops short, the blank runs off the
    edge. Pasting in a rectangle then adds material that does not exist and
    misses notches that do.

    A cylinder is developable, so the honest answer is to unroll the actual
    trimmed face. Every boundary point has a (u, v) on the cylinder; the neutral
    surface at rho = R + K*t is the layer that neither stretches nor compresses,
    so arc length there maps one-to-one onto flat distance:

        flat = tangent_line + v * axis + rho * (u - u_tangent) * across

    At the parent's tangent line u = u_tangent and the offset is zero. At the
    child's, (u - u_tangent) is the bend angle and the offset is rho*theta --
    the bend allowance, falling out of the geometry rather than being asserted.

    Returns a list of developed loops, or None if the face was not captured.
    """
    if bend is None or not bend.inner_loops or bend.axis_x_dir is None:
        return None

    axis_flat = geom.apply_vector(M_frame, fb.axis_dir_flat)[:2]
    n = float(np.linalg.norm(axis_flat))
    if n < 1e-12:
        return None
    axis_flat = axis_flat / n

    rho = bend.neutral_radius()
    out: list[np.ndarray] = []

    for loop in bend.inner_loops:
        if len(loop) < 3:
            continue
        u, v = geom.cylinder_uv(
            loop, bend.axis_point, bend.axis_dir, bend.axis_x_dir, bend.u_mid
        )
        xy = geom.develop_cylinder(
            u, v - bend.v_min, rho, fb.u_tangent, fb.sign,
            origin_2d, axis_flat, across_2d,
        )
        out.append(xy)

    return out or None


def recompute(part: Part, opts, table=None) -> tuple[Flat2D, UnfoldResult]:
    """Reassign allowances at the current K and rebuild the 2D flat.

    This is the whole fast path: everything the K slider triggers, and nothing
    that needs a CAD kernel. Detection does not depend on K at all -- only the
    allowances and the unfold transforms do -- so with the boundary loops cached
    at detection time this is a few numpy transforms and redraws instantly.
    """
    from .unfold import assign_allowances, solve

    assign_allowances(part, opts, table)
    result = solve(part, opts)
    part.base_flange = result.base
    part.transforms = result.transforms
    return build(part, result, opts), result
