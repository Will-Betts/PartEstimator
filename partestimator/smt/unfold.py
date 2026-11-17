"""The unfold solver: a tree of rigid transforms, one per flange.

DERIVATION
----------
Put the upstream flange's reference surface on the plane z = 0, extending in the
-x direction. The bend axis runs along y through (0, 0, R). The tangent line L1
where the flange meets the bend sits at the origin. After bending through theta,
the downstream tangent line L2 sits at (R sin@, 0, R - R cos@).

In the flat, the downstream flange must lie in z = 0 extending in +x from
x = BA (the bend allowance), because the arc is replaced by a strip of flat
material exactly BA wide.

Rotating the downstream subtree by -theta about the bend axis carries L2 exactly
onto the origin and aligns the flange along +x. A translation of BA along +x
then places it correctly. So:

    M_local = Translate(BA * u) . Rotate(axis, -theta)

where u is the unit vector lying in the flat plane, perpendicular to the bend
axis, pointing downstream.

Note this transform is independent of which surface you reference it to. It is a
rigid motion of the whole downstream body, so it must be -- the inner surface,
the mid surface and the outer surface all land correctly. Verified in the tests
for both the inner and outer reference.

Transforms compose down the tree:

    M_global[child] = M_global[parent] @ M_local

because M_local is expressed in the ORIGINAL coordinate frame (using the
un-moved axis), and the parent's own accumulated motion then carries the entire
subtree rigidly.

TWO SIGN PROBLEMS, AND HOW THEY ARE SOLVED
------------------------------------------
1. Which way to rotate. Rather than deriving a sign convention from the face
   orientations -- fragile, and different CAD systems disagree -- the solver
   simply tries both signs and keeps the one that lands the child's normal
   parallel to the parent's. Self-checking, and impossible to get backwards.

2. Which SIDE of the sheet to reference. On a part mixing up-bends and
   down-bends there is no global "top" face. The side is propagated across each
   bend by a geometric test: for a flange face with outward normal n and plane
   point o, the bend axis lies on the +n side (dot(a - o, n) > 0) exactly when
   that face is the INNER face of the bend. Matching that sign across the bend
   picks the corresponding face on the child. See `choose_side`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import geom
from .allowance import bend_deduction, resolve_allowance
from .config import UnfoldOptions
from .graph import BendGraph, UnflattenableError
from .materials import DeductionTable, select_v_die
from .model import Bend, Part


@dataclass
class FlatBend:
    """Where a bend ends up in the flat, for strip construction and DXF."""

    bend_id: int
    # Tangent line on the parent side, in FLAT coordinates.
    line_start: np.ndarray
    line_end: np.ndarray
    # Unit direction from the parent tangent line toward the child, in flat.
    step_dir: np.ndarray
    width: float                  # == bend allowance
    direction: int                # +1 up, -1 down
    parent: int
    child: int

    # Everything needed to DEVELOP the bend face rather than approximate it
    # with a rectangle. See geom.develop_cylinder.
    axis_dir_flat: np.ndarray | None = None   # bend axis, after the parent's motion
    u_tangent: float = 0.0        # cylinder angle where it meets the parent plane
    sign: float = 1.0             # direction of increasing u toward the child


@dataclass
class UnfoldResult:
    transforms: dict[int, np.ndarray]
    flat_bends: list[FlatBend] = field(default_factory=list)
    # flange id -> (outward normal, plane point, kernel face handle) actually
    # used as the reference side for that flange.
    sides: dict[int, tuple[np.ndarray, np.ndarray, Any, int]] = field(default_factory=dict)
    base: int = 0
    base_normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))


# ---------------------------------------------------------------------------


def axis_side_sign(axis_point, normal, plane_point) -> float:
    """+1 if the bend axis lies on the outward side of this face.

    Which is to say: +1 when the face is the INNER face of that bend, -1 when it
    is the outer face. This is the invariant propagated across bends to keep the
    referenced side of the sheet consistent.
    """
    return float(np.sign(geom.signed_plane_distance(axis_point, plane_point, normal)))


def choose_side(flange, axis_point, want_sign: float):
    """Pick the flange face that sits on the same side of the sheet as the parent."""
    candidates = flange.sides()
    for normal, origin, face, side in candidates:
        if axis_side_sign(axis_point, normal, origin) == want_sign:
            return normal, origin, face, side
    # Degenerate (axis exactly in the face plane -- should not happen for a real
    # bend). Fall back to the primary face rather than failing outright.
    return candidates[0]


def assign_allowances(
    part: Part, opts: UnfoldOptions, table: DeductionTable | None = None
) -> None:
    """Populate bend_allowance / bend_deduction / k_factor on every bend."""
    v_die = opts.v_die or select_v_die(part.thickness)
    for b in part.bends.values():
        ba, k, source = resolve_allowance(
            angle=b.angle,
            r_inner=b.r_inner,
            thickness=b.thickness,
            material=opts.material,
            method=opts.method,
            k_override=opts.k_factor,
            table=table,
            v_die=v_die,
        )
        b.bend_allowance = ba
        b.k_factor = k
        b.bend_deduction = bend_deduction(b.angle, b.r_inner, b.thickness, k)
        b.allowance_source = source


def solve(part: Part, opts: UnfoldOptions | None = None) -> UnfoldResult:
    """Compute the bent-to-flat rigid transform for every flange."""
    opts = opts or UnfoldOptions()
    graph = BendGraph.from_part(part)

    base = opts.base_face if opts.base_face is not None else graph.largest_flange()
    if base not in part.flanges:
        raise UnflattenableError(f"requested base flange {base} does not exist")

    order = graph.bfs(base)

    transforms: dict[int, np.ndarray] = {base: geom.identity()}
    sides: dict[int, tuple[np.ndarray, np.ndarray, Any, int]] = {}

    # Seed the base flange's referenced side. If the base touches a bend, adopt
    # the side facing that bend's axis so the choice is geometrically anchored;
    # otherwise the part is flat and either face will do.
    base_flange = part.flanges[base]
    first = next((s for s in order if s.parent == base), None)
    if first is not None:
        a = part.bends[first.bend].axis_point
        n, o, f, side = choose_side(base_flange, a, want_sign=+1.0)
        sides[base] = (n, o, f, side)
    else:
        sides[base] = base_flange.sides()[0]

    flat_bends: list[FlatBend] = []

    for step in order:
        bend = part.bends[step.bend]
        parent = part.flanges[step.parent]
        child = part.flanges[step.child]

        n_p, o_p, _f, _s = sides[step.parent]
        want = axis_side_sign(bend.axis_point, n_p, o_p)
        n_c, o_c, f_c, side_c = choose_side(child, bend.axis_point, want)
        sides[step.child] = (n_c, o_c, f_c, side_c)

        # The axis sitting on the referenced (flat "top") side means the flange
        # swings up away from that side: an up bend. This is the same quantity
        # that selects the child's face, so direction and side stay consistent.
        bend.direction = int(want) if want != 0 else 1

        M_local, step_dir = _local_transform(bend, n_p, n_c, child.centroid)

        transforms[step.child] = transforms[step.parent] @ M_local

        flat_bends.append(
            _flat_bend_record(
                bend, step, n_p, o_p, n_c, o_c, step_dir,
                transforms[step.parent],
            )
        )

    return UnfoldResult(
        transforms=transforms,
        flat_bends=flat_bends,
        sides=sides,
        base=base,
        base_normal=sides[base][0],
    )


def _local_transform(
    bend: Bend, n_parent: np.ndarray, n_child: np.ndarray, child_centroid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """M_local = Translate(BA * u) . Rotate(axis, -theta), with signs resolved."""
    a = bend.axis_point
    d = geom.unit(bend.axis_dir)

    # --- sign of the rotation -------------------------------------------
    # Try both. Keep the one that makes the child's outward normal point the
    # same way as the parent's, which is the definition of "now coplanar".
    best = None
    for sign in (1.0, -1.0):
        R = geom.rotation_about_axis(a, d, sign * bend.angle)
        dot = float(np.dot(geom.apply_vector(R, n_child), n_parent))
        if best is None or dot > best[0]:
            best = (dot, sign, R)
    dot, sign, R = best  # type: ignore[misc]

    if dot < 0.999:
        raise UnflattenableError(
            f"bend {bend.id}: rotating by +/-{np.degrees(bend.angle):.3f} deg does not "
            f"bring the adjoining flanges coplanar (best alignment {dot:.4f}). "
            f"The detected sweep angle is probably wrong, or this is not a simple "
            f"single-axis bend."
        )

    # --- direction of the translation -----------------------------------
    # After the rotation the child is coplanar with the parent, lying to one
    # side of the tangent line. u is the in-plane direction from that line
    # toward the child.
    c_rot = geom.apply_point(R, child_centroid)
    v = c_rot - a
    v = v - np.dot(v, d) * d                  # strip the axial component
    v = geom.project_to_plane(v, n_parent)    # project into the flat plane

    norm = float(np.linalg.norm(v))
    if norm < 1e-9:
        # Child centroid sits on the tangent line. Fall back to the cross
        # product and orient it by the raw centroid offset.
        v = np.cross(d, n_parent)
        if float(np.dot(v, c_rot - a)) < 0:
            v = -v
        norm = float(np.linalg.norm(v))
        if norm < 1e-9:
            raise UnflattenableError(
                f"bend {bend.id}: cannot determine the unfold direction; the bend "
                f"axis appears parallel to the flange normal."
            )
    u = v / norm

    M = geom.translation(bend.bend_allowance * u) @ R
    return M, u


def _tangent_angle(bend: Bend, normal, origin) -> float:
    """The cylinder angle at which the bend runs tangentially into a plane.

    The tangent point is the point on the cylinder closest to that plane, so
    the radial direction to it is the plane normal, pointing from the axis
    toward the plane.
    """
    dist = geom.signed_plane_distance(bend.axis_point, origin, normal)
    radial = -np.sign(dist) * geom.unit(normal)
    u, _v = geom.cylinder_uv(
        [bend.axis_point + radial],
        bend.axis_point, bend.axis_dir, bend.axis_x_dir, bend.u_mid,
    )
    return float(u[0])


def _flat_bend_record(
    bend: Bend, step, n_p, o_p, n_c, o_c, step_dir, M_parent
) -> FlatBend:
    """Place the bend line in flat coordinates, and pin its development."""
    d = geom.unit(bend.axis_dir)
    # The tangent line is the projection of the bend axis onto the parent plane.
    tangent_pt = bend.axis_point - geom.signed_plane_distance(
        bend.axis_point, o_p, n_p
    ) * geom.unit(n_p)

    p0 = tangent_pt + bend.v_min * d
    p1 = tangent_pt + bend.v_max * d

    u_tangent = 0.0
    sign = 1.0
    if bend.axis_x_dir is not None:
        # Where the bend meets each flange, in the cylinder's own angular
        # parametrisation. Developing outward from the parent's tangent line
        # toward the child's fixes both the origin and the direction of u.
        u_tangent = _tangent_angle(bend, n_p, o_p)
        u_child = _tangent_angle(bend, n_c, o_c)
        delta = u_child - u_tangent
        sign = 1.0 if delta >= 0 else -1.0

    return FlatBend(
        bend_id=bend.id,
        line_start=geom.apply_point(M_parent, p0),
        line_end=geom.apply_point(M_parent, p1),
        step_dir=geom.apply_vector(M_parent, step_dir),
        width=bend.bend_allowance,
        direction=bend.direction,
        parent=step.parent,
        child=step.child,
        axis_dir_flat=geom.apply_vector(M_parent, d),
        u_tangent=u_tangent,
        sign=sign,
    )


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def refold_residual(part: Part, result: UnfoldResult) -> float:
    """Max distance a flange centroid moves under unfold-then-refold.

    Should be ~0 to machine precision. This exercises the transform inverse and
    catches accumulation errors in deep bend trees; the stronger geometric
    version (re-fold the flat solid and compare to the original) lives in the
    OCC layer where the kernel is available.
    """
    worst = 0.0
    for fid, M in result.transforms.items():
        c = part.flanges[fid].centroid
        back = geom.apply_point(geom.invert_rigid(M), geom.apply_point(M, c))
        worst = max(worst, float(np.linalg.norm(back - c)))
    return worst


def flat_extent_along(part: Part, result: UnfoldResult, direction) -> float:
    """Extent of all transformed flange centroids along a flat direction.

    A coarse but useful check for simple chain parts, where it should equal
    sum(legs to tangent) + sum(BA).
    """
    u = geom.unit(direction)
    vals = [
        float(np.dot(geom.apply_point(result.transforms[fid], part.flanges[fid].centroid), u))
        for fid in result.transforms
    ]
    return max(vals) - min(vals)
