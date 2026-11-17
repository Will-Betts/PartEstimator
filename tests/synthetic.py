"""Hand-built parts with known-good answers, for testing without a CAD kernel.

These encode geometry worked out by hand so the transform solver can be checked
against arithmetic rather than against itself.

THE U-CHANNEL
-------------
Base 100 mm outside, two 90 degree up-bends, legs 30 mm outside, t = 2, R = 2,
K = 0.44.

    BA   = (pi/2)(2 + 0.44*2)   = 4.5238934
    OSSB = (2 + 2) tan(45)      = 4
    BD   = 2(4) - 4.5238934     = 3.4761066
    flat = 100 + 30 + 30 - 2(3.4761066) = 153.0477868

Cross-check the other way, legs measured to the tangent lines:

    flat = 92 + 26 + 26 + 2(4.5238934) = 153.0477868   agrees

Inner surfaces are placed at z = 0 (base) with material occupying z in [-2, 0],
so the bend axes sit at z = +2.

THE Z-SECTION
-------------
Same dimensions but one bend up and one bend DOWN. The flat length is
identical; what differs is that the solver must flip which side of the sheet it
references as it crosses the down-bend. That is the case that breaks any
implementation assuming a global "top" face.
"""

from __future__ import annotations

import math

import numpy as np

from partestimator.smt.model import Bend, Flange, Part

T = 2.0
R = 2.0
K = 0.44
W = 50.0  # bend line length / part width

BA = (math.pi / 2.0) * (R + K * T)
OSSB = (R + T) * math.tan(math.pi / 4.0)
BD = 2.0 * OSSB - BA
EXPECTED_FLAT = 100.0 + 30.0 + 30.0 - 2.0 * BD


def _rect_loop(corners) -> list[np.ndarray]:
    """A single closed boundary loop, matching what `occ.topo.face_loops` caches."""
    return [np.array(corners, dtype=float)]


X_DIR = np.array([0.0, 0.0, 1.0])       # u measured from +z about the +y axis


def _inner_loop(axis_point, u_lo, u_hi):
    """The inner cylindrical face's boundary, as it would come off the kernel.

    Four corners in (u, v): both tangent lines at both ends of the bend. Lets
    the tests exercise the real development path rather than only the
    rectangular fallback.
    """
    a = np.array(axis_point, dtype=float)
    d = np.array([0.0, 1.0, 0.0])
    x = X_DIR
    y = np.cross(d, x)

    def pt(u, v):
        return a + R * (math.cos(u) * x + math.sin(u) * y) + v * d

    return [np.array([pt(u_lo, 0.0), pt(u_lo, W), pt(u_hi, W), pt(u_hi, 0.0)])]


def _bend(bid, axis_point, flange_a, flange_b, u_lo=None, u_hi=None):
    b = Bend(
        id=bid,
        axis_point=np.array(axis_point, dtype=float),
        axis_dir=np.array([0.0, 1.0, 0.0]),
        r_inner=R,
        thickness=T,
        angle=math.pi / 2.0,
        length_envelope=W,
        length_material=W,
        flange_a=flange_a,
        flange_b=flange_b,
        v_min=0.0,
        v_max=W,
    )
    if u_lo is not None:
        b.axis_x_dir = X_DIR
        b.u_min, b.u_max = u_lo, u_hi
        b.inner_loops = _inner_loop(axis_point, u_lo, u_hi)
    return b


def u_channel() -> Part:
    """Base plus two 90 degree up-bends."""
    part = Part(thickness=T, material="CRS")

    # Base: inner face at z = 0 spanning x 0..92, material below to z = -2.
    part.flanges[0] = Flange(
        id=0,
        normal=np.array([0.0, 0.0, 1.0]),
        origin=np.array([0.0, 0.0, 0.0]),
        origin_b=np.array([0.0, 0.0, -T]),
        centroid=np.array([46.0, W / 2, 0.0]),
        area=92.0 * W,
        loops_a=_rect_loop([(0, 0, 0), (92, 0, 0), (92, W, 0), (0, W, 0)]),
        loops_b=_rect_loop([(0, 0, -T), (92, 0, -T), (92, W, -T), (0, W, -T)]),
    )

    # Right leg: inner face at x = 94, running z 2..28, normal pointing -x.
    part.flanges[1] = Flange(
        id=1,
        normal=np.array([-1.0, 0.0, 0.0]),
        origin=np.array([94.0, 0.0, 2.0]),
        origin_b=np.array([96.0, 0.0, 2.0]),
        centroid=np.array([94.0, W / 2, 15.0]),
        area=26.0 * W,
        loops_a=_rect_loop([(94, 0, 2), (94, W, 2), (94, W, 28), (94, 0, 28)]),
        loops_b=_rect_loop([(96, 0, 2), (96, W, 2), (96, W, 28), (96, 0, 28)]),
    )

    # Left leg: mirror image, inner face at x = -2, normal pointing +x.
    part.flanges[2] = Flange(
        id=2,
        normal=np.array([1.0, 0.0, 0.0]),
        origin=np.array([-2.0, 0.0, 2.0]),
        origin_b=np.array([-4.0, 0.0, 2.0]),
        centroid=np.array([-2.0, W / 2, 15.0]),
        area=26.0 * W,
        loops_a=_rect_loop([(-2, 0, 2), (-2, W, 2), (-2, W, 28), (-2, 0, 28)]),
        loops_b=_rect_loop([(-4, 0, 2), (-4, W, 2), (-4, W, 28), (-4, 0, 28)]),
    )

    part.bends[0] = _bend(
        0, [92.0, 0.0, 2.0], 0, 1, math.pi / 2, math.pi
    )
    part.bends[1] = _bend(
        1, [0.0, 0.0, 2.0], 0, 2, math.pi, 3 * math.pi / 2
    )
    return part


def z_section() -> Part:
    """Base plus one up-bend and one down-bend.

    The down-bend's inner surface is the BOTTOM of the sheet, so its axis sits
    at z = -4 and the solver has to reference the far side of the material for
    that branch.
    """
    part = Part(thickness=T, material="CRS")

    part.flanges[0] = Flange(
        id=0,
        normal=np.array([0.0, 0.0, 1.0]),
        origin=np.array([0.0, 0.0, 0.0]),
        origin_b=np.array([0.0, 0.0, -T]),
        centroid=np.array([46.0, W / 2, 0.0]),
        area=92.0 * W,
        loops_a=_rect_loop([(0, 0, 0), (92, 0, 0), (92, W, 0), (0, W, 0)]),
        loops_b=_rect_loop([(0, 0, -T), (92, 0, -T), (92, W, -T), (0, W, -T)]),
    )

    # Right leg bends UP, same as the channel.
    part.flanges[1] = Flange(
        id=1,
        normal=np.array([-1.0, 0.0, 0.0]),
        origin=np.array([94.0, 0.0, 2.0]),
        origin_b=np.array([96.0, 0.0, 2.0]),
        centroid=np.array([94.0, W / 2, 15.0]),
        area=26.0 * W,
        loops_a=_rect_loop([(94, 0, 2), (94, W, 2), (94, W, 28), (94, 0, 28)]),
        loops_b=_rect_loop([(96, 0, 2), (96, W, 2), (96, W, 28), (96, 0, 28)]),
    )

    # Left leg bends DOWN: inner face is the underside, at x = -2 running
    # z -4..-30, normal +x. Outer (top-side) face at x = -4.
    part.flanges[2] = Flange(
        id=2,
        normal=np.array([1.0, 0.0, 0.0]),
        origin=np.array([-2.0, 0.0, -4.0]),
        origin_b=np.array([-4.0, 0.0, -4.0]),
        centroid=np.array([-2.0, W / 2, -17.0]),
        area=26.0 * W,
        loops_a=_rect_loop([(-2, 0, -4), (-2, W, -4), (-2, W, -30), (-2, 0, -30)]),
        loops_b=_rect_loop([(-4, 0, -4), (-4, W, -4), (-4, W, -30), (-4, 0, -30)]),
    )

    part.bends[0] = _bend(0, [92.0, 0.0, 2.0], 0, 1)
    part.bends[1] = _bend(1, [0.0, 0.0, -4.0], 0, 2)
    return part


def closed_box() -> Part:
    """Four flanges bent into a closed loop -- must be rejected, not unfolded."""
    part = Part(thickness=T, material="CRS")
    for i in range(4):
        part.flanges[i] = Flange(
            id=i,
            normal=np.array([0.0, 0.0, 1.0]),
            origin=np.array([0.0, 0.0, float(i)]),
            origin_b=np.array([0.0, 0.0, float(i) - T]),
            centroid=np.array([float(i), 0.0, 0.0]),
            area=100.0,
        )
    for i in range(4):
        part.bends[i] = _bend(i, [float(i), 0.0, 2.0], i, (i + 1) % 4)
    return part


def twin_tab_flange() -> Part:
    """One bend line, TWO separate tabs on it. The v12 bug, as a fixture.

    Modelled on a real part: a plate with a long bend across the top, but the
    flange material cut away in the middle so only two tabs stand up -- one at
    each end, each with a hole. The tabs are coplanar, share one bend axis, and
    are disconnected from each other.

    Geometry, in millimetres:

        base            0 <= x <= 200, in z = 0, outward normal +Z
        bend            axis along X at (y = 0, z = R), 90 degrees
        tab A           0 <= x <= 40      standing up in y = 0
        tab B         160 <= x <= 200     standing up in y = 0

    The base is one region; the flange above the bend is two. Before the fix,
    only the larger tab reached the flat pattern and the other vanished with no
    warning of any kind.
    """
    part = Part(thickness=T, material="CRS")
    part.measured_thickness = T

    base_a = _rect_loop([(0, -80, 0), (200, -80, 0), (200, 0, 0), (0, 0, 0)])
    base_b = _rect_loop([(0, -80, -T), (200, -80, -T), (200, 0, -T), (0, 0, -T)])

    part.flanges[0] = Flange(
        id=0,
        normal=np.array([0.0, 0.0, 1.0]),
        origin=np.array([0.0, 0.0, 0.0]),
        origin_b=np.array([0.0, 0.0, -T]),
        centroid=np.array([100.0, -40.0, 0.0]),
        area=200.0 * 80.0,
        loops_a=[base_a],
        loops_b=[base_b],
    )

    # The upstanding flange: two coplanar regions in the plane y = 0.
    def tab(x0, x1, offset):
        return _rect_loop([
            (x0, offset, 0.0), (x1, offset, 0.0),
            (x1, offset, 30.0), (x0, offset, 30.0),
        ])

    part.flanges[1] = Flange(
        id=1,
        normal=np.array([0.0, -1.0, 0.0]),
        origin=np.array([0.0, 0.0, 0.0]),
        origin_b=np.array([0.0, T, 0.0]),
        centroid=np.array([100.0, 0.0, 15.0]),
        area=2 * 40.0 * 30.0,
        loops_a=[tab(0.0, 40.0, 0.0), tab(160.0, 200.0, 0.0)],
        loops_b=[tab(0.0, 40.0, T), tab(160.0, 200.0, T)],
    )

    part.bends[0] = _bend(
        0, np.array([0.0, -R, R]), 0, 1, u_lo=0.0, u_hi=200.0,
    )
    part.bends[0].axis_dir = np.array([1.0, 0.0, 0.0])
    part.bends[0].v_min, part.bends[0].v_max = 0.0, 200.0
    return part
