"""Bend allowance mathematics.

    BA   = theta * (R + K*t)          bend allowance, theta in radians
    OSSB = (R + t) * tan(theta/2)     outside setback
    BD   = 2*OSSB - BA                bend deduction

Two equivalent ways to get the flat length:

    flat = sum(leg lengths measured to tangent) + sum(BA)
    flat = sum(outside dimensions)              - sum(BD)

`check_consistency` asserts they agree. It is cheap and it catches sign errors
immediately, so it runs in the test suite on every case.
"""

from __future__ import annotations

import math

from .materials import DeductionTable, Material, get_material


def k_from_curve(r_over_t: float, material: Material) -> float:
    """Linear interpolation over the material's K curve, clamped at the ends."""
    curve = material.k_curve
    if r_over_t <= curve[0][0]:
        return curve[0][1]
    if r_over_t >= curve[-1][0]:
        return curve[-1][1]
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if x0 <= r_over_t <= x1:
            f = (r_over_t - x0) / (x1 - x0)
            return y0 + f * (y1 - y0)
    return curve[-1][1]


# Method multipliers. Bottoming and coining drive the neutral axis outward
# because the material is compressed into the die rather than floated over it.
_METHOD_SHIFT = {"air": 0.0, "bottoming": 0.015, "coining": 0.04}


def k_factor(
    r_inner: float,
    thickness: float,
    material: str | Material = "CRS",
    method: str = "air",
) -> float:
    """K as a function of r/t, material and forming method.

    K is NOT a constant. Treating it as one is the single most common source of
    flat-pattern error. It rises with r/t, from roughly 0.33 at a tight radius
    toward a theoretical ceiling of 0.5.
    """
    mat = get_material(material) if isinstance(material, str) else material
    if thickness <= 0:
        raise ValueError("thickness must be positive")
    k = k_from_curve(r_inner / thickness, mat)
    k += _METHOD_SHIFT.get(method, 0.0)
    return min(k, 0.5)


def bend_allowance(angle: float, r_inner: float, thickness: float, k: float) -> float:
    """Arc length of the neutral surface through the bend. `angle` in radians."""
    if angle <= 0:
        return 0.0
    return angle * (r_inner + k * thickness)


def outside_setback(angle: float, r_inner: float, thickness: float) -> float:
    """Distance from the tangent line to the apex of the outside corner.

    Diverges as the angle approaches 180 degrees -- a flat hem has no meaningful
    outside corner. Returns inf there rather than a huge misleading number.
    """
    if angle <= 0:
        return 0.0
    if angle >= math.pi - 1e-9:
        return math.inf
    return (r_inner + thickness) * math.tan(angle / 2.0)


def bend_deduction(angle: float, r_inner: float, thickness: float, k: float) -> float:
    ossb = outside_setback(angle, r_inner, thickness)
    if math.isinf(ossb):
        return math.inf
    return 2.0 * ossb - bend_allowance(angle, r_inner, thickness, k)


def resolve_allowance(
    angle: float,
    r_inner: float,
    thickness: float,
    material: str = "CRS",
    method: str = "air",
    k_override: float | None = None,
    table: DeductionTable | None = None,
    v_die: float | None = None,
) -> tuple[float, float, str]:
    """Decide the bend allowance for one bend.

    Priority order:
      1. A measured deduction row from the shop table (most trustworthy).
      2. An explicit K override from the caller.
      3. K interpolated from the material curve.

    Returns (bend_allowance, k_used, source_label).
    """
    if table is not None:
        bd = table.lookup(material, thickness, r_inner, v_die)
        if bd is not None:
            ossb = outside_setback(angle, r_inner, thickness)
            if not math.isinf(ossb):
                ba = 2.0 * ossb - bd
                # Back out the K that this measured deduction implies, so the
                # report can show it and you can sanity check the table.
                k_implied = (
                    (ba / angle - r_inner) / thickness if angle > 0 and thickness else 0.0
                )
                return ba, k_implied, "measured table"

    if k_override is not None:
        k = k_override
        source = "user override"
    else:
        k = k_factor(r_inner, thickness, material, method)
        source = f"curve ({material}, {method})"

    return bend_allowance(angle, r_inner, thickness, k), k, source


def check_consistency(
    legs_to_tangent: list[float],
    outside_dims: list[float],
    bends: list[tuple[float, float, float, float]],
    tol: float = 1e-6,
) -> bool:
    """Assert the two flat-length formulations agree.

    `bends` entries are (angle_rad, r_inner, thickness, k).
    """
    ba_total = sum(bend_allowance(a, r, t, k) for a, r, t, k in bends)
    bd_total = sum(bend_deduction(a, r, t, k) for a, r, t, k in bends)
    via_legs = sum(legs_to_tangent) + ba_total
    via_outside = sum(outside_dims) - bd_total
    return abs(via_legs - via_outside) < tol
