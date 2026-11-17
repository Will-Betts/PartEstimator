"""Developing a bend on its neutral surface.

WHY THIS REPLACED THE RECTANGLE
-------------------------------
Every bend used to contribute a rectangle of width BA spanning the full bend
line. That is right only when the bend zone really is a rectangle -- flanges
meeting it squarely along its whole length, nothing cut out of it. Real parts
routinely break all three conditions.

A cylinder is a developable surface, so the honest construction is to unroll the
actual trimmed face. The neutral surface at rho = R + K*t is the layer that
neither stretches nor compresses, which is precisely what the K-factor asserts,
so arc length there maps one-to-one onto flat distance:

    flat = tangent_line + v * axis + rho * (u - u_tangent) * across

The bend allowance stops being a formula bolted onto a rectangle and becomes
what it physically is: the arc length rho*theta of that development. These tests
check that the two agree exactly.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from partestimator.smt.allowance import bend_allowance
from partestimator.smt.geom import cylinder_uv, develop_cylinder

# The same bend as the U-channel used everywhere else: t = 2, R = 2, K = 0.44.
T, R, K = 2.0, 2.0, 0.44
RHO = R + K * T                     # 2.88
THETA = math.pi / 2
BA = THETA * RHO                    # 4.5238934

# Axis along +y at (92, *, 2); inner surface tangent to z = 0 at x = 92.
AXIS_P = np.array([92.0, 0.0, 2.0])
AXIS_D = np.array([0.0, 1.0, 0.0])
X_DIR = np.array([0.0, 0.0, 1.0])   # u measured from +z


def inner_surface_point(u: float, v: float) -> np.ndarray:
    """A point on the inner cylinder at angle u, axial position v."""
    d = AXIS_D
    x = X_DIR
    y = np.cross(d, x)
    return AXIS_P + R * (math.cos(u) * x + math.sin(u) * y) + v * d


class TestCylinderUV(unittest.TestCase):
    def test_round_trip(self):
        for u_in in (-1.2, 0.0, 0.5, math.pi / 2, 2.0):
            for v_in in (0.0, 17.5, 50.0):
                p = inner_surface_point(u_in, v_in)
                u, v = cylinder_uv([p], AXIS_P, AXIS_D, X_DIR, u_reference=u_in)
                self.assertAlmostEqual(float(u[0]), u_in, places=9)
                self.assertAlmostEqual(float(v[0]), v_in, places=9)

    def test_v_is_distance_along_the_axis(self):
        p = inner_surface_point(0.3, 25.0)
        _u, v = cylinder_uv([p], AXIS_P, AXIS_D, X_DIR)
        self.assertAlmostEqual(float(v[0]), 25.0, places=9)

    def test_unwrapping_across_the_branch_cut(self):
        """A bend straddling atan2's discontinuity must not tear in half.

        Without unwrapping, points either side of the -pi/+pi boundary come back
        2*pi apart and develop to opposite ends of the blank.
        """
        us = np.array([math.pi - 0.1, math.pi + 0.1])   # spans the cut
        pts = [inner_surface_point(u, 0.0) for u in us]
        u, _v = cylinder_uv(pts, AXIS_P, AXIS_D, X_DIR, u_reference=math.pi)
        self.assertLess(abs(float(u[1] - u[0])), 0.5,
                        "points either side of the cut must stay adjacent")

    def test_reference_keeps_a_full_sweep_contiguous(self):
        """Referencing the middle of the sweep gives +/- pi of headroom."""
        u_lo, u_hi = -2.5, 2.5          # a 286 degree sweep
        u_mid = 0.5 * (u_lo + u_hi)
        us = np.linspace(u_lo, u_hi, 25)
        pts = [inner_surface_point(x, 0.0) for x in us]
        u, _v = cylinder_uv(pts, AXIS_P, AXIS_D, X_DIR, u_reference=u_mid)
        np.testing.assert_allclose(u, us, atol=1e-9)


class TestDevelopment(unittest.TestCase):
    """The core claim: arc length on the neutral surface is the flat length."""

    def setUp(self):
        self.origin = np.array([0.0, 0.0])
        self.axis = np.array([0.0, 1.0])     # bend line runs along flat +y
        self.across = np.array([1.0, 0.0])   # downstream is flat +x

    def _dev(self, u, v, u_tangent=0.0, sign=1.0):
        return develop_cylinder(
            np.atleast_1d(u), np.atleast_1d(v), RHO, u_tangent, sign,
            self.origin, self.axis, self.across,
        )

    def test_tangent_line_lands_on_the_origin(self):
        xy = self._dev(0.0, 0.0)
        np.testing.assert_allclose(xy[0], [0.0, 0.0], atol=1e-12)

    def test_far_tangent_line_lands_at_the_bend_allowance(self):
        """The whole point: developing through theta gives exactly BA."""
        xy = self._dev(THETA, 0.0)
        self.assertAlmostEqual(float(xy[0][0]), BA, places=9)
        self.assertAlmostEqual(float(xy[0][0]), 4.5238934, places=6)

    def test_matches_the_allowance_formula_exactly(self):
        """Development and the BA formula must not be able to disagree."""
        for r in (0.5, 2.0, 6.0, 25.0):
            for k in (0.33, 0.44, 0.50):
                for theta in (math.radians(30), math.pi / 2, math.radians(135)):
                    rho = r + k * T
                    xy = develop_cylinder(
                        [theta], [0.0], rho, 0.0, 1.0,
                        self.origin, self.axis, self.across,
                    )
                    self.assertAlmostEqual(
                        float(xy[0][0]),
                        bend_allowance(theta, r, T, k),
                        places=9,
                        msg=f"R{r} K{k} theta{math.degrees(theta):.0f}",
                    )

    def test_axial_position_passes_through_unchanged(self):
        """Bending does not stretch along the bend line."""
        xy = self._dev(0.4, 37.0)
        self.assertAlmostEqual(float(xy[0][1]), 37.0, places=9)

    def test_sign_flips_the_development_direction(self):
        a = self._dev(THETA, 0.0, sign=1.0)
        b = self._dev(THETA, 0.0, sign=-1.0)
        self.assertAlmostEqual(float(a[0][0]), -float(b[0][0]), places=12)

    def test_u_tangent_offsets_the_origin(self):
        """Developing from a non-zero tangent angle still starts at zero."""
        xy = self._dev(1.3, 0.0, u_tangent=1.3)
        np.testing.assert_allclose(xy[0], [0.0, 0.0], atol=1e-12)

    def test_development_preserves_arc_length(self):
        """Isometry: equal steps in u give equal steps in the flat."""
        us = np.linspace(0.0, THETA, 11)
        xy = develop_cylinder(
            us, np.zeros_like(us), RHO, 0.0, 1.0,
            self.origin, self.axis, self.across,
        )
        steps = np.diff(xy[:, 0])
        np.testing.assert_allclose(steps, steps[0], atol=1e-12)
        self.assertAlmostEqual(float(steps.sum()), BA, places=9)


class TestNonRectangularBendZone(unittest.TestCase):
    """The case the rectangle got wrong.

    A slot crossing the bend leaves a notch in the bend zone. Developing the
    real boundary keeps the notch; a rectangle of width BA fills it in with
    material that is not there.
    """

    def setUp(self):
        self.origin = np.array([0.0, 0.0])
        self.axis = np.array([0.0, 1.0])
        self.across = np.array([1.0, 0.0])

    def test_a_notched_boundary_develops_with_its_notch(self):
        # Boundary of a bend zone 50 long with a 10-wide slot through the
        # middle, expressed in (u, v) on the cylinder.
        loop_u = [0.0, 0.0, THETA, THETA, 0.0, 0.0, THETA, THETA]
        loop_v = [0.0, 20.0, 20.0, 30.0, 30.0, 50.0, 50.0, 0.0]

        xy = develop_cylinder(
            loop_u, loop_v, RHO, 0.0, 1.0, self.origin, self.axis, self.across
        )

        # The developed patch still spans 0..BA across and 0..50 along.
        self.assertAlmostEqual(float(xy[:, 0].max()), BA, places=9)
        self.assertAlmostEqual(float(xy[:, 1].max()), 50.0, places=9)

        # ...and the notch survives: some boundary points sit at v between 20
        # and 30 at the far edge, which a plain rectangle would never contain.
        far_edge = xy[np.isclose(xy[:, 0], BA)]
        self.assertTrue(
            any(20.0 - 1e-9 <= p[1] <= 30.0 + 1e-9 for p in far_edge),
            "the notch must appear in the developed boundary",
        )

    def test_a_skewed_flange_boundary_develops_as_a_trapezoid(self):
        """A flange meeting the bend at an angle is not a rectangle.

        The tangent boundary is straight, but the far boundary need not be --
        develop it and the zone comes out trapezoidal, which is correct.
        """
        loop_u = [0.0, 0.0, THETA, THETA * 0.5]
        loop_v = [0.0, 50.0, 50.0, 0.0]
        xy = develop_cylinder(
            loop_u, loop_v, RHO, 0.0, 1.0, self.origin, self.axis, self.across
        )
        widths = {round(float(p[0]), 9) for p in xy}
        self.assertGreater(len(widths), 2, "a trapezoid has more than two x values")
        self.assertAlmostEqual(max(widths), BA, places=9)


class TestDevelopmentAgreesWithTheRectangleWhenItShould(unittest.TestCase):
    """A genuinely rectangular bend zone must develop to the old answer.

    The development is a generalisation, not a replacement -- where the old
    rectangle was right it has to stay right, or every previously correct part
    would shift.
    """

    def test_rectangular_zone_matches_the_rectangle(self):
        origin = np.array([0.0, 0.0])
        axis = np.array([0.0, 1.0])
        across = np.array([1.0, 0.0])
        length = 50.0

        loop_u = [0.0, 0.0, THETA, THETA]
        loop_v = [0.0, length, length, 0.0]
        developed = develop_cylinder(
            loop_u, loop_v, RHO, 0.0, 1.0, origin, axis, across
        )

        rectangle = np.array([
            [0.0, 0.0], [0.0, length], [BA, length], [BA, 0.0],
        ])
        np.testing.assert_allclose(developed, rectangle, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
