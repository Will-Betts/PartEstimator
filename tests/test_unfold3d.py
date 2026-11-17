"""The geometry that decides where the solid gets cut.

This is the riskiest calculation in the solid unfold, because a wrong answer
does not raise -- it produces a plausible-looking flat part with a flange cut in
the wrong place, which is exactly the class of bug that has reached the shop
floor before in this project.

It lives in `smt.geom` rather than in `smt.occ` precisely so it can be tested
here, on a machine with no CAD kernel.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from partestimator.smt.geom import bend_cut_plane, signed_plane_distance, unit


class TestBendCutPlane(unittest.TestCase):
    """A 90 degree bend at the edge of a plate lying in z = 0."""

    def setUp(self):
        self.t = 2.0
        self.r = 3.0
        # Flange in the z = 0 plane, outward normal +Z, running out to x = 0.
        self.normal = np.array([0.0, 0.0, 1.0])
        self.origin = np.array([0.0, 0.0, 0.0])
        # Bend axis parallel to Y, one inner radius above the plane, so the
        # tangent line is the y axis at x = 0.
        self.axis_point = np.array([0.0, 0.0, self.r])
        self.axis_dir = np.array([0.0, 1.0, 0.0])

    def test_the_cut_lands_on_the_tangent_line(self):
        point, _n = bend_cut_plane(
            self.axis_point, self.axis_dir, self.normal, self.origin
        )
        # Dropped straight onto the flange plane, directly under the axis.
        np.testing.assert_allclose(point, [0.0, 0.0, 0.0], atol=1e-12)

    def test_the_cut_plane_is_square_to_the_sheet(self):
        _p, n = bend_cut_plane(
            self.axis_point, self.axis_dir, self.normal, self.origin
        )
        self.assertAlmostEqual(abs(float(np.dot(n, self.normal))), 0.0, places=12)

    def test_the_cut_plane_contains_the_axis_direction(self):
        _p, n = bend_cut_plane(
            self.axis_point, self.axis_dir, self.normal, self.origin
        )
        self.assertAlmostEqual(float(np.dot(n, self.axis_dir)), 0.0, places=12)

    def test_the_normal_is_a_unit_vector(self):
        _p, n = bend_cut_plane(
            self.axis_point, self.axis_dir, self.normal, self.origin
        )
        self.assertAlmostEqual(float(np.linalg.norm(n)), 1.0, places=12)

    def test_the_radius_is_never_needed(self):
        """The tangency is found by projection, not by stepping off a radius.

        Which is the point: inner and outer radius differ by the thickness, and
        a version that stepped off 'the radius' would be wrong by a whole
        material thickness whenever it picked the other one.
        """
        for radius in (0.5, 3.0, 12.0):
            with self.subTest(radius=radius):
                point, _n = bend_cut_plane(
                    np.array([0.0, 0.0, radius]), self.axis_dir,
                    self.normal, self.origin,
                )
                np.testing.assert_allclose(point, [0.0, 0.0, 0.0], atol=1e-12)

    def test_it_works_on_a_flange_that_is_not_axis_aligned(self):
        """Real parts are not drawn conveniently."""
        n = unit(np.array([1.0, 2.0, 3.0]))
        origin = np.array([5.0, -2.0, 1.0])
        axis = unit(np.cross(n, np.array([0.0, 1.0, 0.0])))   # lies in the plane
        axis_point = origin + n * 4.0 + axis * 7.0

        point, cut_n = bend_cut_plane(axis_point, axis, n, origin)

        # On the flange plane...
        self.assertAlmostEqual(
            float(signed_plane_distance(point, origin, n)), 0.0, places=10
        )
        # ...square to it, and containing the axis.
        self.assertAlmostEqual(float(np.dot(cut_n, n)), 0.0, places=10)
        self.assertAlmostEqual(float(np.dot(cut_n, axis)), 0.0, places=10)

    def test_an_axis_normal_to_the_sheet_is_rejected(self):
        """That describes a hole, not a bend, and there is no cut plane."""
        with self.assertRaises(ValueError):
            bend_cut_plane(
                self.axis_point, self.normal, self.normal, self.origin
            )

    def test_the_side_the_axis_is_on_does_not_move_the_cut(self):
        """An up bend and a down bend cut in the same place.

        The tangent line is where the flat surface ends, and that is a property
        of the flange, not of which way the flange swings.
        """
        up, _ = bend_cut_plane(
            np.array([0.0, 0.0, self.r]), self.axis_dir, self.normal, self.origin
        )
        down, _ = bend_cut_plane(
            np.array([0.0, 0.0, -self.r]), self.axis_dir, self.normal, self.origin
        )
        np.testing.assert_allclose(up, down, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
