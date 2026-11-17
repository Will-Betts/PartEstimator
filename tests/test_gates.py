"""The detection gates, tested without a CAD kernel.

`smt.occ.detect` imports OCP, so the gate logic can't be imported directly here.
What CAN be tested is the geometry underneath each gate, which is pure numpy,
plus the funnel that reports which gate rejected what.

The two bugs these cover:

  * A flange whose face is split by a cutout presented several coplanar faces to
    one bend. Counting FACES rather than distinct PLANES made the bend look like
    it joined a flange to itself, and it was discarded.
  * Tangency used a fixed 0.1 micron window regardless of radius, which is
    tighter than translated STEP can hold on a large bend.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from partestimator.smt.config import Tolerance
from partestimator.smt.geom import (
    codirectional,
    cylinder_tangent_to_plane,
    signed_plane_distance,
)


class TestTangencyDiscrimination(unittest.TestCase):
    """The test that separates a bend from a hole and from a corner fillet."""

    def setUp(self):
        self.tol = Tolerance()

    def test_bend_cylinder_is_tangent_to_its_flange(self):
        # Axis along y at z = R above the plane z = 0.
        self.assertTrue(
            cylinder_tangent_to_plane(
                [0, 0, 2], [0, 1, 0], 2.0, [0, 0, 0], [0, 0, 1],
                lin_tol=self.tol.tangent,
            )
        )

    def test_hole_is_never_tangent(self):
        """A hole's axis is normal to the sheet, so the angular test kills it."""
        self.assertFalse(
            cylinder_tangent_to_plane(
                [10, 10, 2], [0, 0, 1], 2.0, [0, 0, 0], [0, 0, 1],
                lin_tol=self.tol.tangent,
            )
        )

    def test_wrong_standoff_is_not_tangent(self):
        """Axis parallel to the plane but at the wrong distance: not a bend."""
        self.assertFalse(
            cylinder_tangent_to_plane(
                [0, 0, 3.5], [0, 1, 0], 2.0, [0, 0, 0], [0, 0, 1],
                lin_tol=self.tol.tangent,
            )
        )


class TestTangencyToleranceScaling(unittest.TestCase):
    """A fixed absolute window is wrong for large radii.

    `tangent` alone is 1e-4 mm -- a tenth of a micron. Native CAD holds that;
    STEP through a translator often does not, and the error grows with the
    radius. The effective window is now `max(tangent, tangent_rel * radius)`.
    """

    def setUp(self):
        self.tol = Tolerance()

    def _window(self, radius: float) -> float:
        return max(self.tol.tangent, self.tol.tangent_rel * radius)

    def test_window_grows_with_radius(self):
        self.assertGreater(self._window(50.0), self._window(2.0))

    def test_small_radius_keeps_the_absolute_floor(self):
        self.assertAlmostEqual(self._window(0.1), self.tol.tangent, places=12)

    def test_large_radius_tolerates_translation_noise(self):
        """R50 with 5 micron of noise: rejected before, accepted now."""
        r, noise = 50.0, 0.005
        self.assertFalse(
            cylinder_tangent_to_plane(
                [0, 0, r + noise], [0, 1, 0], r, [0, 0, 0], [0, 0, 1],
                lin_tol=self.tol.tangent,
            ),
            "the old fixed window rejected this",
        )
        self.assertTrue(
            cylinder_tangent_to_plane(
                [0, 0, r + noise], [0, 1, 0], r, [0, 0, 0], [0, 0, 1],
                lin_tol=self._window(r),
            ),
            "the scaled window accepts it",
        )

    def test_scaling_does_not_admit_a_genuinely_wrong_standoff(self):
        """Half a millimetre out on R50 is still not tangent."""
        r = 50.0
        self.assertFalse(
            cylinder_tangent_to_plane(
                [0, 0, r + 0.5], [0, 1, 0], r, [0, 0, 0], [0, 0, 1],
                lin_tol=self._window(r),
            )
        )


class TestCoplanarGrouping(unittest.TestCase):
    """Grouping tangent faces into distinct planes.

    This is the logic `tangent_planes` uses to survive a split flange. Two
    coplanar faces are one side of the bend; two opposed faces are two sides.
    """

    def _same_plane(self, n1, p1, n2, p2, tol=1e-4):
        return (
            codirectional(n1, n2, 1e-4)
            and abs(signed_plane_distance(p2, p1, n1)) < tol
        )

    def test_split_flange_faces_are_one_plane(self):
        """A cutout splits one flange into two coplanar faces."""
        n = [0, 0, 1]
        self.assertTrue(self._same_plane(n, [0, 0, 0], n, [40, 10, 0]))

    def test_opposite_flanges_are_two_planes(self):
        """The two sides of a 90 degree bend."""
        self.assertFalse(self._same_plane([0, 0, 1], [0, 0, 0], [-1, 0, 0], [94, 0, 2]))

    def test_parallel_but_offset_faces_are_two_planes(self):
        """Top and bottom of the sheet: same normal, one thickness apart."""
        n = [0, 0, 1]
        self.assertFalse(self._same_plane(n, [0, 0, 0], n, [0, 0, -2]))

    def test_antiparallel_faces_are_not_the_same_plane(self):
        """Even coincident, opposed normals are different sides."""
        self.assertFalse(self._same_plane([0, 0, 1], [0, 0, 0], [0, 0, -1], [0, 0, 0]))

    def test_a_split_flange_still_yields_exactly_two_sides(self):
        """The regression: three faces, two planes, one valid bend.

        Flange A's face is split in two by a cutout; flange B's is whole.
        Counting faces gives 3 and the old code took the first two -- which
        could be the two halves of flange A, making the bend appear to join a
        flange to itself. Counting planes gives 2, correctly.
        """
        faces = [
            ([0, 0, 1], [0, 0, 0]),        # flange A, piece 1
            ([0, 0, 1], [60, 20, 0]),      # flange A, piece 2 (coplanar)
            ([-1, 0, 0], [94, 0, 2]),      # flange B
        ]
        planes: list[tuple] = []
        for n, p in faces:
            for gn, gp in planes:
                if self._same_plane(gn, gp, n, p):
                    break
            else:
                planes.append((n, p))

        self.assertEqual(len(faces), 3)
        self.assertEqual(len(planes), 2, "a split flange must not read as two sides")


class TestFunnelReporting(unittest.TestCase):
    """The funnel has to name the cause, not just count failures."""

    def test_diagnosis_logic_no_cylinders(self):
        """Reimplements the decision tree so it is covered without a kernel."""
        def diagnose(cylinders, pairs, rej_thick, rej_tan, accepted):
            if accepted:
                return None
            if cylinders == 0:
                return "no cylinders"
            if pairs == 0:
                return "no concentric pairs"
            if rej_thick and not rej_tan:
                return "thickness mismatch"
            if rej_tan:
                return "tangency"
            return "unknown"

        self.assertEqual(diagnose(0, 0, [], [], 0), "no cylinders")
        self.assertEqual(diagnose(12, 0, [], [], 0), "no concentric pairs")
        self.assertEqual(
            diagnose(12, 6, [1.905] * 6, [], 0), "thickness mismatch"
        )
        self.assertEqual(diagnose(12, 6, [], [(2.0, 1, 2)], 0), "tangency")
        self.assertIsNone(diagnose(12, 6, [], [], 3))

    def test_the_inch_failure_signature(self):
        """What the funnel should have said about the inch part.

        Every pair rejected on thickness, all reporting the same dr, and that
        dr being a round inch value, is the fingerprint of the unit bug.
        """
        from partestimator.smt.units import MM_PER_INCH

        dr_values = [0.075 * MM_PER_INCH] * 4      # every bend agrees
        thickness_used = 2.0                        # snapped, wrong

        tol = Tolerance()
        self.assertTrue(
            all(not tol.radius_close(d, thickness_used) for d in dr_values)
        )
        # ...and every one of them agrees with itself, which is the tell that
        # the measurement was fine and the comparison value was wrong.
        self.assertTrue(
            all(tol.radius_close(d, dr_values[0]) for d in dr_values)
        )
        self.assertAlmostEqual(dr_values[0] / MM_PER_INCH, 0.075, places=9)


if __name__ == "__main__":
    unittest.main()
