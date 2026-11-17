"""Blank placement: the bounding box has to be the one you would actually cut.

Nothing in the unfold decides how the blank is oriented -- the solver holds one
flange fixed and swings the rest into its plane, so the flat pattern inherits
whatever direction that flange happened to face in the model. That is fine for
the geometry and wrong for the number a material estimate is built on.

A 400 x 100 blank drawn on the diagonal measures roughly 396 x 296 -- three
times the area, for a part that has not changed at all.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from partestimator.smt import orient

RECT = np.array([[0.0, 0.0], [400.0, 0.0], [400.0, 100.0], [0.0, 100.0]])
L_BRACKET = np.array([
    [0.0, 0.0], [300.0, 0.0], [300.0, 40.0],
    [80.0, 40.0], [80.0, 220.0], [0.0, 220.0],
])


def turned(pts, deg):
    return pts @ orient.rotation_2d(math.radians(deg)).T


class TestMinimumArea(unittest.TestCase):
    def test_a_diagonal_rectangle_comes_back_square(self):
        for deg in (0, 7.5, 12.5, 30, 45, 60, 89, 137, 213):
            with self.subTest(drawn_at=deg):
                pts = turned(RECT, deg)
                w, h = orient.envelope(
                    orient.apply(pts, orient.choose_angle(pts))
                )
                self.assertAlmostEqual(w, 400.0, places=6)
                self.assertAlmostEqual(h, 100.0, places=6)

    def test_the_long_side_always_lands_on_x(self):
        """Nobody wants a blank reported 400 x 100 one day and 100 x 400 the next.

        The quarter-turn that puts the long side on X has to be decided AFTER
        the rotation is folded into a quarter, not before -- folding first
        throws away exactly the quarter turn the decision just added, which is
        how a part drawn at 90 degrees came back on its side.
        """
        for deg in (0, 45, 90, 135, 180, 270):
            with self.subTest(drawn_at=deg):
                pts = turned(RECT, deg)
                w, h = orient.envelope(
                    orient.apply(pts, orient.choose_angle(pts))
                )
                self.assertGreaterEqual(w, h)

    def test_an_already_square_part_is_left_alone(self):
        """No rotation, and honest about having saved nothing."""
        angle = orient.choose_angle(RECT)
        self.assertAlmostEqual(angle, 0.0, places=9)
        self.assertAlmostEqual(orient.saving(RECT, angle), 0.0, places=9)

    def test_the_saving_is_real_and_reported(self):
        pts = turned(RECT, 45.0)
        saving = orient.saving(pts, orient.choose_angle(pts))
        # A 4:1 rectangle at 45 degrees: envelope 353.55 square, area 125000,
        # against 40000 turned back. Saving = 1 - 40000/125000 = 0.68.
        self.assertAlmostEqual(saving, 0.68, places=6)

    def test_a_non_convex_part_still_converges(self):
        sizes = set()
        for deg in (0, 17, 33, 51, 90, 122):
            pts = turned(L_BRACKET, deg)
            w, h = orient.envelope(orient.apply(pts, orient.choose_angle(pts)))
            sizes.add((round(w, 6), round(h, 6)))
        self.assertEqual(len(sizes), 1, f"orientation not stable: {sizes}")

    def test_rotation_is_rigid(self):
        """The part must not change -- only the axes it is measured against."""
        pts = turned(L_BRACKET, 31.0)
        moved = orient.apply(pts, orient.choose_angle(pts))

        def perimeter(p):
            d = np.diff(np.vstack([p, p[:1]]), axis=0)
            return float(np.linalg.norm(d, axis=1).sum())

        self.assertAlmostEqual(perimeter(pts), perimeter(moved), places=9)


class TestGrainWins(unittest.TestCase):
    """Sheet is anisotropic, and no envelope saving is worth a cracked bend."""

    def test_a_grain_constraint_blocks_the_free_rotation(self):
        pts = turned(RECT, 30.0)
        free = orient.choose_angle(pts)
        locked = orient.choose_angle(pts, grain_deg=0)
        self.assertNotAlmostEqual(free, locked, places=6)
        self.assertAlmostEqual(locked, 0.0, places=9)

    def test_grain_still_picks_the_better_of_the_two_legal_placements(self):
        """Square to the grain leaves one choice: 0 or 90. Take the wider."""
        pts = turned(RECT, 90.0)
        angle = orient.choose_angle(pts, grain_deg=0)
        w, h = orient.envelope(orient.apply(pts, angle))
        self.assertGreaterEqual(w, h)

    def test_grain_mode_without_a_direction_does_nothing(self):
        self.assertAlmostEqual(
            orient.choose_angle(turned(RECT, 30.0), mode="grain"), 0.0, places=9
        )


class TestAsModelled(unittest.TestCase):
    def test_as_modelled_leaves_it_exactly_where_it_was(self):
        pts = turned(RECT, 30.0)
        self.assertAlmostEqual(
            orient.choose_angle(pts, mode="as_modelled"), 0.0, places=9
        )


class TestFrameComposition(unittest.TestCase):
    def test_the_3d_frame_matches_the_2d_rotation(self):
        """The whole design rests on these being the same rotation.

        The angle is folded into the flat frame so the preview, the DXF and the
        extruded solid all inherit it and cannot disagree. If the 4x4 and the
        2x2 ever diverge, they silently would.
        """
        angle = math.radians(23.0)
        M = orient.frame_rotation_3d(angle)
        pts = np.array([[3.0, 7.0], [-11.0, 2.5], [0.0, 0.0]])

        via_2d = orient.apply(pts, angle)
        pts3 = np.column_stack([pts, np.zeros(len(pts))])
        via_3d = (pts3 @ M[:3, :3].T + M[:3, 3])[:, :2]

        np.testing.assert_allclose(via_2d, via_3d, atol=1e-12)

    def test_the_3d_frame_leaves_z_alone(self):
        M = orient.frame_rotation_3d(math.radians(23.0))
        np.testing.assert_allclose(M[2], [0.0, 0.0, 1.0, 0.0], atol=1e-15)


class TestOriginPlacement(unittest.TestCase):
    def test_the_blank_lands_in_the_positive_quadrant(self):
        moved = orient.translate_to_origin(turned(RECT, 30.0))
        np.testing.assert_allclose(moved.min(axis=0), [0.0, 0.0], atol=1e-12)


class TestDegenerateInput(unittest.TestCase):
    def test_two_points_do_not_raise(self):
        self.assertEqual(orient.choose_angle(np.zeros((2, 2))), 0.0)

    def test_no_points_do_not_raise(self):
        self.assertEqual(orient.envelope(np.zeros((0, 2))), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
