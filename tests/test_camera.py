"""Which way the model turns when you drag. Pinned, because it was wrong.

v7 inverted the vertical drag and clamped the tumble to 180 degrees. Both are
one-line properties that are obvious the moment anything checks them and
invisible to a careful reading, which is exactly what tests are for.

The rule every test below encodes: THE SURFACE UNDER THE CURSOR FOLLOWS THE
CURSOR. Drag down, the face you are looking at goes down. Drag right, it goes
right.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from partestimator.smt import camera

#: The point on the front of the model, in camera space: straight at the viewer.
FRONT_POINT = np.array([0.0, 0.0, 1.0])


def moved(R_before, R_after, world_point):
    """Where a fixed world point sits in camera space, before and after."""
    return R_before @ world_point, R_after @ world_point


class TestDragDirection(unittest.TestCase):
    """The signs. All four, individually, so a flip cannot hide."""

    def setUp(self):
        self.R = camera.FRONT.copy()
        # The world point currently facing the viewer.
        self.p_world = np.linalg.inv(self.R) @ FRONT_POINT

    def test_dragging_down_moves_the_front_face_down(self):
        after = camera.tumble(self.R, dx=0.0, dy=25.0)
        _b, a = moved(self.R, after, self.p_world)
        self.assertLess(a[1], -1e-6, "front face did not move DOWN on a down-drag")

    def test_dragging_up_moves_the_front_face_up(self):
        after = camera.tumble(self.R, dx=0.0, dy=-25.0)
        _b, a = moved(self.R, after, self.p_world)
        self.assertGreater(a[1], 1e-6, "front face did not move UP on an up-drag")

    def test_dragging_right_moves_the_front_face_right(self):
        after = camera.tumble(self.R, dx=25.0, dy=0.0)
        _b, a = moved(self.R, after, self.p_world)
        self.assertGreater(a[0], 1e-6, "front face did not move RIGHT")

    def test_dragging_left_moves_the_front_face_left(self):
        after = camera.tumble(self.R, dx=-25.0, dy=0.0)
        _b, a = moved(self.R, after, self.p_world)
        self.assertLess(a[0], -1e-6, "front face did not move LEFT")

    def test_dragging_down_brings_the_top_of_the_part_toward_you(self):
        """The same rule stated the other way round, as a sanity cross-check.

        World +Z is up. Drag down and you should end up looking at the top, so
        the world up axis must swing toward the viewer -- camera +Z.
        """
        after = camera.tumble(self.R, dx=0.0, dy=30.0)
        world_up = np.array([0.0, 0.0, 1.0])
        self.assertGreater((after @ world_up)[2], (self.R @ world_up)[2])


class TestUnlimitedRotation(unittest.TestCase):
    """No clamp, no pole, no wall to hit."""

    def test_it_keeps_turning_well_past_a_quarter_turn(self):
        """Total angular travel, not distinct positions.

        The first version of this test counted distinct heights and expected
        them all to differ, which is wrong for a reason worth recording: a full
        turn is periodic, so a point legitimately revisits the same height twice
        per revolution. The test failed on correct code. What actually
        distinguishes turning from stalling is how far the point TRAVELS, so
        that is what is measured -- the angle swept between successive samples,
        summed.
        """
        R = camera.FRONT.copy()
        p_world = np.linalg.inv(R) @ FRONT_POINT

        travelled = 0.0
        previous = R @ p_world
        for _ in range(40):                       # 40 x 25 px x 0.36 = 360 deg
            R = camera.tumble(R, dx=0.0, dy=25.0)
            current = R @ p_world
            cos = float(np.clip(np.dot(previous, current), -1.0, 1.0))
            travelled += math.degrees(math.acos(cos))
            previous = current

        self.assertAlmostEqual(travelled, 360.0, delta=1.0)

    def test_a_turntable_clamp_would_fail_the_test_above(self):
        """Documents what the old behaviour did, so the regression stays visible.

        v7 clamped elevation to +/-90 degrees. Under that rule the same drag
        travels a quarter turn and then stops dead, which is what a 180 degree
        limit feels like from the outside.
        """
        elevation = 0.0
        travelled = 0.0
        for _ in range(40):
            before = elevation
            elevation = float(np.clip(
                elevation + math.radians(25.0 * camera.DEGREES_PER_PIXEL),
                -math.pi / 2, math.pi / 2,
            ))
            travelled += math.degrees(abs(elevation - before))
        self.assertAlmostEqual(travelled, 90.0, delta=1.0)

    def test_a_full_turn_returns_to_the_start(self):
        """360 degrees of dragging must land back where it began."""
        R = camera.FRONT.copy()
        steps = int(round(360.0 / camera.DEGREES_PER_PIXEL))
        for _ in range(steps):
            R = camera.tumble(R, dx=0.0, dy=1.0)
        np.testing.assert_allclose(R, camera.FRONT, atol=1e-6)

    def test_it_can_tumble_right_over_the_top_and_carry_on(self):
        """Past the pole the part should keep going, not stop or jump."""
        R = camera.FRONT.copy()
        world_up = np.array([0.0, 0.0, 1.0])

        # A quarter turn puts world up straight at the viewer.
        quarter = int(round(90.0 / camera.DEGREES_PER_PIXEL))
        for _ in range(quarter):
            R = camera.tumble(R, dx=0.0, dy=1.0)
        self.assertAlmostEqual(float((R @ world_up)[2]), 1.0, places=3)

        # Another quarter turn puts it pointing down the screen. A turntable
        # cannot reach here at all.
        for _ in range(quarter):
            R = camera.tumble(R, dx=0.0, dy=1.0)
        self.assertAlmostEqual(float((R @ world_up)[1]), -1.0, places=3)


class TestMatrixStaysARotation(unittest.TestCase):
    def test_drift_does_not_accumulate(self):
        """Thousands of increments must not shear the model.

        Without re-orthonormalising, composing small rotations slowly loses
        orthogonality and the matrix stops being a rotation -- at which point
        it scales and skews the part instead of turning it.
        """
        R = camera.FRONT.copy()
        rng = np.random.default_rng(0)
        for _ in range(4000):
            R = camera.tumble(R, dx=float(rng.uniform(-9, 9)),
                              dy=float(rng.uniform(-9, 9)))
        self.assertTrue(camera.is_rotation(R, tol=1e-9))

    def test_orthonormalise_never_produces_a_reflection(self):
        """A reflection would turn the part inside out and mirror the flat."""
        bad = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        out = camera.orthonormalise(bad)
        self.assertGreater(float(np.linalg.det(out)), 0.0)


class TestNamedViews(unittest.TestCase):
    def test_every_named_view_is_a_rotation(self):
        for name in camera.NAMED_VIEWS:
            with self.subTest(view=name):
                self.assertTrue(camera.is_rotation(camera.named(name)))

    def test_front_puts_world_z_up_the_screen(self):
        R = camera.named("front")
        np.testing.assert_allclose(R @ np.array([0.0, 0.0, 1.0]),
                                   [0.0, 1.0, 0.0], atol=1e-12)

    def test_front_puts_world_x_to_the_right(self):
        R = camera.named("front")
        np.testing.assert_allclose(R @ np.array([1.0, 0.0, 0.0]),
                                   [1.0, 0.0, 0.0], atol=1e-12)

    def test_top_looks_straight_down(self):
        """From the top view, world -Z must point at the viewer."""
        R = camera.named("top")
        self.assertAlmostEqual(float((R @ np.array([0.0, 0.0, 1.0]))[2]),
                               1.0, places=9)

    def test_right_and_left_are_opposite(self):
        r = camera.named("right") @ np.array([0.0, 1.0, 0.0])
        l = camera.named("left") @ np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(r, -l, atol=1e-12)

    def test_a_named_view_after_tumbling_resets_completely(self):
        R = camera.FRONT.copy()
        for _ in range(50):
            R = camera.tumble(R, dx=7.0, dy=-4.0)
        np.testing.assert_allclose(camera.named("front"), camera.FRONT, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
