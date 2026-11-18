"""Turning the part in its stock: the geometry behind the orientation controls.

numpy and scipy only, no CAD kernel or Qt.
"""
from __future__ import annotations

import unittest

import numpy as np

try:
    from partestimator.machining.geometry import orientation as o

    HAVE_SCIPY = True
except ImportError:  # the light CI environment has no scipy
    HAVE_SCIPY = False


def block(length, width, thickness, turn_deg=0.0):
    """The eight corners of a block, turned about Z the way a file might
    store it."""
    pts = np.array([[x, y, z] for x in (0, length) for y in (0, width)
                    for z in (0, thickness)], dtype=float)
    return pts @ o.rotation_matrix(2, turn_deg).T


@unittest.skipUnless(HAVE_SCIPY, "needs scipy")
class TestFrames(unittest.TestCase):
    def test_box_in_frame_measures_in_that_frame(self):
        pts = block(6, 4, 1, turn_deg=20)
        square = o.box_in_frame(pts, o.rotation_matrix(2, 20).T)
        np.testing.assert_allclose(sorted(square.extents_mm), [1, 4, 6], atol=1e-9)
        skew = o.box_in_frame(pts, np.eye(3))
        self.assertGreater(skew.volume_mm3, square.volume_mm3 * 1.5)

    def test_box_contains_every_point(self):
        rng = np.random.default_rng(3)
        pts = rng.normal(size=(500, 3)) * [5, 2, 1]
        box = o.box_in_frame(pts, o.rotated(np.eye(3), (10, -25, 40)))
        local = (pts - box.center_mm) @ box.axes.T
        self.assertTrue(np.all(np.abs(local) <= box.extents_mm / 2 + 1e-9))

    def test_sorted_frame_is_length_width_thickness(self):
        box = o.box_in_frame(block(1, 6, 4), np.eye(3))
        frame = o.sorted_frame(box)
        ext = o.box_in_frame(block(1, 6, 4), frame).extents_mm
        np.testing.assert_allclose(ext, [6, 4, 1], atol=1e-9)
        self.assertAlmostEqual(np.linalg.det(frame), 1.0)

    def test_rotation_is_right_handed_and_turns_the_part(self):
        # Turning the part +90 about Z swaps its length and width.
        frame = o.rotated(np.eye(3), (0, 0, 90))
        ext = o.box_in_frame(block(6, 4, 1), frame).extents_mm
        np.testing.assert_allclose(ext, [4, 6, 1], atol=1e-9)
        self.assertAlmostEqual(np.linalg.det(frame), 1.0)
        # Coordinates in the new frame are the old ones turned by +90 about Z.
        np.testing.assert_allclose(frame @ [1, 0, 0], [0, 1, 0], atol=1e-12)

    def test_aligning_to_a_face_squares_the_part_up(self):
        pts = block(6, 4, 1, turn_deg=20)
        frame = o.align_axis_to_face(np.eye(3), 2, [0, 0, -1], pts)
        np.testing.assert_allclose(frame[2], [0, 0, 1], atol=1e-12)   # sense kept
        np.testing.assert_allclose(o.box_in_frame(pts, frame).extents_mm, [6, 4, 1],
                                   atol=1e-9)
        self.assertAlmostEqual(np.linalg.det(frame), 1.0)

    def test_aligning_another_axis_keeps_it_in_its_row(self):
        pts = block(6, 4, 1)
        frame = o.align_axis_to_face(np.eye(3), 0, [0, 0, 1], pts)
        np.testing.assert_allclose(abs(frame[0] @ [0, 0, 1]), 1.0)
        self.assertAlmostEqual(np.linalg.det(frame), 1.0)
        np.testing.assert_allclose(sorted(o.box_in_frame(pts, frame).extents_mm),
                                   [1, 4, 6], atol=1e-9)


@unittest.skipUnless(HAVE_SCIPY, "needs scipy")
class TestFaces(unittest.TestCase):
    def setUp(self):
        # A flat square (two triangles) and, apart from it, one bent pair.
        self.pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                             [5, 0, 0], [6, 0, 0], [5, 1, 0], [6, 1, 1]], dtype=float)
        self.tris = np.array([[0, 1, 2], [1, 3, 2], [4, 5, 6], [5, 7, 6]])
        self.labels = o.face_labels(self.tris, len(self.pts))

    def test_triangles_sharing_vertices_are_one_face(self):
        self.assertEqual(self.labels[0], self.labels[1])
        self.assertEqual(self.labels[2], self.labels[3])
        self.assertNotEqual(self.labels[0], self.labels[2])

    def test_a_flat_face_gives_its_normal_and_area(self):
        normal, area, centroid = o.planar_face(self.pts, self.tris, self.labels, 1)
        np.testing.assert_allclose(normal, [0, 0, 1])
        self.assertAlmostEqual(area, 1.0)
        np.testing.assert_allclose(centroid, [0.5, 0.5, 0])

    def test_a_bent_face_is_refused(self):
        self.assertIsNone(o.planar_face(self.pts, self.tris, self.labels, 2))

    def test_out_of_range_is_refused(self):
        self.assertIsNone(o.planar_face(self.pts, self.tris, self.labels, 99))


@unittest.skipUnless(HAVE_SCIPY, "needs scipy")
class TestPickingTheBox(unittest.TestCase):
    def setUp(self):
        self.box = o.box_in_frame(block(6, 4, 1), np.eye(3))

    def test_ray_enters_by_the_face_it_meets_first(self):
        self.assertEqual(o.ray_box_face([3, 2, 10], [0, 0, -1], self.box), (2, 1))
        self.assertEqual(o.ray_box_face([-10, 2, 0.5], [1, 0, 0], self.box), (0, -1))
        self.assertEqual(o.ray_box_face([3, 20, 0.5], [0, -1, 0], self.box), (1, 1))

    def test_a_miss_is_none(self):
        self.assertIsNone(o.ray_box_face([-10, 20, 0.5], [1, 0, 0], self.box))
        self.assertIsNone(o.ray_box_face([3, 2, 10], [0, 0, 1], self.box))  # behind

    def test_larger_stock_extents_widen_the_target(self):
        # Passes over the part's box, but through the larger stock round it.
        self.assertIsNone(o.ray_box_face([-10, 2, 1.4], [1, 0, 0], self.box))
        bigger = o.ray_box_face([-10, 2, 1.4], [1, 0, 0], self.box, extents=[8, 6, 3])
        self.assertEqual(bigger, (0, -1))

    def test_face_corners_lie_on_that_face(self):
        c = o.box_face_corners(self.box.center_mm, self.box.axes, self.box.extents_mm, 2, 1)
        np.testing.assert_allclose(c[:, 2], 1.0)
        self.assertEqual(len({tuple(p) for p in c}), 4)


if __name__ == "__main__":
    unittest.main()
