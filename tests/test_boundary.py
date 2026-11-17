"""Tracing a boundary from a set of edges.

THE FAILURE THIS EXISTS TO PREVENT
----------------------------------
Two separate versions of this went wrong in a row, and each one looked plausible
on screen.

First: `TopExp_Explorer` over a wire yields edges in storage order, not
connection order, and an edge sampled without honouring its orientation
contributes a run that runs backwards. Result -- an outline built from the right
pieces joined in the wrong sequence, with stray sweeps across the part.

Then, fixing that: `BRepTools_WireExplorer` walks correctly but returns NOTHING
on wires it cannot interpret, and an empty result was treated as "no outline to
draw". Whole flanges silently vanished from the flat pattern while the bend
lines around them stayed perfectly correct.

The lesson in both is the same: **validate the output, do not trust the walker.**
A boundary loop has to come back closed. If it does not, fall through to
geometric chaining, which knows nothing about topology and cannot be defeated by
it.
"""

from __future__ import annotations

import unittest

import numpy as np

from partestimator.smt.geom import chain_point_runs, join_point_runs, loop_is_closed


def square_runs():
    """A unit square as four separate edge runs."""
    return [
        np.array([[0.0, 0, 0], [1, 0, 0]]),
        np.array([[1.0, 0, 0], [1, 1, 0]]),
        np.array([[1.0, 1, 0], [0, 1, 0]]),
        np.array([[0.0, 1, 0], [0, 0, 0]]),
    ]


class TestChaining(unittest.TestCase):
    def test_already_ordered_stays_ordered(self):
        runs = square_runs()
        out = chain_point_runs(runs)
        joined = join_point_runs(out)
        self.assertTrue(loop_is_closed(joined))
        self.assertEqual(len(joined), 5)   # 4 corners + the closing point

    def test_scrambled_order_is_repaired(self):
        """The exact failure mode of walking a wire with TopExp_Explorer."""
        runs = square_runs()
        scrambled = [runs[0], runs[2], runs[1], runs[3]]
        joined = join_point_runs(chain_point_runs(scrambled))
        self.assertTrue(loop_is_closed(joined), "chaining must close the loop")
        self.assertEqual(len(joined), 5)

    def test_reversed_runs_are_flipped(self):
        """The other half of the bug: an edge sampled backwards."""
        runs = square_runs()
        runs[1] = runs[1][::-1]
        runs[3] = runs[3][::-1]
        joined = join_point_runs(chain_point_runs(runs))
        self.assertTrue(loop_is_closed(joined))

        # Every consecutive pair must actually be adjacent -- no jumps.
        steps = np.linalg.norm(np.diff(joined, axis=0), axis=1)
        self.assertLess(float(steps.max()), 1.01,
                        "a jump across the shape means the chain is wrong")

    def test_scrambled_and_reversed_together(self):
        runs = square_runs()
        runs = [runs[2][::-1], runs[0], runs[3], runs[1][::-1]]
        joined = join_point_runs(chain_point_runs(runs))
        self.assertTrue(loop_is_closed(joined))
        steps = np.linalg.norm(np.diff(joined, axis=0), axis=1)
        self.assertLess(float(steps.max()), 1.01)

    def test_traversal_visits_every_corner_once(self):
        runs = square_runs()
        joined = join_point_runs(chain_point_runs([runs[3], runs[1], runs[2], runs[0]]))
        corners = {tuple(np.round(p, 9)) for p in joined}
        self.assertEqual(len(corners), 4)

    def test_area_is_preserved_by_correct_chaining(self):
        """A scrambled outline has the wrong area; a chained one does not."""
        def shoelace(poly):
            x, y = poly[:, 0], poly[:, 1]
            return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

        runs = square_runs()
        good = join_point_runs(chain_point_runs(runs))[:, :2]
        self.assertAlmostEqual(shoelace(good), 1.0, places=9)

        # Joining them in storage order without chaining gives a bow-tie.
        bad = join_point_runs([runs[0], runs[2], runs[1], runs[3]])[:, :2]
        self.assertNotAlmostEqual(shoelace(bad), 1.0, places=3)

    def test_empty_input(self):
        self.assertEqual(chain_point_runs([]), [])
        self.assertEqual(len(join_point_runs([])), 0)

    def test_single_point_runs_are_dropped(self):
        runs = [np.array([[0.0, 0, 0]]), *square_runs()]
        out = chain_point_runs(runs)
        self.assertEqual(len(out), 4)


class TestClosureValidation(unittest.TestCase):
    """The check that turns a silent failure into a fallback."""

    def test_a_closed_loop_passes(self):
        joined = join_point_runs(square_runs())
        self.assertTrue(loop_is_closed(joined))

    def test_an_open_loop_fails(self):
        runs = square_runs()[:-1]           # drop the closing edge
        self.assertFalse(loop_is_closed(join_point_runs(runs)))

    def test_too_few_points_fails(self):
        self.assertFalse(loop_is_closed(np.array([[0.0, 0, 0], [1, 0, 0]])))

    def test_tolerance_is_respected(self):
        pts = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0.0004, 0, 0]])
        self.assertTrue(loop_is_closed(pts, tol=1e-3))
        self.assertFalse(loop_is_closed(pts, tol=1e-5))


class TestJoining(unittest.TestCase):
    def test_shared_vertices_are_dropped(self):
        runs = square_runs()
        joined = join_point_runs(runs)
        # 4 edges x 2 points = 8, minus 3 shared joins = 5.
        self.assertEqual(len(joined), 5)

    def test_non_touching_runs_keep_both_endpoints(self):
        runs = [
            np.array([[0.0, 0, 0], [1, 0, 0]]),
            np.array([[5.0, 0, 0], [6, 0, 0]]),
        ]
        self.assertEqual(len(join_point_runs(runs)), 4)

    def test_tolerance_controls_the_drop(self):
        runs = [
            np.array([[0.0, 0, 0], [1, 0, 0]]),
            np.array([[1.0000001, 0, 0], [2, 0, 0]]),
        ]
        self.assertEqual(len(join_point_runs(runs, tol=1e-5)), 3)
        self.assertEqual(len(join_point_runs(runs, tol=1e-9)), 4)


class TestRealisticFlangeOutline(unittest.TestCase):
    def test_a_flange_with_a_radiused_corner(self):
        """Straight edges plus a discretised arc, delivered out of order.

        This is the shape of a real flange boundary, and the case where a
        scrambled traversal produces the smooth stray sweep across the part
        rather than an obviously broken zig-zag -- which is what made it look
        like a curve problem instead of an ordering problem.
        """
        arc = np.array([
            [10.0, 8 + 2 * np.sin(t), 0] if False else
            [10 - 2 + 2 * np.cos(t), 8 + 2 * np.sin(t), 0]
            for t in np.linspace(0, np.pi / 2, 9)
        ])
        runs = [
            np.array([[0.0, 0, 0], [10, 0, 0]]),
            np.array([[10.0, 0, 0], [10, 8, 0]]),
            arc[::-1],
            np.array([[8.0, 10, 0], [0, 10, 0]]),
            np.array([[0.0, 10, 0], [0, 0, 0]]),
        ]
        scrambled = [runs[4], runs[1], runs[3], runs[0], runs[2]]
        joined = join_point_runs(chain_point_runs(scrambled))

        self.assertTrue(loop_is_closed(joined, tol=1e-6))
        steps = np.linalg.norm(np.diff(joined, axis=0), axis=1)
        self.assertLess(
            float(steps.max()), 10.5,
            "no step should leap across the part",
        )


if __name__ == "__main__":
    unittest.main()
