"""Axis grouping, which the bend detector depends on for correctness.

`group_by_line` is an optimization, and optimizations that change answers are
how a tool starts quietly producing wrong flat patterns. The case that matters:
a hole crossing a bend splits its cylindrical face into pieces that MUST be
regrouped, or one bend reads as several and every downstream number is wrong.

An earlier draft bucketed axes by rounding their coordinates. These tests
include the case that killed it -- two axes closer together than the tolerance
but straddling a rounding boundary.
"""

from __future__ import annotations

import unittest

import numpy as np

from partestimator.smt.geom import group_by_line, same_axis


def axis(p, d):
    return (np.array(p, dtype=float), np.array(d, dtype=float))


class TestGroupByLine(unittest.TestCase):
    def test_identical_axes_group(self):
        items = [axis((0, 0, 2), (0, 1, 0)) for _ in range(4)]
        groups = group_by_line(items, lambda x: x)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 4)

    def test_same_line_different_point_and_direction(self):
        """A split bend face records its own location and may run either way.

        These are the same infinite line and must land together, or the merge
        that repairs a hole-split bend never happens.
        """
        items = [
            axis((0, 0, 2), (0, 1, 0)),
            axis((0, 50, 2), (0, -1, 0)),     # reversed, offset along itself
            axis((0, -30, 2), (0, 1, 0)),
        ]
        groups = group_by_line(items, lambda x: x)
        self.assertEqual(len(groups), 1, groups)

    def test_parallel_but_offset_axes_do_not_group(self):
        """Two bends at the same angle on opposite ends of a part."""
        items = [axis((0, 0, 2), (0, 1, 0)), axis((92, 0, 2), (0, 1, 0))]
        groups = group_by_line(items, lambda x: x)
        self.assertEqual(len(groups), 2)

    def test_perpendicular_axes_do_not_group(self):
        items = [axis((0, 0, 2), (0, 1, 0)), axis((0, 0, 2), (1, 0, 0))]
        groups = group_by_line(items, lambda x: x)
        self.assertEqual(len(groups), 2)

    def test_within_tolerance_still_groups_across_a_rounding_boundary(self):
        """The case that ruled out coordinate-rounding buckets.

        These two axes differ by 2e-7 mm -- far inside any sane tolerance, so
        they are the same bend. But they straddle x.xxxx5, so any scheme that
        rounds coordinates into buckets puts them in different ones and never
        compares them. Exact testing has no such boundary.
        """
        items = [
            axis((0, 0, 2.00049990), (0, 1, 0)),
            axis((0, 0, 2.00050010), (0, 1, 0)),
        ]
        groups = group_by_line(items, lambda x: x, lin_tol=1e-5)
        self.assertEqual(len(groups), 1, "tolerance-close axes must group")

    def test_beyond_tolerance_separates(self):
        items = [axis((0, 0, 2.0), (0, 1, 0)), axis((0, 0, 2.5), (0, 1, 0))]
        groups = group_by_line(items, lambda x: x, lin_tol=1e-5)
        self.assertEqual(len(groups), 2)

    def test_every_item_appears_exactly_once(self):
        """Grouping must partition, not drop or duplicate."""
        items = [
            axis((0, 0, 2), (0, 1, 0)),
            axis((0, 9, 2), (0, 1, 0)),
            axis((92, 0, 2), (0, 1, 0)),
            axis((0, 0, 2), (1, 0, 0)),
            axis((5, 5, 5), (1, 1, 1)),
        ]
        groups = group_by_line(items, lambda x: x)
        self.assertEqual(sum(len(g) for g in groups), len(items))

    def test_grouping_agrees_with_pairwise_truth(self):
        """Cross-check against the O(n^2) definition it replaces."""
        rng = np.random.default_rng(20260804)
        items = []
        for _ in range(30):
            d = rng.choice([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
            p = rng.choice([0.0, 2.0, 5.0], size=3)
            items.append(axis(p, d))

        groups = group_by_line(items, lambda x: x)
        lookup = {}
        for gi, g in enumerate(groups):
            for it in g:
                lookup[id(it)] = gi

        for a in items:
            for b in items:
                truth = same_axis(a[0], a[1], b[0], b[1])
                grouped = lookup[id(a)] == lookup[id(b)]
                # Grouping is by representative, so it is transitive; equality
                # holds because collinearity is an equivalence relation here.
                self.assertEqual(truth, grouped, (a, b))


class TestPartitionOfRealisticCase(unittest.TestCase):
    def test_hole_split_bend_regroups(self):
        """Three fragments of one bend face, plus unrelated holes.

        This is the geometry of a bend line interrupted twice by cutouts. The
        three fragments share an axis and must come back as one group; the
        hole axes are perpendicular and must stay separate.
        """
        bend_fragments = [
            axis((0, 0, 2), (0, 1, 0)),
            axis((0, 20, 2), (0, 1, 0)),
            axis((0, 40, 2), (0, 1, 0)),
        ]
        holes = [
            axis((10, 10, 0), (0, 0, 1)),
            axis((30, 25, 0), (0, 0, 1)),
        ]
        groups = group_by_line(bend_fragments + holes, lambda x: x)

        sizes = sorted(len(g) for g in groups)
        self.assertEqual(sizes, [1, 1, 3], f"got groups of {sizes}")


if __name__ == "__main__":
    unittest.main()
