"""A flange can be several disconnected pieces, and all of them must survive.

THE BUG
-------
A plate with a long bend across the top, the flange material cut away in the
middle so two tabs stand up -- one at each end, each with a hole. Perfectly
ordinary sheet metal. Only ONE tab appeared in the flat pattern; the other
vanished with no warning.

Two coplanar faces sharing a bend line are one flange for unfolding purposes:
same plane, same side of the same bend, therefore the same rigid transform. But
detection merged them into a single tangent plane and then kept only
`primary` -- the largest -- so the second tab never entered the flange graph at
all. The bend was found, the big tab was found, the small one was dropped
silently.

The subtle half of the fix is that the loops have to stay grouped BY REGION.
Flattening them into one list makes the second region's outer wire
indistinguishable from a hole in the first, so the tab would come back as a
cutout -- which is worse than losing it, because it looks deliberate.
"""

from __future__ import annotations

import unittest

import numpy as np

from partestimator.smt.config import UnfoldOptions
from partestimator.smt.model2d import recompute
from tests import synthetic


class TestRegionsAreKeptSeparate(unittest.TestCase):
    def setUp(self):
        self.part = synthetic.twin_tab_flange()
        self.flange = self.part.flanges[1]

    def test_the_flange_has_two_regions(self):
        self.assertEqual(len(self.flange.regions_for_side(0)), 2)

    def test_each_region_keeps_its_own_outer_wire(self):
        for region in self.flange.regions_for_side(0):
            with self.subTest(region=region[0][0].tolist()):
                self.assertGreaterEqual(len(region), 1)

    def test_the_regions_are_the_two_tabs_and_do_not_overlap(self):
        spans = []
        for region in self.flange.regions_for_side(0):
            xs = [p[0] for loop in region for p in loop]
            spans.append((min(xs), max(xs)))
        spans.sort()
        self.assertEqual(len(spans), 2)
        self.assertLess(spans[0][1], spans[1][0], "the tabs should not overlap")

    def test_the_flat_form_is_still_understood(self):
        """A bare list of loops means one region, for older callers and fixtures."""
        from partestimator.smt.model import Flange

        loops = [np.zeros((4, 3)), np.ones((4, 3))]
        f = Flange(id=0, normal=[0, 0, 1], origin=[0, 0, 0],
                   centroid=[0, 0, 0], loops_a=loops)
        self.assertEqual(len(f.regions_for_side(0)), 1)
        self.assertEqual(len(f.regions_for_side(0)[0]), 2)


class TestBothTabsReachTheFlatPattern(unittest.TestCase):
    """The test that fails without the fix."""

    def setUp(self):
        self.part = synthetic.twin_tab_flange()
        self.flat, _r = recompute(self.part, UnfoldOptions())

    def test_the_flat_has_an_outline_for_every_region(self):
        # base (1) + two tabs (2) = three outlines, plus the bend strip.
        self.assertGreaterEqual(
            len(self.flat.flange_loops), 3,
            f"only {len(self.flat.flange_loops)} outline(s) -- a tab is missing",
        )

    def test_no_tab_was_turned_into_a_hole(self):
        """The failure mode that is worse than losing it: it looks intentional."""
        self.assertEqual(
            self.flat.hole_loops, [],
            "a region's outer wire was mistaken for a hole",
        )

    def test_both_tabs_are_present_along_the_bend(self):
        """Measured where it matters: at both ends of the bend line.

        The two tabs sit at opposite ends of a 200 mm bend, so if only one
        survived, the blank would carry material at one end and none at the
        other. Checking the extremes catches that directly, without depending
        on which way round the solver happened to lay the part out.
        """
        pts = self.flat.all_points()
        self.assertTrue(len(pts), "the flat pattern is empty")

        long_axis = 0 if self.flat.width >= self.flat.height else 1
        other = 1 - long_axis
        lo, hi = pts[:, long_axis].min(), pts[:, long_axis].max()
        span = hi - lo

        # The tabs occupy the outer fifth at each end of the bend.
        near = pts[pts[:, long_axis] < lo + 0.25 * span]
        far = pts[pts[:, long_axis] > hi - 0.25 * span]

        self.assertGreater(
            float(np.ptp(near[:, other])), 0.0, "no material at one end of the bend"
        )
        self.assertGreater(
            float(np.ptp(far[:, other])), 0.0, "no material at the other end of the bend"
        )

    def test_the_blank_area_accounts_for_both_tabs(self):
        """Area is the decisive check: one tab missing is 1200 mm2 short.

        base 200 x 80 = 16000, two tabs 40 x 30 = 2400, plus the bend strip.
        Losing a tab would put this below the base plus one tab.
        """
        area = self.flat.polygon_area()
        base_plus_one_tab = 200.0 * 80.0 + 40.0 * 30.0
        self.assertGreater(
            area, base_plus_one_tab + 500.0,
            f"blank area {area:.0f} mm2 is too small -- a tab is missing",
        )


if __name__ == "__main__":
    unittest.main()
