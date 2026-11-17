"""Sheet metal vs machined: the rule, then real parts through the kernel."""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from partestimator.classify import MACHINED, SHEET, decide

try:
    import OCP  # noqa: F401

    HAVE_OCP = True
except Exception:  # noqa: BLE001
    HAVE_OCP = False


class TestRule(unittest.TestCase):
    """`decide` is the user's rule, on numbers alone."""

    def d(self, **kw):
        base = dict(uniform=True, flat=True, bend_count=0, thickness_in=0.09,
                    cutoff_in=0.190)
        base.update(kw)
        return decide(**base)[0]

    def test_bent_is_sheet_at_any_thickness(self):
        self.assertEqual(self.d(flat=False, bend_count=2, thickness_in=0.25), SHEET)

    def test_flat_at_or_under_cutoff_is_sheet(self):
        self.assertEqual(self.d(thickness_in=0.190), SHEET)
        self.assertEqual(self.d(thickness_in=0.036), SHEET)

    def test_flat_over_cutoff_is_machined(self):
        self.assertEqual(self.d(thickness_in=0.1901), MACHINED)
        self.assertEqual(self.d(thickness_in=0.250), MACHINED)

    def test_cutoff_is_honoured(self):
        self.assertEqual(self.d(thickness_in=0.250, cutoff_in=0.250), SHEET)
        self.assertEqual(self.d(thickness_in=0.125, cutoff_in=0.100), MACHINED)

    def test_non_uniform_is_machined_even_if_thin(self):
        self.assertEqual(self.d(uniform=False, thin_fraction=0.2, thickness_in=0.06), MACHINED)

    def test_not_flat_without_bends_is_machined(self):
        self.assertEqual(self.d(flat=False, bend_count=0), MACHINED)

    def test_several_bodies_is_machined(self):
        self.assertEqual(self.d(solids=2), MACHINED)


@unittest.skipUnless(HAVE_OCP, "needs OCP")
class TestRealParts(unittest.TestCase):
    EXPECT = {
        # name: (builder, kwargs, expected kind, expected thickness in or None)
        "l_bracket": ("l_bracket", {}, SHEET, 0.060),
        "z_section": ("z_section", {}, SHEET, 0.060),
        "csk_l": ("countersunk_l_bracket", {}, SHEET, 0.060),
        "flat_090": ("flat_blank", {"thickness_in": 0.090, "name": "f090"}, SHEET, 0.090),
        "flat_190": ("flat_blank", {"thickness_in": 0.190, "name": "f190"}, SHEET, 0.190),
        "flat_250": ("flat_blank", {"thickness_in": 0.250, "name": "f250"}, MACHINED, 0.250),
        "pocketed_500": ("machined_plate", {}, MACHINED, None),
        "pocketed_125": ("thin_pocketed_plate", {}, MACHINED, None),
        "spotface_125": ("plate_with_spotface", {}, MACHINED, None),
        "block": ("block", {}, MACHINED, None),
        "turned": ("turned_part", {}, MACHINED, None),
    }

    @classmethod
    def setUpClass(cls):
        from partestimator.classify import classify_file
        from tests import occparts

        cls.tmp = Path(tempfile.mkdtemp(prefix="classify_"))
        # The STEP writer prints a banner straight to fd 1; keep the run quiet.
        devnull = os.open(os.devnull, os.O_WRONLY)
        saved = os.dup(1)
        os.dup2(devnull, 1)
        try:
            paths = {k: getattr(occparts, b)(cls.tmp, **kw)
                     for k, (b, kw, _e, _t) in cls.EXPECT.items()}
        finally:
            os.dup2(saved, 1)
            os.close(devnull)
            os.close(saved)
        cls.results = {k: classify_file(p) for k, p in paths.items()}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_every_part_lands_where_expected(self):
        for name, (_b, _kw, kind, t) in self.EXPECT.items():
            with self.subTest(part=name):
                c = self.results[name]
                self.assertEqual(c.kind, kind, c.reason)
                if t is not None:
                    self.assertAlmostEqual(c.thickness_in, t, delta=0.001)

    def test_bent_parts_report_their_bends(self):
        self.assertEqual(self.results["l_bracket"].bend_count, 1)
        self.assertEqual(self.results["z_section"].bend_count, 2)

    def test_cutoff_moves_the_line(self):
        from partestimator.classify import _judge  # noqa: F401 -- rule is in decide
        c = self.results["flat_250"]
        kind, _r = decide(uniform=True, flat=c.flat, bend_count=0,
                          thickness_in=c.thickness_in, cutoff_in=0.250)
        self.assertEqual(kind, SHEET)

    def test_it_is_quick(self):
        for name, c in self.results.items():
            with self.subTest(part=name):
                self.assertLess(c.seconds, 5.0)


if __name__ == "__main__":
    unittest.main()
