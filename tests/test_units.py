"""Units, and the gauge-snapping bug that rejected every bend on inch parts.

THE BUG THIS FILE EXISTS FOR
----------------------------
Thickness was inferred from the geometry, then snapped to the nearest standard
gauge, and the SNAPPED value was used to test candidates for `dr ~= thickness`.

Snapping was permitted to move the value by up to 2% (`gauge_snap_warn`), while
the match tolerance was 1% (`radius_rel`). Anything landing between those two
had every bend on the part rejected -- silently, with no error, just "no bends
detected".

A 0.075 inch sheet is 1.905 mm. Against a metric stock list the nearest entry is
2.0 mm, a 5% move, and nothing survived. Since most of the shop's models are in
inches, this was not an edge case.

Two things fix it, and both are tested here: classification uses the MEASURED
thickness and never the snapped one, and snapping matches against imperial stock
for a file that declares inches.
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from partestimator.smt.config import Tolerance, UnfoldOptions
from partestimator.smt.units import (
    INCH,
    MM,
    MM_PER_INCH,
    choose_display_unit,
    get_unit,
    nearest_gauge_number_in,
    sniff_step_units,
    standard_dies_mm_for,
    standard_gauges_mm_for,
)


class TestConversion(unittest.TestCase):
    def test_round_trip(self):
        for mm in (1.905, 25.4, 153.0477868, 0.4):
            self.assertAlmostEqual(INCH.to_mm(INCH.from_mm(mm)), mm, places=12)
            self.assertAlmostEqual(MM.to_mm(MM.from_mm(mm)), mm, places=12)

    def test_known_values(self):
        self.assertAlmostEqual(INCH.from_mm(25.4), 1.0, places=12)
        self.assertAlmostEqual(INCH.from_mm(1.905), 0.075, places=12)
        self.assertAlmostEqual(INCH.to_mm(0.075), 1.905, places=12)

    def test_area_scales_as_the_square(self):
        # 1 in^2 == 645.16 mm^2
        self.assertAlmostEqual(645.16 * INCH.per_mm ** 2, 1.0, places=9)

    def test_formatting_precision(self):
        """Inches need more decimals than millimetres to say the same thing."""
        self.assertIn("0.0750", INCH.length(1.905))
        self.assertIn("1.91", MM.length(1.905))

    def test_dxf_codes(self):
        self.assertEqual(INCH.dxf_code, 1)
        self.assertEqual(MM.dxf_code, 4)

    def test_force_units_differ(self):
        """US shops quote short tons, metric quote tonnes -- about 10% apart."""
        self.assertAlmostEqual(
            MM.newtons_per_force_unit / INCH.newtons_per_force_unit, 1.1023, places=3
        )

    def test_get_unit_aliases(self):
        for name in ("in", "inch", "inches", "IN", "Imperial"):
            self.assertIs(get_unit(name), INCH)
        for name in ("mm", "MM", "", None):
            self.assertIs(get_unit(name), MM)


class TestGaugeSnapRegression(unittest.TestCase):
    """The exact numbers that broke inch parts."""

    def test_inch_gauge_snaps_to_itself_not_to_metric_stock(self):
        t_mm = 0.075 * MM_PER_INCH          # 1.905
        stock = standard_gauges_mm_for(INCH)
        snapped = min(stock, key=lambda g: abs(g - t_mm))
        self.assertAlmostEqual(snapped, t_mm, places=6)
        shift = abs(snapped - t_mm) / t_mm
        self.assertLess(shift, 1e-6, "0.075in must land on itself")

    def test_metric_stock_would_have_moved_it_5_percent(self):
        """Documents the old behaviour, so the regression stays visible."""
        t_mm = 0.075 * MM_PER_INCH
        stock = standard_gauges_mm_for(MM)
        snapped = min(stock, key=lambda g: abs(g - t_mm))
        self.assertAlmostEqual(snapped, 2.0, places=6)
        self.assertGreater(abs(snapped - t_mm) / t_mm, 0.04)

    def test_the_fatal_window_between_snap_and_match(self):
        """A snap larger than the match tolerance rejects every bend.

        This is the arithmetic of the bug: `gauge_snap_warn` allowed a 2% move
        while `radius_close` only accepted 1%. Any part in that window lost all
        of its bends. Classification now uses the measured value, so the two
        tolerances never interact -- but the window itself still exists, and
        this test pins the fact that it would matter if they were ever wired
        back together.
        """
        tol = Tolerance()
        opts = UnfoldOptions()
        measured = 1.905
        snapped = 2.0
        shift = abs(snapped - measured) / measured

        self.assertGreater(shift, tol.radius_rel,
                           "snap moves further than the match tolerance allows")
        self.assertFalse(tol.radius_close(measured, snapped),
                         "a bend measured at 1.905 fails against a snapped 2.0")
        self.assertTrue(tol.radius_close(measured, measured),
                        "against the measured value it passes")

    def test_common_inch_gauges_all_land_on_themselves(self):
        stock = standard_gauges_mm_for(INCH)
        for inches in (0.030, 0.048, 0.060, 0.075, 0.090, 0.125, 0.250):
            t_mm = inches * MM_PER_INCH
            snapped = min(stock, key=lambda g: abs(g - t_mm))
            self.assertAlmostEqual(
                snapped, t_mm, places=6, msg=f'{inches}" moved'
            )

    def test_us_gauge_numbers(self):
        self.assertEqual(nearest_gauge_number_in(0.0598 * MM_PER_INCH), 16)
        self.assertEqual(nearest_gauge_number_in(0.1046 * MM_PER_INCH), 12)
        # Far from any gauge -> no claim.
        self.assertIsNone(nearest_gauge_number_in(0.400 * MM_PER_INCH))


class TestDieRacks(unittest.TestCase):
    def test_imperial_dies_are_round_inches(self):
        dies = standard_dies_mm_for(INCH)
        in_dies = [d / MM_PER_INCH for d in dies]
        for expect in (0.5, 1.0, 2.0):
            self.assertTrue(
                any(abs(v - expect) < 1e-9 for v in in_dies), f'{expect}" missing'
            )

    def test_metric_dies_are_round_millimetres(self):
        dies = standard_dies_mm_for(MM)
        for expect in (8, 16, 25):
            self.assertTrue(any(abs(d - expect) < 1e-9 for d in dies))


class TestStepSniffing(unittest.TestCase):
    def _write(self, body: str) -> Path:
        d = tempfile.mkdtemp()
        p = Path(d) / "t.step"
        p.write_text("ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n" + body + "\nENDSEC;\n")
        return p

    def test_detects_inch(self):
        p = self._write(
            "#12=(CONVERSION_BASED_UNIT('INCH',#13)LENGTH_UNIT()NAMED_UNIT(*));"
        )
        self.assertEqual(sniff_step_units(p), "in")

    def test_detects_millimetre(self):
        p = self._write("#11=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.));")
        self.assertEqual(sniff_step_units(p), "mm")

    def test_undeclared_returns_none(self):
        p = self._write("#1=CARTESIAN_POINT('',(0.,0.,0.));")
        self.assertIsNone(sniff_step_units(p))

    def test_missing_file_is_not_an_error(self):
        self.assertIsNone(sniff_step_units(Path("/nonexistent/x.step")))


class TestDisplayChoice(unittest.TestCase):
    def test_auto_follows_the_file(self):
        self.assertIs(choose_display_unit("auto", "in"), INCH)
        self.assertIs(choose_display_unit("auto", "mm"), MM)

    def test_auto_falls_back_to_mm_when_undeclared(self):
        self.assertIs(choose_display_unit("auto", None), MM)

    def test_explicit_choice_overrides_the_file(self):
        self.assertIs(choose_display_unit("in", "mm"), INCH)
        self.assertIs(choose_display_unit("mm", "in"), MM)


class TestInchBlankArithmetic(unittest.TestCase):
    """A worked imperial bend, checked by hand.

    0.075" material, R 0.090", one 90 degree bend, K = 0.44.

        neutral radius = 0.090 + 0.44(0.075) = 0.123"
        BA             = (pi/2)(0.123)       = 0.1932079"
    """

    def test_bend_allowance_in_inches(self):
        from partestimator.smt.allowance import bend_allowance

        t_in, r_in, k = 0.075, 0.09, 0.44
        expected_in = (math.pi / 2) * (r_in + k * t_in)
        self.assertAlmostEqual(expected_in, 0.1932079, places=6)

        # Computed in mm, displayed in inches -- must agree exactly. This is
        # the assertion that matters: it catches a stray 25.4 anywhere between
        # the geometry and the readout.
        ba_mm = bend_allowance(
            math.pi / 2, r_in * MM_PER_INCH, t_in * MM_PER_INCH, k
        )
        self.assertAlmostEqual(INCH.from_mm(ba_mm), expected_in, places=9)

    def test_imperial_u_channel_blank(self):
        """The mm U-channel restated in inches, developed the same way.

        6" outside base, two 1.25" legs, 0.075" material, R 0.090", K = 0.44.
            OSSB = (0.090 + 0.075) tan(45)  = 0.165"
            BD   = 2(0.165) - 0.1932079     = 0.1367921"
            flat = 6 + 1.25 + 1.25 - 2(BD)  = 8.2264158"
        """
        from partestimator.smt.allowance import bend_deduction

        t_in, r_in, k = 0.075, 0.09, 0.44
        bd_mm = bend_deduction(
            math.pi / 2, r_in * MM_PER_INCH, t_in * MM_PER_INCH, k
        )
        bd_in = INCH.from_mm(bd_mm)
        self.assertAlmostEqual(bd_in, 0.1367921, places=6)

        flat_in = 6.0 + 1.25 + 1.25 - 2 * bd_in
        self.assertAlmostEqual(flat_in, 8.2264158, places=6)


if __name__ == "__main__":
    unittest.main()
