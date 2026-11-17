"""Reporting, material tables and warning logic."""

from __future__ import annotations

import math
import unittest

import numpy as np

from partestimator.smt.config import UnfoldOptions
from partestimator.smt.materials import (
    air_bend_tonnage,
    get_material,
    min_flange_length,
    natural_air_bend_radius,
    select_v_die,
    snap_to_gauge,
)
from partestimator.smt.model import FlatPattern
from partestimator.smt.report import build_report, manufacturability_warnings
from partestimator.smt.unfold import assign_allowances

from tests import synthetic


def _with_flat(part):
    """Attach a plausible flat pattern so the report layer has something to chew."""
    part.flat = FlatPattern(
        area=153.05 * 50.0,
        perimeter=2 * (153.05 + 50.0),
        bbox=(153.05, 50.0),
        min_rect=(153.05, 50.0, 0.0),
    )
    part.volume = part.flat.area * part.thickness
    return part


class TestMaterialTables(unittest.TestCase):
    def test_gauge_snapping(self):
        self.assertEqual(snap_to_gauge(1.98)[0], 2.0)
        self.assertEqual(snap_to_gauge(3.05)[0], 3.0)
        # Near-misses snap silently...
        self.assertLess(snap_to_gauge(1.98)[1], 0.02)
        # ...but a value off the end of the sheet range reports a big shift, so
        # the caller can refuse to snap and warn about the units instead.
        _, shift = snap_to_gauge(20.0)
        self.assertGreater(shift, 0.1)

    def test_v_die_scales_with_thickness(self):
        self.assertLess(select_v_die(1.0), select_v_die(3.0))
        self.assertLess(select_v_die(3.0), select_v_die(10.0))
        # 2 mm wants ~16 mm (8t) and 16 is a stock die.
        self.assertEqual(select_v_die(2.0), 16.0)

    def test_natural_radius_is_the_20_percent_rule(self):
        mat = get_material("CRS")
        self.assertAlmostEqual(natural_air_bend_radius(16.0, mat), 2.496, places=3)

    def test_tonnage_against_published_chart(self):
        """2 mm mild steel in a 16 mm vee is about 15-17 t per metre."""
        mat = get_material("CRS")
        tons = air_bend_tonnage(1000.0, 2.0, 16.0, mat)
        self.assertTrue(13.0 < tons < 19.0, tons)

    def test_tonnage_scales_as_t_squared(self):
        mat = get_material("CRS")
        a = air_bend_tonnage(1000.0, 2.0, 16.0, mat)
        b = air_bend_tonnage(1000.0, 4.0, 16.0, mat)
        self.assertAlmostEqual(b / a, 4.0, places=6)

    def test_min_flange(self):
        self.assertAlmostEqual(min_flange_length(16.0), 10.4, places=6)


class TestMeasuredDeductions(unittest.TestCase):
    """A measured shop table must override the K-factor curve."""

    def setUp(self):
        from partestimator.smt.materials import DeductionTable

        self.table = DeductionTable(
            rows=[
                {"material": "CRS", "thickness": 2.0, "radius": 2.0,
                 "v_die": 16.0, "angle_deg": 90.0, "deduction": 3.60},
            ]
        )

    def test_table_row_wins_over_curve(self):
        from partestimator.smt.allowance import resolve_allowance

        ba, k, source = resolve_allowance(
            angle=math.pi / 2, r_inner=2.0, thickness=2.0,
            material="CRS", table=self.table, v_die=16.0,
        )
        self.assertEqual(source, "measured table")
        # BD = 2*OSSB - BA, so BA = 2*4 - 3.60 = 4.40
        self.assertAlmostEqual(ba, 4.40, places=6)

    def test_implied_k_is_backed_out(self):
        """The report shows the K the measured deduction implies, so the shop
        table itself can be sanity checked."""
        from partestimator.smt.allowance import resolve_allowance

        _ba, k, _src = resolve_allowance(
            angle=math.pi / 2, r_inner=2.0, thickness=2.0,
            material="CRS", table=self.table, v_die=16.0,
        )
        self.assertTrue(0.30 < k < 0.50, k)

    def test_falls_through_when_no_row_matches(self):
        from partestimator.smt.allowance import resolve_allowance

        _ba, _k, source = resolve_allowance(
            angle=math.pi / 2, r_inner=9.9, thickness=2.0,
            material="CRS", table=self.table, v_die=16.0,
        )
        self.assertIn("curve", source)

    def test_csv_round_trip(self):
        import csv
        import tempfile
        from pathlib import Path

        from partestimator.smt.materials import load_deduction_table

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ded.csv"
            with open(p, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["material", "thickness", "radius", "v_die",
                            "angle_deg", "deduction"])
                w.writerow(["CRS", "2.0", "2.5", "16", "90", "3.45"])
            t = load_deduction_table(p)
            self.assertAlmostEqual(t.lookup("CRS", 2.0, 2.5, 16.0), 3.45)
            self.assertIsNone(t.lookup("AL6061", 2.0, 2.5, 16.0))


class TestReport(unittest.TestCase):
    def setUp(self):
        self.part = _with_flat(synthetic.u_channel())
        self.opts = UnfoldOptions(k_factor=0.44, material="CRS", cost_per_kg=1.20)
        assign_allowances(self.part, self.opts)
        for b in self.part.bends.values():
            b.direction = 1

    def test_report_builds_and_renders(self):
        rep = build_report(self.part, self.opts)
        text = rep.to_text()
        self.assertIn("SHEET METAL FLAT PATTERN REPORT", text)
        self.assertIn("TOOLING", text)
        self.assertEqual(len(rep.bends), 2)

    def test_distinct_radii_reported(self):
        rep = build_report(self.part, self.opts)
        self.assertEqual(rep.distinct_radii, [2.0])

    def test_tonnage_uses_material_length_not_envelope(self):
        """A bend line half cut away needs about half the force."""
        b = self.part.bends[0]
        b.length_material = b.length_envelope / 2.0
        rep = build_report(self.part, self.opts)
        rows = {r.n: r for r in rep.bends}
        self.assertAlmostEqual(rows[1].tonnage, rows[2].tonnage / 2.0, places=1)

    def test_weight_matches_density(self):
        rep = build_report(self.part, self.opts)
        mat = get_material("CRS")
        expected = self.part.flat.area * self.part.thickness * mat.density
        self.assertAlmostEqual(rep.material.weight_kg, round(expected, 4), places=4)

    def test_cost_tracks_weight(self):
        rep = build_report(self.part, self.opts)
        self.assertAlmostEqual(rep.material.cost, round(rep.material.weight_kg * 1.2, 2))


class TestWarnings(unittest.TestCase):
    def _warnings(self, part, **kw):
        opts = UnfoldOptions(k_factor=0.44, material=kw.pop("material", "CRS"), **kw)
        assign_allowances(part, opts)
        return manufacturability_warnings(_with_flat(part), opts)

    def test_radius_below_material_minimum_is_flagged(self):
        part = synthetic.u_channel()
        for b in part.bends.values():
            b.r_inner = 0.2       # 0.1t on 2 mm -- far under any minimum
        w = self._warnings(part)
        self.assertTrue(any("below the minimum" in x for x in w), w)

    def test_6061_is_stricter_than_5052(self):
        """6061-T6 needs a much larger minimum radius; the table must reflect it.

        At R2.5 on 2 mm (r/t = 1.25), 5052 is fine across the grain and only
        marginal with it -- a nesting constraint, not a redesign. 6061-T6 wants
        3t transverse, so the same radius is a hard reject.
        """
        soft = synthetic.u_channel()
        hard = synthetic.u_channel()
        for part in (soft, hard):
            for b in part.bends.values():
                b.r_inner = 2.5

        w5052 = self._warnings(soft, material="AL5052")
        w6061 = self._warnings(hard, material="AL6061")

        self.assertTrue(any("with the grain" in x for x in w5052), w5052)
        self.assertFalse(any("below the minimum" in x for x in w5052), w5052)
        self.assertTrue(any("below the minimum" in x for x in w6061), w6061)

    def test_generous_radius_is_clean_in_both_alloys(self):
        part = synthetic.u_channel()
        for b in part.bends.values():
            b.r_inner = 10.0
        for mat in ("AL5052", "AL6061"):
            p = synthetic.u_channel()
            for b in p.bends.values():
                b.r_inner = 10.0
            w = self._warnings(p, material=mat)
            self.assertFalse(any("minimum" in x or "grain" in x for x in w), (mat, w))

    def test_natural_air_bend_radius_mismatch_is_flagged(self):
        """The expensive one: modelled radius the tooling will not produce."""
        part = synthetic.u_channel()
        # 2 mm CRS in the default 16 mm vee naturally makes R2.5, not R6.
        for b in part.bends.values():
            b.r_inner = 6.0
        w = self._warnings(part)
        self.assertTrue(any("naturally air bends" in x for x in w), w)

    def test_natural_radius_warning_quantifies_the_error(self):
        """The warning must state how far out the blank actually lands.

        R6 modelled vs R2.5 natural on 2 mm CRS: the allowance differs by
        theta*(dR + t*dK), which is several millimetres per bend -- enough to
        scrap a batch, so the number has to be real and not a placeholder.
        """
        from partestimator.smt.allowance import bend_allowance, k_factor

        part = synthetic.u_channel()
        for b in part.bends.values():
            b.r_inner = 6.0
        w = [x for x in self._warnings(part) if "naturally air bends" in x]
        self.assertEqual(len(w), 2)

        t, ang = 2.0, math.pi / 2
        ba_model = bend_allowance(ang, 6.0, t, 0.44)
        ba_nat = bend_allowance(ang, 2.496, t, k_factor(2.496, t, "CRS", "air"))
        expected = abs(ba_nat - ba_model)
        self.assertGreater(expected, 1.0)
        self.assertIn(f"{expected:.3f} mm out", w[0])

    def test_matching_radius_is_not_flagged(self):
        part = synthetic.u_channel()
        for b in part.bends.values():
            b.r_inner = 2.5       # what a 16 mm vee actually produces
        w = self._warnings(part)
        self.assertFalse(any("naturally air bends" in x for x in w), w)

    def test_press_capacity_is_flagged(self):
        part = synthetic.u_channel()
        w = self._warnings(part, press_capacity_t=0.1)
        self.assertTrue(any("rated" in x for x in w), w)

    def test_ample_press_capacity_is_not_flagged(self):
        part = synthetic.u_channel()
        w = self._warnings(part, press_capacity_t=500.0)
        self.assertFalse(any("rated" in x for x in w), w)


if __name__ == "__main__":
    unittest.main()
