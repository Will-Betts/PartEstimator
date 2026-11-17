"""Hole cycle time, peck penalty, tapping and tap-drill detection."""
from __future__ import annotations

import unittest

from partestimator.machining.geometry import cycle_time as ct
from partestimator.machining.geometry import shop

from .support import MAT, MILL_RATES, FakeHole, make_profile, make_table

SFM = MILL_RATES["drill_sfm"]
IPR_PER_IN = MILL_RATES["drill_ipr_per_inch_dia"]


def drill_minutes(dia_in, depth_in):
    feed = ct.rpm_from_sfm(SFM, dia_in) * IPR_PER_IN * dia_in
    return depth_in / feed


def hole_time(holes, profile=None, features=None, tapped_override=None):
    return ct._hole_time(make_table(), MAT, "mill", {"holes": holes}, features or {},
                         profile, tapped_override)


class DrillTime(unittest.TestCase):
    def test_single_hole_cut_time(self):
        drill, tap, note, _ = hole_time([FakeHole.inches(0.25, 0.5)])
        self.assertAlmostEqual(drill, drill_minutes(0.25, 0.5))
        self.assertEqual(tap, 0.0)
        self.assertIn("1 hole(s)", note)

    def test_feed_independent_of_diameter_before_clamp(self):
        # rpm ~ 1/D and ipr ~ D, so feed in ipm is the same for any size.
        self.assertAlmostEqual(drill_minutes(0.25, 1.0), drill_minutes(0.5, 1.0))

    def test_peck_penalty_past_ratio(self):
        shallow = FakeHole.inches(0.25, 0.25 * ct.PECK_DEPTH_RATIO)          # exactly 4xD
        deep = FakeHole.inches(0.25, 0.25 * ct.PECK_DEPTH_RATIO * 1.5)       # 6xD
        t_shallow, *_ = hole_time([shallow])
        t_deep, _, note, _ = hole_time([deep])
        self.assertAlmostEqual(t_shallow, drill_minutes(0.25, 1.0))
        self.assertAlmostEqual(t_deep, drill_minutes(0.25, 1.5) * (1 + ct.PECK_TIME_PENALTY))
        self.assertIn("1 pecked", note)

    def test_per_hole_overhead(self):
        prof = make_profile(hole_position_s=2.0, hole_approach_s=1.0, hole_retract_s=0.6,
                            spot_drill_enabled=1, spot_drill_s=2.5,
                            chamfer_s=3.0, chamfer_fraction=0.5)
        holes = [FakeHole.inches(0.25, 0.5)] * 4
        drill, *_ = hole_time(holes, prof)
        per_hole_s = 2.0 + 1.0 + 0.6 + 2.5 + 3.0 * 0.5
        self.assertAlmostEqual(drill, 4 * (drill_minutes(0.25, 0.5) + per_hole_s / 60.0))

    def test_spot_drill_only_when_enabled(self):
        h = [FakeHole.inches(0.25, 0.5)]
        off, *_ = hole_time(h, make_profile(spot_drill_s=2.5))
        on, *_ = hole_time(h, make_profile(spot_drill_s=2.5, spot_drill_enabled=1))
        self.assertAlmostEqual(on - off, 2.5 / 60.0)

    def test_rpm_clamp_slows_small_drills(self):
        h = [FakeHole.inches(0.125, 0.5)]
        free, *_ = hole_time(h)
        clamped, *_ = hole_time(h, make_profile(spindle_max_rpm=10_000))
        self.assertAlmostEqual(clamped, 0.5 / (10_000 * IPR_PER_IN * 0.125))
        self.assertGreater(clamped, free)

    def test_degenerate_holes_skipped(self):
        drill, *_ = hole_time([FakeHole(0.0, 10.0), FakeHole(5.0, 0.0)])
        self.assertEqual(drill, 0.0)

    def test_aggregate_fallback_without_hole_list(self):
        feats = {"hole_total_depth_in": 3.0, "hole_dia_mean_in": 0.25, "hole_count": 6}
        prof = make_profile(hole_position_s=2.0)
        drill, tap, note, _ = hole_time([], prof, feats)
        self.assertAlmostEqual(drill, drill_minutes(0.25, 3.0) + 6 * 2.0 / 60.0)
        self.assertEqual(tap, 0.0)
        self.assertIn("aggregate", note)

    def test_no_holes_at_all(self):
        self.assertEqual(hole_time([]), (0.0, 0.0, "", ""))

    def test_no_drill_data(self):
        table = make_table(mill={"unit_power": 0.3})
        out = ct._hole_time(table, MAT, "mill", {"holes": [FakeHole.inches(0.25, 0.5)]},
                            {}, None)
        self.assertEqual(out, (0.0, 0.0, "", ""))


class Tapping(unittest.TestCase):
    PROF = dict(tap_rpm=600.0, tap_retract_ratio=1.0, tap_overhead_s=3.0,
                tap_depth_fraction=0.75)

    def test_quarter_twenty_tap_time(self):
        hole = FakeHole.inches(0.2010, 0.5)                # 1/4-20 tap drill
        _, tap, _, note = hole_time([hole], make_profile(**self.PROF))
        threaded = 0.5 * 0.75
        feed = 600.0 / 20.0
        self.assertAlmostEqual(tap, 2 * threaded / feed + 3.0 / 60.0)
        self.assertIn("1/4-20", note)

    def test_faster_retract(self):
        hole = [FakeHole.inches(0.2010, 0.5)]
        _, base, *_ = hole_time(hole, make_profile(**self.PROF))
        _, fast, *_ = hole_time(hole, make_profile(**{**self.PROF, "tap_retract_ratio": 2.0}))
        threaded_in = 0.375 / 30.0
        self.assertAlmostEqual(base - fast, threaded_in - threaded_in / 2.0)

    def test_override_limits_tapped_count(self):
        holes = [FakeHole.inches(0.2010, 0.5)] * 3
        prof = make_profile(**self.PROF)
        _, all3, *_ = hole_time(holes, prof)
        _, one, *_ = hole_time(holes, prof, tapped_override=1)
        _, none, *_ = hole_time(holes, prof, tapped_override=0)
        self.assertAlmostEqual(one, all3 / 3)
        self.assertEqual(none, 0.0)

    def test_clearance_hole_not_tapped(self):
        _, tap, *_ = hole_time([FakeHole.inches(0.2660, 0.5)], make_profile(**self.PROF))
        self.assertEqual(tap, 0.0)


class TapDrillDetection(unittest.TestCase):
    def test_every_tap_drill_maps_to_its_own_label(self):
        for dia, label in shop.TAP_DRILLS_IN.items():
            found = shop.identify_tapped_holes([FakeHole.inches(dia, 0.5)])
            self.assertEqual(len(found), 1, label)
            self.assertEqual(found[0][1], label)
            self.assertIn(label, shop.TAP_PITCH_IN)

    def test_tolerance(self):
        tol = shop.TAP_DRILL_TOLERANCE_IN
        near = FakeHole.inches(0.2010 + tol * 0.9, 0.5)
        far = FakeHole.inches(0.2010 + tol * 1.5, 0.5)
        self.assertEqual(len(shop.identify_tapped_holes([near])), 1)
        # 0.2010 + 0.006 = 0.2070: nowhere near another tap drill either
        self.assertEqual(shop.identify_tapped_holes([far]), [])

    def test_pitch(self):
        (_, label, pitch), = shop.identify_tapped_holes([FakeHole.inches(0.2008, 0.5)])
        self.assertEqual(label, "M6x1.0")
        self.assertAlmostEqual(pitch, 1.0 / 25.4)


class HolesInFullEstimate(unittest.TestCase):
    def test_tap_adds_a_tool_and_terms(self):
        feats = {"volume_in3": 1, "stock_volume_in3": 2, "solidity": 1,
                 "hole_count": 2, "hole_distinct_diameters": 1}
        holes = [FakeHole.inches(0.2010, 0.5)] * 2
        prof = make_profile(**Tapping.PROF)
        est = ct.estimate_cycle_time(feats, {"holes": holes}, make_table(), MAT, 125.0,
                                     "mill", profile=prof)
        names = [t.name for t in est.terms]
        self.assertIn("Holes", names)
        self.assertIn("Tapping", names)
        # rougher + 1 drill + spot + chamfer + tap
        self.assertEqual(est.tool_count, 5)


if __name__ == "__main__":
    unittest.main()
