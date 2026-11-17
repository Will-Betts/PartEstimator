"""estimate_cycle_time(): the term-by-term mill estimate and its totals."""
from __future__ import annotations

import unittest

from partestimator.machining.geometry import cycle_time as ct

from .support import MAT, FakeHole, make_profile, make_table, prismatic_features


def term(est, name):
    found = [t for t in est.terms if t.name == name]
    return found[0] if found else None


def run(features=None, profile=None, finish=125.0, holes=(), **kw):
    return ct.estimate_cycle_time(
        features or prismatic_features(), {"holes": list(holes)}, make_table(),
        MAT, finish, "mill", profile=profile, **kw)


class Accounting(unittest.TestCase):
    """Identities that must hold for any input."""

    CASES = [
        dict(),
        dict(finish=16.0),
        dict(holes=[FakeHole.inches(0.25, 0.5)] * 4),
        dict(operations=3),
    ]

    def profiles(self):
        yield None
        yield make_profile(
            rough_open_duty=0.7, rough_pocket_duty=0.5, finish_duty=0.8,
            semi_finish_enabled=1, rest_rough_fraction=0.25, deburr_enabled=1,
            deburr_rate_ipm=15.0, deburr_convex_fraction=1.0, spring_pass_below_uin=32,
            tool_change_s=12.0, rapid_fraction=0.15, hole_position_s=2.0,
            index_time_s=8.0, calibrate_overall=1.2)

    def test_totals_add_up(self):
        feats = prismatic_features(area_freeform_in2=12.0, convex_edge_length_in=60.0,
                                   hole_axis_orientations_5p0deg=3)
        for prof in self.profiles():
            for case in self.CASES:
                with self.subTest(profile=prof is not None, **{k: str(v) for k, v in case.items()}):
                    est = run(feats, prof, **case)
                    self.assertAlmostEqual(
                        est.machine_cycle_minutes,
                        est.cut_minutes + est.rapid_minutes + est.tool_change_minutes)
                    self.assertAlmostEqual(
                        est.door_to_door_minutes,
                        (est.machine_cycle_minutes + est.load_unload_minutes
                         + est.deburr_minutes) * est.calibration)
                    self.assertAlmostEqual(
                        est.cut_minutes,
                        sum(t.minutes for t in est.terms if t.name != "Deburring"))
                    for t in est.terms:
                        self.assertGreaterEqual(t.minutes, 0.0, t.name)

    def test_as_rows_ends_with_door_to_door(self):
        est = run()
        label, value, _ = est.as_rows()[-1]
        self.assertEqual(label, "DOOR TO DOOR")
        self.assertEqual(value, f"{est.door_to_door_minutes:,.2f} min")


class BaselineArithmetic(unittest.TestCase):
    """No shop profile: every term checked against hand arithmetic."""

    def setUp(self):
        self.est = run()
        self.mrr, _, _ = ct._mrr_open(make_table(), MAT, "mill", ct.DEFAULT_SPINDLE_HP)

    def test_outside_roughing(self):
        # stock 30 - hull (10 / 0.5 = 20) = 10 in^3
        t = term(self.est, "Outside roughing")
        self.assertAlmostEqual(t.quantity, 10.0)
        self.assertAlmostEqual(t.minutes, 10.0 / self.mrr)
        self.assertEqual(t.limited_by, "tool")

    def test_pocket_roughing_is_derated(self):
        # hull 20 - part 10 = 10 in^3 at half the open rate
        t = term(self.est, "Pocket roughing")
        self.assertAlmostEqual(t.quantity, 10.0)
        self.assertAlmostEqual(t.rate, self.mrr * ct.POCKET_MRR_FRACTION)
        self.assertAlmostEqual(t.minutes, 10.0 / (self.mrr * ct.POCKET_MRR_FRACTION))

    def test_analytic_finishing(self):
        rate, _ = ct._finish_rate_analytic(make_table(), MAT, "mill", None, 125.0)
        t = term(self.est, "Finishing, analytic")
        self.assertAlmostEqual(t.minutes, 40.0 / rate)

    def test_no_freeform_or_holes_terms_when_absent(self):
        names = {t.name for t in self.est.terms}
        self.assertNotIn("Finishing, freeform", names)
        self.assertNotIn("Holes", names)
        self.assertNotIn("Deburring", names)

    def test_non_cutting_defaults(self):
        est = self.est
        self.assertAlmostEqual(est.rapid_minutes, est.cut_minutes * ct.RAPID_FRACTION)
        # rougher + analytic finisher
        self.assertEqual(est.tool_count, 2)
        self.assertAlmostEqual(est.tool_change_minutes, 2 * ct.TOOL_CHANGE_S / 60.0)
        weight = 30.0 * 0.098
        self.assertAlmostEqual(est.load_unload_minutes,
                               ct.MIN_LOAD_UNLOAD_MIN + weight * ct.LOAD_UNLOAD_PER_LB_MIN)

    def test_solidity_zero_means_hull_equals_part(self):
        est = run(prismatic_features(solidity=0.0))
        self.assertIsNone(term(est, "Pocket roughing"))
        self.assertAlmostEqual(term(est, "Outside roughing").quantity, 20.0)

    def test_explicit_load_unload_overrides_weight_model(self):
        self.assertEqual(run(load_unload_min=2.5).load_unload_minutes, 2.5)


class ShopProfileTerms(unittest.TestCase):
    def test_pocket_duty_ratio(self):
        base = run(profile=make_profile(rough_open_duty=0.7, rough_pocket_duty=0.7))
        worse = run(profile=make_profile(rough_open_duty=0.7, rough_pocket_duty=0.35))
        self.assertAlmostEqual(term(worse, "Pocket roughing").minutes,
                               2.0 * term(base, "Pocket roughing").minutes)
        self.assertAlmostEqual(term(worse, "Outside roughing").minutes,
                               term(base, "Outside roughing").minutes)

    def test_rest_roughing(self):
        prof = make_profile(rest_rough_fraction=0.25, rest_rough_rate_fraction=0.35)
        est = run(profile=prof)
        pocket = term(est, "Pocket roughing")
        rest = term(est, "Rest roughing")
        self.assertAlmostEqual(rest.quantity, 0.25 * pocket.quantity)
        self.assertAlmostEqual(rest.rate, 0.35 * pocket.rate)
        self.assertEqual(est.tool_count, 3)

    def test_semi_finishing_adds_a_term_and_a_tool(self):
        off = run(profile=make_profile())
        on = run(profile=make_profile(semi_finish_enabled=1,
                                      semi_finish_stepover_fraction=0.4))
        self.assertIsNone(term(off, "Semi-finishing"))
        self.assertIsNotNone(term(on, "Semi-finishing"))
        self.assertEqual(on.tool_count, off.tool_count + 1)

    def test_thin_wall_derate_slows_finishing_only(self):
        feats = prismatic_features(thin_area_fraction=0.5)
        off = run(feats, make_profile())
        on = run(feats, make_profile(thin_wall_derate_enabled=1, thin_wall_feed_derate=0.5))
        # (1 - 0.5) + 0.5 / 0.5 = 1.5x on finishing
        self.assertAlmostEqual(term(on, "Finishing, analytic").minutes,
                               1.5 * term(off, "Finishing, analytic").minutes)
        self.assertAlmostEqual(term(on, "Outside roughing").minutes,
                               term(off, "Outside roughing").minutes)

    def test_small_cutter_area_moves_out_of_analytic(self):
        feats = prismatic_features(area_radius_under_1_4in_in2=10.0,
                                   min_internal_radius_in=0.0625)
        prof = make_profile(small_cutter_derate_enabled=1, spindle_max_rpm=12_000)
        est = run(feats, prof)
        small = term(est, "Finishing, small cutter")
        self.assertAlmostEqual(small.quantity, 10.0)
        self.assertAlmostEqual(term(est, "Finishing, analytic").quantity, 30.0)
        self.assertLess(small.rate, term(est, "Finishing, analytic").rate)

    def test_spring_pass_at_or_below_threshold(self):
        prof = make_profile(spring_pass_below_uin=32, spring_pass_fraction=1.0,
                            finish_ra_factor_125uin=1.0)
        above = run(profile=prof, finish=63.0)
        at = run(profile=prof, finish=32.0)
        self.assertIsNone(term(above, "Spring pass"))
        sp = term(at, "Spring pass")
        self.assertAlmostEqual(sp.minutes, term(at, "Finishing, analytic").minutes)

    def test_spring_pass_ignores_freeform(self):
        feats = prismatic_features(area_analytic_in2=0.0, area_freeform_in2=40.0)
        est = run(feats, make_profile(spring_pass_below_uin=32), finish=16.0)
        self.assertIsNone(term(est, "Spring pass"))

    def test_deburring_is_outside_machine_cycle_and_rapids(self):
        feats = prismatic_features(convex_edge_length_in=60.0, concave_edge_length_in=40.0)
        prof = make_profile(deburr_enabled=1, deburr_rate_ipm=15.0,
                            deburr_convex_fraction=1.0, deburr_concave_fraction=0.25)
        est = run(feats, prof)
        d = term(est, "Deburring")
        self.assertAlmostEqual(d.quantity, 60.0 + 0.25 * 40.0)
        self.assertAlmostEqual(est.deburr_minutes, 70.0 / 15.0)
        self.assertAlmostEqual(est.cut_minutes,
                               sum(t.minutes for t in est.terms if t.name != "Deburring"))
        # No rapid allowance on hand deburring: rapids follow cut time alone.
        self.assertAlmostEqual(est.rapid_minutes, est.cut_minutes * ct.RAPID_FRACTION)
        self.assertAlmostEqual(est.door_to_door_minutes,
                               est.machine_cycle_minutes + est.load_unload_minutes
                               + est.deburr_minutes)

    def test_deburring_not_double_counted_with_indexing(self):
        # Regression: adding the 5-axis term re-summed cut time over ALL
        # terms, pulling Deburring into the machine cycle as well as adding
        # it to door-to-door separately.
        feats = prismatic_features(convex_edge_length_in=60.0,
                                   hole_axis_orientations_5p0deg=3)
        deburr = dict(deburr_enabled=1, deburr_rate_ipm=15.0, deburr_convex_fraction=1.0)
        without = run(feats, make_profile(**deburr))
        with_idx = run(feats, make_profile(index_time_s=8.0, **deburr))
        # The only difference should be two indexes (3 orientations - 1).
        self.assertAlmostEqual(with_idx.door_to_door_minutes - without.door_to_door_minutes,
                               2 * 8.0 / 60.0)
        self.assertAlmostEqual(with_idx.deburr_minutes, 4.0)

    def test_rapids_not_charged_on_hole_time(self):
        prof = make_profile(rapid_fraction=0.2, hole_position_s=2.0)
        est = run(profile=prof, holes=[FakeHole.inches(0.25, 0.5)] * 5)
        holes = term(est, "Holes").minutes
        self.assertAlmostEqual(est.rapid_minutes, (est.cut_minutes - holes) * 0.2)

    def test_five_axis_indexing(self):
        feats = prismatic_features(hole_axis_orientations_5p0deg=4)
        est = run(feats, make_profile(index_time_s=8.0))
        self.assertAlmostEqual(est.index_minutes, 3 * 8.0 / 60.0)
        self.assertIsNotNone(term(est, "5-axis indexing"))
        self.assertIsNone(term(run(feats, make_profile()), "5-axis indexing"))


class OperationsAndCalibration(unittest.TestCase):
    def test_extra_operations_repeat_overheads_not_cutting(self):
        one = run(operations=1)
        two = run(operations=2)
        self.assertAlmostEqual(two.cut_minutes, one.cut_minutes)
        self.assertAlmostEqual(two.tool_change_minutes, 2 * one.tool_change_minutes)
        self.assertAlmostEqual(two.load_unload_minutes, 2 * one.load_unload_minutes)
        self.assertTrue(any("2 operations" in w for w in two.warnings))

    def test_operations_from_profile(self):
        est = run(profile=make_profile(operations=3))
        self.assertEqual(est.operations, 3)

    def test_operations_floor_at_one(self):
        self.assertEqual(run(operations=0).operations, 1)

    def test_calibrate_overall_scales_total_only(self):
        base = run(profile=make_profile())
        cal = run(profile=make_profile(calibrate_overall=1.5))
        self.assertAlmostEqual(cal.door_to_door_minutes, 1.5 * base.door_to_door_minutes)
        self.assertAlmostEqual(cal.machine_cycle_minutes, base.machine_cycle_minutes)
        self.assertEqual(cal.calibration, 1.5)

    def test_per_term_calibration(self):
        holes = [FakeHole.inches(0.25, 0.5)] * 3
        base = run(profile=make_profile(), holes=holes)
        cal = run(profile=make_profile(calibrate_roughing=2.0, calibrate_finishing=3.0,
                                       calibrate_holes=4.0), holes=holes)
        for name, k in (("Outside roughing", 2.0), ("Pocket roughing", 2.0),
                        ("Finishing, analytic", 3.0), ("Holes", 4.0)):
            self.assertAlmostEqual(term(cal, name).minutes, k * term(base, name).minutes,
                                   msg=name)


class Behaviour(unittest.TestCase):
    def test_finer_finish_never_faster(self):
        feats = prismatic_features(area_freeform_in2=30.0)
        prof = make_profile(finish_ra_factor_250uin=0.85, finish_ra_factor_125uin=1.0,
                            finish_ra_factor_63uin=1.25, finish_ra_factor_32uin=1.7,
                            finish_ra_factor_16uin=2.4)
        times = [run(feats, prof, finish=ra).door_to_door_minutes
                 for ra in ct.FINISH_TIERS_UIN]
        self.assertEqual(times, sorted(times))

    def test_more_stock_takes_longer(self):
        small = run(prismatic_features(stock_volume_in3=25.0))
        big = run(prismatic_features(stock_volume_in3=60.0))
        self.assertGreater(big.door_to_door_minutes, small.door_to_door_minutes)

    def test_unknown_material_returns_empty_estimate_with_warning(self):
        est = ct.estimate_cycle_time(prismatic_features(), {"holes": []}, make_table(),
                                     "Unobtainium", 125.0, "mill")
        self.assertEqual(est.terms, [])
        self.assertEqual(est.door_to_door_minutes, 0.0)
        self.assertTrue(any("No rate data" in w for w in est.warnings))

    def test_warnings(self):
        feats = prismatic_features(has_thin_walls=True, hole_captured_count=2,
                                   min_internal_radius_in=0.05)
        text = " ".join(run(feats).warnings)
        self.assertIn("Thin walls", text)
        self.assertIn("2 captured hole", text)
        self.assertIn("Smallest internal radius", text)

    def test_non_cutting_dominates_warning(self):
        tiny = prismatic_features(volume_in3=0.1, stock_volume_in3=0.12, solidity=1.0,
                                  area_analytic_in2=0.5)
        self.assertTrue(any("Non-cutting time dominates" in w for w in run(tiny).warnings))


class ToolCount(unittest.TestCase):
    def test_counts(self):
        f = {"area_analytic_in2": 1, "area_freeform_in2": 1, "hole_distinct_diameters": 3,
             "hole_counterbored_count": 5, "hole_count": 9}
        # rougher, finisher, ball, 3 drills, 1 counterbore, spot + chamfer
        self.assertEqual(ct._tool_count(f), 9)
        self.assertEqual(ct._tool_count(f, tapped=True, semi_finish=True, rest_rough=True), 12)
        self.assertEqual(ct._tool_count({}), 1)

    def test_counterbores_add_one_tool_however_many(self):
        one = ct._tool_count({"hole_counterbored_count": 1})
        many = ct._tool_count({"hole_counterbored_count": 50})
        self.assertEqual(one, many)


if __name__ == "__main__":
    unittest.main()
