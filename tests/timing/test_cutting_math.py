"""The building blocks of the mill timing model: spindle speed, feed, the
two MRR bounds, scallop stepover and the finish-tier factor."""
from __future__ import annotations

import math
import unittest

from partestimator.machining.geometry import cycle_time as ct

from .support import MAT, MILL_RATES, make_profile, make_table


class RpmAndClamps(unittest.TestCase):
    def test_rpm_from_sfm_formula(self):
        self.assertAlmostEqual(ct.rpm_from_sfm(800, 0.625), 3.82 * 800 / 0.625)

    def test_rpm_from_sfm_rejects_non_positive_diameter(self):
        self.assertEqual(ct.rpm_from_sfm(800, 0.0), 0.0)
        self.assertEqual(ct.rpm_from_sfm(800, -1.0), 0.0)

    def test_clamp_rpm_without_profile_is_identity(self):
        self.assertEqual(ct.clamp_rpm(50_000.0, None), (50_000.0, False))

    def test_clamp_rpm_at_spindle_ceiling(self):
        prof = make_profile(spindle_max_rpm=12_000)
        # The example from the docstring: a 0.190" drill at 680 sfm.
        wanted = ct.rpm_from_sfm(680, 0.190)
        self.assertGreater(wanted, 12_000)
        self.assertEqual(ct.clamp_rpm(wanted, prof), (12_000, True))
        self.assertEqual(ct.clamp_rpm(5_000.0, prof), (5_000.0, False))

    def test_zero_ceiling_means_unlimited(self):
        prof = make_profile(spindle_max_rpm=0)
        self.assertEqual(ct.clamp_rpm(99_999.0, prof), (99_999.0, False))

    def test_clamp_feed(self):
        self.assertEqual(ct.clamp_feed(900.0, None), 900.0)
        self.assertEqual(ct.clamp_feed(900.0, make_profile(max_cutting_feed_ipm=400)), 400)
        self.assertEqual(ct.clamp_feed(150.0, make_profile(max_cutting_feed_ipm=400)), 150.0)
        self.assertEqual(ct.clamp_feed(900.0, make_profile(max_cutting_feed_ipm=0)), 900.0)


class ScallopStepover(unittest.TestCase):
    def test_cusp_height_round_trips(self):
        # The stepover should leave a cusp of exactly h = 4 x Ra.
        R = ct.BALL_TOOL_RADIUS_IN
        for ra in ct.FINISH_TIERS_UIN:
            ae = ct.scallop_stepover_in(ra)
            h = R - math.sqrt(R * R - (ae / 2.0) ** 2)
            self.assertAlmostEqual(h, 4.0 * ra * 1e-6, places=9, msg=f"Ra {ra}")

    def test_finer_finish_means_smaller_stepover(self):
        steps = [ct.scallop_stepover_in(ra) for ra in ct.FINISH_TIERS_UIN]  # 250 .. 16
        self.assertEqual(steps, sorted(steps, reverse=True))

    def test_halving_ra_roughly_divides_stepover_by_sqrt2(self):
        # For h << R, ae ~ sqrt(8 R h), so ae scales with sqrt(Ra).
        ratio = ct.scallop_stepover_in(64.0) / ct.scallop_stepover_in(32.0)
        self.assertAlmostEqual(ratio, math.sqrt(2.0), places=3)

    def test_ra_below_one_is_floored(self):
        self.assertEqual(ct.scallop_stepover_in(0.0), ct.scallop_stepover_in(1.0))

    def test_cusp_capped_at_half_radius(self):
        R = 0.25
        self.assertAlmostEqual(ct.scallop_stepover_in(1e9, R), math.sqrt(3.0) * R)


class FinishRaFactor(unittest.TestCase):
    TIERS = {"finish_ra_factor_250uin": 0.85, "finish_ra_factor_125uin": 1.00,
             "finish_ra_factor_63uin": 1.25, "finish_ra_factor_32uin": 1.70,
             "finish_ra_factor_16uin": 2.40}

    def setUp(self):
        self.prof = make_profile(**self.TIERS)

    def test_no_profile_is_neutral(self):
        self.assertEqual(ct.finish_ra_factor(16.0, None), 1.0)

    def test_profile_without_tiers_is_neutral(self):
        self.assertEqual(ct.finish_ra_factor(16.0, make_profile(spindle_hp=20)), 1.0)

    def test_exact_tiers(self):
        for key, val in self.TIERS.items():
            ra = float(key.split("_")[-1].removesuffix("uin"))
            self.assertAlmostEqual(ct.finish_ra_factor(ra, self.prof), val, msg=key)

    def test_interpolates_in_log_ra(self):
        ra = math.sqrt(63.0 * 125.0)          # geometric midpoint
        self.assertAlmostEqual(ct.finish_ra_factor(ra, self.prof), (1.25 + 1.00) / 2)

    def test_clamped_outside_tier_range(self):
        self.assertAlmostEqual(ct.finish_ra_factor(2.0, self.prof), 2.40)
        self.assertAlmostEqual(ct.finish_ra_factor(1000.0, self.prof), 0.85)

    def test_monotonic_tighter_finish_costs_more(self):
        ras = [250, 200, 125, 90, 63, 40, 32, 20, 16]
        factors = [ct.finish_ra_factor(r, self.prof) for r in ras]
        self.assertEqual(factors, sorted(factors))


class OpenRoughingMrr(unittest.TestCase):
    def setUp(self):
        self.table = make_table()
        r = MILL_RATES
        feed = ct.rpm_from_sfm(r["rough_open_sfm"], ct.ROUGH_TOOL_DIA_IN) \
            * ct.ROUGH_TOOL_FLUTES * r["rough_open_ipt"]
        self.tool_bound = (ct.ROUGH_TOOL_DIA_IN * ct.ROUGH_RADIAL_FRACTION
                           * ct.ROUGH_TOOL_DIA_IN * ct.ROUGH_AXIAL_FRACTION * feed)

    def test_tool_bound_wins_on_a_big_spindle(self):
        mrr, bound, _ = ct._mrr_open(self.table, MAT, "mill", 22.0)
        self.assertEqual(bound, "tool")
        self.assertAlmostEqual(mrr, self.tool_bound)

    def test_power_bound_wins_on_a_small_spindle(self):
        mrr, bound, _ = ct._mrr_open(self.table, MAT, "mill", 1.0)
        self.assertEqual(bound, "power")
        self.assertAlmostEqual(mrr, 1.0 * ct.DRIVE_EFFICIENCY / MILL_RATES["unit_power"])

    def test_profile_drive_efficiency_used_for_power_bound(self):
        prof = make_profile(drive_efficiency=0.5)
        mrr, bound, _ = ct._mrr_open(self.table, MAT, "mill", 1.0, prof)
        self.assertEqual(bound, "power")
        self.assertAlmostEqual(mrr, 0.5 / MILL_RATES["unit_power"])

    def test_duty_scales_the_peak(self):
        peak, _, _ = ct._mrr_open(self.table, MAT, "mill", 22.0, make_profile())
        avg, _, detail = ct._mrr_open(self.table, MAT, "mill", 22.0,
                                      make_profile(rough_open_duty=0.7))
        self.assertAlmostEqual(avg, 0.7 * peak)
        self.assertIn("duty 70%", detail)

    def test_feed_cap_lowers_tool_bound(self):
        prof = make_profile(max_cutting_feed_ipm=10.0)
        mrr, bound, _ = ct._mrr_open(self.table, MAT, "mill", 22.0, prof)
        self.assertEqual(bound, "tool")
        ae = ct.ROUGH_TOOL_DIA_IN * ct.ROUGH_RADIAL_FRACTION
        ap = ct.ROUGH_TOOL_DIA_IN * ct.ROUGH_AXIAL_FRACTION
        self.assertAlmostEqual(mrr, ae * ap * 10.0)

    def test_missing_rate_data(self):
        table = make_table(mill={"unit_power": 0.3})
        self.assertEqual(ct._mrr_open(table, MAT, "mill", 22.0), (None, "", ""))


class FinishingRates(unittest.TestCase):
    def setUp(self):
        self.table = make_table()
        r = MILL_RATES
        self.sfm = r["rough_open_sfm"] * ct.FINISH_SFM_FRACTION
        self.ipt = r["rough_open_ipt"] * ct.FINISH_IPT_FRACTION

    def test_analytic_rate_is_stepover_times_feed(self):
        rate, _ = ct._finish_rate_analytic(self.table, MAT, "mill")
        feed = ct.rpm_from_sfm(self.sfm, ct.FINISH_TOOL_DIA_IN) * ct.FINISH_TOOL_FLUTES * self.ipt
        self.assertAlmostEqual(
            rate, ct.FINISH_TOOL_DIA_IN * ct.FINISH_ANALYTIC_STEPOVER_FRACTION * feed)

    def test_analytic_rate_divided_by_ra_factor_and_scaled_by_duty(self):
        base, _ = ct._finish_rate_analytic(self.table, MAT, "mill", make_profile(), 125.0)
        prof = make_profile(finish_duty=0.8, finish_ra_factor_125uin=2.0)
        rate, _ = ct._finish_rate_analytic(self.table, MAT, "mill", prof, 125.0)
        self.assertAlmostEqual(rate, base * 0.8 / 2.0)

    def test_small_cutter_slower_than_diameter_alone_once_rpm_clamps(self):
        prof = make_profile(spindle_max_rpm=12_000)
        big, _ = ct._finish_rate_analytic(self.table, MAT, "mill", prof)
        small, detail = ct._finish_rate_analytic(self.table, MAT, "mill", prof,
                                                 tool_dia_in=0.125)
        self.assertIn("rpm clamped", detail)
        # Unclamped, rate is independent of diameter (rpm ~ 1/D, stepover ~ D);
        # the clamp is what makes the small tool genuinely slower.
        self.assertLess(small, big * 0.5)

    def test_freeform_rate_uses_scallop_stepover(self):
        rate, _ = ct._finish_rate_freeform(self.table, MAT, "mill", 63.0)
        feed = (ct.rpm_from_sfm(self.sfm, 2 * ct.BALL_TOOL_RADIUS_IN)
                * ct.FINISH_TOOL_FLUTES * self.ipt)
        self.assertAlmostEqual(rate, ct.scallop_stepover_in(63.0) * feed)

    def test_freeform_rate_drops_with_finer_finish(self):
        coarse, _ = ct._finish_rate_freeform(self.table, MAT, "mill", 125.0)
        fine, _ = ct._finish_rate_freeform(self.table, MAT, "mill", 16.0)
        self.assertLess(fine, coarse)

    def test_missing_data_gives_zero_rate(self):
        table = make_table(mill={"unit_power": 0.3})
        self.assertEqual(ct._finish_rate_analytic(table, MAT, "mill"), (0.0, ""))
        self.assertEqual(ct._finish_rate_freeform(table, MAT, "mill", 63.0), (0.0, ""))


if __name__ == "__main__":
    unittest.main()
