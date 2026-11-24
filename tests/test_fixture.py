"""Tabbed fixturing: the layout (geometry/fixture.py), the settings that set
the shop standard, and what the cycle-time estimate charges for it.

numpy and scipy on synthetic meshes; no CAD kernel or Qt. The solid bodies
and the window are in test_fixture_cad.py and test_fixture_gui.py.
"""
from __future__ import annotations

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import scipy  # noqa: F401

    from partestimator.machining.geometry import fixture as fx

    HAVE = True
except ImportError:  # the light CI environment has no scipy
    HAVE = False

from partestimator.machining.geometry import stock as st

MM = 25.4


def box_mesh(l_in, w_in, t_in, origin=(0.0, 0.0, 0.0)):
    """Points (mm) and triangles of a closed box."""
    o = np.asarray(origin, dtype=float)
    pts = np.array([[x, y, z] for z in (0, t_in) for y in (0, w_in) for x in (0, l_in)],
                   dtype=float) * MM + o * MM
    quads = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    tris = []
    for a, b, c, d in quads:
        tris += [(a, b, c), (a, c, d)]
    return pts, np.array(tris)


def l_mesh(l_in=4.0, a_in=3.0, b_in=2.0, t_in=0.25):
    """An L section a x b, legs t thick, run l along X: as two boxes."""
    p1, t1 = box_mesh(l_in, a_in, t_in)
    p2, t2 = box_mesh(l_in, t_in, b_in)
    return np.vstack([p1, p2]), np.vstack([t1, t2 + len(p1)])


def flat_planes(l_in, w_in):
    area = l_in * w_in * MM * MM
    return [((0, 0, 1), area), ((0, 0, -1), area)]


def plan_box(l_in, w_in, t_in, stock=None, **kw):
    pts, tris = box_mesh(l_in, w_in, t_in)
    rect_ext = np.array([l_in, w_in, t_in]) * MM
    stock = stock or st.rectilinear_stock(rect_ext)
    return fx.plan(pts, tris, np.eye(3), rect_ext, stock, l_in * w_in * t_in,
                   planes=flat_planes(l_in, w_in), **kw)


@unittest.skipUnless(HAVE, "needs scipy")
class TestProcedure(unittest.TestCase):
    def test_thick_plate_stands_in_the_vise(self):
        p = plan_box(6.0, 4.0, 0.5)
        self.assertEqual((p.procedure, p.operations), (fx.VERTICAL, 1))
        self.assertIn("above the jaws", p.why)

    def test_thin_plate_lies_flat(self):
        # 0.5" stock (0.25" part + cleanup) under a 0.6" minimum: a floor to face off.
        p = plan_box(6.0, 4.0, 0.25, settings=fx.FixtureSettings(vertical_min_thickness_in=0.6))
        self.assertEqual((p.procedure, p.operations), (fx.FLAT, 2))
        self.assertIn("minimum", p.why)

    def test_sheet_is_one_flat_operation_cut_through(self):
        # Sheet is bought to the part's thickness: no floor, nothing for op 2.
        p = plan_box(6.0, 4.0, 0.15)
        self.assertEqual((p.procedure, p.operations), (fx.FLAT, 1))
        self.assertTrue(any("spoilboard" in w for w in p.warnings))

    def test_tall_stickout_lies_flat(self):
        # 0.5" plate sticking 9" out of the jaws: more than 10x its thickness.
        p = plan_box(10.0, 8.0, 0.375, grow_stock=True)
        self.assertEqual(p.procedure, fx.FLAT)
        self.assertIn("more than", p.why)

    def test_the_limits_are_settings(self):
        s = fx.FixtureSettings(vertical_min_thickness_in=0.1, max_stickout_ratio=50.0)
        self.assertEqual(plan_box(6.0, 4.0, 0.15, settings=s).procedure, fx.VERTICAL)
        s = fx.FixtureSettings(max_stickout_ratio=2.0)
        self.assertEqual(plan_box(6.0, 4.0, 0.5, settings=s).procedure, fx.FLAT)

    def test_clamping_a_short_edge_turns_the_plan(self):
        long_ = plan_box(6.0, 3.0, 1.0)
        short = plan_box(6.0, 3.0, 1.0, clamp_edge=fx.CLAMP_EDGES[2])
        # X runs along the clamped edge.
        self.assertGreater(long_.stock_in[0], long_.stock_in[1])
        self.assertLess(short.stock_in[0], short.stock_in[1])


@unittest.skipUnless(HAVE, "needs scipy")
class TestLayout(unittest.TestCase):
    def test_the_stock_grows_for_the_frame_and_the_vise(self):
        s = fx.FixtureSettings()
        p = plan_box(6.0, 4.0, 0.5, grow_stock=True)
        w = p.channel_width_in
        self.assertEqual(w, 0.250)                     # 0.25" cut from each side
        need_x = 6.0 + 2 * w + 2 * s.min_frame_in
        need_y = 4.0 + 2 * w + 2 * s.min_frame_in + s.vise_grip_in + s.vise_clearance_in
        self.assertEqual(p.stock_in[0], st.ceil_to_increment(need_x))
        self.assertEqual(p.stock_in[1], st.ceil_to_increment(need_y))
        self.assertEqual(p.warnings, [])

    def test_a_small_stock_is_warned_not_grown(self):
        p = plan_box(6.0, 4.0, 0.5)
        self.assertEqual(p.stock_in[:2], [7.0, 5.0])
        self.assertTrue(any("frame" in w for w in p.warnings))

    def test_the_part_is_centred(self):
        p = plan_box(6.0, 4.0, 0.5, grow_stock=True)
        s = fx.FixtureSettings()
        lo, hi = np.array(p.part_lo_in), np.array(p.part_hi_in)
        sx, sy, sz = p.stock_in
        grip = s.vise_grip_in + s.vise_clearance_in
        np.testing.assert_allclose((lo + hi) / 2, [sx / 2, grip + (sy - grip) / 2, sz / 2])
        flat = plan_box(6.0, 4.0, 0.15, grow_stock=True)
        lo, hi = np.array(flat.part_lo_in), np.array(flat.part_hi_in)
        np.testing.assert_allclose((lo + hi) / 2, np.array(flat.stock_in) / 2)

    def test_outline_and_channel_areas(self):
        p = plan_box(6.0, 4.0, 0.5)
        w = p.channel_width_in
        self.assertAlmostEqual(p.footprint_area_in2, 24.0, delta=0.15)
        rounded = (6 + 2 * w) * (4 + 2 * w) - (4 - math.pi) * w * w
        self.assertAlmostEqual(p.channel_area_in2, rounded, delta=0.2)
        self.assertAlmostEqual(p.perimeter_in, 20.0, delta=0.1)
        self.assertEqual(len(p.outline), 1)
        self.assertLessEqual(len(p.outline[0]), 6)       # a rectangle, straightened

    def test_volumes_add_up(self):
        p = plan_box(6.0, 4.0, 0.5, grow_stock=True)
        part = 12.0
        self.assertAlmostEqual(p.removed_volume_in3,
                               p.stock_volume_in3 - part - p.frame_volume_in3, places=6)
        sx, sy, sz = p.stock_in
        self.assertAlmostEqual(p.facing_volume_in3, sx * sy * (sz - 0.5), places=6)
        self.assertAlmostEqual(p.channel_volume_in3,
                               (p.channel_area_in2 - p.footprint_area_in2) * 0.5, places=6)

    def test_channel_width_table(self):
        s = fx.FixtureSettings()
        self.assertEqual([s.channel_width(d) for d in (0.25, 0.5, 0.75, 1.0, 2.0)],
                         [0.25, 0.25, 0.375, 0.375, 0.5])
        self.assertEqual(plan_box(6.0, 4.0, 1.5).channel_width_in, 0.375)   # 0.875/side
        self.assertEqual(plan_box(6.0, 4.0, 0.15).channel_width_in, 0.25)   # flat: full depth
        self.assertEqual(plan_box(6.0, 4.0, 0.5, channel_width_in=0.5).channel_width_in, 0.5)


@unittest.skipUnless(HAVE, "needs scipy")
class TestTabs(unittest.TestCase):
    def test_count_from_the_outline(self):
        self.assertEqual(plan_box(6.0, 4.0, 0.5).tab_count, 5)      # 20" / 4"
        self.assertEqual(plan_box(2.0, 1.0, 0.5).tab_count, 3)      # the minimum
        self.assertEqual(plan_box(6.0, 4.0, 0.5, tab_count=8).tab_count, 8)

    def test_tabs_sit_on_straight_edges_and_cross_the_channel(self):
        p = plan_box(6.0, 4.0, 0.5)
        lo, hi = np.array(p.part_lo_in), np.array(p.part_hi_in)
        for t in p.tabs:
            self.assertTrue(t.on_straight)
            on_x = min(abs(t.x - lo[0]), abs(t.x - hi[0])) < 0.02
            on_y = min(abs(t.y - lo[1]), abs(t.y - hi[1])) < 0.02
            self.assertTrue(on_x or on_y, (t.x, t.y))
            # Away from the corners.
            along = t.y if on_x else t.x
            ends = (lo[1], hi[1]) if on_x else (lo[0], hi[0])
            self.assertGreaterEqual(min(abs(along - e) for e in ends),
                                    fx.TAB_CORNER_MARGIN_IN + t.width / 2 - 0.02)
            self.assertAlmostEqual(t.length, p.channel_width_in + 2 * fx.TAB_OVERLAP_IN)

    def test_vertical_tabs_are_centred_flat_tabs_on_the_bottom(self):
        v = plan_box(6.0, 4.0, 0.5)
        zm = (v.part_lo_in[2] + v.part_hi_in[2]) / 2
        for t in v.tabs:
            self.assertAlmostEqual((t.z0 + t.z1) / 2, zm)
            self.assertAlmostEqual(t.z1 - t.z0, 0.060)
        f = plan_box(6.0, 4.0, 0.15)
        for t in f.tabs:
            self.assertAlmostEqual(t.z0, f.part_lo_in[2])
            self.assertAlmostEqual(t.z1, f.part_lo_in[2] + 0.060)

    def test_tabs_only_where_the_wall_is_at_their_height(self):
        # An L standing in the vise: at mid-thickness only the upright leg's
        # wall is there, so every tab goes on it.
        pts, tris = l_mesh(4.0, 3.0, 2.0, 0.25)
        ext = np.array([4.0, 3.0, 2.0]) * MM
        p = fx.plan(pts, tris, np.eye(3), ext, st.rectilinear_stock(ext), 4 * (0.75 + 0.4375),
                    grow_stock=True)
        self.assertEqual(p.procedure, fx.VERTICAL)
        self.assertGreater(p.tab_count, 0)
        section = fx.section_points(
            np.asarray(pts) @ np.asarray(p.frame).T / MM - np.asarray(p.origin_mm) @
            np.asarray(p.frame).T / MM, tris, (p.tabs[0].z0 + p.tabs[0].z1) / 2, 0.01)
        for t in p.tabs:
            d = np.min(np.hypot(section[:, 0] - t.x, section[:, 1] - t.y))
            self.assertLess(d, 0.05, (t.x, t.y))

    def test_flip_swaps_the_sides(self):
        a, b = plan_box(6.0, 4.0, 0.5), plan_box(6.0, 4.0, 0.5, flipped=True)
        self.assertAlmostEqual(float(np.dot(a.frame[2], b.frame[2])), -1.0)


@unittest.skipUnless(HAVE, "needs scipy")
class TestPlanRoundTrip(unittest.TestCase):
    def test_to_dict_and_back(self):
        p = plan_box(6.0, 4.0, 0.5, grow_stock=True)
        d = json.loads(json.dumps(p.to_dict()))
        q = fx.FixturePlan.from_dict(d)
        self.assertEqual(q.tab_count, p.tab_count)
        self.assertEqual(q.stock_in, p.stock_in)
        np.testing.assert_allclose(q.tabs[0].corners(), p.tabs[0].corners())

    def test_matches_stock(self):
        p = plan_box(6.0, 4.0, 0.5, grow_stock=True)
        self.assertTrue(fx.matches_stock(p, fx.grown_stock(p)))
        self.assertTrue(fx.matches_stock(p.to_dict(), fx.grown_stock(p)))
        self.assertFalse(fx.matches_stock(p, st.custom_rectilinear(1.0, 6.0, 8.0)))

    def test_applies(self):
        plate = st.custom_rectilinear(1.0, 6.0, 8.0)
        bar = st.custom_cylindrical(1.0, 4.0)
        self.assertTrue(fx.applies(plate, "mill"))
        self.assertFalse(fx.applies(plate, "lathe"))
        self.assertFalse(fx.applies(bar, "mill"))
        self.assertFalse(fx.applies(plate, "mill", fx.FixtureSettings(enabled=False)))

    def test_overlay_is_in_the_files_coordinates(self):
        pts, tris = box_mesh(6.0, 4.0, 0.5, origin=(10.0, -3.0, 2.0))
        ext = np.array([6.0, 4.0, 0.5]) * MM
        p = fx.plan(pts, tris, np.eye(3), ext, st.rectilinear_stock(ext), 12.0)
        loops, tabs = fx.overlay(p)
        allpts = np.vstack(loops) / MM
        self.assertTrue(np.all(allpts.min(axis=0) >= np.array([10 - 0.3, -3 - 0.3, 2 - 1e-6])))
        self.assertTrue(np.all(allpts.max(axis=0) <= np.array([16 + 0.3, 1 + 0.3, 2.5 + 1e-6])))
        self.assertEqual(len(tabs), p.tab_count)


@unittest.skipUnless(HAVE, "needs scipy")
class TestStandardInSettings(unittest.TestCase):
    def setUp(self):
        self.addCleanup(fx.set_standard, None)
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def test_round_trip_and_apply(self):
        from partestimator.settings import Settings

        s = Settings(tab_width_in=0.375, min_tabs=4, channel_width_3_in=0.75,
                     fixture_enabled=False, max_stickout_ratio=6.0)
        s.save(self.dir / "s.json")
        back = Settings.load(self.dir / "s.json")
        self.assertEqual(back, s)
        back.apply()
        std = fx.standard()
        self.assertEqual((std.tab_width_in, std.min_tabs, std.enabled, std.max_stickout_ratio),
                         (0.375, 4, False, 6.0))
        self.assertEqual(std.channel_width(5.0), 0.75)

    def test_bad_values_fall_back(self):
        from partestimator.settings import Settings

        (self.dir / "s.json").write_text(json.dumps({
            "tab_height_in": -1, "min_tabs": 0, "fixture_enabled": "yes",
            "channel_depth_1_in": 2.0, "channel_depth_2_in": 1.0}), encoding="utf-8")
        s = Settings.load(self.dir / "s.json")
        d = Settings()
        self.assertEqual((s.tab_height_in, s.min_tabs, s.fixture_enabled),
                         (d.tab_height_in, d.min_tabs, d.fixture_enabled))
        self.assertEqual((s.channel_depth_1_in, s.channel_depth_2_in),
                         (d.channel_depth_1_in, d.channel_depth_2_in))

    def test_settings_dict_round_trip(self):
        s = fx.FixtureSettings(min_frame_in=0.5)
        self.assertEqual(fx.FixtureSettings.from_dict(json.loads(json.dumps(s.to_dict()))), s)


@unittest.skipUnless(HAVE, "needs scipy")
class TestEstimate(unittest.TestCase):
    """What estimate_cycle_time charges for a tabbed part."""

    @classmethod
    def setUpClass(cls):
        from partestimator.machining.geometry import cycle_time, rates, shop

        cls.ct = cycle_time
        cls.table = rates.load_rates()
        cls.prof = shop.load_profile()

    def estimate(self, plan, stock, ops=None):
        feats = {"volume_in3": 12.0, "solidity": 1.0, "stock_volume_in3": stock.volume_in3,
                 "area_analytic_in2": 60.0, "material_density_lb_in3": 0.098}
        results = {"stock": stock, "holes": [], "fixture": plan}
        return self.ct.estimate_cycle_time(feats, results, self.table, "Aluminum 6061-T6",
                                           63.0, "mill", profile=self.prof, operations=ops)

    def test_vertical(self):
        p = plan_box(6.0, 4.0, 0.5, grow_stock=True)
        stock = fx.grown_stock(p)
        est = self.estimate(p, stock, ops=3)
        names = {t.name: t for t in est.terms}
        self.assertIn("Tab channel", names)
        self.assertIn("Turn 180° (tabbed)", names)
        self.assertAlmostEqual(names["Tab break-off"].minutes,
                               p.tab_count * self.prof.get("tab_breakoff_s") / 60)
        self.assertEqual(est.operations, 3)                   # a count given by hand stands
        self.assertEqual(self.estimate(p, stock).operations, 1)   # else the procedure's
        # A solid block has no pockets: outside + channel is all that comes out,
        # the frame stays on.
        self.assertNotIn("Pocket roughing", names)
        removed = names["Outside roughing"].quantity + names["Tab channel"].quantity
        self.assertAlmostEqual(removed, p.removed_volume_in3, places=6)
        self.assertAlmostEqual(names["Tab channel"].quantity, p.channel_volume_in3, places=6)
        # Break-off is hand work: listed after the machine cycle, and in
        # door-to-door but not in the cut time.
        rows = [r[0] for r in est.as_rows()]
        self.assertGreater(rows.index("Tab break-off"), rows.index("Machine cycle"))
        self.assertAlmostEqual(est.cut_minutes, sum(
            t.minutes for t in est.terms if t.name not in ("Deburring", "Tab break-off")))
        self.assertAlmostEqual(est.door_to_door_minutes,
                               est.machine_cycle_minutes + est.load_unload_minutes
                               + est.deburr_minutes, places=6)
        self.assertGreaterEqual(est.deburr_minutes, names["Tab break-off"].minutes)

    def test_flat_is_two_setups_and_no_turn(self):
        p = plan_box(6.0, 4.0, 0.25, grow_stock=True,
                     settings=fx.FixtureSettings(vertical_min_thickness_in=0.6))
        est = self.estimate(p, fx.grown_stock(p))
        self.assertEqual(est.operations, 2)
        self.assertNotIn("Turn 180° (tabbed)", [t.name for t in est.terms])

    def test_without_a_plan_nothing_changes(self):
        stock = st.custom_rectilinear(1.0, 5.0, 7.0)
        est = self.estimate(None, stock, ops=2)
        self.assertEqual(est.operations, 2)
        self.assertFalse({"Tab channel", "Tab break-off"} & {t.name for t in est.terms})

    def test_lathe_ignores_it(self):
        p = plan_box(6.0, 4.0, 0.5)
        self.assertIsNone(self.ct.fixture_plan({"fixture": None}))
        self.assertIsNotNone(self.ct.fixture_plan({"fixture": p.to_dict()}))


if __name__ == "__main__":
    unittest.main()
