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


def prism_mesh(poly_in, h_in):
    """A closed prism: the convex polygon `poly_in` (inches, CCW) run up h."""
    poly = np.asarray(poly_in, dtype=float)
    n = len(poly)
    pts = np.vstack([np.column_stack([poly, np.zeros(n)]),
                     np.column_stack([poly, np.full(n, h_in)])]) * MM
    tris = []
    for k in range(1, n - 1):
        tris += [(0, k + 1, k), (n, n + k, n + k + 1)]           # bottom, top fans
    for k in range(n):
        j = (k + 1) % n
        tris += [(k, j, n + j), (k, n + j, n + k)]
    return pts, np.array(tris)


def layers_mesh(layers, l_in=6.0, w_in=4.0):
    """A part built as a stack of centred boxes, [(z0, z1, inset), ...]:
    each layer is the l x w outline less `inset` all round."""
    pts, tris = [], []
    for z0, z1, inset in layers:
        p, t = box_mesh(l_in - 2 * inset, w_in - 2 * inset, z1 - z0,
                        origin=(inset, inset, z0))
        tris.append(t + sum(len(q) for q in pts))
        pts.append(p)
    return np.vstack(pts), np.vstack(tris)


def plan_layers(layers, **kw):
    pts, tris = layers_mesh(layers)
    top = max(z1 for _z0, z1, _i in layers)
    ext = np.array([6.0, 4.0, top]) * MM
    vol = sum((6 - 2 * i) * (4 - 2 * i) * (z1 - z0) for z0, z1, i in layers)
    return fx.plan(pts, tris, np.eye(3), ext, st.rectilinear_stock(ext), vol,
                   planes=flat_planes(6.0, 4.0), **kw)


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

    def test_tabs_sit_on_the_edges_and_cross_the_channel(self):
        p = plan_box(6.0, 4.0, 0.5)
        lo, hi = np.array(p.part_lo_in), np.array(p.part_hi_in)
        for t in p.tabs:
            self.assertTrue(t.on_straight)
            self.assertLess(t.inset, 0.03)               # the wall is at the outline
            on_x = min(abs(t.x - lo[0]), abs(t.x - hi[0])) < 0.02
            on_y = min(abs(t.y - lo[1]), abs(t.y - hi[1])) < 0.02
            self.assertTrue(on_x or on_y, (t.x, t.y))
            # Away from the corners.
            along = t.y if on_x else t.x
            ends = (lo[1], hi[1]) if on_x else (lo[0], hi[0])
            self.assertGreaterEqual(min(abs(along - e) for e in ends),
                                    fx.TAB_CORNER_MARGIN_IN + t.width / 2 - 0.02)
            self.assertAlmostEqual(t.length,
                                   p.channel_width_in + t.inset + 2 * fx.TAB_OVERLAP_IN)

    def test_a_straight_wall_gets_its_tabs_in_the_middle(self):
        for p in (plan_box(6.0, 4.0, 0.5), plan_box(6.0, 4.0, 0.15)):
            zm = (p.part_lo_in[2] + p.part_hi_in[2]) / 2
            for t in p.tabs:
                self.assertAlmostEqual((t.z0 + t.z1) / 2, zm, delta=0.02)
                self.assertAlmostEqual(t.z1 - t.z0, 0.060)

    def test_a_contoured_edge_gets_them_where_the_part_is_widest(self):
        # Widest between 0.30 and 0.40 of a 1" thick part, tapering both ways.
        p = plan_layers([(0.0, 0.1, 0.20), (0.1, 0.2, 0.10), (0.2, 0.3, 0.05),
                         (0.3, 0.4, 0.0), (0.4, 0.6, 0.10), (0.6, 1.0, 0.25)])
        zb = p.part_lo_in[2]
        self.assertGreater(p.tab_count, 0)
        for t in p.tabs:
            self.assertAlmostEqual((t.z0 + t.z1) / 2 - zb, 0.35, delta=0.03)
            self.assertLess(t.inset, 0.03)

    def test_a_groove_round_the_edge_is_skipped(self):
        # A 0.2" deep groove between 0.4 and 0.6: tabs go above or below it.
        p = plan_layers([(0.0, 0.4, 0.0), (0.4, 0.6, 0.2), (0.6, 1.0, 0.0)])
        zb = p.part_lo_in[2]
        for t in p.tabs:
            zc = (t.z0 + t.z1) / 2 - zb
            self.assertTrue(abs(zc - 0.2) < 0.03 or abs(zc - 0.8) < 0.03, zc)
            self.assertLess(t.inset, 0.03)

    def test_a_tab_reaches_in_to_the_nearest_surface(self):
        # A lip thinner than a tab makes the outline; the body is 0.1" in.
        # Every tab reaches in across the lip to join the body.
        p = plan_layers([(0.0, 0.95, 0.1), (0.95, 0.975, 0.0)])
        self.assertGreater(p.tab_count, 0)
        for t in p.tabs:
            self.assertAlmostEqual(t.inset, 0.1, delta=0.03)
            self.assertAlmostEqual(t.length,
                                   p.channel_width_in + t.inset + 2 * fx.TAB_OVERLAP_IN)

    def test_tabs_only_where_the_wall_is_at_their_height(self):
        # An L standing in the vise: away from the corner, only the upright
        # leg's wall is near the outline at mid-height; each tab sits on
        # solid part at its own height.
        pts, tris = l_mesh(4.0, 3.0, 2.0, 0.25)
        ext = np.array([4.0, 3.0, 2.0]) * MM
        p = fx.plan(pts, tris, np.eye(3), ext, st.rectilinear_stock(ext), 4 * (0.75 + 0.4375),
                    grow_stock=True)
        self.assertEqual(p.procedure, fx.VERTICAL)
        self.assertGreater(p.tab_count, 0)
        plan_pts = (np.asarray(pts) - np.asarray(p.origin_mm)) @ np.asarray(p.frame).T / MM
        for t in p.tabs:
            section = fx.section_points(plan_pts, tris, (t.z0 + t.z1) / 2, 0.01)
            root = np.array([t.x, t.y]) - np.array([t.nx, t.ny]) * t.inset
            d = np.min(np.hypot(section[:, 0] - root[0], section[:, 1] - root[1]))
            self.assertLess(d, 0.05, (t.x, t.y, t.z0))

    def test_slanted_edges_take_tabs(self):
        # A hexagonal plate: four of its six edges run at 60 degrees to the
        # grid the outline is traced on. Tabs must not mistake the traced
        # staircase for corners.
        r = 2.0
        hexagon = [(r * math.cos(math.radians(a)), r * math.sin(math.radians(a)))
                   for a in range(0, 360, 60)]
        pts, tris = prism_mesh(hexagon, 0.75)
        ext = np.array([2 * r, math.sqrt(3) * r, 0.75]) * MM
        p = fx.plan(pts, tris, np.eye(3), ext, st.rectilinear_stock(ext), 7.8,
                    planes=flat_planes(4.0, 3.5))
        self.assertEqual(len(p.outline[0]), 6)
        self.assertEqual(p.tab_count, 3)                       # 12" / 4"
        slanted = [t for t in p.tabs if 0.2 < abs(t.nx) < 0.95]
        self.assertTrue(slanted, [(t.nx, t.ny) for t in p.tabs])
        for t in p.tabs:
            self.assertLess(t.inset, 0.03)

    def test_a_round_part_takes_tabs_on_its_curve(self):
        circle = [(1.5 * math.cos(a), 1.5 * math.sin(a))
                  for a in np.linspace(0, 2 * math.pi, 96, endpoint=False)]
        pts, tris = prism_mesh(circle, 0.5)
        ext = np.array([3.0, 3.0, 0.5]) * MM
        p = fx.plan(pts, tris, np.eye(3), ext, st.rectilinear_stock(ext), 3.5,
                    planes=flat_planes(3.0, 3.0))
        self.assertEqual(p.tab_count, 3)                       # max(3, round(9.4 / 4))
        centre = np.mean(np.asarray(p.outline[0]), axis=0)
        for t in p.tabs:
            self.assertFalse(t.on_straight)
            # The normal points out from the centre.
            c = np.array([t.x, t.y]) - centre
            self.assertGreater(float(np.dot(c / np.linalg.norm(c), [t.nx, t.ny])), 0.98)

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
        self.assertFalse(fx.applies(plate, "mill"))              # off by default
        on = fx.FixtureSettings(enabled=True)
        self.assertTrue(fx.applies(plate, "mill", on))
        self.assertFalse(fx.applies(plate, "lathe", on))
        self.assertFalse(fx.applies(bar, "mill", on))

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


@unittest.skipUnless(HAVE, "needs scipy")
class TestLayoutEdges(unittest.TestCase):
    def test_settings_and_plan_helpers(self):
        s = fx.FixtureSettings(channel_table=((0.5, 0.25), (1.0, 0.375)))
        self.assertEqual(s.channel_width(3.0), 0.375)          # past the table: its last
        self.assertEqual(fx.FixtureSettings.from_dict(None), fx.FixtureSettings())
        flat1 = plan_box(6.0, 4.0, 0.15)
        self.assertIn("flat, one operation", flat1.summary())
        self.assertEqual(fx._plain({"a": (np.int64(2), np.float64(np.inf), np.bool_(True))}),
                         {"a": [2, None, True]})
        self.assertIsNone(fx.plan_for_results({}, None))
        self.assertFalse(fx.matches_stock(flat1, st.custom_cylindrical(1, 2)))
        self.assertFalse(fx.matches_stock(None, fx.grown_stock(flat1)))

    def test_outline_tracing_edges(self):
        # Two cells touching only at a corner: traced as two loops, not one.
        grid = fx.Grid(np.zeros(2), np.array([1.0, 1.0]), 0.25)
        mask = np.zeros(grid.shape, dtype=bool)
        mask[1, 1] = mask[2, 2] = True
        loops = fx.loops(grid, mask, tolerance=0.0)
        self.assertEqual(len(loops), 2)
        tri = np.array([[0, 0], [1, 0], [0, 1.0]])
        np.testing.assert_array_equal(fx.simplify(tri, 0.1), tri)
        closed = np.array([[0, 0], [1, 0], [1, 1.0], [0, 0]])
        self.assertEqual(len(fx._dp(closed, 0.1)), 4)          # ends coincide: all corners kept

    def test_degenerate_slicing_and_probing(self):
        pts, tris = box_mesh(2.0, 1.0, 0.5)
        local = pts / MM
        self.assertEqual(fx.section_points(local, tris, 5.0, 0.01).shape, (0, 2))
        sl = fx.Slices(local, tris, 0.0, 0.5)
        sl.segments[0] = (np.zeros((0, 2)), np.zeros((0, 2)))
        entry, solid = sl.probe(np.array([[50.0, 50.0]]), np.array([[1.0, 0.0]]), 1.0)
        self.assertTrue(np.isinf(entry).all())
        spots = fx._outline_spots([np.array([[0, 0], [0, 0], [2, 0], [2, 1], [0, 1.0]])],
                                  0.25, 0.2)
        self.assertTrue(spots)

    def test_nowhere_for_tabs(self):
        grid = fx.Grid(np.zeros(2), np.ones(2), 0.1)
        none, notes = fx.place_tabs([np.zeros((3, 2))], grid, np.zeros(grid.shape, bool),
                                    fx.FixtureSettings(), 0.25, 0.0, 0.5)
        self.assertEqual(none, [])
        self.assertIn("no outline", notes[0])
        tiny = np.array([[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1]])
        none, notes = fx.place_tabs([tiny], grid, np.zeros(grid.shape, bool),
                                    fx.FixtureSettings(), 0.25, 0.0, 0.5)
        self.assertIn("No room", notes[0])

    def test_a_stock_too_small_for_the_part(self):
        p = plan_box(6.0, 4.0, 0.5, stock=st.custom_rectilinear(0.25, 3.0, 5.0))
        self.assertIn("The stock does not enclose the part.", p.warnings)


if __name__ == "__main__":
    unittest.main()
