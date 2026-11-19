"""machining/tuning.py: the shop tuning library, its files and the fit.

Pure Python and numpy, so it runs in the light CI environment: the parts are
hand-built captured records, not analysed STEP files (tests/test_tuning_gui.py
does the real thing).
"""
from __future__ import annotations

import contextlib
import io
import math
import unittest.mock
import shutil
import tempfile
import unittest
from pathlib import Path

from partestimator.machining import tuning
from partestimator.machining.geometry import shop

from .support import MAT, make_table, prismatic_features

FIT_KEYS = ["calibrate_roughing", "tool_change_s", "load_unload_base_min"]


class TempDir(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)


def parts_for(table, profile, n=20, outlier=None):
    """`n` varied milled parts whose measured time is exactly what `profile`
    estimates (times `outlier`'s factor for the part it names)."""
    out = []
    for i in range(n):
        vol = 2.0 + 1.5 * i
        feats = prismatic_features(
            volume_in3=vol, stock_volume_in3=vol * (2.0 + (i % 4)),
            solidity=0.5 + 0.04 * (i % 5), area_analytic_in2=10.0 + 7.0 * (i % 6),
            convex_edge_length_in=20.0 + 3 * i, hole_distinct_diameters=(i * 7) % 5)
        e = tuning.Entry(part=f"p{i}", id=f"id{i}", material=MAT,
                         operations=1 + i % 3)
        p = tuning.Prepared(entry=e, features=feats, results={"holes": []}, material=MAT,
                            finish_uin=125.0, removal="mill", operations=e.operations,
                            tapped=None, measured=None)
        p.measured = tuning.estimate(p, table, profile).door_to_door_minutes
        if outlier and outlier[0] == i:
            p.measured *= outlier[1]
        e.measured_min = p.measured
        out.append(p)
    return out


class TestFit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = make_table()
        cls.base = shop.load_profile()
        cls.truth = tuning._profile_with(cls.base, {"calibrate_roughing": 1.8,
                                                    "tool_change_s": 25.0})

    def knob(self, fit, key):
        return next(k for k in fit.knobs if k.key == key)

    def test_recovers_a_shops_settings(self):
        parts = parts_for(self.table, self.truth)
        fit = tuning.fit(parts, self.table, self.base, FIT_KEYS, cross_check=False)
        self.assertAlmostEqual(self.knob(fit, "calibrate_roughing").fitted, 1.8, delta=0.15)
        self.assertAlmostEqual(self.knob(fit, "tool_change_s").fitted, 25.0, delta=4.0)
        self.assertGreater(fit.before.median_error_pct, 10.0)
        self.assertLess(fit.after.median_error_pct, 3.0)
        self.assertTrue(self.knob(fit, "calibrate_roughing").pinned)

    def test_a_setting_no_part_uses_stays_put(self):
        parts = parts_for(self.table, self.truth, n=10)
        fit = tuning.fit(parts, self.table, self.base,
                         FIT_KEYS + ["lathe_load_unload_min"], cross_check=False)
        lathe = self.knob(fit, "lathe_load_unload_min")
        self.assertAlmostEqual(lathe.fitted, lathe.baseline, places=6)
        self.assertFalse(lathe.pinned)
        self.assertTrue(any("Loosely set" in w and "lathe" in w for w in fit.warnings))

    def test_one_bad_record_does_not_drag_the_fit(self):
        parts = parts_for(self.table, self.truth, outlier=(7, 4.0))
        fit = tuning.fit(parts, self.table, self.base, FIT_KEYS, cross_check=False)
        self.assertAlmostEqual(self.knob(fit, "calibrate_roughing").fitted, 1.8, delta=0.3)
        self.assertTrue(any("p7" in w for w in fit.warnings), fit.warnings)

    def test_cross_check_estimates_every_part_unseen(self):
        parts = parts_for(self.table, self.truth, n=8)
        fit = tuning.fit(parts, self.table, self.base, FIT_KEYS)
        self.assertIsNotNone(fit.cross_checked)
        self.assertTrue(all(p.held_out for p in fit.parts))
        self.assertLess(fit.cross_checked.median_error_pct, 10.0)

    def test_few_parts_are_called_provisional(self):
        parts = parts_for(self.table, self.truth, n=4)
        fit = tuning.fit(parts, self.table, self.base, FIT_KEYS)
        self.assertIsNone(fit.cross_checked)
        self.assertTrue(any("provisional" in w for w in fit.warnings))

    def test_a_setting_switched_off_is_not_fitted(self):
        base = tuning._profile_with(self.base, {"index_time_s": 0.0})
        self.assertEqual(tuning.usable_knobs(base, ["index_time_s", "tool_change_s"]),
                         ["tool_change_s"])

    def test_nothing_measured_is_an_error(self):
        parts = parts_for(self.table, self.truth, n=3)
        for p in parts:
            p.measured = None
        with self.assertRaises(ValueError):
            tuning.fit(parts, self.table, self.base, FIT_KEYS)

    def test_report_lists_parts_and_settings(self):
        fit = tuning.fit(parts_for(self.table, self.truth, n=6), self.table, self.base,
                         FIT_KEYS, cross_check=False)
        text = tuning.format_report(fit)
        self.assertIn("Roughing time", text)
        self.assertIn("p5", text)


class TestStats(unittest.TestCase):
    def test_errors_are_ratios(self):
        s = tuning.Stats.of([1.0, 1.1, 0.8, 2.0])
        self.assertEqual(s.n, 4)
        self.assertAlmostEqual(s.median_error_pct, 15.0)
        self.assertAlmostEqual(s.within_20_pct, 75.0)
        self.assertGreater(s.bias_pct, 0)

    def test_empty(self):
        self.assertEqual(tuning.Stats.of([]).text(), "—")


class TestOverrides(TempDir):
    def write(self, text):
        (self.dir / tuning.TUNING_FILE).write_text(text, encoding="utf-8")

    def test_no_file_is_the_shipped_profile(self):
        prof, state = tuning.load_tuned_profile(self.dir)
        self.assertEqual(prof.values, shop.load_profile().values)
        self.assertFalse(state.active)
        self.assertEqual(state.describe(), "")

    def test_fit_and_hand_set_rows_are_laid_over(self):
        self.write("key,value,source,notes\n"
                   "calibrate_roughing,1.4,fit,\n"
                   "spindle_hp,30,manual,our VF-4\n")
        prof, state = tuning.load_tuned_profile(self.dir)
        self.assertEqual(prof.get("calibrate_roughing"), 1.4)
        self.assertEqual(prof.get("spindle_hp"), 30.0)
        self.assertEqual(state.fitted, {"calibrate_roughing": 1.4})
        self.assertEqual(state.manual, {"spindle_hp": 30.0})
        self.assertIn("set by hand", state.describe())
        base = tuning.fit_baseline(self.dir)
        self.assertEqual(base.get("calibrate_roughing"), 1.0)
        self.assertEqual(base.get("spindle_hp"), 30.0)

    def test_a_bad_value_falls_back_to_shipped_and_says_so(self):
        self.write("key,value,source,notes\nrough_open_duty,2.0,manual,\n")
        prof, state = tuning.load_tuned_profile(self.dir)
        self.assertEqual(prof.get("rough_open_duty"), shop.load_profile().get("rough_open_duty"))
        self.assertTrue(state.errors)

    def test_save_replaces_the_fit_and_keeps_hand_set_rows(self):
        self.write("key,value,source,notes\n"
                   "spindle_hp,30,manual,\n"
                   "calibrate_holes,1.3,fit,\n")
        table, base = make_table(), tuning.fit_baseline(self.dir)
        truth = tuning._profile_with(base, {"calibrate_roughing": 1.5})
        fit = tuning.fit(parts_for(table, truth, n=8), table, base, ["calibrate_roughing"],
                         cross_check=False)
        tuning.save_fit(fit, self.dir)
        overrides, _ = tuning.read_overrides(self.dir)
        self.assertEqual(set(overrides), {"spindle_hp", "calibrate_roughing"})
        self.assertEqual(overrides["calibrate_roughing"].source, "fit")
        _prof, state = tuning.load_tuned_profile(self.dir)
        self.assertEqual(state.summary["parts"], 8)
        self.assertIn("tuned to 8 parts", state.describe())

        tuning.clear_fit(self.dir)
        overrides, _ = tuning.read_overrides(self.dir)
        self.assertEqual(set(overrides), {"spindle_hp"})
        self.assertFalse((self.dir / tuning.SUMMARY_FILE).exists())

    def test_clearing_the_only_rows_removes_the_file(self):
        self.write("key,value,source,notes\ncalibrate_holes,1.3,fit,\n")
        tuning.clear_fit(self.dir)
        self.assertFalse((self.dir / tuning.TUNING_FILE).exists())


def snapshot(**features):
    """A captured part, as tuning.capture would write it."""
    feats = prismatic_features(**features)
    return {"format": tuning.CAPTURE_FORMAT, "features": feats,
            "holes": [{"nominal_diameter_mm": 6.35, "depth_mm": 12.7,
                       "axis_direction": [0, 0, 1], "axis_point": [0, 0, 0]}],
            "turning": None}


class TestLibrary(TempDir):
    def entry(self, part="bracket", pid="abc123", **kw):
        kw.setdefault("stock_size", "1 x 4 x 6")
        return tuning.Entry(part=part, id=pid, material=MAT, **kw)

    def test_add_round_trips_without_any_step_file(self):
        lib = tuning.Library(self.dir / "lib")
        e = lib.add(self.entry(operations=2, measured_min=12.5, tapped_holes=3,
                               source_file="C:/jobs/bracket.step"), snapshot())
        lib.save()
        self.assertTrue(lib.snapshot_path(e).is_file())
        self.assertEqual({p.name for p in lib.folder.iterdir()},
                         {"README.txt", tuning.LIBRARY_FILE, tuning.CAPTURED_DIR})
        again = tuning.Library.load(lib.folder)
        self.assertEqual(again.entries, lib.entries)
        self.assertEqual(again.errors, [])
        self.assertEqual(again.load_snapshot(again.entries[0])["holes"][0]["depth_mm"], 12.7)

    def test_adding_the_same_part_again_replaces_its_geometry_only(self):
        lib = tuning.Library(self.dir / "lib")
        first = lib.add(self.entry(measured_min=5.0), snapshot(volume_in3=1.0))
        same = lib.add(self.entry(measured_min=9.0), snapshot(volume_in3=2.0))
        self.assertIs(same, first)
        self.assertEqual(len(lib.entries), 1)
        self.assertEqual(lib.load_snapshot(first)["features"]["volume_in3"], 2.0)

    def test_remove_deletes_the_captured_geometry(self):
        lib = tuning.Library(self.dir / "lib")
        e = lib.add(self.entry(), snapshot())
        lib.remove(e)
        self.assertFalse(lib.snapshot_path(e).exists())

    def test_bad_rows_are_reported_and_the_rest_load(self):
        (self.dir / tuning.LIBRARY_FILE).write_text(
            "part,id,process,stock_type,measured_min\n"
            "good,a1,mill,plate,10\n"
            "bad,b1,drill,plate,10\n"
            "noid,,mill,plate,5\n"
            "auto,c1,mill,auto,5\n", encoding="utf-8")
        lib = tuning.Library.load(self.dir)
        self.assertEqual([e.part for e in lib.entries], ["good"])
        self.assertEqual(len(lib.errors), 3)

    def test_excel_style_csv_reads(self):
        (self.dir / tuning.LIBRARY_FILE).write_bytes(
            "\ufeffinclude,part,id,measured_min,tapped_holes\r\n"
            "0,a,a1,12.5,\r\n".encode("utf-8"))
        e = tuning.Library.load(self.dir).entries[0]
        self.assertFalse(e.include)
        self.assertEqual(e.measured_min, 12.5)
        self.assertIsNone(e.tapped_holes)

    def test_problems(self):
        table = make_table()
        e = tuning.Entry(part="a", id="a1", stock_size="1 x 2 x 3")
        self.assertEqual(tuning.problems(e, table), ["no material", "no measured time"])
        e.material, e.measured_min = MAT, 3.0
        self.assertEqual(tuning.problems(e, table), [])
        e.material = "Unobtainium"
        self.assertEqual(tuning.problems(e, table), ["no rates for Unobtainium"])
        e.material, e.stock_size = MAT, "1 x 2"
        self.assertEqual(len(tuning.problems(e, table)), 1)


class TestPrepare(unittest.TestCase):
    """A captured part with its inputs applied estimates exactly as the same
    feature vector does when handed to the model directly."""

    def setUp(self):
        self.table, self.prof = make_table(), shop.load_profile()

    def test_matches_the_model_fed_directly(self):
        from partestimator.machining.geometry import cycle_time
        from partestimator.machining.geometry import stock as st

        snap = snapshot(volume_in3=10.0)
        e = tuning.Entry(part="a", id="a1", material=MAT, operations=2,
                         stock_type="plate", stock_size="2 x 4 x 6")
        prep = tuning.prepare(e, snap, self.prof)
        stock = st.custom_rectilinear(2, 4, 6)
        feats = dict(snap["features"], **st.stock_features(stock, 10.0))
        feats.pop("material_density_lb_in3")          # Testium has no density
        direct = cycle_time.estimate_cycle_time(
            feats, {"holes": tuning._holes(snap), "stock": stock}, self.table, MAT, 63.0,
            "mill", profile=self.prof, operations=2)
        self.assertAlmostEqual(tuning.estimate(prep, self.table, self.prof).door_to_door_minutes,
                               direct.door_to_door_minutes, places=9)
        self.assertEqual(prep.features["stock_volume_in3"], 48.0)

    def test_inputs_are_reapplied_to_the_captured_geometry(self):
        snap = snapshot(volume_in3=10.0)

        def minutes(**kw):
            e = tuning.Entry(part="a", id="a1", material=MAT, stock_type="plate",
                             stock_size="2 x 4 x 6")
            for k, v in kw.items():
                setattr(e, k, v)
            p = tuning.prepare(e, snap, self.prof)
            return tuning.estimate(p, self.table, self.prof).door_to_door_minutes

        base = minutes()
        self.assertGreater(minutes(stock_size="3 x 5 x 8"), base)
        self.assertGreater(minutes(operations=3), base)
        self.assertGreater(minutes(stock_type="round", stock_size="4 x 6"), 0)

    def test_auto_process_follows_the_captured_turned_share(self):
        snap = snapshot(volume_in3=10.0, turned_area_fraction=0.9)
        e = tuning.Entry(part="a", id="a1", material=MAT, process="auto",
                         stock_type="round", stock_size="2 x 6")
        self.assertEqual(tuning.prepare(e, snap, self.prof).removal, "lathe")
        e.stock_type, e.stock_size = "plate", "2 x 2 x 6"
        self.assertEqual(tuning.prepare(e, snap, self.prof).removal, "mill")

    def test_json_safe(self):
        import json

        import numpy as np

        data = tuning._plain({"a": np.float64(1.5), "b": np.arange(3), "c": np.bool_(True),
                              "d": float("nan"), "e": (np.int32(2), "x")})
        self.assertEqual(json.loads(json.dumps(data)),
                         {"a": 1.5, "b": [0, 1, 2], "c": True, "d": None, "e": [2, "x"]})


class TestText(unittest.TestCase):
    def test_lengths(self):
        for text, want in (("1.25", 1.25), ("1-1/4", 1.25), ("1 1/4", 1.25), ("5/8", 0.625),
                           ('2"', 2.0), ("2 in", 2.0), ("3", 3.0)):
            self.assertAlmostEqual(tuning.parse_length(text), want, msg=text)
        for bad in ("", "abc", "1/"):
            with self.assertRaises(ValueError):
                tuning.parse_length(bad)

    def test_sizes(self):
        self.assertEqual(tuning.parse_sizes("1.25 x 4 x 6"), [1.25, 4.0, 6.0])
        self.assertEqual(tuning.parse_sizes("1-1/2X3*10"), [1.5, 3.0, 10.0])
        self.assertEqual(tuning.parse_sizes("2 × 6"), [2.0, 6.0])

    def test_stock_text_round_trips(self):
        for fields in (("plate", "1 x 4 x 6", ""), ("round", "2 x 6", ""),
                       ("extrusion", "6.5", "AND10133-2004"),
                       ("extrusion", "12", "Angle 2 x 2 x 1/4")):
            e = tuning.Entry(part="a", id="a1", stock_type=fields[0], stock_size=fields[1],
                             stock_profile=fields[2])
            self.assertEqual(tuning.parse_stock_text(tuning.stock_text(e)), fields)

    def test_bad_stock_text_is_refused(self):
        for bad in ("block 1 x 2 x 3", "plate 1 x 2", "round 2", "extrusion AND10133-2004",
                    "extrusion NOPE @ 6", "plate 0 x 2 x 3"):
            with self.assertRaises(ValueError, msg=bad):
                tuning.parse_stock_text(bad)

    def test_extrusion_stock_volume_is_section_times_length(self):
        e = tuning.Entry(part="a", id="a1", stock_type="extrusion", stock_size="10",
                         stock_profile="AND10133-2004")
        s = tuning.build_stock(e)
        self.assertTrue(s.is_extrusion)
        self.assertAlmostEqual(s.volume_in3, s.profile.area_in2 * 10.0)

    def test_tuning_folder(self):
        with unittest.mock.patch.dict("os.environ", {tuning.ENV_DIR: "C:/shop/tuning"}):
            self.assertEqual(tuning.tuning_dir(), Path("C:/shop/tuning"))
        with unittest.mock.patch.dict("os.environ", {tuning.ENV_DIR: ""}):
            # In the tool's own folder, beside the launcher.
            self.assertEqual(tuning.tuning_dir(), tuning.tool_dir() / "tuning")
            self.assertTrue((tuning.tool_dir() / "pe.ps1").is_file())


class TestCli(TempDir):
    def run_cli(self, *argv):
        from partestimator import __main__ as cli

        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = cli.main(["tune", "--dir", str(self.dir / "lib"), *argv])
        return rc, out.getvalue(), err.getvalue()

    def test_an_empty_library_says_how_to_start(self):
        rc, out, _ = self.run_cli()
        self.assertEqual(rc, 1)
        self.assertIn("--add", out)
        self.assertTrue((self.dir / "lib" / tuning.LIBRARY_FILE).is_file())

    def test_fits_a_library_and_saves(self):
        lib = tuning.Library(self.dir / "lib")
        for i in range(6):
            lib.add(tuning.Entry(part=f"p{i}", id=f"id{i}", material="Aluminum 6061-T6",
                                 stock_size=f"{1 + i * 0.25} x 4 x 6", measured_min=8.0 + i),
                    snapshot(volume_in3=3.0 + i))
        lib.save()
        rc, out, _ = self.run_cli("--save")
        self.assertEqual(rc, 0, out)
        self.assertIn("Fitted to 6 part(s)", out)
        _prof, state = tuning.load_tuned_profile(self.dir / "lib")
        self.assertIn("tuned to 6 parts", state.describe())



if __name__ == "__main__":
    unittest.main()
