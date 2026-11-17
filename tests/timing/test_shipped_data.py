"""The shipped machining_rates.csv and shop_profile.csv: they load, they are
complete, the loaders reject bad input, and the estimates they produce
behave sensibly. No exact minutes are pinned here -- the data is meant to be
tuned, and these tests must survive that."""
from __future__ import annotations

import dataclasses
import shutil
import tempfile
import textwrap
import unittest
from pathlib import Path

from partestimator import materials_db
from partestimator.machining.geometry import cycle_time as ct
from partestimator.machining.geometry import rates, shop

from .support import FakeHole, prismatic_features, shaft

MILL_OPS = {"unit_power", "rough_open_sfm", "rough_open_ipt", "drill_sfm",
            "drill_ipr_per_inch_dia"}
LATHE_OPS = {"turn_rough_sfm", "turn_rough_ipr", "turn_rough_doc_in",
             "turn_finish_sfm", "turn_finish_ipr_max"}


class ShippedFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = rates.load_rates()
        cls.profile = shop.load_profile()

    def test_every_machining_material_has_full_rate_set(self):
        for name in materials_db.machining_names():
            with self.subTest(material=name):
                mill = {r.operation for r in self.table.rows
                        if r.material == name and r.removal_type == "mill"}
                lathe = {r.operation for r in self.table.rows
                         if r.material == name and r.removal_type == "lathe"}
                self.assertEqual(mill, MILL_OPS)
                self.assertEqual(lathe, LATHE_OPS)

    def test_unit_power_is_never_derated(self):
        for r in self.table.rows:
            if r.operation == "unit_power":
                self.assertEqual(r.derate, 1.0, r.material)

    def test_legacy_names_resolve(self):
        for old in materials_db.LEGACY_NAMES:
            self.assertIsNotNone(self.table.get(old, "mill", "unit_power"), old)

    def test_profile_has_keys_the_model_relies_on(self):
        for key in ("spindle_hp", "spindle_max_rpm", "rough_open_duty", "rough_pocket_duty",
                    "finish_duty", "tool_change_s", "rapid_fraction", "lathe_max_rpm",
                    "calibrate_overall"):
            self.assertIn(key, self.profile.values)

    def test_every_material_estimates_mill_and_lathe(self):
        feats = prismatic_features(area_freeform_in2=10.0, hole_count=4,
                                   hole_distinct_diameters=1)
        holes = [FakeHole.inches(0.25, 0.5)] * 4
        tg = shaft(1.5, 3.0, step_diameter_in=1.0)
        lathe_feats = {"volume_in3": tg.envelope_volume_mm3 / 25.4 ** 3}
        for name in materials_db.machining_names():
            with self.subTest(material=name):
                m = ct.estimate_cycle_time(feats, {"holes": holes}, self.table, name, 63.0,
                                           "mill", profile=self.profile)
                t = ct.estimate_cycle_time(lathe_feats, {"turning": tg, "holes": []},
                                           self.table, name, 63.0, "lathe",
                                           profile=self.profile)
                for est in (m, t):
                    self.assertGreater(est.door_to_door_minutes, 0.0)
                    self.assertFalse(any("No rate data" in w or "No turning data" in w
                                         for w in est.warnings))

    def test_harder_materials_take_longer(self):
        feats = prismatic_features()
        order = ["Aluminum 6061-T6", "Steel 1018", "Stainless 304", "Titanium Ti-6Al-4V",
                 "Inconel 718 (aged)"]
        times = [ct.estimate_cycle_time(feats, {"holes": []}, self.table, m, 125.0, "mill",
                                        profile=self.profile).door_to_door_minutes
                 for m in order]
        self.assertEqual(times, sorted(times), dict(zip(order, times)))

    def test_turning_beats_milling_on_a_round_part(self):
        tg = shaft(2.0, 3.5, step_diameter_in=1.25)
        part_in3 = tg.envelope_volume_mm3 / 25.4 ** 3
        stock_in3 = 3.14159 * (2.25 / 2) ** 2 * 4.5
        feats = {"volume_in3": part_in3, "stock_volume_in3": stock_in3,
                 "solidity": 1.0, "area_analytic_in2": 30.0}
        res = {"turning": tg, "holes": []}
        lathe = ct.estimate_cycle_time(feats, res, self.table, "Aluminum 6061-T6", 63.0,
                                       "lathe", profile=self.profile)
        mill = ct.estimate_cycle_time(feats, res, self.table, "Aluminum 6061-T6", 63.0,
                                      "mill", profile=self.profile)
        self.assertLess(lathe.door_to_door_minutes, mill.door_to_door_minutes)


class LoaderValidation(unittest.TestCase):
    HEADER = ("material,iso_group,removal_type,operation,published_value,units,derate,"
              "effective_value,notes,source,source_url,retrieved\n")
    GOOD = "X,N,mill,unit_power,0.3,hp,1.0,0.3,,src,https://e.invalid,2026-01-01\n"

    def write(self, text: str) -> Path:
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        p = Path(d) / "f.csv"
        p.write_text(text, encoding="utf-8")
        return p

    def test_good_rate_file_loads(self):
        t = rates.load_rates(self.write(self.HEADER + self.GOOD))
        self.assertEqual(t.value("X", "mill", "unit_power"), 0.3)

    def test_rate_file_rejections(self):
        bad = {
            "derate": "X,N,mill,rough_open_sfm,800,sfm,1.5,800,,s,https://e.invalid,d\n",
            "positive": "X,N,mill,rough_open_sfm,800,sfm,0.8,0,,s,https://e.invalid,d\n",
            "url": "X,N,mill,rough_open_sfm,800,sfm,0.8,640,,s,not-a-url,d\n",
            "number": "X,N,mill,rough_open_sfm,fast,sfm,0.8,640,,s,https://e.invalid,d\n",
        }
        for why, row in bad.items():
            with self.subTest(why), self.assertRaises(ValueError):
                rates.load_rates(self.write(self.HEADER + row))
        with self.assertRaises(ValueError):
            rates.load_rates(self.write("material,operation\nX,unit_power\n"))
        with self.assertRaises(ValueError):
            rates.load_rates(self.write(self.HEADER))
        with self.assertRaises(FileNotFoundError):
            rates.load_rates(Path(tempfile.gettempdir()) / "does-not-exist.csv")

    def test_lathe_lookup_falls_back_to_mill(self):
        t = rates.load_rates(self.write(self.HEADER + self.GOOD))
        self.assertEqual(t.value("X", "lathe", "unit_power"), 0.3)
        self.assertIsNone(t.get("X", "mill", "drill_sfm"))

    def test_override_detection(self):
        row = rates.RateRow("X", "N", "mill", "rough_open_sfm", 800, "sfm", 0.8, 640,
                            "", "", "", "")
        self.assertFalse(row.is_overridden)
        self.assertTrue(dataclasses.replace(row, effective_value=700).is_overridden)

    def profile_text(self, **extra):
        rows = {"spindle_hp": "22", "rough_open_duty": "0.7", "rough_pocket_duty": "0.5"}
        rows.update(extra)
        return "key,value,units,notes\n" + "".join(f"{k},{v},,\n" for k, v in rows.items())

    def test_good_profile_loads(self):
        p = shop.load_profile(self.write(self.profile_text(semi_finish_enabled="1")))
        self.assertEqual(p.get("spindle_hp"), 22.0)
        self.assertTrue(p.flag("semi_finish_enabled"))
        self.assertFalse(p.flag("deburr_enabled"))
        self.assertEqual(p.get("missing", 7.0), 7.0)

    def test_profile_rejections(self):
        cases = {
            "fraction": dict(finish_duty="1.5"),
            "calibrate": dict(calibrate_overall="0"),
            "ra factor": dict(finish_ra_factor_16uin="-1"),
            "number": dict(spindle_hp="lots"),
        }
        for why, extra in cases.items():
            with self.subTest(why), self.assertRaises(ValueError):
                shop.load_profile(self.write(self.profile_text(**extra)))
        missing = textwrap.dedent("""\
            key,value,units,notes
            spindle_hp,22,,
        """)
        with self.assertRaises(ValueError):
            shop.load_profile(self.write(missing))

    def test_ra_factor_above_one_is_allowed(self):
        # finish_ra_factor_* are multipliers, not fractions.
        p = shop.load_profile(self.write(self.profile_text(finish_ra_factor_16uin="2.4")))
        self.assertEqual(p.get("finish_ra_factor_16uin"), 2.4)


if __name__ == "__main__":
    unittest.main()
