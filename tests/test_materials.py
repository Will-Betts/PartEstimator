"""The shared material library: one list, both tools, old names still work."""
from __future__ import annotations

import unittest

from partestimator import materials_db as db


class TestLibrary(unittest.TestCase):
    def test_size_and_families(self):
        self.assertGreaterEqual(len(db.load()), 50)
        for fam in ("Aluminum", "Steel", "Stainless", "Superalloy", "Titanium",
                    "Copper", "Plastic"):
            self.assertIn(fam, db.families())

    def test_every_sheet_material_is_complete(self):
        for m in db.load():
            if m.sheet_metal:
                with self.subTest(m=m.name):
                    self.assertIsNotNone(m.min_bend_t_across)
                    self.assertIsNotNone(m.min_bend_t_with)
                    self.assertGreaterEqual(m.min_bend_t_with, m.min_bend_t_across)
                    self.assertTrue(m.air_radius_coeff and 0.1 < m.air_radius_coeff < 0.3)
                    self.assertIn(m.k_curve, ("steel", "alu", "ss"))
                    self.assertGreater(m.uts_mpa, 50)

    def test_densities_are_plausible(self):
        for m in db.load():
            with self.subTest(m=m.name):
                self.assertTrue(0.8 < m.density_g_cc < 9.5)

    def test_legacy_names_resolve(self):
        self.assertEqual(db.canonical("CRS"), "Steel 1008/1010 CRS")
        self.assertEqual(db.canonical("AL5052"), "Aluminum 5052-H32")
        self.assertEqual(db.canonical("Aluminum 6061"), "Aluminum 6061-T6")
        self.assertEqual(db.canonical("nylon"), "Nylon 6/6")
        self.assertIsNone(db.canonical("Unobtainium"))


class TestSheetMetalSide(unittest.TestCase):
    def test_smtv13_values_unchanged(self):
        from partestimator.smt.materials import get_material

        crs = get_material("CRS")
        self.assertAlmostEqual(crs.uts, 400.0)
        self.assertEqual(crs.min_radius_t, (1.0, 1.5))
        self.assertAlmostEqual(crs.air_radius_coeff, 0.156)
        al = get_material("AL6061")
        self.assertEqual(al.min_radius_t, (3.0, 4.0))

    def test_new_sheet_materials_available(self):
        from partestimator.smt.materials import MATERIALS

        for name in ("Aluminum 2024-T3", "Aluminum 7075-T6", "Titanium Ti-6Al-4V",
                     "Stainless 17-7PH (Cond A)", "Inconel 625 (annealed)"):
            self.assertIn(name, MATERIALS)
        self.assertGreaterEqual(len(MATERIALS), 25)

    def test_deduction_rows_match_across_old_and_new_names(self):
        from partestimator.smt.materials import DeductionTable

        t = DeductionTable(rows=[{"material": "CRS", "thickness": 2.0, "radius": 2.5,
                                  "v_die": 16.0, "angle_deg": 90.0, "deduction": 3.45}])
        hit = t.lookup("Steel 1008/1010 CRS", 2.0, 2.5, 16.0)
        self.assertIsNotNone(hit)


class TestMachiningSide(unittest.TestCase):
    def test_dropdown_is_the_library(self):
        from partestimator.machining.geometry.materials import MATERIALS

        self.assertEqual(MATERIALS[:-1], db.machining_names())
        self.assertEqual(MATERIALS[-1], "Other")

    def test_density_follows_library_and_old_names(self):
        from partestimator.machining.geometry.materials import density_lb_per_in3

        self.assertAlmostEqual(density_lb_per_in3("Aluminum 6061"),
                               density_lb_per_in3("Aluminum 6061-T6"))
        self.assertAlmostEqual(density_lb_per_in3("Titanium Ti-6Al-4V"), 4.43 * 0.0361273, 6)
