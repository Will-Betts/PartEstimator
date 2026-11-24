"""The guards in the extrusion catalogue and the stock rules: profiles that
cannot exist, duplicate and unreadable library files, empty point sets, and
stock checks handed nothing. Plain Python.
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from partestimator.machining.geometry import extrusions as ex
from partestimator.machining.geometry import stock as st

MM = 25.4


class TestProfileGuards(unittest.TestCase):
    def test_descriptions(self):
        self.assertIn("OD x", ex.ExtrusionProfile("T", "round_tube", 2.0, t=0.125).size_text)
        self.assertIn("A/F", ex.ExtrusionProfile("H", "hex_bar", 1.0).size_text)
        with self.assertRaisesRegex(ValueError, "unknown extrusion shape"):
            ex.ExtrusionProfile("X", "blob", 1.0, 1.0, 0.1).outline_loops()

    def test_impossible_profiles(self):
        cases = [(dict(shape="angle", a=0, b=1, t=0.1), "a must be"),
                 (dict(shape="angle", a=1, b=1, t=0), "t must be"),
                 (dict(shape="channel", a=1, b=1, t=0.1, t2=0), "t2 must be"),
                 (dict(shape="zee", a=1, b=1, t=0.1, c=0), "c must be"),
                 (dict(shape="bulb_angle", a=1, b=1, t=0.1, r=0), "bulb radius")]
        for kw, msg in cases:
            with self.subTest(msg=msg):
                with self.assertRaisesRegex(ValueError, msg):
                    ex.validate(ex.ExtrusionProfile(name="bad", **kw))

    def test_point_helpers(self):
        self.assertEqual(ex._inside_fraction([], np.zeros((0, 2))), 1.0)
        self.assertIsNone(ex._sample(None))
        self.assertIsNone(ex._sample(np.zeros((0, 3))))
        self.assertIsNone(ex.surface_samples(None, None))
        pts = np.random.default_rng(0).random((5, 3))
        np.testing.assert_array_equal(ex.surface_samples(pts, None), pts)
        np.testing.assert_array_equal(ex.surface_samples(pts, np.zeros((0, 3))), pts)
        flat = np.zeros((3, 3))
        np.testing.assert_array_equal(ex.surface_samples(flat, np.array([[0, 1, 2]])), flat)
        angle = ex.find("angle", "AND10133-2004")
        self.assertFalse(ex.fills_outline(angle, None))
        self.assertIsNone(ex._shortest_fitting_length(angle, None, None, loose=False))
        self.assertEqual(ex._fit(angle, None, None, None, None, None), (None, None))


class TestLibraryFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_duplicates_are_refused(self):
        src = Path(ex.DATA_PATH).read_text(encoding="utf-8").splitlines()
        dup = self.tmp / "ext.csv"
        dup.write_text("\n".join(src + [src[1]]) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate shape"):
            ex.load_catalog(str(dup))
        std = Path(ex.STANDARD_PATH).read_text(encoding="utf-8").splitlines()
        dup2 = self.tmp / "std.csv"
        dup2.write_text("\n".join(std + [std[1]]) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate part number"):
            ex.load_standard(str(dup2))

    def test_custom_library_edges(self):
        bad = self.tmp / "custom.csv"
        bad.write_bytes(b"name,shape\n\xff\xfe broken")
        lib = ex.read_custom(bad)
        self.assertTrue(lib.errors)
        self.assertFalse(ex.read_custom(self.tmp / "none.csv").exists)
        neg = self.tmp / "neg.csv"
        neg.write_text("name,shape,a_in,b_in,c_in,t_in,t2_in,r_in,area_in2,notes\n"
                       "MINE,angle,2,2,,0.25,,,-1,\n", encoding="utf-8")
        self.assertTrue(any("negative" in e for e in ex.read_custom(neg).errors))


class TestStockGuards(unittest.TestCase):
    def test_rules_handed_nothing(self):
        self.assertEqual(st.ceil_to_increment(1.13, 0.0), 1.13)
        bar = st.custom_cylindrical(1.0, 4.0)
        self.assertEqual(bar.dims_mm(), (1.0 * MM, 4.0 * MM))
        plate = st.custom_rectilinear(0.5, 2.0, 3.0)
        self.assertEqual(plate.dims_mm(), (3.0 * MM, 2.0 * MM, 0.5 * MM))
        self.assertIsNone(st.snap("rectilinear", st.RECOMMENDED, None, None))
        self.assertFalse(st.same_size(None, plate))
        self.assertIsNone(st.classify_basis(None, None, None))
        prof = ex.find("angle", "AND10133-2004")
        extr = st.StockSize(kind="extrusion", profile=prof, basis=st.EXTRUSION)
        self.assertIs(st.classify_basis(extr, None, None), extr)
        self.assertEqual(st.fit_warnings(None, None, None), [])
        self.assertEqual(st.fit_warnings(plate, None, None), [])

    def test_a_bar_too_thin(self):
        cyl = SimpleNamespace(radius_mm=1.0 * MM, height_mm=3.0 * MM)
        w = st.fit_warnings(st.custom_cylindrical(1.5, 2.0), None, cyl)
        self.assertTrue(any("Diameter" in x for x in w))
        self.assertTrue(any("Length" in x for x in w))


if __name__ == "__main__":
    unittest.main()
