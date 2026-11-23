"""settings.json and the stock allowances it carries: the recommended stock
is the part plus the shop's own length, width, thickness and diameter
allowances, rounded up to the next 1/4 in.

The settings and stock parts are numpy-free; the panel needs Qt and the
worker hand-off the CAD kernel, and skip without them.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from partestimator.machining.geometry import stock as st
from partestimator.settings import Settings, settings_path

if os.name != "nt" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

MM = 25.4


def box(l_in, w_in, t_in):
    return SimpleNamespace(extents_mm=[l_in * MM, w_in * MM, t_in * MM],
                           axes=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], center_mm=[0, 0, 0],
                           volume_mm3=l_in * w_in * t_in * MM ** 3)


def cyl(d_in, h_in):
    return SimpleNamespace(radius_mm=d_in / 2 * MM, height_mm=h_in * MM,
                           axis_direction=[0, 0, 1], axis_point_mm=[0, 0, 0])


class Restore(unittest.TestCase):
    def setUp(self):
        self.addCleanup(st.set_allowances)


class TestAllowances(Restore):
    def test_shipped_values(self):
        self.assertEqual(st.allowances(), {"length_in": 1.0, "width_in": 1.0,
                                           "thickness_in": 0.125, "diameter_in": 0.125})
        s = st.rectilinear_stock(box(4.0, 2.5, 0.75).extents_mm)
        self.assertEqual((s.thickness_or_diameter_in, s.width_in, s.length_in), (1.0, 3.5, 5.0))

    def test_plate_uses_each_allowance(self):
        st.set_allowances(length_in=0.5, width_in=0.25, thickness_in=0.0)
        s = st.rectilinear_stock(box(4.0, 2.5, 0.75).extents_mm)
        self.assertEqual((s.thickness_or_diameter_in, s.width_in, s.length_in), (0.75, 2.75, 4.5))
        self.assertAlmostEqual(s.volume_in3, 0.75 * 2.75 * 4.5)

    def test_bar_uses_length_and_diameter(self):
        st.set_allowances(length_in=2.0, diameter_in=0.5)
        s = st.cylindrical_stock(1.0 * MM, 3.0 * MM)
        self.assertEqual((s.thickness_or_diameter_in, s.length_in), (1.5, 5.0))

    def test_sheet_is_still_used_as_is(self):
        st.set_allowances(thickness_in=0.5)
        s = st.rectilinear_stock(box(4.0, 2.5, 0.09).extents_mm)
        self.assertAlmostEqual(s.thickness_or_diameter_in, 0.09)

    def test_names_left_out_go_back_to_shipped(self):
        st.set_allowances(length_in=3.0)
        st.set_allowances(width_in=2.0)
        self.assertEqual(st.allowances()["length_in"], 1.0)
        self.assertEqual(st.allowances()["width_in"], 2.0)

    def test_the_recommended_size_follows_and_is_still_recognised(self):
        st.set_allowances(length_in=2.0, width_in=2.0)
        rect = box(4.0, 2.5, 0.75)
        s = st.snap("rectilinear", st.RECOMMENDED, rect, None)
        self.assertEqual((s.width_in, s.length_in), (4.5, 6.0))
        typed = st.custom_rectilinear(1.0, 4.5, 6.0)
        self.assertEqual(st.classify_basis(typed, rect, None).basis, st.RECOMMENDED)

    def test_bad_values_are_refused(self):
        for kw in ({"length_in": -0.1}, {"width_in": 13.0}, {"height_in": 1.0}):
            with self.assertRaises(ValueError):
                st.set_allowances(**kw)
        self.assertEqual(st.allowances(), st.DEFAULT_ALLOWANCES)


class TestSettingsFile(Restore):
    def setUp(self):
        super().setUp()
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.path = self.dir / "settings.json"

    def test_the_tests_never_read_the_real_file(self):
        self.assertEqual(settings_path().name, "no_settings.json")
        self.assertFalse(settings_path().exists())

    def test_round_trip(self):
        s = Settings(stock_length_pad_in=0.5, stock_width_pad_in=0.75,
                     stock_thickness_pad_in=0.0625, stock_diameter_pad_in=0.25)
        s.save(self.path)
        back = Settings.load(self.path)
        self.assertEqual(back, s)
        self.assertEqual(back.stock_allowances(), {"length_in": 0.5, "width_in": 0.75,
                                                   "thickness_in": 0.0625,
                                                   "diameter_in": 0.25})

    def test_an_old_file_keeps_the_shipped_allowances(self):
        self.path.write_text(json.dumps({"sheet_cutoff_in": 0.1}), encoding="utf-8")
        s = Settings.load(self.path)
        self.assertEqual(s.sheet_cutoff_in, 0.1)
        self.assertEqual(s.stock_allowances(), st.DEFAULT_ALLOWANCES)

    def test_bad_values_fall_back_one_by_one(self):
        self.path.write_text(json.dumps({"stock_length_pad_in": -1, "stock_width_pad_in": "x",
                                         "stock_thickness_pad_in": 0.25}), encoding="utf-8")
        s = Settings.load(self.path)
        self.assertEqual((s.stock_length_pad_in, s.stock_width_pad_in, s.stock_thickness_pad_in),
                         (1.0, 1.0, 0.25))

    def test_apply(self):
        Settings(stock_length_pad_in=0.5).apply()
        self.assertEqual(st.allowances()["length_in"], 0.5)


try:
    from PySide6.QtWidgets import QApplication

    HAVE_QT = True
except Exception:  # noqa: BLE001
    HAVE_QT = False


@unittest.skipUnless(HAVE_QT, "needs PySide6")
class TestOpenPartFollows(Restore):
    def setUp(self):
        super().setUp()
        from partestimator.machining.ui import stock_panel as sp

        self.app = QApplication.instance() or QApplication([])
        self.w = sp.StockControls()
        self.emitted = []
        self.w.stockChanged.connect(self.emitted.append)

    def test_a_recommended_stock_is_resized(self):
        self.w.set_part(box(4.0, 2.5, 0.75), None, "rectilinear")
        st.set_allowances(length_in=0.5, width_in=0.5)
        self.w.refresh_allowances()
        self.assertEqual((self.w.spin_b.value(), self.w.spin_l.value()), (3.0, 4.5))
        self.assertEqual(self.emitted[-1].basis, st.RECOMMENDED)

    def test_a_typed_size_is_kept(self):
        self.w.set_part(box(4.0, 2.5, 0.75), None, "rectilinear")
        self.w.spin_l.setValue(8.0)
        st.set_allowances(length_in=0.5)
        self.w.refresh_allowances()
        self.assertEqual(self.w.spin_l.value(), 8.0)
        self.assertEqual(self.w.stock().basis, st.CUSTOM)

    def test_a_round_bar_is_resized(self):
        self.w.set_part(box(4.0, 2.0, 2.0), cyl(2.0, 4.0), "cylindrical")
        st.set_allowances(length_in=2.0, diameter_in=0.5)
        self.w.refresh_allowances()
        self.assertEqual((self.w.spin_a.value(), self.w.spin_l.value()), (2.5, 6.0))


try:
    import OCP  # noqa: F401

    HAVE_OCP = True
except Exception:  # noqa: BLE001
    HAVE_OCP = False


def _analyse_in_child(path, options):
    from partestimator.machining.geometry import pipeline

    s = pipeline.compute_analysis(path, options=options)["stock"]
    return s.thickness_or_diameter_in, s.width_in, s.length_in


@unittest.skipUnless(HAVE_OCP, "needs the CAD kernel")
class TestWorkerProcess(Restore):
    def test_the_allowances_reach_a_fresh_process(self):
        import concurrent.futures as cf
        import contextlib
        import io
        import multiprocessing as mp

        from tests import occparts

        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        with contextlib.redirect_stdout(io.StringIO()):
            part = occparts.block(tmp)              # 3 x 2 x 1.5
        opts = {"stock_allowances": {"length_in": 0.25, "width_in": 0.25,
                                     "thickness_in": 0.0, "diameter_in": 0.125}}
        with cf.ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as pool:
            got = pool.submit(_analyse_in_child, str(part), opts).result(timeout=300)
        self.assertEqual(got, (1.5, 2.25, 3.25))


if __name__ == "__main__":
    unittest.main()
