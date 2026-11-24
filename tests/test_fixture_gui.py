"""Tabbed fixturing with real parts: the solid bodies and their STEP file
(geometry/fixture_cad.py), the Part Inputs controls in the real window, the
export through a worker process, and tabbed parts in the tuning library.

Needs the CAD kernel and Qt; skipped in the light CI environment.
"""
from __future__ import annotations

import contextlib
import io
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

if os.name != "nt" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    import OCP  # noqa: F401
    from PySide6.QtWidgets import QApplication, QFileDialog

    HAVE = True
except Exception:  # noqa: BLE001
    HAVE = False


def _read_names(path) -> list[str]:
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDF import TDF_LabelSequence
    from OCP.TDocStd import TDocStd_Document
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.XCAFDoc import XCAFDoc_DocumentTool

    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    XCAFApp_Application.GetApplication_s().InitDocument(doc)
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    reader.ReadFile(str(path))
    reader.Transfer(doc)
    labels = TDF_LabelSequence()
    XCAFDoc_DocumentTool.ShapeTool_s(doc.Main()).GetFreeShapes(labels)
    names = []
    for i in range(1, labels.Length() + 1):
        attr = TDataStd_Name()
        if labels.Value(i).FindAttribute(TDataStd_Name.GetID_s(), attr):
            names.append(attr.Get().ToExtString())
    return names


@unittest.skipUnless(HAVE, "needs the CAD kernel and Qt")
class TestBodies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from partestimator.machining.geometry import pipeline
        from tests import occparts

        cls.tmp = Path(tempfile.mkdtemp())
        with contextlib.redirect_stdout(io.StringIO()):
            cls.plate = occparts.machined_plate(cls.tmp)          # 6 x 4 x 0.5: vertical
            cls.thin = occparts.thin_pocketed_plate(cls.tmp)      # 0.125: flat
        cls.res = {p: pipeline.compute_analysis(str(p)) for p in (cls.plate, cls.thin)}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def export(self, path, settings=None):
        from partestimator.machining.geometry import fixture, fixture_cad

        res = self.res[path]
        plan = fixture.plan_for_results(res, res["stock"], settings, grow_stock=True)
        out = self.tmp / f"{path.stem}_fixture.step"
        return plan, fixture_cad.export(str(path), plan.to_dict(), str(out)), res

    def check(self, path, procedure, names, settings=None):
        plan, got, res = self.export(path, settings)
        self.assertEqual(plan.procedure, procedure)
        self.assertEqual(got["problems"], [])
        self.assertEqual(_read_names(got["path"]), names)
        v = got["volumes_in3"]
        part = res["shape_props"].volume_mm3 / 25.4 ** 3
        self.assertAlmostEqual(v["Stock"], plan.stock_volume_in3, places=4)
        self.assertAlmostEqual(v["Part"], part, places=4)
        # Part, frame and tabs: the plan's frame plus the part, give or take
        # the tabs and the outline being measured on a grid.
        held = names[-2]
        self.assertAlmostEqual(v[held], plan.frame_volume_in3 + part, delta=0.02 * v[held])
        if len(names) == 4:
            self.assertGreater(v[names[1]], v[held])
            self.assertGreater(v["Stock"], v[names[1]])
        return plan

    def test_vertical_bodies(self):
        self.check(self.plate, "vertical", ["Stock", "After side 1", "After side 2", "Part"])

    def test_flat_bodies(self):
        from partestimator.machining.geometry import fixture

        plan = self.check(self.plate, "flat", ["Stock", "After op 1", "After op 2", "Part"],
                          fixture.FixtureSettings(vertical_min_thickness_in=1.0))
        self.assertEqual(plan.operations, 2)

    def test_sheet_is_three_bodies(self):
        plan = self.check(self.thin, "flat", ["Stock", "After op 1", "Part"])
        self.assertEqual(plan.operations, 1)

    def test_a_tab_that_misses_the_part_is_reported(self):
        from partestimator.machining.geometry import fixture, fixture_cad, step_io

        res = self.res[self.plate]
        plan = fixture.plan_for_results(res, res["stock"], grow_stock=True)
        t = plan.tabs[0]
        t.x, t.y = t.x + t.nx * 0.2, t.y + t.ny * 0.2      # out into the channel
        _bodies, problems = fixture_cad.build_bodies(step_io.load_step_file(str(self.plate)),
                                                     plan)
        self.assertTrue(any("Tab 1" in p for p in problems), problems)


@unittest.skipUnless(HAVE, "needs the CAD kernel and Qt")
class TestInTheWindow(unittest.TestCase):
    def setUp(self):
        from partestimator.machining.geometry import pipeline
        from partestimator.machining.ui.main_window import MainWindow
        from tests import occparts

        self.app = QApplication.instance() or QApplication([])
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        with contextlib.redirect_stdout(io.StringIO()):
            self.part = occparts.machined_plate(self.tmp)
        self.w = MainWindow()
        self.addCleanup(self.w.close)
        res = pipeline.compute_analysis(str(self.part), options=self.w._analysis_options())
        self.w._current_path = str(self.part)
        self.w._on_analysis_finished(res)

    def test_a_plate_part_opens_tabbed_on_a_grown_stock(self):
        w = self.w
        plan, stock = w.results["fixture"], w.results["stock"]
        self.assertIsNotNone(plan)
        self.assertEqual(plan.procedure, "vertical")
        self.assertEqual(stock.basis, "recommended")
        self.assertEqual(stock.table_rows(), [("OD/ST", "0.750"), ("W", "5.750"),
                                              ("L", "7.000")])
        self.assertTrue(w.operations_spin.isEnabled())
        self.assertEqual(w.operations_spin.value(), 1)
        names = [t.name for t in w._current_estimate().terms]
        self.assertIn("Tab channel", names)
        self.assertIn("Vertical", w.fixture_controls.lbl_plan.text())

    def test_off_and_on(self):
        w = self.w
        on = w._current_estimate().door_to_door_minutes
        w.fixture_controls.chk.setChecked(False)
        self.assertIsNone(w.results["fixture"])
        self.assertEqual(w.results["stock"].width_in, 5.0)      # the plain rule again
        self.assertFalse(w.fixture_controls.details.isVisibleTo(w))   # just the checkbox
        self.assertTrue(w.fixture_controls.chk.isVisibleTo(w))
        self.assertNotAlmostEqual(w._current_estimate().door_to_door_minutes, on)
        w.fixture_controls.chk.setChecked(True)
        self.assertAlmostEqual(w._current_estimate().door_to_door_minutes, on)

    def test_lathe_and_bar_are_not_tabbed(self):
        w = self.w
        w.removal_combo.setCurrentText("Lathe")
        self.assertIsNone(w.results["fixture"])
        self.assertFalse(w.fixture_controls.isVisibleTo(w))      # the section goes
        self.assertFalse(w.fixture_title.isVisibleTo(w))
        w.removal_combo.setCurrentText("Mill")
        self.assertIsNotNone(w.results["fixture"])
        self.assertTrue(w.fixture_controls.details.isVisibleTo(w))

    def test_operations_can_still_be_changed(self):
        w = self.w
        w.operations_spin.setValue(3)
        self.assertEqual(w._current_estimate().operations, 3)
        w.fixture_controls.btn_flip.setChecked(True)            # same procedure: kept
        self.assertEqual(w.operations_spin.value(), 3)
        w.fixture_controls.clamp.setCurrentIndex(2)             # now flat: suggests 2
        self.assertEqual(w.operations_spin.value(), 2)
        w.operations_spin.setValue(4)
        self.assertEqual(w._current_estimate().operations, 4)

    def test_untouched_operations_go_back_when_tabs_go(self):
        w = self.w
        w.fixture_controls.chk.setChecked(False)
        w.operations_spin.setValue(2)
        w.fixture_controls.chk.setChecked(True)
        self.assertEqual(w.operations_spin.value(), 1)          # the procedure's
        w.fixture_controls.chk.setChecked(False)
        self.assertEqual(w.operations_spin.value(), 2)          # what it was

    def test_a_short_edge_clamp_can_lay_it_flat_and_back(self):
        w = self.w
        w.fixture_controls.clamp.setCurrentIndex(2)
        self.assertEqual(w.results["fixture"].procedure, "flat")
        self.assertEqual(w.operations_spin.value(), 2)
        self.assertTrue(w.fixture_controls.clamp.isEnabled())
        w.fixture_controls.clamp.setCurrentIndex(0)
        self.assertEqual(w.results["fixture"].procedure, "vertical")

    def test_a_typed_stock_is_kept_and_warned(self):
        w = self.w
        w.stock_controls.spin_l.setValue(6.5)
        self.assertEqual(w.results["stock"].basis, "custom")
        self.assertTrue(any("frame" in x for x in w.results["fixture"].warnings))
        w.stock_controls.btn_recommended.click()
        self.assertEqual(w.results["stock"].length_in, 7.0)

    def test_export_through_a_worker(self):
        w = self.w
        out = self.tmp / "out.step"
        with mock.patch.object(QFileDialog, "getSaveFileName", return_value=(str(out), "")):
            w.export_fixture()
        t0 = time.time()
        while w.fixture_busy():
            self.app.processEvents()
            time.sleep(0.05)
            self.assertLess(time.time() - t0, 180)
        self.app.processEvents()
        self.assertEqual(_read_names(out), ["Stock", "After side 1", "After side 2", "Part"])
        self.assertTrue(w.fixture_controls.btn_export.isEnabled())

    def test_settings_change_relays_it_out(self):
        from partestimator.machining.geometry import fixture

        self.addCleanup(fixture.set_standard, None)
        fixture.set_standard(fixture.FixtureSettings(min_tabs=8))
        self.w.refresh_stock_allowances()
        self.assertEqual(self.w.results["fixture"].tab_count, 8)
        # Off by default: the open part keeps its own choice; the next part
        # opens untabbed, and can still be tabbed by hand.
        fixture.set_standard(fixture.FixtureSettings(enabled=False))
        self.w.refresh_stock_allowances()
        self.assertIsNotNone(self.w.results["fixture"])
        self.w._on_analysis_finished(dict(self.w.results))
        self.assertIsNone(self.w.results["fixture"])
        self.assertFalse(self.w.fixture_controls.chk.isChecked())
        self.w.fixture_controls.chk.setChecked(True)
        self.assertIsNotNone(self.w.results["fixture"])

    def test_saved_to_the_library_it_estimates_the_same(self):
        from partestimator.machining import tuning
        from partestimator.machining.geometry import rates
        from partestimator.machining.ui import tuning_window as tw

        folder = self.tmp / "lib"
        with mock.patch.dict(os.environ, {tuning.ENV_DIR: str(folder)}), \
                mock.patch.object(tw.SaveToLibraryDialog, "exec", lambda _d: 1):
            self.w.save_to_library()
            lib = tuning.Library.load(folder)
            e = lib.entries[-1]
            snap = lib.load_snapshot(e)
            self.assertEqual(snap["fixture"]["procedure"], "vertical")
            prof, _state = tuning.load_tuned_profile(folder)
            table = rates.load_rates()
            prep = tuning.prepare(e, snap, prof)
            self.assertAlmostEqual(tuning.estimate(prep, table, prof).door_to_door_minutes,
                                   self.w._current_estimate().door_to_door_minutes, places=6)
            # Another stock size in the library has no layout: not tabbed.
            e.stock_size = "1 x 6 x 9"
            self.assertNotIn("fixture", tuning.prepare(e, snap, prof).results)


if __name__ == "__main__":
    unittest.main()
