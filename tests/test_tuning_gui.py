"""Shop tuning in the real window: real STEP files analysed in worker
processes and captured, the Tuning window driven as a user would, and the
Machining tab's "Save to tuning library" and "Open in Machining tab".

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
    from PySide6.QtWidgets import QApplication, QMessageBox

    HAVE = True
except Exception:  # noqa: BLE001
    HAVE = False


@unittest.skipUnless(HAVE, "needs the CAD kernel and Qt")
class TestTuningLibrary(unittest.TestCase):
    """One library, built up step by step: the tests run in name order."""

    @classmethod
    def setUpClass(cls):
        from partestimator.machining import tuning
        from partestimator.machining.geometry import rates
        from tests import occparts

        cls.app = QApplication.instance() or QApplication([])
        cls.tmp = Path(tempfile.mkdtemp())
        cls.folder = cls.tmp / "lib"
        cls.env = mock.patch.dict(os.environ, {tuning.ENV_DIR: str(cls.folder)})
        cls.env.start()
        src = cls.tmp / "src"
        src.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            cls.files = [occparts.machined_plate(src), occparts.block(src),
                         occparts.turned_part(src), occparts.l_bracket(src)]
            cls.extra = occparts.plate_with_spotface(src)
        cls.table = rates.load_rates()
        cls.tuning = tuning

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def spin(self, busy, timeout=300):
        t0 = time.time()
        while busy():
            self.app.processEvents()
            time.sleep(0.02)
            self.assertLess(time.time() - t0, timeout, "never finished")
        self.app.processEvents()

    def window(self):
        from partestimator.machining.ui.tuning_window import TuningWindow

        defaults = {"material": "Aluminum 6061-T6", "finish_uin": 63.0, "operations": 2}
        return TuningWindow(self.table, defaults=lambda: defaults, folder=self.folder)

    def test_1_add_captures_and_keeps_no_step_files(self):
        from partestimator.machining.ui import tuning_window as tw

        w = self.window()
        w.add_paths(self.files)
        self.spin(w.is_busy)
        lib = w.library
        self.assertEqual(len(lib.entries), 4, w.lbl_busy.text())
        self.assertTrue(all(lib.has_snapshot(e) for e in lib.entries))
        self.assertEqual(list(self.folder.rglob("*.step")), [])
        self.assertEqual({e.material for e in lib.entries}, {"Aluminum 6061-T6"})
        self.assertEqual({e.operations for e in lib.entries}, {2})
        turned = next(e for e in lib.entries if e.part == "turned")
        self.assertEqual((turned.process, turned.stock_type), ("lathe", "round"))
        self.assertEqual(w.parts.item(0, tw.C_STATUS).text(), "no measured time")
        self.assertTrue(w.parts.item(0, tw.C_EST).text())
        # The same files again add nothing.
        w.add_paths(self.files[:2])
        self.spin(w.is_busy)
        self.assertEqual(len(w.library.entries), 4)
        self.assertIn("Already in the library", w.lbl_busy.text())

    def test_2_edit_in_the_table(self):
        from partestimator.machining.ui import tuning_window as tw

        w = self.window()
        for r in range(w.parts.rowCount()):
            est = float(w.parts.item(r, tw.C_EST).text().replace(",", ""))
            w.parts.item(r, tw.C_MEASURED).setText(f"{est * 1.5:.2f}")
        before = w.parts.item(1, tw.C_EST).text()
        w.parts.item(1, tw.C_OPS).setText("3")
        w.parts.item(1, tw.C_TAPPED).setText("nonsense")       # put back as it was
        w.parts.item(0, tw.C_STOCK).setText("plate 1 x 6 x 8")
        w.parts.item(3, tw.C_STOCK).setText("plate 1 x 6")     # refused, kept
        lib = self.tuning.Library.load(self.folder)            # saved as it goes
        self.assertTrue(all(e.measured_min for e in lib.entries))
        self.assertEqual(lib.entries[1].operations, 3)
        self.assertIsNone(lib.entries[1].tapped_holes)
        self.assertEqual((lib.entries[0].stock_type, lib.entries[0].stock_size),
                         ("plate", "1 x 6 x 8"))
        self.assertNotEqual(lib.entries[3].stock_size, "1 x 6")
        self.assertIn("T x W x L", w.lbl_busy.text())
        self.assertNotEqual(w.parts.item(1, tw.C_EST).text(), before)
        self.assertEqual(w.parts.item(2, tw.C_STATUS).text(), "ready")
        self.assertEqual(w.parts.item(2, tw.C_ERR).text(), "-33%")

    def test_3_fit_and_save(self):
        w = self.window()
        w.run_fit()
        self.spin(w.is_busy)
        self.assertIsNotNone(w._fit, w.lbl_busy.text())
        self.assertEqual(len(w._fit.parts), 4)
        self.assertLess(w._fit.after.median_error_pct, w._fit.before.median_error_pct)
        seen = []
        w.tuningChanged.connect(lambda: seen.append(1))
        w.save_fit()
        self.assertEqual(seen, [1])
        prof, state = self.tuning.load_tuned_profile(self.folder)
        self.assertIn("tuned to 4 parts", state.describe())
        self.assertGreater(prof.get("calibrate_roughing"), 1.0)

    def test_4_machining_tab_save_and_open(self):
        from partestimator.machining.geometry import pipeline
        from partestimator.machining.ui import tuning_window as tw
        from partestimator.machining.ui.main_window import MainWindow

        m = MainWindow()
        self.addCleanup(m.close)
        self.assertIn("tuned to 4 parts", m.lbl_tuning.text())
        res = pipeline.compute_analysis(str(self.extra), options=m._analysis_options())
        m._current_path = str(self.extra)
        m._on_analysis_finished(res)
        m.material_combo.setCurrentText("Aluminum 7075-T6")
        m.operations_spin.setValue(3)
        m.stock_controls.spin_l.setValue(m.stock_controls.spin_l.value() + 1.0)
        stock_rows = m.stock_controls.stock().table_rows()
        on_screen = m._current_estimate().door_to_door_minutes

        def accept(dlg):
            dlg.measured.setValue(42.0)
            return 1

        with mock.patch.object(tw.SaveToLibraryDialog, "exec", accept):
            m.save_to_library()
        lib = self.tuning.Library.load(self.folder)
        e = lib.entries[-1]
        self.assertEqual((e.material, e.operations, e.measured_min, e.stock_type),
                         ("Aluminum 7075-T6", 3, 42.0, "plate"))
        self.assertEqual(Path(e.source_file), self.extra.resolve())
        self.assertEqual(list(self.folder.rglob("*.step")), [])
        # The library estimates it exactly as the window did.
        prof, _state = self.tuning.load_tuned_profile(self.folder)
        prep = self.tuning.prepare(e, lib.load_snapshot(e), prof)
        self.assertAlmostEqual(self.tuning.estimate(prep, self.table, prof).door_to_door_minutes,
                               on_screen, places=6)

        # Change everything, then open it back from the library.
        m.material_combo.setCurrentText("Aluminum 6061-T6")
        m.operations_spin.setValue(1)
        opened = []
        m.fileOpened.connect(opened.append)
        m.open_library_entry(e, lib)
        self.spin(lambda: m._runner is not None)
        self.assertEqual(opened, [e.source_file])
        self.assertEqual(m.material_combo.currentText(), "Aluminum 7075-T6")
        self.assertEqual(m.operations_spin.value(), 3)
        self.assertEqual(m.stock_controls.stock().table_rows(), stock_rows)

        # Saving again updates the same entry rather than adding one.
        with mock.patch.object(tw.SaveToLibraryDialog, "exec", accept):
            m.save_to_library()
        self.assertEqual(len(self.tuning.Library.load(self.folder).entries), len(lib.entries))

    def test_5_a_moved_step_file_still_fits(self):
        w = self.window()
        e = w.library.entries[0]
        e.source_file = str(self.tmp / "gone.step")
        w.library.save()
        w.reload()
        self.assertEqual(w.parts.item(0, 11).text(), "ready")
        with mock.patch("partestimator.machining.ui.tuning_window.QMessageBox.information") as info:
            w.parts.selectRow(0)
            w.open_selected()
        info.assert_called_once()

    def test_6_back_to_shipped(self):
        w = self.window()
        with mock.patch("partestimator.machining.ui.tuning_window.QMessageBox.question",
                        return_value=QMessageBox.Yes):
            w.clear_fit()
        _prof, state = self.tuning.load_tuned_profile(self.folder)
        self.assertFalse(state.active)

    def test_7_command_line_add(self):
        from partestimator import __main__ as cli

        folder = self.tmp / "cli"
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            rc = cli.main(["tune", "--dir", str(folder), "--add", str(self.files[1]),
                           "--material", "Aluminum 6061-T6", "--operations", "2"])
        self.assertEqual(rc, 0, out.getvalue())
        lib = self.tuning.Library.load(folder)
        self.assertEqual([(e.part, e.material, e.operations) for e in lib.entries],
                         [("block", "Aluminum 6061-T6", 2)])
        self.assertTrue(lib.has_snapshot(lib.entries[0]))
        self.assertEqual(list(folder.rglob("*.step")), [])


if __name__ == "__main__":
    unittest.main()
