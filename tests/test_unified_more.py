"""The unified window's own logic: settings, tabs, classification and its
failures, the menus, drag and drop, and start-up.

The two tools' actual loading is stubbed (test_unified_window opens real
parts end to end); what is checked here is how the window decides, and
what it tells you.
"""
from __future__ import annotations

import concurrent.futures as cf
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

if os.name != "nt" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    import OCP  # noqa: F401
    from PySide6.QtCore import QMimeData, QUrl
    from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

    HAVE = True
except Exception:  # noqa: BLE001
    HAVE = False


@unittest.skipUnless(HAVE, "needs OCP and PySide6")
class TestUnifiedLogic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from partestimator import app as app_mod

        cls.mod = app_mod
        cls.app = QApplication.instance() or QApplication([])
        cls.tmp = Path(tempfile.mkdtemp(prefix="unified_more_"))
        cls.env = mock.patch.dict(os.environ, {"PE_SETTINGS": str(cls.tmp / "settings.json")})
        cls.env.start()
        cls.w = app_mod.UnifiedWindow()

    @classmethod
    def tearDownClass(cls):
        cls.w.close()
        cls.env.stop()
        from partestimator.machining.geometry import fixture, stock

        stock.set_allowances()
        fixture.set_standard(None)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        w = self.w
        # The tools' loading is not under test here.
        for target, name, attr in ((w.sheet, "load", "load"),
                                   (w.mach, "_load_and_process", "process")):
            p = mock.patch.object(target, name)
            setattr(self, attr, p.start())
            self.addCleanup(p.stop)
        w._future = None
        w.path = None
        w.detected = None
        w.mode = None
        w._loaded_into = {"sheet_metal": None, "machined": None}

    # -- settings -----------------------------------------------------------

    def test_settings_dialog_round_trip(self):
        from partestimator.settings import Settings

        s = Settings()
        dlg = self.mod.SettingsDialog(s)
        dlg.cutoff.setValue(0.12)
        dlg.pads["stock_length_pad_in"].setValue(0.5)
        dlg.fx["tab_width_in"].setValue(0.375)
        dlg.min_tabs.setValue(5)
        dlg.fixture_enabled.setChecked(True)
        dlg.fx["channel_depth_2_in"].setValue(0.25)      # below depth 1: put right
        dlg.apply_to(s)
        self.assertEqual((s.sheet_cutoff_in, s.stock_length_pad_in, s.tab_width_in,
                          s.min_tabs, s.fixture_enabled), (0.12, 0.5, 0.375, 5, True))
        self.assertGreater(s.channel_depth_2_in, s.channel_depth_1_in)

    def test_edit_settings_cancelled(self):
        before = self.w.settings.sheet_cutoff_in
        with mock.patch.object(self.mod.SettingsDialog, "exec", return_value=QDialog.Rejected):
            self.w.edit_settings()
        self.assertEqual(self.w.settings.sheet_cutoff_in, before)

    def accept_with(self, **values):
        def run(dlg):
            for key, v in values.items():
                spin = dlg.pads.get(key) or dlg.fx.get(key)
                (spin or getattr(dlg, key)).setValue(v)
            return QDialog.Accepted
        return mock.patch.object(self.mod.SettingsDialog, "exec", run)

    def test_edit_settings_saves_and_relays_the_stock_out(self):
        w = self.w
        with self.accept_with(stock_width_pad_in=0.75), \
                mock.patch.object(w.mach, "refresh_stock_allowances") as refresh:
            w.edit_settings()
        refresh.assert_called_once()
        self.assertIn('"stock_width_pad_in": 0.75',
                      (self.tmp / "settings.json").read_text(encoding="utf-8"))
        from partestimator.machining.geometry import stock

        self.assertEqual(stock.allowances()["width_in"], 0.75)

    def test_a_new_cutoff_offers_to_recheck_the_open_part(self):
        w = self.w
        w.path = self.tmp / "part.step"
        with self.accept_with(cutoff=w.settings.sheet_cutoff_in + 0.01), \
                mock.patch.object(QMessageBox, "question", return_value=QMessageBox.Yes), \
                mock.patch.object(w, "open_path") as reopen:
            w.edit_settings()
        reopen.assert_called_once_with(w.path)

    def test_a_failed_save_is_reported(self):
        w = self.w
        with self.accept_with(), \
                mock.patch.object(type(w.settings), "save", side_effect=OSError("read-only")), \
                mock.patch.object(QMessageBox, "warning") as warn:
            w.edit_settings()
        self.assertIn("read-only", warn.call_args[0][2])

    # -- tabs and opening ---------------------------------------------------

    def test_tab_clicks(self):
        w = self.w
        w._tab_clicked(-1)                                    # nothing
        w._tab_clicked(1)                                     # no part: just switches
        self.assertEqual((w.mode, w.stack.currentIndex()), ("machined", self.mod.PAGE_MACHINED))
        w.path = self.tmp / "p.step"
        w._future = object()                                  # still classifying: ignored
        with mock.patch.object(w, "set_mode") as sm:
            w._tab_clicked(0)
            sm.assert_not_called()
            w._future = None
            w._tab_clicked(0)
            sm.assert_called_once_with("sheet_metal", user=True)

    def test_open_dialog(self):
        w = self.w
        with mock.patch.object(self.mod.QFileDialog, "getOpenFileName",
                               return_value=("", "")), \
                mock.patch.object(w, "open_path") as op:
            w.open_dialog()
            op.assert_not_called()
        with mock.patch.object(self.mod.QFileDialog, "getOpenFileName",
                               return_value=("C:/x/part.step", "")), \
                mock.patch.object(w, "open_path") as op:
            w.open_dialog()
            op.assert_called_once_with(Path("C:/x/part.step"))

    def test_one_open_at_a_time(self):
        w = self.w
        w._future = object()
        w.open_path(self.tmp / "a.step")
        self.assertIsNone(w.path)

    def classify_in_process(self, result=None, error=None):
        w = self.w
        saved = w._pool
        w._pool = None
        self.addCleanup(setattr, w, "_pool", saved)
        if error is not None:
            return mock.patch.object(self.mod, "classify_to_dict", side_effect=error)
        return mock.patch.object(self.mod, "classify_to_dict", return_value=result)

    def test_classified_in_process(self):
        from partestimator.classify import Classification, classify_to_dict  # noqa: F401

        w = self.w
        result = {"kind": "sheet_metal", "reason": "bent, uniform 0.060"}
        with self.classify_in_process(result), \
                mock.patch.object(self.mod, "from_dict",
                                  return_value=SimpleNamespace(kind="sheet_metal",
                                                               reason="bent")):
            w.open_path(self.tmp / "b.step")
        self.assertEqual(w.mode, "sheet_metal")
        self.load.assert_called_once()
        self.assertIn("bent", w.banner.text())

    def test_a_file_that_cannot_be_classified_opens_on_machining(self):
        w = self.w
        with self.classify_in_process(error=ValueError("bad file")):
            w.open_path(self.tmp / "c.step")
        self.assertEqual(w.mode, "machined")
        self.assertIn("Could not classify", w.banner.text())
        self.process.assert_called_once()

    def test_a_pool_that_will_not_take_work_is_dropped(self):
        w = self.w
        saved = w._pool
        self.addCleanup(setattr, w, "_pool", saved)
        w._pool = mock.Mock(submit=mock.Mock(side_effect=RuntimeError("broken")))
        with mock.patch.object(self.mod, "classify_to_dict", side_effect=ValueError("x")):
            w.open_path(self.tmp / "d.step")
        self.assertIsNone(w._pool)

    def test_checking_on_the_pool(self):
        w = self.w
        w._pending_path = w.path = self.tmp / "e.step"
        w._future = None
        w._check_classification()                             # nothing pending
        pending = cf.Future()
        w._future = pending
        w._check_classification()                             # not done yet
        self.assertIs(w._future, pending)
        pending.set_exception(cf.process.BrokenProcessPool("died"))
        with mock.patch.object(w, "_warm_pool") as warm:
            w._check_classification()
        warm.assert_called_once()
        self.assertIsNone(w._future)
        self.assertIn("Could not classify", w.banner.text())
        done = cf.Future()
        done.set_result({"kind": "machined"})
        w._future = done
        with mock.patch.object(self.mod, "from_dict",
                               return_value=SimpleNamespace(kind="machined", reason="solid")):
            w._check_classification()
        self.assertEqual(w.mode, "machined")

    def test_warm_pool_falls_back(self):
        w = self.w
        saved = w._pool
        self.addCleanup(setattr, w, "_pool", saved)
        with mock.patch.object(self.mod.cf, "ProcessPoolExecutor", side_effect=OSError("no")):
            w._warm_pool()
        self.assertIsNone(w._pool)
        self.assertTrue(self.mod._warm_child())

    def test_set_mode(self):
        w = self.w
        w.set_mode("sheet_metal", user=True)                  # no part: nothing
        self.load.assert_not_called()
        w.path = self.tmp / "f.step"
        w.detected = SimpleNamespace(kind="machined", reason="solid block")
        with mock.patch.object(w.sheet.view3d, "set_highlight", side_effect=RuntimeError):
            w.set_mode("sheet_metal", user=True)
        self.assertIn("by hand", w.banner.text())
        self.assertIn("solid block", w.banner.text())
        self.load.assert_called_once_with(w.path)
        w.set_mode("sheet_metal", user=True)                  # already loaded there
        self.load.assert_called_once()
        w.detected = None
        w.set_mode("machined", user=True)
        self.assertIn("by hand", w.banner.text())
        self.process.assert_called_once()

    # -- the tuning library, menus, drag and drop ---------------------------

    def test_tuning(self):
        w = self.w
        with mock.patch.object(w.mach, "open_tuning") as ot:
            w._open_tuning()
        ot.assert_called_once()
        w._opened_from_library(str(self.tmp / "g.step"))
        self.assertEqual((w.mode, w.lbl_file.text()), ("machined", "g.step"))
        self.assertIn("tuning library", w.banner.text())

    def test_export_sheet(self):
        w = self.w
        with mock.patch.object(QMessageBox, "information") as info:
            w._export_sheet()
        info.assert_called_once()
        w.mode = "sheet_metal"
        with mock.patch.object(w.sheet.btn_export, "isEnabled", return_value=True), \
                mock.patch.object(w.sheet, "export") as ex:
            w._export_sheet()
        ex.assert_called_once()

    def test_about(self):
        with mock.patch.object(QMessageBox, "information") as info:
            self.w._about()
        self.assertIn("No network", info.call_args[0][2])

    def test_drag_and_drop(self):
        w = self.w

        def event(*paths):
            data = QMimeData()
            data.setUrls([QUrl.fromLocalFile(p) for p in paths])
            return mock.Mock(mimeData=mock.Mock(return_value=data))

        e = event("C:/x/notes.txt")
        w.dragEnterEvent(e)
        e.acceptProposedAction.assert_not_called()
        e = event("C:/x/notes.txt", "C:/x/part.STEP")
        w.dragEnterEvent(e)
        e.acceptProposedAction.assert_called_once()
        with mock.patch.object(w, "open_path") as op:
            w.dropEvent(event("C:/x/notes.txt", "C:/x/part.stp", "C:/x/other.step"))
        op.assert_called_once_with(Path("C:/x/part.stp"))
        empty = mock.Mock(mimeData=mock.Mock(return_value=QMimeData()))
        w.dragEnterEvent(empty)
        empty.acceptProposedAction.assert_not_called()


@unittest.skipUnless(HAVE, "needs OCP and PySide6")
class TestRunAndClose(unittest.TestCase):
    def test_run_shows_the_window_and_opens_the_file(self):
        from partestimator import app as app_mod

        qapp = QApplication.instance() or QApplication([])
        win = mock.Mock()
        with mock.patch.object(app_mod, "UnifiedWindow", return_value=win), \
                mock.patch.object(QApplication, "exec", return_value=0), \
                mock.patch("partestimator.theme.apply"):
            self.assertEqual(app_mod.run(Path("C:/x/part.step")), 0)
            end = time.time() + 2
            while not win.open_path.called and time.time() < end:
                qapp.processEvents()
                time.sleep(0.02)
        win.show.assert_called_once()
        win.open_path.assert_called_once_with(Path("C:/x/part.step"))

    def test_close_survives_a_failing_tool_and_pool(self):
        from partestimator import app as app_mod

        QApplication.instance() or QApplication([])
        w = app_mod.UnifiedWindow()
        bad = mock.Mock(shutdown=mock.Mock(side_effect=RuntimeError("gone")))
        real = w._pool
        w._pool = bad
        with mock.patch.object(w.sheet, "close", side_effect=RuntimeError("x")):
            w.close()
        bad.shutdown.assert_called_once()
        if real is not None:
            real.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    unittest.main()
