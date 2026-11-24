"""The sheet metal window, driven as a user would: open a bent part, switch
views and colouring, recompute, pick the sheet face, export, cancel, and the
ways each can fail.

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
    from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

    HAVE = True
except Exception:  # noqa: BLE001
    HAVE = False


@unittest.skipUnless(HAVE, "needs the CAD kernel and Qt")
class TestSheetWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from partestimator.smt.gui.app import MainWindow
        from tests import occparts

        cls.app = QApplication.instance() or QApplication([])
        cls.tmp = Path(tempfile.mkdtemp())
        with contextlib.redirect_stdout(io.StringIO()):
            cls.part = occparts.countersunk_l_bracket(cls.tmp)
            cls.blank = occparts.flat_blank(cls.tmp)
        cls.w = MainWindow()

    @classmethod
    def tearDownClass(cls):
        cls.w.close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def wait(self, timeout=120):
        w = self.w
        t0 = time.time()
        self.app.processEvents()
        while w.btn_cancel.isEnabled():
            self.app.processEvents()
            time.sleep(0.02)
            self.assertLess(time.time() - t0, timeout, "never finished")
        for _ in range(5):
            self.app.processEvents()

    def open(self, path=None):
        self.w.load(path or self.part)
        self.wait()
        self.assertIsNotNone(self.w.part)

    def test_open_through_the_dialog(self):
        w = self.w
        with mock.patch.object(QFileDialog, "getOpenFileName", return_value=("", "")), \
                mock.patch.object(w, "load") as load:
            w.open_file()
            load.assert_not_called()
        w.opts.sheet_face = 3
        with mock.patch.object(QFileDialog, "getOpenFileName",
                               return_value=(str(self.part), "")):
            w.open_file()
        self.wait()
        self.assertIsNone(w.opts.sheet_face)           # a new file forgets the picked face
        self.assertIn("bend", w.activity.toPlainText())

    def test_views_colour_and_box(self):
        self.open()
        w = self.w
        w.show_view("flat")
        self.assertEqual(w._current_view, "flat")
        w._colour_toggled(False)
        self.assertFalse(w._colour_pref["flat"])
        w.show_view("folded")
        w.show_view("no such view")                     # falls back to one there is
        # A box that does not sit snug on the blank says so.
        real = w.part.thickness
        w.part.thickness = 9.0
        try:
            w.show_view("flat")
        finally:
            w.part.thickness = real
        self.assertIn("detected", " ".join(w.view3d._box_labels or ()))

    def test_recompute(self):
        self.open()
        w = self.w
        w._recompute()
        self.assertIsNone(w.part.flat)                  # kernel products now stale
        with mock.patch("partestimator.smt.model2d.recompute", side_effect=ValueError("bad K")):
            w._recompute()
        self.assertIn("Recompute failed", w.statusBar().currentMessage())
        with mock.patch.object(w, "_refresh", side_effect=RuntimeError("panel")):
            w._recompute()
        self.assertIn("failed to update", w.statusBar().currentMessage())
        w.opts.cost_per_kg = 3.0
        w._refresh()
        self.assertNotEqual(w.blank._fields["cost"].text(), "--")
        w.opts.cost_per_kg = None
        w._refresh()
        self.assertEqual(w.blank._fields["cost"].text(), "--")

    def test_reopen_and_stall_warning(self):
        self.open(self.blank)
        w = self.w
        w._reopen()
        self.wait()
        self.assertTrue(w.part.is_blank)
        w._warn_stalled()
        self.assertIn("Still working", w.statusBar().currentMessage())

    def test_pick_the_sheet_face(self):
        self.open()
        w = self.w
        w._face_picked(2)                               # not armed: ignored
        self.assertIsNone(w.opts.sheet_face)
        w._arm_picking()
        self.assertEqual(w._current_view, "folded")
        sheet = next(f.index for f in w.part.features.values() if f.role.value == "sheet")
        w._face_picked(sheet)
        self.wait()
        self.assertEqual(w.opts.sheet_face, sheet)
        self.assertTrue(any("you selected" in n for n in w.part.notes))
        w.opts.sheet_face = None

    def test_export_and_cancel(self):
        self.open()
        w = self.w
        out = self.tmp / "export"
        out.mkdir(exist_ok=True)
        with mock.patch.object(QFileDialog, "getExistingDirectory", return_value=""):
            w.export()
        with mock.patch.object(QFileDialog, "getExistingDirectory", return_value=str(out)):
            w.export()
        self.wait()
        self.assertTrue(any(p.suffix == ".dxf" for p in out.iterdir()))
        self.assertIn("Wrote", w.activity.toPlainText())
        job = mock.Mock()
        w._job = job
        w._cancel()
        job.cancel.assert_called_once()
        w._job = None

    def test_failures(self):
        w = self.w
        w._failed("Cancelled.", "")
        self.assertEqual(w.statusBar().currentMessage(), "Cancelled.")
        with mock.patch.object(QMessageBox, "critical") as crit:
            w._failed("Exploded", "Traceback: ...")
        crit.assert_called_once()
        self.assertIn("Traceback", w.activity.toPlainText())
        with mock.patch.object(QMessageBox, "information") as info:
            w._about()
        info.assert_called_once()

    def test_the_3d_view(self):
        self.open()
        v = self.w.view3d
        v.set_show_edges(False)
        self.assertIsNone(v._edges_actor)
        v.set_show_edges(True)
        v.set_show_box(False)
        v.set_show_box(True)
        v.set_box([0, 0, 0], [10, 5, 1], ("10", "5", "1"))
        v.set_box(None)
        v.set_highlight(2)
        v.set_highlight(None)
        for name in ("top", "front", "right", "iso", "nonsense"):
            v.set_view(name)
        v.fit()
        self.assertTrue(v.has_scene())
        # Clicks: a still click picks, a drag does not, a failed read is ignored.
        picked = []
        v.picked.connect(picked.append)
        iren = v.plotter.iren
        with mock.patch.object(iren, "get_event_position", return_value=(5, 5)), \
                mock.patch.object(v, "pick", return_value=7):
            v._on_press()
            v._on_release()
        with mock.patch.object(iren, "get_event_position", side_effect=[(5, 5), (80, 5)]), \
                mock.patch.object(v, "pick", return_value=7):
            v._on_press()
            v._on_release()
        with mock.patch.object(iren, "get_event_position", side_effect=RuntimeError):
            v._on_press()
            v._press = (1, 1)
            v._on_release()
        self.assertEqual(picked, [7])
        size = v.plotter.window_size
        hit = v.pick(size[0] // 2, size[1] // 2)
        self.assertTrue(hit is None or hit >= 0)
        self.assertIsNone(v.pick(-50, -50))
        # An older PyVista without smooth-shading options: drawn plainly.
        real = v.plotter.add_mesh

        def old_add_mesh(*a, **kw):
            if "split_sharp_edges" in kw:
                raise TypeError("unexpected keyword")
            return real(*a, **kw)

        with mock.patch.object(v.plotter, "add_mesh", side_effect=old_add_mesh):
            v.set_scene(v._scene, v._roles)
        self.assertIsNotNone(v._surface)

    def test_nothing_open(self):
        from PySide6.QtGui import QCloseEvent

        w = self.w
        saved = (w.part, w.flat2d, w.scenes)
        w.part, w.flat2d, w.scenes = None, None, {}
        try:
            w._recompute()
            w._arm_picking()
            w.export()
            w.show_view("flat")
            w._apply_box(None)
        finally:
            w.part, w.flat2d, w.scenes = saved
        # Closing cancels a running job and survives VTK refusing to close.
        # (The real teardown is left to tearDownClass: one VTK window per run.)
        job = mock.Mock()
        w._job = job
        with mock.patch.object(w.view3d, "close", side_effect=RuntimeError("vtk")), \
                mock.patch("PySide6.QtWidgets.QMainWindow.closeEvent"):
            w.closeEvent(QCloseEvent())
        job.cancel.assert_called()
        w._job = None


@unittest.skipUnless(HAVE, "needs the CAD kernel and Qt")
class TestLegendAndRun(unittest.TestCase):
    def test_legend(self):
        from partestimator.smt.features import Role
        from partestimator.smt.gui.app import _Legend

        QApplication.instance() or QApplication([])
        legend = _Legend()
        legend.show_roles({Role.SHEET, Role.BEND_OUTER, Role.UNKNOWN})
        texts = [legend._lay.itemAt(i).widget().text() for i in range(legend._lay.count())
                 if legend._lay.itemAt(i).widget() is not None]
        self.assertIn("bend", texts)
        self.assertIn("NOT RECOGNISED", texts)
        legend.show_roles(set())

    def test_run(self):
        from partestimator.smt.gui import app as sheet_app

        qapp = QApplication.instance() or QApplication([])
        win = mock.Mock()
        with mock.patch.object(sheet_app, "MainWindow", return_value=win), \
                mock.patch.object(QApplication, "exec", return_value=0):
            self.assertEqual(sheet_app.run(None, Path("C:/x/p.step")), 0)
            end = time.time() + 2
            while not win.load.called and time.time() < end:
                qapp.processEvents()
                time.sleep(0.02)
        win.load.assert_called_once_with(Path("C:/x/p.step"))


if __name__ == "__main__":
    unittest.main()
