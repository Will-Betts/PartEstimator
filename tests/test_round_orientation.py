"""Round bar set another way round, in the real window: the stepped shaft's
flat set square to the bar turns the bar across the shaft, the turned profile
is measured again about the new axis (in a child process), and Automatic puts
everything back.

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

if os.name != "nt" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    import OCP  # noqa: F401
    from PySide6.QtWidgets import QApplication

    HAVE = True
except Exception:  # noqa: BLE001
    HAVE = False


@unittest.skipUnless(HAVE, "needs the CAD kernel and Qt")
class TestRoundBarInTheWindow(unittest.TestCase):
    def setUp(self):
        import numpy as np

        from partestimator.machining.geometry import orientation, pipeline
        from partestimator.machining.ui.main_window import MainWindow
        from tests import occparts

        self.np = np
        self.app = QApplication.instance() or QApplication([])
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        with contextlib.redirect_stdout(io.StringIO()):
            shaft = occparts.stepped_shaft(self.tmp)
        self.w = MainWindow()
        self.addCleanup(self.w.close)
        self.res = pipeline.compute_analysis(str(shaft), options=self.w._analysis_options())
        self.w._current_path = str(shaft)
        self.w._on_analysis_finished(self.res)
        pts, tris = self.res["render_points"], self.res["render_triangles"]
        labels = orientation.face_labels(tris, len(pts))
        axis = np.asarray(self.res["cylinder"].axis_direction)
        self.flat = next(
            f[0] for f in (orientation.planar_face(pts, tris, labels,
                                                   int(np.flatnonzero(labels == lab)[0]))
                           for lab in np.unique(labels))
            if f is not None and abs(np.dot(f[0], axis)) < 0.05)

    def wait(self):
        t0 = time.time()
        while self.w.turning_busy():
            self.app.processEvents()
            time.sleep(0.02)
            self.assertLess(time.time() - t0, 120)
        self.app.processEvents()

    def test_flat_square_to_the_bar_remeasures_the_turning(self):
        w, sc = self.w, self.w.stock_controls
        self.assertEqual(sc.type_combo.currentText(), "Round bar")
        auto_turning, before = w.results["turning"], w._current_estimate().door_to_door_minutes
        self.assertGreater(auto_turning.turned_fraction, 0.5)

        sc.btn_orient.click()
        sc.set_part_face(self.flat)
        cyl = w.results["cylinder"]
        self.assertIs(cyl, sc.cylinder())
        self.assertAlmostEqual(abs(float(self.np.dot(cyl.axis_direction, self.flat))), 1.0)
        self.assertTrue(w.turning_busy())
        self.wait()
        tg = w.results["turning"]
        self.assertIsNot(tg, auto_turning)
        self.assertAlmostEqual(abs(float(self.np.dot(tg.axis_direction, self.flat))), 1.0,
                               places=3)
        self.assertLess(tg.turned_fraction, auto_turning.turned_fraction)
        self.assertGreater(w._current_estimate().door_to_door_minutes, before)

        sc.orient_dialog.btn_automatic.click()
        self.assertIs(w.results["turning"], auto_turning)
        self.assertFalse(w.turning_busy())
        self.assertAlmostEqual(w._current_estimate().door_to_door_minutes, before)

    def test_a_new_part_cancels_a_remeasure_in_flight(self):
        sc = self.w.stock_controls
        sc.btn_orient.click()
        sc.set_part_face(self.flat)
        self.assertTrue(self.w.turning_busy())
        self.w._on_analysis_finished(dict(self.res))
        self.assertFalse(self.w.turning_busy())


if __name__ == "__main__":
    unittest.main()
