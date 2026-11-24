"""The stock panel's remaining paths: extrusions set, snapped and refreshed,
clearing, the custom library editor and its errors, and the orientation
controls asked to act with nothing to act on.

Needs Qt (and the extrusion catalogue, which is plain Python).
"""
from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

if os.name != "nt" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication

    HAVE = True
except Exception:  # noqa: BLE001
    HAVE = False

MM = 25.4


def box(l_in, w_in, t_in):
    return SimpleNamespace(extents_mm=np.array([l_in, w_in, t_in]) * MM,
                           axes=np.eye(3), center_mm=np.zeros(3),
                           volume_mm3=l_in * w_in * t_in * MM ** 3,
                           corners_mm=np.zeros((8, 3)))


@unittest.skipUnless(HAVE, "needs Qt")
class TestStockPanelMore(unittest.TestCase):
    def setUp(self):
        from partestimator.machining.geometry import extrusions as ex
        from partestimator.machining.geometry import stock as st
        from partestimator.machining.ui import stock_panel as sp

        QApplication.instance() or QApplication([])
        self.ex, self.st, self.sp = ex, st, sp
        self.w = sp.StockControls()
        self.got = []
        self.w.stockChanged.connect(self.got.append)

    def angle(self):
        return self.ex.find("angle", "AND10133-2004")

    def test_set_stock(self):
        w = self.w
        w.set_stock(None)
        self.assertEqual(self.got, [])
        rect = box(6, 2, 2)
        w.set_part(rect, None, "rectilinear")
        w.set_stock(self.ex.extrusion_stock(self.angle(), rect, length_in=7.0))
        self.assertEqual(w.type_combo.currentText(), self.sp.EXTRUSION)
        self.assertIn("AND10133-2004", w.profile.currentText())

    def test_extrusion_snap_and_refresh(self):
        w = self.w
        rect = box(6, 2, 2)
        w.set_part(rect, None, "rectilinear")
        w.type_combo.setCurrentText(self.sp.EXTRUSION)
        w.profile.setCurrentText(f"angle|{self.angle().name}")
        rec = w.stock().recommended_length_in
        w.snap(self.st.MINIMUM)
        self.assertAlmostEqual(w.stock().length_in, w.stock().min_length_in)
        w.snap(self.st.RECOMMENDED)
        self.assertAlmostEqual(w.stock().length_in, rec)
        w.refresh_allowances()                               # at the recommended: recut
        self.assertAlmostEqual(w.stock().length_in, rec)
        w.spin_l.setValue(rec + 2.0)                         # typed: kept
        w.refresh_allowances()
        self.assertAlmostEqual(w.stock().length_in, rec + 2.0)
        w.type_combo.setCurrentText(self.sp.PLATE)
        w._stock = self.st.custom_rectilinear(1, 3, 7)
        w.type_combo.blockSignals(True)
        w.type_combo.setCurrentText(self.sp.EXTRUSION)
        w.type_combo.blockSignals(False)
        before = len(self.got)
        w.snap(self.st.MINIMUM)                              # not an extrusion yet: nothing
        self.assertEqual(len(self.got), before)

    def test_clear_and_nothing_to_act_on(self):
        w = self.w
        w.refresh_allowances()                               # no stock: nothing
        w.align(np.array([0, 0, 1.0]), 0)
        w.square_to_bar(np.array([0, 0, 1.0]))
        w.reset_orientation()
        w._on_angles_changed()
        w._reorient()
        w.set_part(box(6, 2, 1), None, "rectilinear")
        w.clear()
        self.assertFalse(w.isEnabled())
        self.assertIsNone(w.stock())

    def test_custom_library_editing(self):
        w = self.w
        with mock.patch.object(self.ex, "ensure_custom_file", side_effect=OSError("ro")):
            w.edit_custom_library()
        self.assertIn("Cannot create", w.lbl_custom.text())
        with mock.patch("PySide6.QtGui.QDesktopServices.openUrl") as url, \
                mock.patch.object(self.ex, "ensure_custom_file",
                                  return_value=self.ex.CUSTOM_PATH):
            w.edit_custom_library()
        url.assert_called_once()
        self.assertIn("Editing", w.lbl_custom.text())
        lib = SimpleNamespace(profiles=(), errors=[f"line {k}: bad" for k in range(8)],
                              path=self.ex.CUSTOM_PATH, exists=True)
        w._show_custom_status(lib)
        self.assertIn("3 more rows skipped", w.lbl_custom.text())


if __name__ == "__main__":
    unittest.main()
