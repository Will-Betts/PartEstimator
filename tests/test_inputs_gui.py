"""The input widgets (issues #2, #3, #4), driven as a user would.

Real PySide6, no 3D viewer, so this runs offscreen: the material picker and
its search, the mouse-wheel guard, and the stock controls.
"""
from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

if os.name != "nt" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QDoubleSpinBox,
        QScrollArea,
        QVBoxLayout,
        QWidget,
    )

    HAVE_QT = True
except Exception:  # noqa: BLE001
    HAVE_QT = False

MM = 25.4


def box(l_in, w_in, t_in):
    return SimpleNamespace(extents_mm=[l_in * MM, w_in * MM, t_in * MM],
                           axes=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], center_mm=[0, 0, 0],
                           volume_mm3=l_in * w_in * t_in * MM ** 3)


def cyl(d_in, h_in):
    return SimpleNamespace(radius_mm=d_in / 2 * MM, height_mm=h_in * MM,
                           axis_direction=[0, 0, 1], axis_point_mm=[0, 0, 0])


@unittest.skipUnless(HAVE_QT, "needs PySide6")
class QtCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])


class TestMaterialPicker(QtCase):
    def setUp(self):
        from partestimator import materials_db as db
        from partestimator.ui_common import material_picker

        self.names = db.machining_names()
        self.p = material_picker(self.names, ["Other"])
        self.seen = []
        self.p.currentTextChanged.connect(self.seen.append)

    def test_two_short_lists_instead_of_one_long_one(self):
        self.assertGreater(self.p.group_combo.count(), 5)
        self.assertLess(self.p.item_combo.count(), 20)
        self.assertEqual(self.p.group_combo.itemText(0), "Aluminum")

    def test_alloy_labels_drop_the_family_word(self):
        self.p.setCurrentText("Aluminum 6061-T6")
        self.assertEqual(self.p.item_combo.currentText(), "6061-T6")
        self.assertEqual(self.p.currentText(), "Aluminum 6061-T6")
        self.assertEqual(self.seen, ["Aluminum 6061-T6"])

    def test_changing_family_selects_its_first_alloy(self):
        self.p.group_combo.setCurrentText("Titanium")
        self.assertTrue(self.p.currentText().startswith("Titanium"))
        self.assertEqual(self.seen[-1], self.p.currentText())

    def test_search_by_keywords(self):
        self.assertEqual(self.p.matches("stainless 304"), ["Stainless 304"])
        self.assertIn("Aluminum 7075-T6", self.p.matches("7075"))
        self.assertEqual(self.p.matches("unobtainium"), [])

    def test_enter_picks_the_first_match_and_clears_the_box(self):
        self.p.search.setText("inconel 718")
        self.p.search.returnPressed.emit()
        self.assertEqual(self.p.currentText(), "Inconel 718 (annealed)")
        self.assertEqual(self.p.group_combo.currentText(), "Superalloy")
        self.assertEqual(self.p.search.text(), "")

    def test_unknown_value_is_ignored(self):
        before = self.p.currentText()
        self.p.setCurrentText("Unobtainium")
        self.assertEqual(self.p.currentText(), before)

    def test_other_is_available(self):
        self.p.setCurrentText("Other")
        self.assertEqual(self.p.currentText(), "Other")


class TestWheelGuard(QtCase):
    def wheel(self, widget, up=True):
        pos = QPointF(widget.rect().center())
        ev = QWheelEvent(pos, QPointF(widget.mapToGlobal(pos.toPoint())), QPoint(0, 0),
                         QPoint(0, 120 if up else -120), Qt.NoButton, Qt.NoModifier,
                         Qt.NoScrollPhase, False)
        QApplication.sendEvent(widget, ev)

    def test_wheel_over_inputs_changes_nothing(self):
        from partestimator.ui_common import guard_wheel

        area = QScrollArea()
        inner = QWidget()
        lay = QVBoxLayout(inner)
        spin = QDoubleSpinBox()
        spin.setRange(0, 100)
        spin.setValue(50)
        combo = QComboBox()
        combo.addItems(["a", "b", "c"])
        combo.setCurrentIndex(1)
        lay.addWidget(spin)
        lay.addWidget(combo)
        area.setWidget(inner)
        self.assertEqual(guard_wheel(area), 2)
        for up in (True, False):
            self.wheel(spin, up)
            self.wheel(combo, up)
        self.assertEqual(spin.value(), 50)
        self.assertEqual(combo.currentIndex(), 1)

    def test_unguarded_spin_box_would_have_changed(self):
        """The control: proves the wheel events above are real ones."""
        spin = QDoubleSpinBox()
        spin.setRange(0, 100)
        spin.setValue(50)
        spin.show()
        spin.setFocus()
        self.wheel(spin)
        spin.hide()
        self.assertNotEqual(spin.value(), 50)


class TestStockControls(QtCase):
    def setUp(self):
        from partestimator.machining.ui import stock_panel as sp

        self.sp = sp
        self.w = sp.StockControls()
        self.emitted = []
        self.w.stockChanged.connect(self.emitted.append)

    def test_new_part_opens_on_the_recommended_size(self):
        s = self.w.set_part(box(4.0, 2.5, 0.75), cyl(4.5, 0.8), "rectilinear")
        self.assertEqual(s.basis, "recommended")
        self.assertEqual(self.w.type_combo.currentText(), self.sp.PLATE)
        self.assertAlmostEqual(self.w.spin_a.value(), 1.0)       # 0.75 + .125 -> 1.0
        self.assertAlmostEqual(self.w.spin_l.value(), 5.0)
        self.assertEqual(self.emitted[-1].basis, "recommended")

    def test_typing_a_size_makes_it_custom_and_reports_it(self):
        self.w.set_part(box(4.0, 2.5, 0.75), None, "rectilinear")
        self.w.spin_a.setValue(1.5)
        s = self.emitted[-1]
        self.assertEqual(s.basis, "custom")
        self.assertAlmostEqual(s.thickness_or_diameter_in, 1.5)
        self.assertAlmostEqual(s.volume_in3, 1.5 * 3.5 * 5.0)

    def test_snap_to_minimum_and_back(self):
        self.w.set_part(box(4.0, 2.5, 0.75), None, "rectilinear")
        self.w.btn_minimum.click()
        self.assertEqual(self.emitted[-1].basis, "minimum")
        self.assertAlmostEqual(self.w.spin_l.value(), 4.0)
        self.w.btn_recommended.click()
        self.assertEqual(self.emitted[-1].basis, "recommended")
        self.assertAlmostEqual(self.w.spin_l.value(), 5.0)

    def test_too_small_is_warned(self):
        self.w.set_part(box(4.0, 2.5, 0.75), None, "rectilinear")
        self.w.spin_a.setValue(0.5)
        self.assertIn("does not enclose", self.w.lbl_status.text())

    def test_round_bar_needs_a_cylinder_fit(self):
        self.w.set_part(box(4.0, 2.5, 0.75), None, "rectilinear")
        self.assertFalse(self.w.type_combo.model().item(1).isEnabled())
        self.w.set_part(box(4.0, 2.0, 2.0), cyl(2.0, 4.0), "cylindrical")
        self.assertEqual(self.w.type_combo.currentText(), self.sp.ROUND)
        self.assertTrue(self.emitted[-1].is_round)
        self.assertFalse(self.w.spin_b.isVisibleTo(self.w))

    def test_extrusion_picks_the_best_fitting_profile(self):
        """6 x 0.9 x 0.9: the lightest profile that fits, recommended length."""
        self.w.set_part(box(6.0, 0.9, 0.9), None, "rectilinear")
        self.w.type_combo.setCurrentText(self.sp.EXTRUSION)
        s = self.emitted[-1]
        self.assertTrue(s.is_extrusion)
        self.assertTrue(s.fits)
        self.assertAlmostEqual(s.min_length_in, 6.0)
        self.assertAlmostEqual(self.w.spin_l.value(), 7.0)
        self.assertFalse(self.w.spin_a.isEnabled())
        self.assertIn("minimum 6.000", self.w.lbl_status.text())

    def test_a_solid_part_gets_solid_bar_not_an_angle(self):
        self.w.set_part(box(6.0, 0.9, 0.9), None, "rectilinear",
                        part_volume_in3=6.0 * 0.9 * 0.9)
        self.w.type_combo.setCurrentText(self.sp.EXTRUSION)
        s = self.emitted[-1]
        self.assertEqual((s.profile.shape, s.profile.name), ("flat_bar", "1 x 1 square"))
        self.assertTrue(s.enough_material)

    def test_extrusion_length_snaps_and_edits(self):
        self.w.set_part(box(6.0, 0.9, 0.9), None, "rectilinear")
        self.w.type_combo.setCurrentText(self.sp.EXTRUSION)
        self.w.btn_minimum.click()
        self.assertAlmostEqual(self.emitted[-1].length_in, 6.0)
        self.w.spin_l.setValue(12.0)
        s = self.emitted[-1]
        self.assertAlmostEqual(s.length_in, 12.0)
        self.assertAlmostEqual(s.volume_in3, s.profile.area_in2 * 12.0)

    def test_choosing_another_profile(self):
        self.w.set_part(box(6.0, 0.9, 0.9), None, "rectilinear")
        self.w.type_combo.setCurrentText(self.sp.EXTRUSION)
        self.w.profile.setCurrentText("rect_tube|2 x 2 x 1/8")
        s = self.emitted[-1]
        self.assertEqual(s.profile.name, "2 x 2 x 1/8")
        self.assertEqual(s.profile.shape, "rect_tube")


if __name__ == "__main__":
    unittest.main()
