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
        self.first = self.p.values()[0]
        self.p.currentTextChanged.connect(self.seen.append)

    @staticmethod
    def submenus(menu):
        # Not QAction.menu(): PySide6 deletes the submenu when that throwaway
        # wrapper is collected.
        from PySide6.QtWidgets import QMenu

        return {m.title(): m for m in menu.findChildren(QMenu)}

    def test_one_dropdown_starting_on_the_first_alloy(self):
        self.assertTrue(self.first.startswith("Aluminum "))
        from PySide6.QtWidgets import QComboBox as Combo

        self.assertEqual(len(self.p.findChildren(Combo)), 1)
        self.assertEqual(self.p.currentText(), self.first)
        self.assertEqual(self.p.line.text(), self.first)

    def test_menu_is_families_with_alloys_nested(self):
        menu = self.p.build_menu()
        subs = self.submenus(menu)
        self.assertGreater(len(subs), 5)
        self.assertEqual(menu.actions()[0].text(), "Aluminum")
        labels = [a.text() for a in subs["Aluminum"].actions()]
        self.assertIn("6061-T6", labels)            # the family word is dropped
        self.assertLess(len(labels), 20)
        # "Other" is a single entry, not a submenu holding one item.
        self.assertIn("Other", [a.text() for a in menu.actions()])
        self.assertNotIn("Other", subs)

    def test_picking_from_a_submenu(self):
        menu = self.p.build_menu()
        act = self.submenus(menu)["Titanium"].actions()[0]
        act.trigger()
        self.assertTrue(self.p.currentText().startswith("Titanium"))
        self.assertEqual(self.seen, [self.p.currentText()])
        self.assertEqual(self.p.line.text(), self.p.currentText())
        # Reopened, the menu shows where you are.
        menu = self.p.build_menu()
        again = self.submenus(menu)["Titanium"]
        self.assertTrue(again.menuAction().font().bold())
        self.assertTrue(again.actions()[0].isChecked())

    def test_alloy_labels_drop_the_family_word(self):
        self.p.setCurrentText("Stainless 304")
        self.assertEqual(self.p.currentText(), "Stainless 304")
        self.assertEqual(self.seen, ["Stainless 304"])

    def test_search_by_keywords(self):
        self.assertEqual(self.p.matches("stainless 304"), ["Stainless 304"])
        self.assertIn("Aluminum 7075-T6", self.p.matches("7075"))
        self.assertEqual(self.p.matches("unobtainium"), [])

    def test_typing_fires_nothing_until_a_pick(self):
        self.p.line.setText("inconel")
        self.p.line.textEdited.emit("inconel")
        self.assertEqual(self.seen, [])
        self.assertEqual(self.p.currentText(), self.first)

    def test_enter_picks_the_first_match(self):
        self.p.line.setText("inconel 718")
        self.p.line.returnPressed.emit()
        self.assertEqual(self.p.currentText(), "Inconel 718 (annealed)")
        self.assertEqual(self.p.group_of(self.p.currentText()), "Superalloy")
        self.assertEqual(self.p.line.text(), "Inconel 718 (annealed)")

    def test_a_search_that_finds_nothing_puts_the_value_back(self):
        self.p.line.setText("unobtainium")
        self.p.line.returnPressed.emit()
        self.assertEqual(self.p.currentText(), self.first)
        self.assertEqual(self.p.line.text(), self.first)
        self.assertEqual(self.seen, [])

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
        self.assertIn("Minimum 6.000", self.w.spin_l.toolTip())
        self.assertFalse(self.w.lbl_status.isVisibleTo(self.w))     # nothing wrong

    def test_a_solid_part_gets_solid_bar_not_an_angle(self):
        self.w.set_part(box(6.0, 0.9, 0.9), None, "rectilinear",
                        part_volume_in3=6.0 * 0.9 * 0.9)
        self.w.type_combo.setCurrentText(self.sp.EXTRUSION)
        s = self.emitted[-1]
        self.assertEqual((s.profile.shape, s.profile.name), ("hex_bar", "1-1/2 A/F"))
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

    def test_choosing_a_standard_profile_by_part_number(self):
        """Typing the SAE number finds the AND die it replaced."""
        self.w.set_part(box(6.0, 1.2, 0.9), None, "rectilinear")
        self.w.type_combo.setCurrentText(self.sp.EXTRUSION)
        self.assertEqual(self.w.profile.matches("AMS10134-1205"), ["angle|AND10134-1205"])
        self.w.profile.line.setText("AMS10134-1205")
        self.w.profile.line.returnPressed.emit()
        s = self.emitted[-1]
        self.assertEqual(s.profile.name, "AND10134-1205")
        # The box shows the part number and size, not the internal key.
        self.assertTrue(self.w.profile.line.text().startswith("AND10134-1205  Angle"))
        self.assertAlmostEqual(s.profile.area_in2, 0.202)
        self.assertIn("AMS10134-1205", self.w.profile.toolTip())

    def test_custom_library_reloads_into_the_picker(self):
        import shutil
        import tempfile
        from pathlib import Path

        from partestimator.machining.geometry import extrusions as ex

        tmp = Path(tempfile.mkdtemp(prefix="custom_gui_"))
        saved = ex.CUSTOM_PATH
        ex.CUSTOM_PATH = tmp / "custom_extrusions.csv"

        def restore():
            ex.CUSTOM_PATH = saved
            ex.custom_library(reload=True)
            shutil.rmtree(tmp, True)
        self.addCleanup(restore)

        self.w.set_part(box(6.0, 1.2, 0.9), None, "rectilinear")
        self.w.type_combo.setCurrentText(self.sp.EXTRUSION)
        ex.ensure_custom_file()
        with ex.CUSTOM_PATH.open("a", encoding="utf-8") as fh:
            fh.write("SHOP-7,angle,1.5,1.25,,0.1,,,,our die\n")
            fh.write("BROKEN,angle,1.5,,,,,,,\n")
        self.w.reload_custom_library()
        self.assertIn("1 profile.", self.w.btn_reload_custom.toolTip())
        self.assertIn("Skipped line", self.w.lbl_custom.text())
        self.assertEqual(self.w.profile.matches("custom shop-7"), ["angle|SHOP-7"])
        self.w.profile.setCurrentText("angle|SHOP-7")
        s = self.emitted[-1]
        self.assertEqual(s.profile.name, "SHOP-7")
        self.assertTrue(s.profile.custom)

        # The die is deleted from the file: the picker lets go of it too.
        ex.CUSTOM_PATH.write_text("name,shape,a_in\n", encoding="utf-8")
        self.w.reload_custom_library()
        self.assertNotEqual(self.emitted[-1].profile.name, "SHOP-7")
        self.assertNotIn("angle|SHOP-7", self.w.profile.values())


@unittest.skipUnless(HAVE_QT, "needs PySide6")
class TestStockOrientation(QtCase):
    """A 6 x 4 x 1 block stored turned 20 degrees in its file, so the box
    the file's own axes give is too big until the part is squared up."""

    def setUp(self):
        import numpy as np

        from partestimator.machining.geometry import orientation as o
        from partestimator.machining.ui import stock_panel as sp

        self.np, self.o, self.sp = np, o, sp
        pts = np.array([[x, y, z] for x in (0, 6) for y in (0, 4) for z in (0, 1)],
                       dtype=float) * MM
        self.pts = pts @ o.rotation_matrix(2, 20).T
        self.skew = o.box_in_frame(self.pts, np.eye(3))
        self.w = sp.StockControls()
        self.emitted, self.picks, self.turned = [], [], []
        self.w.stockChanged.connect(self.emitted.append)
        self.w.pickRequested.connect(self.picks.append)
        self.w.orientationChanged.connect(lambda: self.turned.append(True))
        self.w.set_part(self.skew, None, "rectilinear", points_mm=self.pts)
        self.addCleanup(self.w.orient_dialog.close)

    def dims(self):
        return sorted(float(e) / MM for e in self.w.rect_box().extents_mm)

    def bottom_axis(self):
        """The row of the current box nearest the part's bottom face normal."""
        return int(self.np.argmax(abs(self.w.rect_box().axes @ [0, 0, 1])))

    def step_text(self):
        return self.w.orient_dialog.lbl_step.text()

    def test_one_button_opens_on_the_automatic_box(self):
        self.assertIs(self.w.rect_box(), self.skew)
        self.assertFalse(self.w.is_reoriented())
        self.assertTrue(self.w.btn_orient.isEnabled())
        self.assertTrue(self.w.orient_row.isVisibleTo(self.w))
        self.assertFalse(self.w.lbl_oriented.isVisibleTo(self.w))
        self.assertFalse(self.w.orient_dialog.isVisible())

    def test_the_window_walks_through_part_face_then_stock_face(self):
        self.w.btn_orient.click()
        self.assertTrue(self.w.orient_dialog.isVisible())
        self.assertEqual(self.picks, ["part"])
        self.assertIn("flat face of the part", self.step_text())

        self.w.set_part_face([0, 0, -1])
        self.assertEqual(self.picks[-1], "stock")
        self.assertIn("face of the stock", self.step_text())

        self.w.set_stock_face(self.bottom_axis())
        for got, want in zip(self.dims(), [1.0, 4.0, 6.0]):
            self.assertAlmostEqual(got, want, places=6)
        self.assertTrue(self.w.is_reoriented())
        self.assertTrue(self.turned)
        self.assertIn("top face", self.step_text())
        self.assertEqual(self.picks[-1], "part")           # ready for another pair
        self.assertTrue(self.w.lbl_oriented.isVisibleTo(self.w))
        s = self.emitted[-1]
        self.assertEqual((s.thickness_or_diameter_in, s.width_in, s.length_in),
                         (1.25, 5.0, 7.0))

    def test_a_stock_face_first_does_nothing(self):
        self.w.btn_orient.click()
        self.w.set_stock_face(0)
        self.assertIs(self.w.rect_box(), self.skew)
        self.assertEqual(self.w.picking(), "part")

    def test_a_bad_click_asks_again(self):
        self.w.btn_orient.click()
        self.w.pick_problem("That face is curved.")
        self.assertIn("curved", self.step_text())
        self.assertIn("flat face of the part", self.step_text())

    def test_start_over_and_closing(self):
        self.w.btn_orient.click()
        self.w.set_part_face([0, 0, 1])
        self.w.orient_dialog.btn_start_over.click()
        self.assertEqual(self.w.picking(), "part")
        self.w.orient_dialog.btn_close.click()
        self.assertEqual(self.w.picking(), "")
        self.assertEqual(self.picks[-1], "")

    def test_fine_turns_about_the_stock_axes(self):
        self.w.align([0, 0, 1], self.bottom_axis())
        self.w.spin_rz.setValue(90.0)            # about the thickness: L and W swap
        for got, want in zip(self.dims(), [1.0, 4.0, 6.0]):
            self.assertAlmostEqual(got, want, places=6)
        self.w.spin_rz.setValue(30.0)
        self.assertGreater(self.dims()[2], 6.5)
        self.assertIn("Z +30°", self.w.lbl_oriented.toolTip())
        self.w.spin_rz.setValue(0.0)
        self.w.spin_rx.setValue(90.0)            # about the length: stands it on edge
        ext = [float(e) / MM for e in self.w.rect_box().extents_mm]
        for got, want in zip(ext, [6.0, 1.0, 4.0]):   # rows stay X, Y, Z
            self.assertAlmostEqual(got, want, places=6)

    def test_a_face_that_ends_up_on_a_different_side_is_explained(self):
        """The part's 6-inch end face laid on the stock's big top face: the
        part is longest that way, so it can only be the stock's end."""
        end_normal = self.o.rotation_matrix(2, 20) @ [1, 0, 0]
        self.w.align(end_normal, self.bottom_axis())
        text = self.step_text()
        self.assertIn("end face", text)
        self.assertIn("you picked its top face", text)

    def test_automatic_goes_back(self):
        self.w.align([0, 0, 1], self.bottom_axis())
        self.w.spin_ry.setValue(10.0)
        self.w.btn_auto_orient.click()
        self.assertIs(self.w.rect_box(), self.skew)
        self.assertFalse(self.w.is_reoriented())
        self.assertEqual(self.w.angles(), (0.0, 0.0, 0.0))
        self.assertFalse(self.w.lbl_oriented.isVisibleTo(self.w))

    def test_custom_size_is_kept_when_turned(self):
        self.w.spin_a.setValue(2.0)
        self.assertEqual(self.emitted[-1].basis, "custom")
        self.w.align([0, 0, 1], self.bottom_axis())
        self.assertEqual(self.emitted[-1].basis, "custom")
        self.assertAlmostEqual(self.emitted[-1].thickness_or_diameter_in, 2.0)

    def test_round_bar_hides_it_and_a_new_part_resets_it(self):
        self.w.btn_orient.click()
        self.w.align([0, 0, 1], self.bottom_axis())
        self.w.set_part(box(4.0, 2.0, 2.0), cyl(2.0, 4.0), "cylindrical")
        self.assertFalse(self.w.orient_row.isVisibleTo(self.w))
        self.assertFalse(self.w.orient_dialog.isVisible())
        self.assertFalse(self.w.is_reoriented())
        # No vertices given: only the automatic box is on offer.
        self.assertFalse(self.w.btn_orient.isEnabled())

    def test_an_extrusion_follows_the_turn(self):
        self.w.type_combo.setCurrentText(self.sp.EXTRUSION)
        before = self.emitted[-1]
        self.w.align([0, 0, 1], self.bottom_axis())
        after = self.emitted[-1]
        self.assertTrue(after.is_extrusion)
        self.assertLessEqual(after.volume_in3, before.volume_in3 + 1e-9)


@unittest.skipUnless(HAVE_QT, "needs PySide6")
class TestOpensOnAnExtrusion(QtCase):
    """The analysis names a profile that beats plate: the part opens on it."""

    def setUp(self):
        from partestimator.machining.ui import stock_panel as sp

        self.sp = sp
        self.w = sp.StockControls()

    def test_a_fitted_profile_is_the_starting_stock(self):
        s = self.w.set_part(box(6.0, 0.9, 0.9), None, "rectilinear",
                            extrusion=("angle", "1 x 1 x 1/8"))
        self.assertEqual(self.w.type_combo.currentText(), self.sp.EXTRUSION)
        self.assertTrue(s.is_extrusion)
        self.assertEqual((s.profile.shape, s.profile.name), ("angle", "1 x 1 x 1/8"))
        self.assertEqual(self.w.profile.currentText(), "angle|1 x 1 x 1/8")

    def test_none_or_an_unknown_profile_opens_on_plate(self):
        for extrusion in (None, ("angle", "no such size")):
            s = self.w.set_part(box(6.0, 0.9, 0.9), None, "rectilinear", extrusion=extrusion)
            self.assertEqual(self.w.type_combo.currentText(), self.sp.PLATE)
            self.assertEqual(s.kind, "rectilinear")

    def test_round_work_stays_on_round_bar(self):
        s = self.w.set_part(box(4.0, 2.0, 2.0), cyl(2.0, 4.0), "cylindrical",
                            extrusion=("angle", "1 x 1 x 1/8"))
        self.assertTrue(s.is_round)


if __name__ == "__main__":
    unittest.main()
