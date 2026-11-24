"""The machining window's remaining paths, in the real window: start-up with
a broken rate table or shop profile, the view buttons and overlays, copying
and exporting, units, the thin-wall scan, tuning-library errors, picking
faces in the 3D view, every kind of stock in the stock table, and failures
along the way.

Needs the CAD kernel and Qt; skipped in the light CI environment.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

if os.name != "nt" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    import OCP  # noqa: F401
    from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

    HAVE = True
except Exception:  # noqa: BLE001
    HAVE = False


def _quiet(fn, *a):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a)


@unittest.skipUnless(HAVE, "needs the CAD kernel and Qt")
class WindowCase(unittest.TestCase):
    PART = "machined_plate"

    @classmethod
    def setUpClass(cls):
        from partestimator.machining.geometry import pipeline
        from partestimator.machining.ui.main_window import MainWindow
        from tests import occparts

        cls.app = QApplication.instance() or QApplication([])
        cls.tmp = Path(tempfile.mkdtemp())
        cls.path = _quiet(getattr(occparts, cls.PART), cls.tmp)
        cls.res = pipeline.compute_analysis(str(cls.path))
        # One window per class: VTK on a software or 1.1 driver does not
        # survive dozens of render windows being made and torn down.
        cls.w = MainWindow()

    @classmethod
    def tearDownClass(cls):
        cls.w.close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        w = self.w
        w.removal_combo.setCurrentText("Mill")
        w._current_path = str(self.path)
        w._on_analysis_finished(dict(self.res))

    def spin(self, busy, timeout=120):
        t0 = time.time()
        while busy():
            self.app.processEvents()
            time.sleep(0.02)
            self.assertLess(time.time() - t0, timeout)
        self.app.processEvents()


class TestStartUp(unittest.TestCase):
    @unittest.skipUnless(HAVE, "needs the CAD kernel and Qt")
    def test_a_broken_rate_table_and_profile_are_reported_not_fatal(self):
        from partestimator.machining.ui import main_window as mw

        QApplication.instance() or QApplication([])
        with mock.patch.object(mw.rates, "load_rates", side_effect=ValueError("bad rates")), \
                mock.patch.object(mw.tuning, "load_tuned_profile", side_effect=OSError("t")), \
                mock.patch.object(mw.shop, "load_profile", side_effect=ValueError("bad shop")):
            w = mw.MainWindow()
        self.addCleanup(w.close)
        text = w.lbl_cycle_warnings.text()
        self.assertIn("bad rates", text)
        self.assertIn("bad shop", text)
        self.assertIsNone(w._current_estimate())
        with mock.patch.object(QMessageBox, "warning") as warn:
            self.assertIsNone(w.open_tuning())
        warn.assert_called_once()
        w._show_rate_sources()                    # no table: nothing to show

    @unittest.skipUnless(HAVE, "needs the CAD kernel and Qt")
    def test_a_broken_tuning_falls_back_to_the_shipped_profile(self):
        from partestimator.machining.ui import main_window as mw

        QApplication.instance() or QApplication([])
        with mock.patch.object(mw.tuning, "load_tuned_profile", side_effect=OSError("t")):
            w = mw.MainWindow()
        self.addCleanup(w.close)
        self.assertIsNotNone(w._shop_profile)
        self.assertIn("shop profile", w.lbl_cycle_warnings.text())


class TestViewAndOverlays(WindowCase):
    def test_view_buttons(self):
        for m in ("view_isometric", "view_xy", "view_xz", "view_yz", None):
            self.w._set_view(m)

    def test_every_overlay(self):
        w = self.w
        for chk in (w.chk_show_box, w.chk_show_captured, w.chk_show_com, w.chk_show_stock):
            chk.setChecked(not chk.isChecked())
            chk.setChecked(not chk.isChecked())
        w.chk_show_box.setChecked(True)
        w.chk_show_com.setChecked(True)
        w.chk_show_captured.setChecked(True)
        thin = SimpleNamespace(worst_points_mm=[(np.zeros(3), 1.0), (np.ones(3), 2.0)])
        w.results["thin_wall"] = thin
        w._draw_overlays()
        self.assertGreater(len(w.viewer._overlay_actors), 3)
        w.results["thin_wall"] = None

    def test_units(self):
        w = self.w
        w._on_unit_changed("in")
        self.assertEqual(w.display_unit, "IN")
        self.assertTrue(w.thin_threshold_spin.suffix().strip() == "in")
        w._on_unit_changed("mm")
        self.assertEqual(w.display_unit, "MM")
        w._sync_thin_threshold_spin_to_unit(reset_to_default=False)


class TestCopyAndExport(WindowCase):
    def test_copy_the_cycle_and_the_stock(self):
        w = self.w
        w._copy_cycle()
        self.assertIn("Term\tMinutes", QApplication.clipboard().text())
        w._copy_stock(with_header=True)
        self.assertTrue(QApplication.clipboard().text().startswith("OD/ST"))
        with mock.patch.object(w, "_current_estimate", return_value=None):
            w._copy_cycle()                                    # nothing to copy
        saved = w.results.pop("stock")
        w._copy_stock()
        w.results["stock"] = saved

    def test_rate_sources(self):
        w = self.w
        with mock.patch.object(QMessageBox, "exec") as shown:
            w._show_rate_sources()
        shown.assert_called_once()
        with mock.patch.object(w._rate_table, "sources_for", return_value=[]), \
                mock.patch.object(QMessageBox, "information") as info:
            w._show_rate_sources()
        info.assert_called_once()

    def test_export_csv_and_json(self):
        w = self.w
        csv_path, json_path = self.tmp / "rows.csv", self.tmp / "row.json"
        with mock.patch.object(QFileDialog, "getSaveFileName", return_value=("", "")):
            w._on_export_csv()
            w._on_export_json()
        self.assertFalse(csv_path.exists())
        with mock.patch.object(QFileDialog, "getSaveFileName", return_value=(str(csv_path), "")):
            w._on_export_csv()
        with mock.patch.object(QFileDialog, "getSaveFileName", return_value=(str(json_path), "")):
            w._on_export_json()
        self.assertIn("estimated_cycle_time_min", csv_path.read_text(encoding="utf-8"))
        self.assertEqual(json.loads(json_path.read_text(encoding="utf-8"))["part_name"],
                         self.path.stem)
        bad = str(self.tmp / "no_such_dir" / "x.csv")
        with mock.patch.object(QFileDialog, "getSaveFileName", return_value=(bad, "")), \
                mock.patch.object(QMessageBox, "critical") as crit:
            w._on_export_csv()
            w._on_export_json()
        self.assertEqual(crit.call_count, 2)

    def test_estimate_failures_never_break_the_window(self):
        from partestimator.machining.ui import main_window as mw

        w = self.w
        with mock.patch.object(mw.cycle_time, "estimate_cycle_time", side_effect=ValueError("x")):
            w._refresh_cycle_time()
        self.assertEqual(w.lbl_cycle_total.text(), "—")
        with mock.patch.object(mw.derived, "compute_derived_features",
                               side_effect=ValueError("y")):
            w._refresh_feature_table()                         # keeps the old features
        w.results["derived"] = {}
        self.assertIsNotNone(w._current_estimate())            # worked out afresh

    def test_lathe_on_plate_is_warned(self):
        w = self.w
        w.removal_combo.setCurrentText("Lathe")
        self.assertIn("round bar", w.lbl_cycle_warnings.text())
        w.removal_combo.setCurrentText("Mill")


class TestActions(WindowCase):
    def test_open_and_cancel(self):
        w = self.w
        with mock.patch.object(QFileDialog, "getOpenFileName", return_value=("", "")), \
                mock.patch.object(w, "_load_and_process") as lp:
            w._on_open_clicked()
            lp.assert_not_called()
        with mock.patch.object(QFileDialog, "getOpenFileName",
                               return_value=(str(self.path), "")), \
                mock.patch.object(w, "_load_and_process") as lp:
            w._on_open_clicked()
            lp.assert_called_once_with(str(self.path))
        runner = mock.Mock()
        w._thin_runner = runner
        w._on_cancel_clicked()
        runner.cancel.assert_called_once()
        w._thin_runner = None
        self.assertEqual(w.statusBar().currentMessage(), "Cancelled.")

    def test_worker_failure(self):
        with mock.patch.object(QMessageBox, "critical") as crit:
            self.w._on_worker_failed("Boom", "it broke")
        crit.assert_called_once()

    def test_thin_wall_scan_in_a_worker(self):
        w = self.w
        w._on_run_thin_wall_scan()
        self.spin(lambda: w._thin_runner is not None)
        self.assertIsNotNone(w.results.get("thin_wall"))
        self.assertTrue(w.lbl_thin_wall.text())
        w.results["thin_wall"] = SimpleNamespace(min_thickness_mm=None)
        w._refresh_thin_wall_label()
        self.assertIn("no opposing-surface hits", w.lbl_thin_wall.text())
        w._current_path = ""
        w._on_run_thin_wall_scan()                             # nothing open: nothing

    def test_close_survives_a_runner_that_will_not_cancel(self):
        from partestimator.machining.ui.main_window import MainWindow

        w = MainWindow()
        w._turn_runner = mock.Mock(cancel=mock.Mock(side_effect=RuntimeError("x")))
        real = w.viewer.close
        with mock.patch.object(w.viewer, "close", side_effect=RuntimeError("y")):
            w.close()
        real()


class TestTuningErrors(WindowCase):
    def test_label_and_reload(self):
        from partestimator.machining.ui import main_window as mw

        w = self.w
        w._tuning_state = SimpleNamespace(describe=lambda: "tuned to 3 parts",
                                          errors=["line 4: bad", "line 7: bad"])
        w._refresh_tuning_label()
        self.assertIn("line 4", w.lbl_tuning.text())
        with mock.patch.object(mw.tuning, "load_tuned_profile", side_effect=OSError("gone")):
            w.reload_shop_profile()
        self.assertIn("gone", w.statusBar().currentMessage())

    def test_save_to_library_failures(self):
        from partestimator.machining.ui import main_window as mw
        from partestimator.machining.ui import tuning_window as tw

        w = self.w
        with mock.patch.object(mw.tuning.Library, "load", side_effect=OSError("locked")), \
                mock.patch.object(QMessageBox, "warning") as warn:
            w.save_to_library()
        self.assertIn("locked", warn.call_args[0][2])
        folder = self.tmp / "lib"
        with mock.patch.dict(os.environ, {mw.tuning.ENV_DIR: str(folder)}):
            with mock.patch.object(tw.SaveToLibraryDialog, "exec", return_value=0):
                w.save_to_library()                            # cancelled
            self.assertFalse((folder / mw.tuning.LIBRARY_FILE).exists())
            with mock.patch.object(tw.SaveToLibraryDialog, "exec", return_value=1), \
                    mock.patch.object(mw.tuning.Library, "save", side_effect=OSError("excel")), \
                    mock.patch.object(QMessageBox, "warning") as warn:
                w.save_to_library()
            self.assertIn("excel", warn.call_args[0][2])
            w.results = {}
            with mock.patch.object(QMessageBox, "information") as info:
                w.save_to_library()
            info.assert_called_once()

    def test_a_library_part_whose_file_is_gone(self):
        with mock.patch.object(QMessageBox, "information") as info:
            self.w.open_library_entry(SimpleNamespace(source_file=str(self.tmp / "gone.step")),
                                      None)
        info.assert_called_once()

    def test_a_library_part_with_a_bad_stock(self):
        from partestimator.machining import tuning

        e = tuning.Entry(part="p", id="x", material="Aluminum 6061-T6", operations=1,
                         stock_type="plate", stock_size="bad", measured_min=3.0)
        self.w._apply_entry_inputs(e)
        self.assertIn("p: opened from the tuning library", self.w.statusBar().currentMessage())


class TestPicking(WindowCase):
    def test_picking_mode(self):
        w = self.w
        w.chk_show_stock.setChecked(False)
        w.chk_show_box.setChecked(False)
        w._on_pick_requested("stock")
        self.assertTrue(w.chk_show_stock.isChecked())          # something to click on
        w._on_pick_requested("")
        saved, w.results = w.results, {}
        w._on_pick_requested("part")
        w._on_view_click(1, 1)                                 # no part: nothing
        w.results = saved

    def test_clicking_the_part(self):
        w = self.w
        w._on_pick_requested("part")
        with mock.patch.object(w.viewer, "pick_part_triangle", return_value=None), \
                mock.patch.object(w.stock_controls, "pick_problem") as problem:
            w._on_view_click(5, 5)
        problem.assert_called_once_with("That missed the part.")
        tris = np.asarray(w.results["render_triangles"])
        pts = np.asarray(w.results["render_points"])
        normals = np.cross(pts[tris[:, 1]] - pts[tris[:, 0]], pts[tris[:, 2]] - pts[tris[:, 0]])
        flat = int(np.argmax(np.abs(normals[:, 2])))
        with mock.patch.object(w.viewer, "pick_part_triangle", return_value=flat), \
                mock.patch.object(w.stock_controls, "set_part_face") as set_face:
            w._on_view_click(5, 5)
        set_face.assert_called_once()
        with mock.patch.object(w.viewer, "pick_part_triangle", return_value=flat), \
                mock.patch("partestimator.machining.geometry.orientation.planar_face",
                           return_value=None), \
                mock.patch.object(w.stock_controls, "pick_problem") as problem:
            w._on_view_click(5, 5)
        problem.assert_called_once_with("That face is curved.")

    def test_clicking_the_stock(self):
        w = self.w
        w._on_pick_requested("stock")
        rect = w.stock_controls.rect_box()
        centre = np.asarray(rect.center_mm, dtype=float)
        up = np.asarray(rect.axes[2], dtype=float)
        with mock.patch.object(w.viewer, "pick_ray",
                               return_value=(centre + up * 500, -up)), \
                mock.patch.object(w.stock_controls, "set_stock_face") as set_face:
            w._on_view_click(5, 5)
        set_face.assert_called_once_with(2)
        side = np.cross(up, rect.axes[0])
        with mock.patch.object(w.viewer, "pick_ray",
                               return_value=(centre + side * 500 + up * 500, side)), \
                mock.patch.object(w.stock_controls, "pick_problem") as problem:
            w._on_view_click(5, 5)
        problem.assert_called_once_with("That missed the stock.")
        with mock.patch.object(w.stock_controls, "rect_box", return_value=None):
            w._on_view_click(5, 5)                             # no box: nothing


class TestStockTable(WindowCase):
    def tip_for(self, stock):
        self.w.results["stock"] = stock
        self.w._refresh_stock_labels()
        return self.w.stock_table.toolTip()

    def test_a_tooltip_for_every_kind(self):
        from partestimator.machining.geometry import extrusions as ex
        from partestimator.machining.geometry import stock as st

        self.assertIn("Minimum", self.tip_for(st.custom_rectilinear(0.5, 4, 6, basis=st.MINIMUM)))
        self.assertIn("Custom", self.tip_for(st.custom_rectilinear(1, 5, 7)))
        self.assertIn("Barstock", self.tip_for(st.cylindrical_stock(25.0, 100.0)))
        self.assertIn("Sheet", self.tip_for(st.rectilinear_stock([150, 100, 2.0])))
        self.assertIn("Plate", self.tip_for(st.rectilinear_stock([150, 100, 20.0])))
        prof = ex.find("angle", "AND10133-2004")
        rect = self.w.results["rect_box"]
        self.assertIn("Extrusion", self.tip_for(ex.extrusion_stock(prof, rect)))
        self.assertEqual(self.tip_for(None), "—")


class TestRoundBarPaths(WindowCase):
    PART = "turned_part"

    def test_remeasure_guards(self):
        w = self.w
        cyl = w.results["cylinder"]
        w._current_path = ""
        other = SimpleNamespace(**{**cyl.__dict__})
        w._remeasure_turning(other)                            # no file: nothing
        self.assertFalse(w.turning_busy())
        w.results["cylinder"] = other
        w._on_turning_measured(cyl, "stale")                   # the bar moved since
        self.assertNotEqual(w.results.get("turning"), "stale")

    def test_the_fixture_section_hides_for_bar(self):
        w = self.w
        self.assertFalse(w.fixture_controls.isVisibleTo(w))
        w._on_fixture_input()                                  # bar: no restock needed
        self.assertIsNone(w.results["fixture"])


class TestFixturePaths(WindowCase):
    def setUp(self):
        super().setUp()
        from partestimator.machining.geometry import fixture

        fixture.set_standard(fixture.FixtureSettings(enabled=True))
        self.addCleanup(fixture.set_standard, None)
        self.w.fixture_controls.chk.setChecked(True)

    def test_exported_with_problems_is_warned(self):
        w = self.w
        with mock.patch.object(QMessageBox, "warning") as warn:
            w._on_fixture_exported({"path": "x.step", "volumes_in3": {"Stock": 1.0},
                                    "problems": ["Tab 2 does not touch the part."]})
        self.assertIn("Tab 2", warn.call_args[0][2])
        w._on_fixture_exported({"path": "x.step", "volumes_in3": {"Stock": 1.0}})

    def test_export_cancelled_or_nothing_to_export(self):
        w = self.w
        with mock.patch.object(QFileDialog, "getSaveFileName", return_value=("", "")):
            w.export_fixture()
        self.assertFalse(w.fixture_busy())
        w.results["fixture"] = None
        w.export_fixture()

    def test_a_layout_that_fails_is_reported(self):
        from partestimator.machining.ui import main_window as mw

        w = self.w
        w._fixture_cache.clear()
        with mock.patch.object(mw.fixture, "plan_for_results", side_effect=ValueError("no")):
            self.assertIsNone(w._fixture_plan(w.results["stock"]))
        self.assertIn("Tab layout failed", w.statusBar().currentMessage())

    def test_settings_change_with_nothing_open(self):
        w = self.w
        w.results = {}
        w.refresh_stock_allowances()
        w._on_fixture_input()


class TestViewer(WindowCase):
    """The 3D viewer's drawing and picking, on the window's own viewer."""

    def test_odd_overlays(self):
        from partestimator.machining.ui import viewer as vmod

        v = self.w.viewer
        before = len(v._overlay_actors)
        holes = [SimpleNamespace(is_captured=True, axis_point=np.zeros(3),
                                 axis_direction=np.array([0, 0, 1.0]), axial_lo=0.0,
                                 axial_hi=5.0),
                 SimpleNamespace(is_captured=True, axis_point=np.zeros(3),
                                 axis_direction=np.zeros(3)),
                 SimpleNamespace(is_captured=False)]
        v.show_captured_hole_axes(holes)
        v.show_captured_hole_axes([])
        self.assertEqual(len(v._overlay_actors), before + 1)
        v.show_thin_wall_points(np.zeros((0, 3)))
        v._add_dashed([])
        self.assertEqual(vmod._dash_segments(np.zeros(3), np.zeros(3), 1.0)[0][0].shape, (3,))
        v._show_cylinder(np.zeros(3), np.array([1.0, 0, 0]), 5.0, 10.0)   # axis along X
        v.show_fixture([np.zeros((1, 3))], [])
        # An older PyVista: plain shading. No mesh: nothing to draw.
        real = v.plotter.add_mesh

        def old_add_mesh(*a, **kw):
            if "split_sharp_edges" in kw:
                raise TypeError("unexpected keyword")
            return real(*a, **kw)

        with mock.patch.object(v.plotter, "add_mesh", side_effect=old_add_mesh):
            v._add_surface_actor()
        mesh, v._mesh = v._mesh, None
        v._add_surface_actor()
        v._mesh = mesh
        polys, v._edge_polylines = v._edge_polylines, [np.zeros((1, 3))]
        v._rebuild_edge_actor()
        v._edge_polylines = polys
        v._rebuild_edge_actor()

    def test_viewer_failures_are_quiet(self):
        v = self.w.viewer
        saved = getattr(v, "_click_observers", False)
        v._click_observers = False
        with mock.patch.object(v.plotter.iren, "add_observer", side_effect=RuntimeError):
            v.set_click_handler(lambda x, y: None)
        v._click_observers = saved
        v.set_click_handler(None)
        with mock.patch("vtk.vtkCellPicker") as picker:
            picker.return_value.Pick.return_value = 0
            self.assertIsNone(v.pick_part_triangle(1, 1))
        real = v.plotter.add_mesh

        def no_mapper(*a, **kw):
            actor = real(*a, **kw)
            return mock.Mock(wraps=actor, GetMapper=mock.Mock(side_effect=RuntimeError))

        with mock.patch.object(v.plotter, "add_mesh", side_effect=no_mapper), \
                mock.patch.object(v.plotter, "remove_actor"):
            v.highlight_triangles([0])
        v._highlight_actors = []

    def test_edges_and_mesh_wireframe(self):
        v = self.w.viewer
        v.set_show_mesh(True)
        self.assertIsNotNone(v._mesh_overlay_actor)
        v.set_show_mesh(False)
        self.assertIsNone(v._mesh_overlay_actor)
        v.set_show_edges(False)
        self.assertIsNone(v._edge_actor)
        v.set_show_edges(True)
        self.assertIsNotNone(v._edge_actor)

    def test_every_stock_drawn(self):
        from partestimator.machining.geometry import extrusions as ex
        from partestimator.machining.geometry import stock as st

        w = self.w
        rect, cyl = w.results["rect_box"], w.results["cylinder"]
        v = w.viewer
        before = len(v._overlay_actors)
        v.show_stock("rectilinear", ex.extrusion_stock(ex.find("angle", "AND10133-2004"), rect),
                     rect, cyl)
        v.show_stock("cylindrical", st.cylindrical_stock(25.0, 100.0), None, cyl)
        v.show_stock("rectilinear", None, rect, cyl)
        v.show_primary_bounding_box("cylindrical", rect, cyl)
        tube = ex.find("round_tube", next(p.name for p in ex.load_profiles()
                                          if p.shape == "round_tube"))
        v.show_stock("rectilinear", ex.extrusion_stock(tube, rect, cyl), rect, cyl)
        self.assertGreater(len(v._overlay_actors), before + 3)
        v.show_fixture([], [])                         # nothing to draw
        v.show_captured_hole_axes([SimpleNamespace(is_captured=True, axis_point=None,
                                                   axis_direction=None)])

    def test_clicks_and_drags(self):
        v = self.w.viewer
        clicks = []
        v.set_click_handler(lambda x, y: clicks.append((x, y)))
        v.set_click_handler(lambda x, y: clicks.append((x, y)))      # observers once
        with mock.patch.object(v.plotter.iren, "get_event_position", return_value=(10, 10)):
            v._on_press()
            v._on_release()
        with mock.patch.object(v.plotter.iren, "get_event_position", side_effect=[(10, 10),
                                                                                  (60, 10)]):
            v._on_press()
            v._on_release()                                          # a drag, not a click
        with mock.patch.object(v.plotter.iren, "get_event_position",
                               side_effect=RuntimeError("no")):
            v._on_press()
            v._on_release()
        self.assertEqual(clicks, [(10, 10)])
        v._press = (1, 1)
        with mock.patch.object(v.plotter.iren, "get_event_position",
                               side_effect=RuntimeError("no")):
            v._on_release()
        v.set_click_handler(None)

    def test_picking_and_highlights(self):
        w = self.w
        v = w.viewer
        v.fit_all()
        size = v.plotter.window_size
        tri = v.pick_part_triangle(size[0] // 2, size[1] // 2)
        self.assertTrue(tri is None or tri >= 0)
        origin, direction = v.pick_ray(size[0] // 2, size[1] // 2)
        self.assertEqual(origin.shape, (3,))
        self.assertGreater(float(np.linalg.norm(direction)), 0)
        v.set_picking_cursor(True)
        v.set_picking_cursor(False)
        v.highlight_triangles([0, 1, 2])
        v.highlight_triangles([])
        rect = w.results["rect_box"]
        v.highlight_quad(np.asarray(rect.corners_mm)[:4])
        v.clear_highlights("part")
        v.clear_highlights()
        saved = v._mesh
        v._mesh = None
        self.assertIsNone(v.pick_part_triangle(1, 1))
        v.highlight_triangles([0])
        v._mesh = saved


class _FakeRunner:
    """Stands in for AnalysisRunner: records start calls, fails on request."""

    made = []

    def __init__(self, parent=None):
        from PySide6.QtCore import QObject, Signal

        class Sig(QObject):
            progress = Signal(str)
            finished = Signal(object)
            failed = Signal(str, str)
            done = Signal()

        self._s = Sig()
        self.progress, self.finished = self._s.progress, self._s.finished
        self.failed, self.done = self._s.failed, self._s.done
        self.cancel = mock.Mock()
        _FakeRunner.made.append(self)

    def start_fixture_export(self, *a):
        self.failed.emit("Boom", "it broke\nmore")

    def start_turning(self, *a):
        self.failed.emit("Kernel", "no")


class TestOddCorners(WindowCase):
    def test_export_with_a_runner_already_going_and_a_failure(self):
        from partestimator.machining.geometry import fixture
        from partestimator.machining.ui import main_window as mw

        fixture.set_standard(fixture.FixtureSettings(enabled=True))
        self.addCleanup(fixture.set_standard, None)
        w = self.w
        w.fixture_controls.chk.setChecked(True)
        old = mock.Mock()
        w._fixture_runner = old
        with mock.patch.object(mw, "AnalysisRunner", _FakeRunner), \
                mock.patch.object(QFileDialog, "getSaveFileName",
                                  return_value=(str(self.tmp / "x.step"), "")), \
                mock.patch.object(QMessageBox, "warning") as warn:
            w.export_fixture()
        old.cancel.assert_called_once()
        self.assertIn("it broke", warn.call_args[0][2])
        w._fixture_runner = None
        w.fixture_controls.chk.setChecked(False)

    def test_turning_remeasure_replaces_and_reports(self):
        from partestimator.machining.ui import main_window as mw

        w = self.w
        cyl = w.results["cylinder"]
        other = SimpleNamespace(**{**cyl.__dict__})
        old = mock.Mock()
        w._turn_runner = old
        with mock.patch.object(mw, "AnalysisRunner", _FakeRunner):
            w._remeasure_turning(other)
        old.cancel.assert_called_once()
        self.assertIn("not re-measured", w.statusBar().currentMessage())
        w._turn_runner = None

    def test_units_weights_and_labels(self):
        w = self.w
        w.finish_unit_combo.setCurrentIndex(
            next(i for i in range(w.finish_unit_combo.count())
                 if w.finish_unit_combo.itemText(i).startswith("µm")))
        w.finish_value.setValue(1.6)
        self.assertAlmostEqual(w._finish_uin(), 1.6 / 0.0254)
        w.finish_unit_combo.setCurrentIndex(0)
        sp, w.results["shape_props"] = w.results["shape_props"], None
        w._refresh_weight_labels()
        self.assertEqual(w.lbl_weight.text(), "—")
        w.results["shape_props"] = sp
        with mock.patch("partestimator.machining.geometry.materials.weight_lb",
                        return_value=None), \
                mock.patch("partestimator.machining.geometry.materials.density_lb_per_in3",
                           return_value=None):
            w._refresh_weight_labels()
        self.assertIn("no density", w.lbl_weight.text())
        self.assertIn("no density", w.lbl_stock_weight.text())
        stock, w.results["stock"] = w.results["stock"], None
        w._refresh_weight_labels()
        self.assertEqual(w.lbl_stock_weight.text(), "—")
        w.results["stock"] = stock
        w.results["thin_wall"] = None
        w._refresh_thin_wall_label()

    def test_stock_and_rect_changes(self):
        w = self.w
        saved = w.results
        w.results = {}
        w._on_stock_changed(saved["stock"])                    # nothing open: nothing
        w._refresh_cycle_time()
        w.results = saved
        turned = SimpleNamespace(**{**saved["rect_box"].__dict__})
        with mock.patch.object(w.stock_controls, "rect_box", return_value=turned):
            w._on_stock_changed(saved["stock"])
        self.assertIs(w.results["rect_box"], turned)

    def test_tuning_window_from_the_machining_tab(self):
        from partestimator.machining import tuning
        from partestimator.machining.ui import tuning_window as tw

        w = self.w
        folder = self.tmp / "tlib"
        with mock.patch.dict(os.environ, {tuning.ENV_DIR: str(folder)}):
            win = w.open_tuning()
            self.addCleanup(lambda: setattr(w, "_tuning_window", None))
            self.addCleanup(win.close)
            self.assertIs(w.open_tuning(), win)
            self.assertEqual(w.library_defaults()["material"], w.material_combo.currentText())
            w.reload_shop_profile()
            with mock.patch.object(tw.SaveToLibraryDialog, "exec", return_value=1), \
                    mock.patch.object(win, "entry_saved") as saved:
                w.save_to_library()
            saved.assert_called_once()
            lib = tuning.Library.load(folder)
            e = lib.entries[0]
            e.stock_type, e.stock_size, e.stock_profile = "extrusion", "7", "angle|AND10133-2004"
            w._apply_entry_inputs(e)
        self.assertTrue(w.stock_controls.stock().is_extrusion)


if __name__ == "__main__":
    unittest.main()
