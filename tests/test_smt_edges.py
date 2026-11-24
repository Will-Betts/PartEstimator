"""Small edges of the sheet metal model and its helpers: degenerate geometry,
the model's convenience accessors, a missing base flange, the panels'
controls, DXF export's fallbacks, and the solid unfold's final gates.
"""
from __future__ import annotations

import contextlib
import io
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

if os.name != "nt" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    import OCP  # noqa: F401

    HAVE_OCP = True
except Exception:  # noqa: BLE001
    HAVE_OCP = False

try:
    from PySide6.QtWidgets import QApplication

    HAVE_QT = True
except Exception:  # noqa: BLE001
    HAVE_QT = False


class TestGeom(unittest.TestCase):
    def test_degenerate_inputs(self):
        from partestimator.smt import geom

        with self.assertRaises(ValueError):
            geom.unit([0.0, 0.0, 0.0])
        a = np.eye(4)
        a[0, 3] = 2.0
        b = np.eye(4)
        b[1, 3] = 3.0
        np.testing.assert_allclose(geom.compose(a, b)[:3, 3], [2.0, 3.0, 0.0])
        lo, hi = geom.bbox_of_points([[1, 2, 3], [-1, 5, 0]])
        np.testing.assert_allclose(lo, [-1, 2, 0])
        np.testing.assert_allclose(hi, [1, 5, 3])
        self.assertEqual(geom.min_area_rect([[0, 0], [3, 1]]), (3.0, 1.0, 0.0))
        self.assertEqual(geom.min_area_rect([[1, 1], [1, 1], [1, 1], [1, 1]])[2], 0.0)
        self.assertEqual(len(geom._convex_hull(np.array([[0.0, 0], [1, 1]]))), 2)
        runs = geom.join_point_runs([np.zeros((0, 3)), np.array([[0, 0, 0], [1, 0, 0.0]]),
                                     np.array([[1, 0, 0], [2, 0, 0.0]])])
        self.assertEqual(len(runs), 3)


class TestModel(unittest.TestCase):
    def test_accessors(self):
        from partestimator.smt.features import Role
        from partestimator.smt.model import Bend, FlatPattern, Flange, Part

        fl = Flange(id=1, normal=np.array([0, 0, 1.0]), origin=np.zeros(3), area=10.0,
                    centroid=np.zeros(3))
        self.assertFalse(fl.has_outline())
        self.assertEqual(fl.region_count, 1)
        self.assertEqual(fl.faces_for_side(0), [])
        fl.face = "face-a"
        self.assertEqual(fl.faces_for_side(0), ["face-a"])
        fl.loops_a = [np.zeros((3, 2))]
        self.assertTrue(fl.has_outline())
        self.assertEqual(fl.regions_for_side(1), [[fl.loops_a[0]]])   # falls back to side a
        b = Bend(id=4, axis_point=np.zeros(3), axis_dir=np.array([0, 1.0, 0]), r_inner=2.0,
                 thickness=1.5, angle=1.0, length_envelope=10, length_material=10,
                 flange_a=1, flange_b=2)
        self.assertEqual(b.r_outer, 3.5)
        self.assertEqual((b.other(1), b.other(2)), (2, 1))
        with self.assertRaises(KeyError):
            b.other(9)
        self.assertIsNone(FlatPattern().face)
        part = Part(thickness=1.5)
        part.bends[4] = b
        self.assertEqual(part.bend_count, 1)
        self.assertEqual(part.secondary_operations(), [])
        del Role

    def test_flat2d_edges(self):
        from partestimator.smt import model2d

        flat = model2d.Flat2D()
        self.assertEqual(flat.min_area_rect(), (0.0, 0.0, 0.0))
        self.assertEqual(model2d._shoelace(np.zeros((2, 2))), 0.0)


@unittest.skipUnless(HAVE_OCP, "needs the CAD kernel")
class TestUnfoldAndExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from partestimator.smt.config import UnfoldOptions
        from partestimator.smt.pipeline import unfold_step
        from tests import occparts

        cls.tmp = Path(tempfile.mkdtemp())
        with contextlib.redirect_stdout(io.StringIO()):
            path = occparts.z_section(cls.tmp)
        cls.opts = UnfoldOptions()
        cls.part = unfold_step(path, cls.opts)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_a_missing_base_flange(self):
        from partestimator.smt.config import UnfoldOptions
        from partestimator.smt.graph import UnflattenableError
        from partestimator.smt.unfold import solve

        with self.assertRaisesRegex(UnflattenableError, "does not exist"):
            solve(self.part, UnfoldOptions(base_face=999))
        other = UnfoldOptions(base_face=min(self.part.flanges))
        self.assertEqual(solve(self.part, other).base, min(self.part.flanges))

    def test_dxf_fallbacks(self):
        from partestimator.smt import dxf

        out = self.tmp / "z.dxf"
        with mock.patch.object(dxf, "_collect_face_exact", side_effect=RuntimeError("arc")):
            dxf.export_dxf(self.part, out)
        self.assertGreater(out.stat().st_size, 0)
        dxf.export_dxf(self.part, self.tmp / "z2.dxf", exact=False)
        flat, self.part.flat = self.part.flat, None
        try:
            with self.assertRaises(ValueError):
                dxf.export_dxf(self.part, out)
        finally:
            self.part.flat = flat
        w, h = dxf.flat_size_inches(self.part)
        self.assertGreater(w * h, 0)
        options = mock.MagicMock()
        type(options).font_cache_directory = mock.PropertyMock(side_effect=RuntimeError)
        dxf._confine_ezdxf(SimpleNamespace(options=options))
        dxf._confine_ezdxf(SimpleNamespace())                  # very old ezdxf: nothing
        import ezdxf

        doc = ezdxf.new()
        dxf._set_extents(doc, doc.modelspace())               # empty: nothing to fit

    def test_the_solid_unfolds_last_gates(self):
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        from partestimator.smt.occ import unfold3d

        box = BRepPrimAPI_MakeBox(10.0, 10.0, 1.5).Shape()
        good = SimpleNamespace(thickness=1.5, volume=150.0)
        unfold3d._strict_gate(good, box)
        unfold3d._sanity_check(good, box)
        with self.assertRaisesRegex(unfold3d.SolidUnfoldError, "deep"):
            unfold3d._strict_gate(SimpleNamespace(thickness=3.0, volume=150.0), box)
        with self.assertRaisesRegex(unfold3d.SolidUnfoldError, "volume"):
            unfold3d._strict_gate(SimpleNamespace(thickness=1.5, volume=300.0), box)
        unfold3d._sanity_check(SimpleNamespace(thickness=1.5, volume=0.0), box)


@unittest.skipUnless(HAVE_QT, "needs Qt")
class TestPanels(unittest.TestCase):
    def setUp(self):
        QApplication.instance() or QApplication([])

    def test_simple_controls(self):
        from partestimator.smt.config import UnfoldOptions
        from partestimator.smt.gui.panels import SimpleControls

        c = SimpleControls()
        seen = []
        c.changed.connect(lambda: seen.append("changed"))
        c.rescan.connect(lambda: seen.append("rescan"))
        c.k_auto.setChecked(False)
        self.assertTrue(c.k_slider.isEnabled())
        c.k_slider.setValue(c.k_slider.value() + 3)
        self.assertNotEqual(c.k_label.text(), "auto")
        c.k_auto.setChecked(True)
        self.assertEqual(c.k_label.text(), "auto")
        c.thickness_auto.setChecked(False)
        self.assertIn("rescan", seen)
        from partestimator.smt.units import choose_display_unit

        c.set_unit(choose_display_unit("in", None))
        c.thickness.setValue(0.125)
        self.assertAlmostEqual(c.thickness_mm(), 3.175, places=3)
        c.thickness_auto.setChecked(True)
        c.apply_to(UnfoldOptions())

    def test_advanced_and_detail(self):
        from partestimator.smt.config import UnfoldOptions
        from partestimator.smt.gui.panels import AdvancedPanel, DetailTable

        a = AdvancedPanel()
        a.orient.setCurrentText("as modelled")
        opts = UnfoldOptions()
        a.apply_to(opts)
        self.assertEqual(opts.orient, "as_modelled")
        t = DetailTable()
        t.show_features(SimpleNamespace(feature_groups=[]), None)
        self.assertEqual(t.item(0, 0).text(), "no secondary features found")


if __name__ == "__main__":
    unittest.main()
