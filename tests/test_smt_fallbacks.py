"""The sheet metal kernel layer's fallbacks, each made to happen: an edge
with no curve, a kernel binding that wants other arguments, a wire explorer
that fails, a face that will not mesh, a fuse that will not build, an empty
scene -- and the diagnostics: path redaction, logging set-up, the crash and
exception hooks, and stage cancelling.

These paths exist because translated STEP and older kernels do produce
them; each is forced here with a mock where real geometry would be fragile.
"""
from __future__ import annotations

import contextlib
import io
import logging
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

try:
    import OCP  # noqa: F401

    HAVE_OCP = True
except Exception:  # noqa: BLE001
    HAVE_OCP = False

IN = 25.4


def _box(x=10.0, y=20.0, z=5.0):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

    return BRepPrimAPI_MakeBox(x, y, z).Shape()


def _first(shape, kind):
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_WIRE

    k = {"edge": TopAbs_EDGE, "face": TopAbs_FACE, "wire": TopAbs_WIRE}[kind]
    cur = TopExp_Explorer(shape, k).Current()
    return {"edge": TopoDS.Edge_s, "face": TopoDS.Face_s, "wire": TopoDS.Wire_s}[kind](cur)


@unittest.skipUnless(HAVE_OCP, "needs the CAD kernel")
class TestTopoFallbacks(unittest.TestCase):
    def setUp(self):
        from partestimator.smt.occ import topo

        self.t = topo

    def test_edges(self):
        t = self.t
        edge = _first(_box(), "edge")
        self.assertTrue(t.edge_has_curve(edge))
        self.assertEqual(t.edge_curve_kind(edge), "line")
        with mock.patch.object(t.BRep_Tool, "Degenerated_s", side_effect=RuntimeError):
            self.assertFalse(t.edge_has_curve(edge))
        with mock.patch.object(t.BRep_Tool, "Degenerated_s", return_value=True):
            self.assertFalse(t.edge_has_curve(edge))
            self.assertEqual(t.discretise_edge(edge), [])
            self.assertEqual(t.edge_curve_kind(edge), "none")
        # One kernel build wants (edge, f, l), another just (edge).
        with mock.patch.object(t.BRep_Tool, "Curve_s",
                               side_effect=[TypeError("args"), (object(), 0.0, 1.0)]):
            self.assertTrue(t.edge_has_curve(edge))
        with mock.patch.object(t.BRep_Tool, "Curve_s", side_effect=TypeError("never")):
            self.assertFalse(t.edge_has_curve(edge))
        with mock.patch.object(t.BRep_Tool, "Curve_s", side_effect=ValueError("broken")):
            self.assertFalse(t.edge_has_curve(edge))
        with mock.patch.object(t, "GCPnts_QuasiUniformDeflection",
                               return_value=mock.Mock(IsDone=mock.Mock(return_value=False))):
            self.assertEqual(len(t.discretise_edge(edge)), 2)
        with mock.patch.object(t, "BRepAdaptor_Curve", side_effect=RuntimeError("bad")):
            self.assertEqual(t.discretise_edge(edge), [])

    def test_circle_and_other_curves(self):
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere

        cyl = BRepPrimAPI_MakeCylinder(5.0, 10.0).Shape()
        kinds = set()
        from OCP.TopAbs import TopAbs_EDGE
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS

        for shape in (cyl, BRepPrimAPI_MakeSphere(5.0).Shape()):
            ex = TopExp_Explorer(shape, TopAbs_EDGE)
            while ex.More():
                kinds.add(self.t.edge_curve_kind(TopoDS.Edge_s(ex.Current())))
                ex.Next()
        self.assertTrue({"circle", "none"} <= kinds)

    def test_faces_and_wires(self):
        t = self.t
        box = _box()
        face = _first(box, "face")
        wire = _first(box, "wire")
        with mock.patch.object(t.BRepTools, "UVBounds_s", side_effect=TypeError):
            bounds = t.uv_bounds(face, t.BRepAdaptor_Surface(face))
        self.assertEqual(len(bounds), 4)
        with mock.patch.object(t, "GProp_GProps", side_effect=RuntimeError):
            self.assertEqual(t.solid_volume(box), 0.0)
        with mock.patch.object(t, "describe_face", side_effect=ValueError("odd face")):
            self.assertEqual(t.describe_all(box), [])
        # The wire explorer fails: the edges are chained by geometry instead.
        with mock.patch.object(t, "BRepTools_WireExplorer", side_effect=RuntimeError):
            pts = t.wire_points(wire, face, 0.1)
        self.assertGreaterEqual(len(pts), 4)
        with mock.patch.object(t, "BRepTools_WireExplorer", side_effect=RuntimeError), \
                mock.patch.object(t, "discretise_edge", return_value=[]):
            self.assertEqual(len(t.wire_points(wire, face, 0.1)), 0)
        with mock.patch.object(t, "face_loops", side_effect=RuntimeError("kernel")):
            self.assertEqual(t.face_loops_safe(face, 3), [])
        self.assertEqual(len(t.face_loops(face)), 1)

    def test_merged_cylinder_edges(self):
        t = self.t
        mc = t.MergedCylinder.__new__(t.MergedCylinder)
        mc.faces = [SimpleNamespace(area=0.0, centroid=np.array([1.0, 2, 3]))]
        self.assertEqual(list(mc.centroid), [1.0, 2.0, 3.0])
        flat = SimpleNamespace(radius=0.0, sweep=0.0, area=5.0)
        self.assertEqual(t.FaceInfo.material_length.fget(flat), 0.0)


@unittest.skipUnless(HAVE_OCP, "needs the CAD kernel")
class TestTessFallbacks(unittest.TestCase):
    def setUp(self):
        from partestimator.smt.occ import tess

        self.t = tess

    def test_empty_scene_and_box(self):
        from OCP.BRep import BRep_Builder
        from OCP.TopoDS import TopoDS_Compound

        t = self.t
        scene = t.Scene()
        self.assertTrue(scene.is_empty)
        self.assertEqual(list(scene.bounds()[0]), [0.0, 0.0, 0.0])
        self.assertEqual(scene.diagonal(), 1.0)
        self.assertEqual(list(scene.centre()), [0.0, 0.0, 0.0])
        self.assertEqual(scene.triangle_normals().shape, (0, 3))
        self.assertEqual(t.FaceRender(index=0).n_triangles, 0)
        empty = TopoDS_Compound()
        BRep_Builder().MakeCompound(empty)
        lo, hi = t.bounding_box(empty)
        self.assertEqual(list(hi - lo), [0.0, 0.0, 0.0])
        with self.assertRaises(t.TessellationError):
            t.build_scene(empty, infos=[])

    def test_faces_and_edges_that_fail(self):
        t = self.t
        box = _box()
        # No face meshes: the flat faces still draw from their exact outlines.
        with mock.patch.object(t, "_face_triangles", side_effect=RuntimeError("mesh")):
            scene = t.build_scene(box)
        self.assertEqual(len(scene.triangles), 0)
        self.assertTrue(all(f.drawable_as_polygon for f in scene.faces))
        with mock.patch.object(t, "edge_polylines", side_effect=RuntimeError("edges")):
            scene = t.build_scene(box)
        self.assertEqual(scene.edges, [])
        self.assertGreater(len(scene.triangles), 0)
        with mock.patch.object(t, "BRepMesh_IncrementalMesh",
                               return_value=mock.Mock(IsDone=mock.Mock(return_value=False))), \
                self.assertLogs("smt", "WARNING"):
            t._mesh_shape(box, 0.1, 0.5)
        self.assertEqual(t._face_triangles(_first(_box(), "face"), 0), (None, None))
        with mock.patch.object(t, "GCPnts_QuasiUniformDeflection",
                               return_value=mock.Mock(IsDone=mock.Mock(return_value=False))):
            lines = t.edge_polylines(box, 0.1)
        self.assertTrue(all(len(p) == 2 for p, _o in lines))
        with mock.patch.object(t, "BRepAdaptor_Curve", side_effect=RuntimeError):
            self.assertEqual(t.edge_polylines(box, 0.1), [])
        with mock.patch.object(t, "edge_has_curve", return_value=False):
            self.assertEqual(t.edge_polylines(box, 0.1), [])


@unittest.skipUnless(HAVE_OCP, "needs the CAD kernel")
class TestFlattenFallbacks(unittest.TestCase):
    def setUp(self):
        from partestimator.smt.occ import flatten

        self.f = flatten

    def test_frames_faces_and_fusing(self):
        f = self.f
        M = f.flat_frame(np.zeros(3), np.array([1.0, 0, 0]))
        np.testing.assert_allclose(M[:3, :3] @ [1.0, 0, 0], [0, 0, 1], atol=1e-12)
        with mock.patch.object(f, "BRepBuilderAPI_MakeFace",
                               return_value=mock.Mock(IsDone=mock.Mock(return_value=False))):
            with self.assertRaises(f.FlattenError):
                f._polygon_face([(0, 0), (1, 0), (1, 1)])
        a = f._polygon_face([(0, 0), (1, 0), (1, 1), (0, 1)])
        b = f._polygon_face([(1, 0), (2, 0), (2, 1), (1, 1)])
        with mock.patch.object(f, "BOPAlgo_Builder", side_effect=RuntimeError("bop")):
            fused = f._fuse_all([a, b])
        self.assertFalse(fused.IsNull())
        with mock.patch.object(f, "BOPAlgo_Builder", side_effect=RuntimeError("bop")), \
                mock.patch.object(f, "BRepAlgoAPI_Fuse",
                                  return_value=mock.Mock(IsDone=mock.Mock(return_value=False))):
            with self.assertRaises(f.FlattenError):
                f._fuse_all([a, b])

    def test_refold_check_without_a_flat(self):
        self.assertTrue(np.isnan(self.f.refold_check(SimpleNamespace(flat=None, volume=1.0),
                                                     None)))


class TestDiagnostics(unittest.TestCase):
    def setUp(self):
        from partestimator.smt import diag

        self.d = diag
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_names_and_redaction(self):
        d = self.d

        class Bad:
            def __str__(self):
                raise ValueError("no")

        self.assertEqual(d.name_only(Bad()), "(unnamed)")
        self.assertEqual(d.name_only("C:/a/b/"), "(unnamed)")
        self.assertEqual(d.name_only(r"C:\Users\me\part.step"), "part.step")
        rec = logging.LogRecord("smt", logging.INFO, "", 0, r"read C:\Users\me\x.step %s",
                                ("now",), None)
        self.assertTrue(d.RedactPaths().filter(rec))
        self.assertNotIn("Users", rec.getMessage())
        broken = logging.LogRecord("smt", logging.INFO, "", 0, "%d", ("not a number",), None)
        self.assertTrue(d.RedactPaths().filter(broken))

    def test_scratch_that_cannot_be_removed(self):
        d = self.d
        with mock.patch.object(d, "scratch_dir", return_value=self.tmp), \
                mock.patch.object(d.shutil, "rmtree", side_effect=OSError("busy")):
            d.clean_scratch()

    def test_logging_set_up_once_into_a_read_only_folder(self):
        d = self.d
        saved = (d._CONFIGURED, list(d.LOGGER.handlers), d.LOGGER.level, d.LOGGER.propagate,
                 d._CRASH_FILE)

        def restore():
            for h in d.LOGGER.handlers[:]:
                if h not in saved[1]:
                    d.LOGGER.removeHandler(h)
                    h.close()
            d._CONFIGURED, _h, lvl, prop, crash = saved
            d.LOGGER.setLevel(lvl)
            d.LOGGER.propagate = prop
            if d._CRASH_FILE is not None and d._CRASH_FILE is not crash:
                import faulthandler

                faulthandler.disable()
                d._CRASH_FILE.close()
            d._CRASH_FILE = crash

        self.addCleanup(restore)
        d._CONFIGURED = False
        d._CRASH_FILE = None
        with mock.patch.object(d, "log_path", return_value=self.tmp / "no" / "x.log"), \
                mock.patch.object(d, "crash_path", return_value=self.tmp / "no" / "c.txt"), \
                mock.patch.dict(sys.modules, {"ezdxf": None, "OCP.Standard": None}), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(d.setup_logging(verbose=False))
            self.assertIsNone(d.install_crash_handler())
        self.assertTrue(d._CONFIGURED)
        with mock.patch.object(d, "log_path", return_value=self.tmp / "x.log"):
            self.assertEqual(d.setup_logging(), self.tmp / "x.log")      # already set up
        with mock.patch.dict(sys.modules, {"OCP": None, "OCP.Standard": None}), \
                contextlib.redirect_stdout(io.StringIO()):
            d._log_environment()
        with mock.patch.object(d, "crash_path", return_value=self.tmp / "c.txt"):
            self.assertEqual(d.install_crash_handler(), self.tmp / "c.txt")
            self.assertEqual(d.install_crash_handler(), self.tmp / "c.txt")   # armed once

    def test_exception_hooks(self):
        d = self.d
        saved = (sys.excepthook, threading.excepthook)
        self.addCleanup(lambda: (setattr(sys, "excepthook", saved[0]),
                                 setattr(threading, "excepthook", saved[1])))
        d.install_excepthook()
        with mock.patch.object(sys, "__excepthook__") as base, \
                self.assertLogs("smt", "ERROR") as logs:
            try:
                raise ValueError("boom")
            except ValueError as exc:
                sys.excepthook(type(exc), exc, exc.__traceback__)
            threading.excepthook(SimpleNamespace(thread=None, exc_type=KeyError,
                                                 exc_value=KeyError("k"), exc_traceback=None))
        base.assert_called_once()
        self.assertIn("UNCAUGHT ValueError", logs.output[0])
        self.assertIn("in thread ?", logs.output[1])

    def test_stages_cancel_at_their_boundaries(self):
        d = self.d
        stage = d.Stage(cancel=lambda: True)
        with self.assertRaises(d.Cancelled):
            stage.check_cancelled()
        quiet = d.Stage()
        with quiet.step("Working", "detail") as s:
            self.assertIs(s, quiet)
        self.assertEqual(quiet.current, "Working")


if __name__ == "__main__":
    unittest.main()
