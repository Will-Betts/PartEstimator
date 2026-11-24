"""Edge cases of the geometry helpers: the 2D hull and enclosing circle, the
sheet metal bend graph, STEP loading (missing, garbage, surfaces, several
bodies), mesh extraction (untessellated, moved, older kernels) and hole
probing (open arcs, cancelling, progress).

The hull and the graph are plain Python; the rest needs the CAD kernel.
"""
from __future__ import annotations

import contextlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    import OCP  # noqa: F401

    HAVE_OCP = True
except Exception:  # noqa: BLE001
    HAVE_OCP = False

IN = 25.4


class TestHull2d(unittest.TestCase):
    def setUp(self):
        from partestimator.machining.geometry import hull2d

        self.h = hull2d

    def test_degenerate_hulls(self):
        self.assertEqual(len(self.h.convex_hull_2d(np.array([[0, 0], [0, 0], [1, 1]]))), 2)
        self.assertEqual(self.h.min_area_rectangle_2d(np.zeros((0, 2)), already_hulled=True)[0],
                         0.0)
        area, centre, wh, _a = self.h.min_area_rectangle_2d(np.array([[2.0, 3.0]]))
        self.assertEqual((area, tuple(centre), wh), (0.0, (2.0, 3.0), (0.0, 0.0)))
        area, centre, wh, angle = self.h.min_area_rectangle_2d(np.array([[0, 0], [3.0, 4.0]]))
        self.assertEqual((area, wh), (0.0, (5.0, 0.0)))
        self.assertAlmostEqual(angle, np.arctan2(4, 3))
        same = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        self.assertEqual(self.h.min_area_rectangle_2d(same, already_hulled=True)[0], 0.0)

    def test_enclosing_circles(self):
        with self.assertRaises(ValueError):
            self.h.min_enclosing_circle(np.zeros((0, 2)))
        c, r = self.h.min_enclosing_circle(np.array([[1.0, 2.0]]))
        self.assertEqual((tuple(c), r), ((1.0, 2.0), 0.0))
        c, r = self.h.min_enclosing_circle(np.array([[0, 0], [4.0, 0]]))
        self.assertAlmostEqual(r, 2.0)
        # Three collinear points: the two farthest apart set the circle.
        c, r = self.h._circle_from_3(np.array([0.0, 0]), np.array([1.0, 0]), np.array([4.0, 0]))
        self.assertAlmostEqual(r, 2.0)
        self.assertAlmostEqual(c[0], 2.0)


class TestBendGraph(unittest.TestCase):
    def graph(self, flanges, bends):
        from partestimator.smt.graph import BendGraph

        fl = {i: SimpleNamespace(id=i, area=a) for i, a in flanges.items()}
        bd = {i: SimpleNamespace(id=i, flange_a=a, flange_b=b) for i, (a, b) in bends.items()}
        return BendGraph(fl, bd)

    def test_walks_and_components(self):
        g = self.graph({0: 5.0, 1: 9.0, 2: 1.0}, {0: (0, 1), 1: (1, 2), 2: (1, 99)})
        self.assertEqual(g.largest_flange(), 1)
        self.assertEqual([s.child for s in g.bfs(1)], [0, 2])
        self.assertEqual(len(g.components()), 1)

    def test_what_cannot_be_flattened(self):
        from partestimator.smt.graph import UnflattenableError

        with self.assertRaises(UnflattenableError):
            self.graph({}, {}).largest_flange()
        with self.assertRaisesRegex(UnflattenableError, "not in graph"):
            self.graph({0: 1.0}, {}).bfs(5)
        with self.assertRaisesRegex(UnflattenableError, "closed section"):
            self.graph({0: 1, 1: 1, 2: 1}, {0: (0, 1), 1: (1, 2), 2: (2, 0)}).bfs(0)
        g = self.graph({0: 1, 1: 1, 2: 1}, {0: (0, 1)})
        self.assertEqual(len(g.components()), 2)
        with self.assertRaisesRegex(UnflattenableError, "not connected"):
            g.bfs(0)


@unittest.skipUnless(HAVE_OCP, "needs the CAD kernel")
class TestStepLoading(unittest.TestCase):
    def setUp(self):
        from tests import occparts

        self.o = occparts._occ()
        self.write = occparts._write
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def box(self, x, y, z, at=(0, 0, 0)):
        return self.o["BRepPrimAPI_MakeBox"](self.o["gp_Pnt"](*at), x, y, z).Shape()

    def compound(self, *shapes):
        from OCP.BRep import BRep_Builder
        from OCP.TopoDS import TopoDS_Compound

        comp, b = TopoDS_Compound(), BRep_Builder()
        b.MakeCompound(comp)
        for s in shapes:
            b.Add(comp, s)
        return comp

    def load(self, shape_or_path):
        from partestimator.smt.occ.load import load_step

        path = shape_or_path
        if not isinstance(path, Path):
            with contextlib.redirect_stdout(io.StringIO()):
                path = self.write(shape_or_path, self.tmp / "x.step")
        return load_step(path)

    def test_missing_and_garbage(self):
        from partestimator.smt.occ.load import LoadError

        with self.assertRaisesRegex(LoadError, "no such file"):
            self.load(self.tmp / "none.step")
        junk = self.tmp / "junk.step"
        junk.write_text("ISO-10303-21; nonsense", encoding="utf-8")
        with self.assertRaises(LoadError):
            self.load(junk)

    def test_an_open_shell_is_sewn_and_a_lone_surface_refused(self):
        from OCP.BRep import BRep_Builder
        from OCP.TopAbs import TopAbs_FACE
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS_Shell
        from partestimator.smt.occ.load import LoadError

        box = self.box(10, 20, 30)
        shell, b = TopoDS_Shell(), BRep_Builder()
        b.MakeShell(shell)
        faces = []
        ex = TopExp_Explorer(box, TopAbs_FACE)
        while ex.More():
            faces.append(ex.Current())
            ex.Next()
        for f in faces:
            b.Add(shell, f)
        solid = self.load(shell)
        self.assertEqual(solid.ShapeType().name, "TopAbs_SOLID")
        with self.assertRaisesRegex(LoadError, "no solid body"):
            self.load(faces[0])

    def test_several_bodies_keep_the_largest(self):
        from OCP.BRepGProp import BRepGProp
        from OCP.GProp import GProp_GProps

        def volume(s):
            p = GProp_GProps()
            BRepGProp.VolumeProperties_s(s, p)
            return p.Mass()

        stray = self.compound(self.box(50, 50, 5), self.box(1, 1, 1, at=(100, 0, 0)))
        self.assertAlmostEqual(volume(self.load(stray)), 50 * 50 * 5, places=3)
        assembly = self.compound(self.box(50, 50, 5), self.box(40, 40, 5, at=(100, 0, 0)))
        with self.assertLogs("smt", "WARNING") as logs:
            self.assertAlmostEqual(volume(self.load(assembly)), 50 * 50 * 5, places=3)
        self.assertTrue(any("assembly" in m for m in logs.output))

    def test_export_to_nowhere(self):
        from partestimator.smt.occ.load import LoadError, export_step

        with self.assertRaises(LoadError):
            export_step(self.box(1, 1, 1), self.tmp / "no_dir" / "x.step")


@unittest.skipUnless(HAVE_OCP, "needs the CAD kernel")
class TestMeshUtils(unittest.TestCase):
    def setUp(self):
        from partestimator.machining.geometry import mesh_utils
        from tests import occparts

        self.m = mesh_utils
        self.o = occparts._occ()

    def test_untessellated_and_empty(self):
        from OCP.TopAbs import TopAbs_FACE
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS, TopoDS_Compound

        box = self.o["BRepPrimAPI_MakeBox"](10.0, 10.0, 10.0).Shape()
        with self.assertRaisesRegex(ValueError, "no triangulation"):
            self.m.extract_mesh(box)
        face = TopoDS.Face_s(TopExp_Explorer(box, TopAbs_FACE).Current())
        self.assertIsNone(self.m.face_area_from_triangulation(face))
        self.assertEqual(self.m.face_area_and_centroid_from_triangulation(face), (None, None))
        empty = TopoDS_Compound()
        from OCP.BRep import BRep_Builder

        BRep_Builder().MakeCompound(empty)
        self.assertEqual(self.m.shape_diagonal_mm(empty), 0.0)
        self.assertEqual(self.m.tessellate(empty), self.m.MIN_DEFLECTION_MM)

    def test_a_moved_shape_comes_out_where_it_is(self):
        from OCP.gp import gp_Trsf, gp_Vec
        from OCP.TopLoc import TopLoc_Location

        box = self.o["BRepPrimAPI_MakeBox"](10.0, 10.0, 10.0).Shape()
        trsf = gp_Trsf()
        trsf.SetTranslation(gp_Vec(100.0, 0.0, 0.0))
        moved = box.Moved(TopLoc_Location(trsf))
        self.m.tessellate(moved)
        pts, _tris, stats = self.m.extract_mesh(moved, with_face_stats=True)
        self.assertGreaterEqual(pts[:, 0].min(), 100.0 - 1e-6)
        self.assertTrue(all(c is None or c[0] >= 100.0 - 1e-6 for _a, c in stats))
        from OCP.TopAbs import TopAbs_FACE
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS

        face = TopoDS.Face_s(TopExp_Explorer(moved, TopAbs_FACE).Current())
        area, centre = self.m.face_area_and_centroid_from_triangulation(face)
        self.assertAlmostEqual(area, 100.0, places=6)
        self.assertGreaterEqual(centre[0], 100.0 - 1e-6)

    def test_older_kernel_accessors(self):
        # An older OCP has no Coord()/Get(): the per-component fallbacks.
        class P:
            def __init__(self, x):
                self.x = x

            def X(self):
                return self.x

            def Y(self):
                return 0.0

            def Z(self):
                return 0.0

        class T:
            def Value(self, i):
                return i

        tri = SimpleNamespace(Node=lambda i: P(float(i)), Triangle=lambda i: T())
        np.testing.assert_allclose(self.m._face_nodes(tri, 3)[:, 0], [1, 2, 3])
        np.testing.assert_array_equal(self.m._face_triangle_indices(tri, 2), [[1, 2, 3]] * 2)


@unittest.skipUnless(HAVE_OCP, "needs the CAD kernel")
class TestHoleProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from partestimator.machining.geometry import pipeline
        from tests import occparts

        o = occparts._occ()
        plate = o["BRepPrimAPI_MakeBox"](4 * IN, 3 * IN, 0.5 * IN).Shape()
        plate = occparts._drill(plate, 1.0 * IN, 1.5 * IN, 0.5 * IN)
        # A hole breaking out of the edge with about 310 degrees of wall left:
        # still a hole, but not a full circle, so its wall is checked.
        plate = occparts._drill(plate, 3.66 * IN, 1.5 * IN, 0.75 * IN)
        cls.tmp = Path(tempfile.mkdtemp())
        with contextlib.redirect_stdout(io.StringIO()):
            cls.path = occparts._write(plate, cls.tmp / "notched.step")
        cls.res = pipeline.compute_analysis(str(cls.path))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_a_broken_out_hole_is_checked_and_kept(self):
        dias = sorted(round(h.nominal_diameter_mm / IN, 3) for h in self.res["holes"])
        self.assertEqual(dias, [0.5, 0.75])
        partial = next(h for h in self.res["holes"]
                       if round(h.nominal_diameter_mm / IN, 3) == 0.75)
        self.assertTrue(partial.validated)
        self.assertIsNotNone(partial.wall_thickness_mm)

    def test_probe_options(self):
        from partestimator.machining.geometry import face_scan, hole_probe, step_io

        shape = step_io.load_step_file(str(self.path))
        from partestimator.machining.geometry import mesh_utils

        mesh_utils.tessellate(shape)
        holes = list(face_scan.analyze_faces(shape).holes)
        diag = mesh_utils.shape_diagonal_mm(shape)
        seen = []
        kept, stats = hole_probe.probe_holes(shape, holes, diag,
                                             progress=lambda d, t: seen.append((d, t)))
        self.assertTrue(seen)
        self.assertEqual(len(kept), len(holes))
        # A wall the material check says is not there: rejected, and why.
        from unittest import mock

        with mock.patch.object(hole_probe, "_check_material_around",
                               return_value=(False, None)):
            kept, stats = hole_probe.probe_holes(shape, holes, diag)
        self.assertEqual(stats.holes_rejected, 1)
        self.assertEqual(len(kept), len(holes) - 1)
        gone = next(h for h in holes if h not in kept)
        self.assertIn("not enclosed", gone.reject_reason)
        kept, stats = hole_probe.probe_holes(shape, holes, diag, should_cancel=lambda: True)
        self.assertTrue(stats.cancelled)
        self.assertEqual(len(kept), len(holes))                  # nothing lost
        loose = SimpleNamespace(axis_direction=None, axis_point=None)
        kept, _s = hole_probe.probe_holes(shape, [loose], diag)
        self.assertEqual(kept, [loose])
        self.assertEqual(hole_probe._swept_degrees(SimpleNamespace(
            steps=[SimpleNamespace(angular_sweep_deg=90.0)])), 90.0)
        self.assertEqual(hole_probe._swept_degrees(SimpleNamespace(steps=[])), 360.0)


if __name__ == "__main__":
    unittest.main()
