"""Guards in the machining geometry: edge convexity cancelled or fed edges it
cannot read, thin-wall and undercut scans on awkward inputs, a turned part
with a spherical end and a filleted shoulder, and the axis statistics
without scipy.
"""
from __future__ import annotations

import contextlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

try:
    import OCP  # noqa: F401

    HAVE = True
except Exception:  # noqa: BLE001
    HAVE = False

IN = 25.4


class TestAxisStats(unittest.TestCase):
    def test_without_scipy_and_empty(self):
        from partestimator.machining.geometry import axis_stats as ax

        axes = ax.canonical_axes([[0, 0, 1.0], [0, 0, -1.0], [1, 0, 0.0], None])
        with mock.patch.object(ax, "cKDTree", None):
            labels, sizes, reps = ax.cluster_axes(axes, 1.0)
        self.assertEqual(sorted(sizes), [1, 2])
        self.assertEqual(ax.canonical_axes([None]).shape, (0, 3))
        labels, sizes, reps = ax.cluster_axes(np.zeros((0, 3)), 1.0)
        self.assertEqual(sizes, [])
        self.assertEqual(ax.angular_deviations_deg(np.zeros((0, 3)), [0, 0, 1]).size, 0)


@unittest.skipUnless(HAVE, "needs the CAD kernel")
class TestWeightedHelpers(unittest.TestCase):
    def test_percentiles_and_fractions(self):
        from partestimator.machining.geometry import features

        self.assertTrue(np.isnan(features._weighted_percentile(np.zeros(0), np.zeros(0), 0.1)))
        self.assertEqual(features._weighted_percentile(np.array([1.0, 2, 3]),
                                                        np.zeros(3), 0.5), 2.0)
        self.assertEqual(features.ThinWallResult().thin_area_fraction, 0.0)


@unittest.skipUnless(HAVE, "needs the CAD kernel")
class TestKernelGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from partestimator.machining.geometry import mesh_utils, step_io
        from tests import occparts

        cls.tmp = Path(tempfile.mkdtemp())
        with contextlib.redirect_stdout(io.StringIO()):
            path = occparts.many_holes_plate(cls.tmp)
        cls.plate = step_io.load_step_file(str(path))
        mesh_utils.tessellate(cls.plate)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_edge_convexity(self):
        from partestimator.machining.geometry import edge_convexity as ec

        full = ec.compute_edge_convexity(self.plate)
        stopped = ec.compute_edge_convexity(self.plate, should_cancel=lambda: True)
        self.assertLess(stopped.convex_edge_count + stopped.concave_edge_count,
                        full.convex_edge_count + full.concave_edge_count)
        with mock.patch.object(ec, "_uv_on_face", return_value=None):
            self.assertEqual(ec.compute_edge_convexity(self.plate).convex_edge_count, 0)
        with mock.patch.object(ec, "sampled_outward_normal", return_value=None):
            self.assertEqual(ec.compute_edge_convexity(self.plate).convex_edge_count, 0)
        with mock.patch.object(ec, "BRepAdaptor_Curve", side_effect=RuntimeError):
            self.assertEqual(ec.compute_edge_convexity(self.plate).convex_edge_count, 0)
        with mock.patch.object(ec.BRep_Tool, "CurveOnSurface_s", return_value=None):
            self.assertIsNone(ec._uv_on_face(None, None, 0.0))
        with mock.patch.object(ec.BRep_Tool, "CurveOnSurface_s", return_value=(None, 0, 1)):
            self.assertIsNone(ec._uv_on_face(None, None, 0.0))

    def test_thin_wall_and_undercut_scans(self):
        from partestimator.machining.geometry import features

        r = features.detect_thin_walls(self.plate, 50.0, 2.0, should_cancel=lambda: True)
        self.assertEqual(r.samples_checked, 0)
        everything = features.detect_thin_walls(self.plate, 50.0, 2.0, quick=True)
        big_first = [1e6] + [1e-4] * 10000                     # all but one are slivers
        r = features.detect_thin_walls(self.plate, 50.0, 2.0, quick=True,
                                       face_areas=big_first)
        self.assertLess(r.samples_checked, everything.samples_checked)
        with mock.patch.object(features, "sampled_outward_normal", return_value=None):
            u = features.detect_undercut_faces(self.plate)
        self.assertEqual(u.flagged_face_count, 0)
        hit = mock.Mock(Face=mock.Mock(side_effect=RuntimeError))
        self.assertTrue(features._is_opposing_hit(hit, np.array([0, 0, 1.0])))

    def test_edges(self):
        from partestimator.machining.geometry import edges

        self.assertFalse(edges._is_seam([object()]))
        lines = edges.extract_edge_polylines(self.plate, diagonal_mm=0.0)
        self.assertTrue(lines)
        with mock.patch.object(edges, "GCPnts_QuasiUniformDeflection",
                               return_value=mock.Mock(IsDone=mock.Mock(return_value=False))):
            self.assertEqual(edges.extract_edge_polylines(self.plate), [])
        with mock.patch.object(edges, "BRepAdaptor_Curve", side_effect=RuntimeError):
            self.assertEqual(edges.extract_edge_polylines(self.plate), [])
        pts, cells = edges.polylines_to_line_cells([])
        self.assertEqual((pts.shape, cells.shape), ((0, 3), (0,)))

    def test_a_turned_part_with_a_ball_end_and_a_filleted_shoulder(self):
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
        from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere
        from OCP.BRepAdaptor import BRepAdaptor_Curve
        from OCP.GeomAbs import GeomAbs_Circle
        from OCP.TopAbs import TopAbs_EDGE
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS
        from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
        from partestimator.machining.geometry import bounding_box, mesh_utils, turning

        big = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 20.0, 40.0)
        small = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, 40), gp_Dir(0, 0, 1)), 10.0, 30.0)
        s = BRepAlgoAPI_Fuse(big.Shape(), small.Shape()).Shape()
        s = BRepAlgoAPI_Fuse(s, BRepPrimAPI_MakeSphere(gp_Pnt(0, 0, 70), 10.0).Shape()).Shape()
        fil = BRepFilletAPI_MakeFillet(s)
        ex = TopExp_Explorer(s, TopAbs_EDGE)
        while ex.More():
            e = TopoDS.Edge_s(ex.Current())
            c = BRepAdaptor_Curve(e)
            if c.GetType() == GeomAbs_Circle and abs(c.Circle().Radius() - 10.0) < 1e-6 \
                    and abs(c.Circle().Location().Z() - 40.0) < 1e-6:
                fil.Add(3.0, e)
            ex.Next()
        s = fil.Shape()
        mesh_utils.tessellate(s)
        pts, tris = mesh_utils.extract_mesh(s)
        cyl = bounding_box.minimum_enclosing_cylinder(pts)
        holes = [SimpleNamespace(axis_direction=None, axis_point=None)]
        tg = turning.analyze_turning(s, cyl, pts, tris, holes)
        self.assertGreater(tg.turned_fraction, 0.9)
        self.assertAlmostEqual(tg.max_diameter_mm, 40.0, delta=0.5)
        self.assertIsNone(turning.analyze_turning(s, None, pts, tris))
        self.assertIsNone(turning.analyze_turning(s, cyl, np.zeros((0, 3)), np.zeros((0, 3))))
        np.testing.assert_array_equal(turning._fill_gaps(np.full(4, -1.0)), np.zeros(4))


if __name__ == "__main__":
    unittest.main()
