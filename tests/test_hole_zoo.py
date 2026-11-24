"""Every kind of hole, on one plate (occparts.hole_zoo): through, blind with
a drill point, counterbored, countersunk, an internal groove, blind from
the bottom with a flat floor -- and a boss whose base fillet is a concave
torus. Then the face scan's slower modes (robust concavity, the undercut
check), its progress and cancelling, and the bounding-box reductions a big
mesh triggers.

Needs the CAD kernel.
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


@unittest.skipUnless(HAVE, "needs the CAD kernel")
class TestHoleZoo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from partestimator.machining.geometry import pipeline
        from tests import occparts

        cls.tmp = Path(tempfile.mkdtemp())
        with contextlib.redirect_stdout(io.StringIO()):
            cls.path = occparts.hole_zoo(cls.tmp)
        cls.res = pipeline.compute_analysis(str(cls.path))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def hole_at(self, x, y, dia=None):
        for h in self.res["holes"]:
            p = np.asarray(h.axis_point) / IN
            if abs(p[0] - x) < 0.02 and abs(p[1] - y) < 0.02 and \
                    (dia is None or abs(h.nominal_diameter_mm / IN - dia) < 0.01):
                return h
        self.fail(f"no hole at ({x}, {y})")

    def test_each_hole_is_what_it_is(self):
        cases = [((0.75, 1), "plain", "through"),
                 ((1.75, 1), "plain", "blind_tapered"),
                 ((2.75, 1), "counterbore", "through"),
                 ((3.75, 1), "countersink", "through"),
                 ((1.0, 3), "plain", "through"),
                 ((2.5, 3), "plain", "blind_flat")]
        for (x, y), entry, term in cases:
            with self.subTest(at=(x, y)):
                h = self.hole_at(x, y)
                self.assertEqual((h.entry_type, h.termination), (entry, term))
        drilled = self.hole_at(1.75, 1)
        self.assertAlmostEqual(drilled.point_angle_deg, 118.0, delta=0.5)
        self.assertFalse(drilled.is_through)
        self.assertTrue(self.hole_at(2.75, 1).is_counterbored)
        self.assertTrue(self.hole_at(3.75, 1).is_countersunk)
        self.assertEqual(self.hole_at(1.0, 3).internal_groove_count, 1)

    def test_the_counts_reach_the_features(self):
        d = self.res["derived"]
        self.assertEqual(d["hole_counterbored_count"], 1)
        self.assertGreaterEqual(d["hole_countersunk_count"], 1)
        self.assertGreaterEqual(d["hole_blind_count"], 2)

    def test_the_boss_fillet_is_a_concave_torus(self):
        tori = [f for f in self.res["fillets"] if f.surface_type == "torus"]
        self.assertEqual([round(f.radius_mm / IN, 3) for f in tori], [0.125])

    def test_classify_holes_on_its_own(self):
        from partestimator.machining.geometry import derived

        bore = SimpleNamespace(axis_point=np.zeros(3), axis_direction=np.array([0, 0, 1.0]),
                               radius_mm=3.0)
        wide = SimpleNamespace(axis_point=np.array([0, 0, 5.0]),
                               axis_direction=np.array([0, 0, -1.0]), radius_mm=5.0)
        loose = SimpleNamespace(axis_point=None, axis_direction=None, radius_mm=1.0)
        point = SimpleNamespace(axis_point=np.zeros(3), axis_direction=np.array([0, 0, 1.0]),
                                radius_min_mm=0.0, radius_max_mm=3.0)
        flare = SimpleNamespace(axis_point=np.zeros(3), axis_direction=np.array([0, 0, 1.0]),
                                radius_min_mm=3.0, radius_max_mm=6.0)
        out = derived.classify_holes([bore, wide, loose], [point, flare])
        # The wide bore shares the drill point's axis, so it reads blind too.
        self.assertEqual(out["blind"], [True, True, False])
        self.assertEqual(out["countersunk"], [True, True, False])
        self.assertEqual(out["counterbored"], [True, False, False])
        self.assertEqual(derived.DerivedFeatures({"a": 1})["a"], 1)
        self.assertEqual(derived.DerivedFeatures({"a": 1}).get("b", 2), 2)
        self.assertEqual(derived.count_planar_levels([]), (0, 0))

    def test_slow_scan_modes_agree(self):
        from partestimator.machining.geometry import face_scan, mesh_utils, step_io

        shape = step_io.load_step_file(str(self.path))
        mesh_utils.tessellate(shape)
        fast = face_scan.analyze_faces(shape)
        robust = face_scan.analyze_faces(shape, robust=True, do_undercut=True)
        self.assertEqual(len(robust.holes), len(fast.holes))
        self.assertEqual(len(robust.fillets), len(fast.fillets))
        self.assertGreaterEqual(robust.undercut.flagged_face_count, 0)
        empty = face_scan.FaceScanResult(holes=[], fillets=[], surface_split={},
                                         undercut=None, face_count=0)
        self.assertEqual((empty.cones, empty.planes), ([], []))

    def test_reference_finder_agrees(self):
        from partestimator.machining.geometry import round_surfaces, step_io

        shape = step_io.load_step_file(str(self.path))
        holes, fillets = round_surfaces.find_hole_and_fillet_faces(shape)
        self.assertEqual(sorted(round(f.radius_mm / IN, 3) for f in fillets
                                if f.surface_type == "torus"), [0.125])
        groups = round_surfaces.group_hole_area_by_radius(holes)
        self.assertEqual(sum(g["hole_count"] for g in groups), len(holes))
        self.assertEqual(groups, sorted(groups, key=lambda g: g["nominal_diameter_mm"]))
        self.assertAlmostEqual(round_surfaces.smallest_internal_radius_mm(fillets) / IN,
                               min(f.radius_mm for f in fillets) / IN)
        self.assertIsNone(round_surfaces.smallest_internal_radius_mm([]))


@unittest.skipUnless(HAVE, "needs the CAD kernel")
class TestScanProgressAndCancel(unittest.TestCase):
    def test_progress_and_cancel(self):
        from partestimator.machining.geometry import face_scan, mesh_utils, step_io
        from tests import occparts

        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        with contextlib.redirect_stdout(io.StringIO()):
            path = occparts.many_holes_plate(tmp)
        shape = step_io.load_step_file(str(path))
        mesh_utils.tessellate(shape)
        seen = []
        scan = face_scan.analyze_faces(shape, progress=lambda d, t: seen.append(d))
        self.assertGreater(scan.face_count, 256)
        self.assertEqual(seen[-1], scan.face_count)
        with self.assertRaises(face_scan.ScanCancelled):
            face_scan.analyze_faces(shape, should_cancel=lambda: True)


@unittest.skipUnless(HAVE, "needs the CAD kernel")
class TestAngularUnion(unittest.TestCase):
    def test_old_records_and_wrapping(self):
        from partestimator.machining.geometry import round_surfaces as rs

        def face(sweep, start=0.0, x_dir=(1.0, 0, 0)):
            return SimpleNamespace(sweep_rad=sweep, sweep_start=start,
                                   x_dir=None if x_dir is None else np.array(x_dir),
                                   axis_direction=np.array([0, 0, 1.0]))

        self.assertAlmostEqual(rs._angular_union([face(4.0, x_dir=None), face(4.0)]),
                               2 * np.pi)
        self.assertAlmostEqual(rs._angular_union([face(1.0), face(1.0, x_dir=None)]), 2.0)
        self.assertAlmostEqual(rs._angular_union([face(2 * np.pi)]), 2 * np.pi)
        self.assertEqual(rs._angular_union([face(0.0)]), 0.0)
        wrapped = rs._angular_union([face(1.0, start=2 * np.pi - 0.5), face(0.5, start=3.0)])
        self.assertAlmostEqual(wrapped, 1.5)


@unittest.skipUnless(HAVE, "needs the CAD kernel")
class TestBigMeshBox(unittest.TestCase):
    def test_reduced_hull_and_subsampled_search(self):
        from partestimator.machining.geometry import bounding_box as bb

        rng = np.random.default_rng(1)
        pts = rng.normal(size=(30000, 3)) * np.array([30.0, 20.0, 5.0])
        with mock.patch.object(bb, "HULL_INPUT_LIMIT", 1000):
            hull = bb.prepare_hull(pts)
        self.assertTrue(hull.hull_is_approximate)
        box = bb.minimum_oriented_box(pts, hull)
        proj = pts @ box.axes.T
        centre = box.center_mm @ box.axes.T
        self.assertTrue(np.all(np.abs(proj - centre) <= box.extents_mm / 2 + 1e-6))
        self.assertEqual(len(box.dims_sorted_desc), 3)
        self.assertEqual(box.longest_axis_direction.shape, (3,))
        few = bb._reduce_for_hull(pts, n_directions=16, scan_limit=500, chunk=8)
        self.assertLess(len(few), 600)

    def test_degenerate_points_cannot_be_boxed(self):
        from partestimator.machining.geometry import bounding_box as bb

        hull = SimpleNamespace(points=np.zeros((4, 3)), search_points=np.zeros((4, 3)),
                               candidate_normals=np.array([[0, 0, 1.0]]),
                               all_points=np.zeros((4, 3)))
        with self.assertRaises(ValueError):
            bb.minimum_oriented_box(np.zeros((4, 3)), hull)


if __name__ == "__main__":
    unittest.main()
