"""Sheet metal detection's gates, on the parts that trip them: a solid block,
a tube, a bent section too thick to be sheet, a face picked by hand that has
nothing behind it, and full (not simple) mode with its hole pass. Plus the
funnel's report and diagnosis, and thickness inference on hand-made inputs.

The real parts need the CAD kernel; the rest is plain Python.
"""
from __future__ import annotations

import contextlib
import math
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


@unittest.skipUnless(HAVE_OCP, "needs the CAD kernel")
class TestFunnel(unittest.TestCase):
    def funnel(self, **kw):
        from partestimator.smt.occ.detect import Funnel

        f = Funnel(faces=10, planes=6, cylinders=4, merged=4, pairs=2, thickness_used=1.5)
        for k, v in kw.items():
            setattr(f, k, v)
        return f

    def test_report_lists_every_rejection(self):
        text = self.funnel(rejected_thickness=[2.0, 2.0, 3.1], rejected_sweep=[0.01],
                           rejected_tangency=[(3.0, 1, 2)], accepted=1).report()
        for want in ("dr != thickness 3", "dr values: 2, 3.1", "sweep too small 1",
                     "not tangent", "R3.000: inner touched 1", "ACCEPTED as bends         1"):
            self.assertIn(want, text)

    def test_diagnosis(self):
        self.assertIsNone(self.funnel(accepted=1).diagnosis())
        self.assertIn("No cylindrical faces", self.funnel(cylinders=0).diagnosis())
        self.assertIn("CONCENTRIC PAIRS", self.funnel(pairs=0).diagnosis())
        self.assertIn("did not match the thickness",
                      self.funnel(rejected_thickness=[2.0]).diagnosis())
        self.assertIn("not tangent",
                      self.funnel(rejected_tangency=[(3.0, 1, 1)]).diagnosis())
        self.assertIn("see the funnel", self.funnel().diagnosis())


@unittest.skipUnless(HAVE_OCP, "needs the CAD kernel")
class TestInferThickness(unittest.TestCase):
    def setUp(self):
        from partestimator.smt.config import Tolerance
        from partestimator.smt.occ import detect

        self.d = detect
        self.tol = Tolerance()
        self.part = SimpleNamespace(warnings=[], warn=lambda m: self.part.warnings.append(m))

    def cand(self, dr):
        return SimpleNamespace(delta_r=dr)

    def plane(self, z, up=True):
        return SimpleNamespace(normal=np.array([0, 0, 1.0 if up else -1.0]),
                               point=np.array([0, 0, z]))

    def test_the_biggest_plausible_cluster_wins_and_is_cross_checked(self):
        cands = [self.cand(1.5), self.cand(1.5), self.cand(40.0), self.cand(40.0),
                 self.cand(40.0)]
        planes = [self.plane(0.0), self.plane(2.0, up=False)]      # 2 mm apart: no match
        t = self.d.infer_thickness(cands, planes, self.tol, self.part)
        self.assertAlmostEqual(t, 1.5)
        self.assertTrue(any("not confirmed" in w for w in self.part.warnings))

    def test_equal_support_is_ambiguous(self):
        with self.assertRaises(self.d.DetectionError) as cm:
            self.d.infer_thickness([self.cand(1.5), self.cand(3.0)], [], self.tol)
        self.assertIn("ambiguous", str(cm.exception))

    def test_nothing_plausible(self):
        with self.assertRaises(self.d.DetectionError) as cm:
            self.d.infer_thickness([self.cand(40.0)], [], self.tol)
        self.assertIn("no plausible sheet thickness", str(cm.exception))

    def test_no_bends_and_no_planes(self):
        with self.assertRaises(self.d.DetectionError) as cm:
            self.d.infer_thickness([], [], self.tol)
        self.assertIn("no coaxial cylinder pairs", str(cm.exception))


@unittest.skipUnless(HAVE_OCP, "needs the CAD kernel")
class TestGatesOnHandMadeCandidates(unittest.TestCase):
    """classify_bends, coaxial_pairs and pair_leftovers on inputs built by
    hand, with the tangency search stubbed: each gate tripped on purpose."""

    def setUp(self):
        from unittest import mock

        from partestimator.smt.config import Tolerance
        from partestimator.smt.occ import detect

        self.d, self.mock, self.tol = detect, mock, Tolerance()
        self.part = SimpleNamespace(warnings=[], warn=lambda m: self.part.warnings.append(m))

    def cand(self, dr=1.5, sweep=1.57, outer_sweep=None, radius=3.0):
        inner = SimpleNamespace(sweep=sweep, radius=radius)
        outer = SimpleNamespace(sweep=sweep if outer_sweep is None else outer_sweep,
                                radius=radius + dr)
        return SimpleNamespace(delta_r=dr, inner=inner, outer=outer)

    def classify(self, cands, planes_each):
        funnel = self.d.Funnel()
        with self.mock.patch.object(self.d, "tangent_planes",
                                    return_value=[object()] * planes_each):
            kept = self.d.classify_bends(cands, 1.5, {}, {}, self.tol, self.part, funnel)
        return kept, funnel

    def test_rejections(self):
        kept, funnel = self.classify([self.cand(dr=2.5), self.cand(sweep=0.001)], 2)
        self.assertEqual(kept, [])
        self.assertEqual((len(funnel.rejected_thickness), len(funnel.rejected_sweep)), (1, 1))
        kept, funnel = self.classify([self.cand()], 1)
        self.assertEqual(len(funnel.rejected_tangency), 1)

    def test_warnings_on_accepted_bends(self):
        cands = [self.cand(), self.cand(outer_sweep=1.2), self.cand(sweep=math.radians(170))]
        kept, funnel = self.classify(cands, 3)
        self.assertEqual(len(kept), 3)
        text = " ".join(self.part.warnings)
        self.assertIn("distinct planes", text)
        self.assertIn("sweeps differ", text)
        self.assertIn("hem or", text)

    def test_same_radius_cylinders_do_not_pair(self):
        axis = (np.zeros(3), np.array([0, 0, 1.0]))
        a = SimpleNamespace(radius=5.0, axis_point=axis[0], axis_dir=axis[1])
        b = SimpleNamespace(radius=5.0, axis_point=axis[0], axis_dir=axis[1])
        self.assertEqual(self.d.coaxial_pairs([a, b], self.tol), [])

    def test_leftover_faces_pair_by_thickness(self):
        uf = self.d._UnionFind()
        planes = {0: SimpleNamespace(normal=np.array([0, 0, 1.0]), point=np.zeros(3)),
                  1: SimpleNamespace(normal=np.array([0, 0, -1.0]), point=np.array([0, 0, -1.5])),
                  2: SimpleNamespace(normal=np.array([1.0, 0, 0]), point=np.zeros(3))}
        for k in planes:
            uf.find(k)
        self.d.pair_leftovers(uf, planes, 1.5, self.tol)
        self.assertEqual(uf.find(0), uf.find(1))
        self.assertNotEqual(uf.find(0), uf.find(2))

    def test_an_ambiguous_flat_blank(self):
        plane = SimpleNamespace(to_desc=lambda: None)
        with self.mock.patch.object(self.d, "thickness_candidates",
                                    return_value=[(1.5, 10.0), (3.0, 9.0)]):
            t = self.d.infer_thickness([], [plane, plane], self.tol, self.part)
        self.assertEqual(t, 1.5)
        self.assertTrue(any("could be" in w for w in self.part.warnings))


@unittest.skipUnless(HAVE_OCP, "needs the CAD kernel")
class TestDetectionOnParts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests import occparts

        cls.tmp = Path(tempfile.mkdtemp())
        o = occparts._occ()
        with contextlib.redirect_stdout(io.StringIO()):
            outer = o["BRepPrimAPI_MakeCylinder"](
                o["gp_Ax2"](o["gp_Pnt"](0, 0, 0), o["gp_Dir"](0, 0, 1)), IN, 3 * IN).Shape()
            inner = o["BRepPrimAPI_MakeCylinder"](
                o["gp_Ax2"](o["gp_Pnt"](0, 0, -1), o["gp_Dir"](0, 0, 1)),
                0.94 * IN, 3 * IN + 2).Shape()
            cls.tube = occparts._write(o["BRepAlgoAPI_Cut"](outer, inner).Shape(),
                                       cls.tmp / "tube.step")
            thick = occparts.section([(3.0, 0.4), (0.4, 0.4), (0.4, 2.0)], 0.5, 0.8, 3.0)
            cls.thick = occparts._write(thick, cls.tmp / "thick_l.step")
            cls.block = occparts.block(cls.tmp)
            cls.csk = occparts.countersunk_l_bracket(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def unfold(self, path, **opts):
        from partestimator.smt.config import UnfoldOptions
        from partestimator.smt.pipeline import unfold_step

        return unfold_step(path, UnfoldOptions(**opts))

    def test_a_block_is_a_flat_blank(self):
        part = self.unfold(self.block)
        self.assertTrue(part.is_blank)
        self.assertTrue(any("flat blank" in n for n in part.notes))

    def test_a_tube_has_no_flange(self):
        from partestimator.smt.occ.detect import DetectionError

        with self.assertRaises(DetectionError) as cm:
            self.unfold(self.tube)
        self.assertIn("no flange", str(cm.exception))

    def test_too_thick_for_sheet_unless_picked_by_hand(self):
        from partestimator.smt.occ.detect import DetectionError

        with self.assertRaises(DetectionError) as cm:
            self.unfold(self.thick)
        self.assertIn("no plausible sheet thickness", str(cm.exception))
        part = self.unfold(self.thick, sheet_face=0)
        self.assertEqual(len(part.bends), 1)
        self.assertTrue(any("outside the range" in w for w in part.warnings))

    def test_a_picked_face_with_nothing_behind_it(self):
        from partestimator.smt.features import Role

        part = self.unfold(self.csk)
        bend_face = next(f.index for f in part.features.values()
                         if f.role in (Role.BEND_INNER, Role.BEND_OUTER))
        again = self.unfold(self.csk, sheet_face=bend_face)
        self.assertTrue(any("nothing lies parallel" in w for w in again.warnings))
        self.assertEqual(len(again.bends), 1)                   # fell back to automatic

    def test_full_mode_finds_the_holes(self):
        from tests import occparts

        with contextlib.redirect_stdout(io.StringIO()):
            bracket = occparts.l_bracket(self.tmp)
            blank = occparts.flat_blank(self.tmp)
        part = self.unfold(bracket, simple_mode=False)
        self.assertEqual(len(part.holes), 1)
        self.assertAlmostEqual(part.holes[0].diameter, 0.25 * IN, places=2)
        self.assertEqual(len(self.unfold(blank, simple_mode=False).holes), 2)
        # A countersink is its own feature, not a plain hole.
        self.assertEqual(len(self.unfold(self.csk, simple_mode=False).holes), 0)


if __name__ == "__main__":
    unittest.main()
