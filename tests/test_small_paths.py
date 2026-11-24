"""Small paths in classification, the rate table, STEP reading and the
calibration report: several bodies, no measurement possible, a scan out of
time, a bend detector that declines, odd rate files, files that are not STEP,
and calibrating from one part or without numpy.
"""
from __future__ import annotations

import contextlib
import csv
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

try:
    import OCP  # noqa: F401

    HAVE_OCP = True
except Exception:  # noqa: BLE001
    HAVE_OCP = False


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)


@unittest.skipUnless(HAVE_OCP, "needs the CAD kernel")
class TestClassifyPaths(Tmp):
    def test_two_bodies_and_no_measurement(self):
        from OCP.BRep import BRep_Builder
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeSphere
        from OCP.TopoDS import TopoDS_Compound
        from partestimator import classify

        comp, b = TopoDS_Compound(), BRep_Builder()
        b.MakeCompound(comp)
        b.Add(comp, BRepPrimAPI_MakeBox(10.0, 10.0, 1.0).Shape())
        from OCP.gp import gp_Pnt

        b.Add(comp, BRepPrimAPI_MakeBox(gp_Pnt(20.0, 0.0, 0.0), 10.0, 10.0, 1.0).Shape())
        c = classify.classify_shape(comp)
        self.assertEqual((c.kind, c.solids), (classify.MACHINED, 2))
        self.assertFalse(c.is_sheet)
        empty = classify._judge([], 0.19, None)
        self.assertIn("No through-thickness", empty.reason)
        sphere = BRepPrimAPI_MakeSphere(10.0).Shape()
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.TopAbs import TopAbs_FACE
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS

        face = TopoDS.Face_s(TopExp_Explorer(sphere, TopAbs_FACE).Current())
        self.assertIsNone(classify._band_width(face, BRepAdaptor_Surface(face, True)))

    def test_out_of_time_and_a_declining_detector(self):
        from partestimator import classify
        from tests import occparts

        with contextlib.redirect_stdout(io.StringIO()):
            path = occparts.l_bracket(self.tmp)
        with mock.patch.object(classify, "TIME_BUDGET_S", -1.0):
            c = classify.classify_file(path)
        self.assertTrue(any("time budget" in n for n in c.notes))
        with mock.patch("partestimator.smt.occ.detect.detect", side_effect=ValueError("no")):
            n, why = classify._bend_count(None)
        self.assertEqual(n, 0)
        self.assertIn("declined", why)
        self.assertEqual(classify._weighted_quantile([3.0, 1.0, 2.0], [0, 0, 0], 0.5), 2.0)


class TestRates(Tmp):
    def test_table_helpers_and_bad_files(self):
        from partestimator.machining.geometry import rates

        table = rates.load_rates()
        self.assertIn("Aluminum 6061-T6", table.materials)
        self.assertEqual(len(table.materials), len(set(table.materials)))
        self.assertEqual(table.iso_group("Unobtainium"), "?")
        src = Path(rates.__file__).resolve().parents[1] / "data" / "machining_rates.csv"
        lines = src.read_text(encoding="utf-8").splitlines()
        bad = self.tmp / "rates.csv"
        short = "Aluminum 6061-T6,N,mill"                    # a row cut short
        bad.write_text("\n".join(lines[:2] + [short]) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "wrong number of fields"):
            rates.load_rates(bad)


@unittest.skipUnless(HAVE_OCP, "needs the CAD kernel")
class TestStepIo(Tmp):
    def test_not_step_and_unparseable(self):
        from partestimator.machining.geometry import step_io

        txt = self.tmp / "a.txt"
        txt.write_text("x", encoding="utf-8")
        with self.assertRaisesRegex(step_io.StepLoadError, "Not a STEP file"):
            step_io.load_step_file(txt)
        junk = self.tmp / "junk.step"
        junk.write_text("not step at all", encoding="utf-8")
        with self.assertRaises(step_io.StepLoadError):
            step_io.load_step_file(junk)
        self.assertAlmostEqual(step_io.mm5_to_display(25.4 ** 5, "IN"), 1.0)
        self.assertEqual(step_io.mm5_to_display(7.0, "MM"), 7.0)
        from OCP.IFSelect import IFSelect_RetDone

        reader = mock.Mock(ReadFile=mock.Mock(return_value=IFSelect_RetDone),
                           TransferRoots=mock.Mock(return_value=0))
        good = self.tmp / "ok.step"
        good.write_text("x", encoding="utf-8")
        with mock.patch.object(step_io, "STEPControl_Reader", return_value=reader):
            with self.assertRaisesRegex(step_io.StepLoadError, "no transferable roots"):
                step_io.load_step_file(good)
            reader.TransferRoots.return_value = 1
            reader.OneShape.return_value = mock.Mock(IsNull=mock.Mock(return_value=True))
            with self.assertRaisesRegex(step_io.StepLoadError, "no geometry"):
                step_io.load_step_file(good)


class TestCalibrate(Tmp):
    def table(self, n):
        path = self.tmp / "t.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["part_name", "cycle_time_min",
                                               "estimated_cycle_time_min",
                                               "est_rough_outside_min", "est_hole_min"])
            w.writeheader()
            for k in range(n):
                w.writerow({"part_name": f"p{k}", "cycle_time_min": 10 + k,
                            "estimated_cycle_time_min": 8 + k * 0.5,
                            "est_rough_outside_min": 4 + k * 0.3, "est_hole_min": 1})
        return path

    def run_main(self, argv):
        from partestimator.machining import calibrate

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = calibrate.main(argv)
        return rc, out.getvalue()

    def test_usage_one_part_and_no_numpy(self):
        rc, out = self.run_main([])
        self.assertEqual(rc, 2)
        self.assertIn("Usage", out)
        self.assertEqual(self.run_main(["--help"])[0], 0)
        rc, out = self.run_main([str(self.table(1))])
        self.assertEqual(rc, 0)
        self.assertIn("calibrate_overall", out)
        with mock.patch.dict(sys.modules, {"numpy": None}):
            rc, out = self.run_main([str(self.table(30))])
        self.assertIn("numpy not available", out)


if __name__ == "__main__":
    unittest.main()
