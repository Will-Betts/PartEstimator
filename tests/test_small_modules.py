"""Small modules the bigger tests never reach: the sheet metal command line
and its window entry point, the feature export, the theme, and the
deprecated machining worker stub.

The command line needs the CAD kernel; the theme needs Qt. Each skips
without them.
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)


@unittest.skipUnless(HAVE_OCP, "needs the CAD kernel")
class TestSheetCommandLine(Tmp):
    def setUp(self):
        super().setUp()
        from tests import occparts

        with contextlib.redirect_stdout(io.StringIO()):
            self.part = occparts.l_bracket(self.tmp)

    def run_cli(self, *args):
        from partestimator.smt import cli

        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = cli.main([str(self.part), "-o", str(self.tmp / "out"), *args])
        return rc, out.getvalue(), err.getvalue()

    def test_unfolds_and_writes_everything(self):
        rc, out, _err = self.run_cli("--material", "CRS", "--k", "0.44", "--scrap", "0.1",
                                     "--cost-per-kg", "2.5", "--press-tons", "40")
        self.assertEqual(rc, 0, out)
        written = sorted(p.suffix for p in (self.tmp / "out").iterdir())
        self.assertIn(".dxf", written)
        self.assertIn(".json", written)
        self.assertIn("wrote", out)

    def test_quiet_and_nothing_written(self):
        rc, out, _err = self.run_cli("-q", "--no-dxf", "--no-step", "--no-json",
                                     "--orient", "as_modelled", "--no-origin",
                                     "--no-gauge-snap", "--grain", "90")
        self.assertEqual(rc, 0)
        self.assertEqual(out, "")

    def test_a_deduction_table(self):
        table = self.tmp / "ded.csv"
        shutil.copy(Path(__file__).resolve().parents[1] / "deductions_example.csv", table)
        rc, _out, _err = self.run_cli("-q", "--deductions", str(table), "--no-step")
        self.assertEqual(rc, 0)

    def test_a_bad_file_is_an_error(self):
        from partestimator.smt import cli

        bad = self.tmp / "nope.step"
        bad.write_text("not a step file", encoding="utf-8")
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            rc = cli.main([str(bad), "-o", str(self.tmp)])
        self.assertEqual(rc, 1)
        self.assertIn("error:", err.getvalue())

    def test_the_parser_names_its_program(self):
        from partestimator.smt import cli

        self.assertEqual(cli.build_parser("x").prog, "x")


@unittest.skipUnless(HAVE_QT, "needs Qt")
class TestSheetWindowEntry(unittest.TestCase):
    def test_python_dash_m_smt_opens_the_window(self):
        from partestimator.smt import __main__ as smt_main

        with mock.patch("partestimator.smt.gui.app.run", return_value=0) as run, \
                mock.patch("partestimator.smt.diag.install_excepthook"), \
                mock.patch("partestimator.smt.diag.setup_logging"), \
                mock.patch.object(sys, "argv", ["smt", "part.step"]):
            self.assertEqual(smt_main.main(), 0)
        opts, path = run.call_args[0]
        self.assertEqual(path, "part.step")


class TestFeatureExport(Tmp):
    def estimate(self):
        term = lambda name, m: SimpleNamespace(name=name, minutes=m)  # noqa: E731
        return SimpleNamespace(
            door_to_door_minutes=12.3456, machine_cycle_minutes=10.0, cut_minutes=8.0,
            load_unload_minutes=1.5,
            terms=[term("Outside roughing", 3.0), term("Holes", 1.25), term("Other", 9.0)])

    def test_estimate_columns(self):
        from partestimator.machining.geometry import export

        self.assertEqual(export.estimate_columns(None), {})
        cols = export.estimate_columns(self.estimate())
        self.assertEqual(cols["estimated_cycle_time_min"], 12.346)
        self.assertEqual(cols["est_rough_outside_min"], 3.0)
        self.assertEqual(cols["est_hole_min"], 1.25)
        self.assertEqual(cols["est_finish_freeform_min"], 0.0)       # not in this estimate

    def test_build_row(self):
        from partestimator.machining.geometry import export

        stock = SimpleNamespace(basis="extrusion",
                                profile=SimpleNamespace(describe=lambda: "Angle 2 x 2"))
        row = export.build_row({"path": "C:/parts/bracket.step", "stock": stock},
                               {"volume_in3": 2.0}, material="6061", finish_value=63,
                               finish_units="uin", tolerance_in=0.005,
                               quality={"face_count": 40}, removal_type="mill",
                               estimate=self.estimate())
        self.assertEqual((row["part_name"], row["stock_basis"], row["stock_profile"]),
                         ("bracket", "extrusion", "Angle 2 x 2"))
        self.assertEqual((row["face_count"], row["volume_in3"]), (40, 2.0))
        blank = export.build_row({}, {})
        self.assertEqual((blank["part_name"], blank["target_finish"]), ("", ""))
        plain = export.build_row({"stock": SimpleNamespace(basis="recommended", profile=None)},
                                 {})
        self.assertEqual(plain["stock_profile"], "")

    def test_write_json(self):
        from partestimator.machining.geometry import export

        path = self.tmp / "row.json"
        export.write_json(path, {"a": 1, "when": Path("x")})
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"a": 1, "when": "x"})

    def test_append_csv_grows_its_columns_and_keeps_old_rows(self):
        from partestimator.machining.geometry import export

        path = self.tmp / "rows.csv"
        export.append_csv(path, {"part_name": "a", "volume_in3": 1})
        export.append_csv(path, {"part_name": "b", "volume_in3": 2})          # same columns
        export.append_csv(path, {"part_name": "c", "area_in2": 9})            # a new one
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual([r["part_name"] for r in rows], ["a", "b", "c"])
        self.assertEqual((rows[0]["area_in2"], rows[2]["volume_in3"]), ("", ""))
        with path.open(newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh))
        self.assertEqual(header[0], "part_name")                 # metadata first
        self.assertEqual(header[1:], ["area_in2", "volume_in3"])  # then features, sorted


@unittest.skipUnless(HAVE_QT, "needs Qt")
class TestTheme(unittest.TestCase):
    def test_apply(self):
        from partestimator import theme

        app = QApplication.instance() or QApplication([])
        old_style, old_sheet, old_palette = app.style().name(), app.styleSheet(), app.palette()
        try:
            theme.apply(app)
            self.assertIn(theme.ACCENT, app.styleSheet())
            from PySide6.QtGui import QPalette

            self.assertEqual(app.palette().color(QPalette.Highlight).name(), theme.ACCENT)
        finally:
            app.setStyleSheet(old_sheet)
            app.setPalette(old_palette)
            app.setStyle(old_style)


class TestDeprecatedWorkers(unittest.TestCase):
    def test_a_stale_import_fails_with_the_reason(self):
        with self.assertRaises(ImportError) as cm:
            from partestimator.machining.ui.workers import AnalysisWorker  # noqa: F401
        self.assertIn("proc_workers", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
