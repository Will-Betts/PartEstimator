"""The tuning library and its window, off the happy path: bad rows and files,
missing geometry, every table edit, removing and opening parts, fitting with
nothing ready, a real fit with a cross-check, saving failures, and the
command line's reports.

A part is captured once and its geometry reused for every entry, so the
library is quick to build. Needs the CAD kernel and Qt.
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

if os.name != "nt" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    import OCP  # noqa: F401
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

    HAVE = True
except Exception:  # noqa: BLE001
    HAVE = False

MAT = "Aluminum 6061-T6"


@unittest.skipUnless(HAVE, "needs the CAD kernel and Qt")
class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from partestimator.machining import tuning
        from partestimator.machining.geometry import pipeline, rates
        from tests import occparts

        cls.t = tuning
        cls.app = QApplication.instance() or QApplication([])
        cls.src = Path(tempfile.mkdtemp())
        with contextlib.redirect_stdout(io.StringIO()):
            cls.part = occparts.machined_plate(cls.src)
        res = pipeline.compute_analysis(str(cls.part), options=dict(tuning.ANALYSIS_OPTIONS))
        cls.snap = tuning.capture(res, res["stock"], MAT)
        cls.table = rates.load_rates()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.src, ignore_errors=True)

    def setUp(self):
        self.folder = Path(tempfile.mkdtemp()) / "lib"
        self.addCleanup(shutil.rmtree, self.folder.parent, ignore_errors=True)

    def library(self, n=7, extra=()):
        lib = self.t.Library.load(self.folder)
        for k in range(n):
            e = self.t.Entry(part=f"p{k}", id=f"id{k:02d}", material=MAT, operations=1 + k % 2,
                             stock_type="plate", stock_size=f"0.75 x {5 + k * 0.25} x 7",
                             measured_min=8.0 + 1.5 * k, source_file=str(self.part))
            lib.add(e, self.snap)
        for e, snap in extra:
            lib.add(e, snap) if snap is not None else lib.entries.append(e)
        lib.save()
        return lib


class TestLibraryEdges(Base):
    def test_bad_rows_and_snapshots(self):
        lib = self.library(2)
        csv_path = lib.csv_path
        text = csv_path.read_text(encoding="utf-8-sig")
        text += ",,,,,,,,,,,,,,,\n"                                       # a blank row
        text += "1,neg,idneg,,auto,63,,1,,plate,1 x 2 x 3,,-4,,,\n"       # negative time
        csv_path.write_text(text, encoding="utf-8-sig")
        again = self.t.Library.load(self.folder)
        self.assertEqual(len(again.entries), 2)
        self.assertTrue(any("negative" in e for e in again.errors))
        # A snapshot that is not JSON, or not a capture, is missing.
        bad = again.snapshot_path(again.entries[0])
        bad.write_text("{not json", encoding="utf-8")
        self.assertIsNone(again.load_snapshot(again.entries[0]))
        bad.write_text(json.dumps({"no": "features"}), encoding="utf-8")
        self.assertIsNone(again.load_snapshot(again.entries[0]))
        # Removing one whose snapshot is locked is still a removal.
        with mock.patch.object(Path, "unlink", side_effect=OSError("locked")):
            again.remove(again.entries[1])
        self.assertEqual(len(again.entries), 1)

    def test_parsing_helpers(self):
        t = self.t
        self.assertEqual(t._num("", 5), 5)
        self.assertEqual(t._num("x", 5), 5)
        self.assertEqual(t._num("3.0", 0, int), 3)
        self.assertIsNone(t._plain(float("inf")))
        self.assertEqual(t._plain(Path("a")), "a")
        self.assertIsNone(t.find_profile(""))
        self.assertIsNotNone(t.find_profile("angle|AND10133-2004"))
        with self.assertRaises(ValueError):
            t.stock_fields(None)
        for stype, size, prof in (("round", "1 x 2 x 3", ""), ("tube", "1", ""),
                                  ("extrusion", "6", "no such"),
                                  ("extrusion", "6 x 2", "angle|AND10133-2004")):
            with self.subTest(stype=stype, size=size):
                with self.assertRaises(ValueError):
                    t.build_stock(t.Entry(part="x", id="x", stock_type=stype,
                                          stock_size=size, stock_profile=prof))

    def test_overrides_file_with_bad_numbers_and_a_bad_summary(self):
        t = self.t
        self.folder.mkdir(parents=True)
        (self.folder / t.TUNING_FILE).write_text(
            "key,value,source,notes\n# a comment,1,,\ncalibrate_roughing,abc,fit,\n"
            "calibrate_finishing,1.2,fit,\n", encoding="utf-8")
        (self.folder / t.SUMMARY_FILE).write_text("{broken", encoding="utf-8")
        _prof, state = t.load_tuned_profile(self.folder)
        self.assertTrue(any("not a number" in e for e in state.errors))
        self.assertIsNone(state.summary)

    def test_prepare_library_reasons(self):
        t = self.t
        lib = self.library(1, extra=[
            (t.Entry(part="nomat", id="nm", stock_type="plate", stock_size="1 x 2 x 3",
                     measured_min=5.0), self.snap),
            (t.Entry(part="nosnap", id="ns", material=MAT, stock_type="plate",
                     stock_size="1 x 2 x 3", measured_min=5.0), None),
            (t.Entry(part="out", id="out", material=MAT, include=False, stock_type="plate",
                     stock_size="1 x 2 x 3", measured_min=5.0), self.snap)])
        base = t.fit_baseline(self.folder)
        with mock.patch.object(t, "prepare", side_effect=[ValueError("odd")] + [mock.DEFAULT] * 9,
                               wraps=t.prepare):
            ready, skipped = t.prepare_library(lib, self.table, base)
        reasons = sorted(skipped.values())
        self.assertTrue(any("no material" in r for r in reasons))
        self.assertIn("captured geometry missing", reasons)
        self.assertTrue(any("odd" in r for r in reasons))
        self.assertEqual(len(ready), 0)

    def test_fit_warnings_and_cancelling(self):
        t = self.t
        lib = self.library(7)
        base = t.fit_baseline(self.folder)
        ready, _s = t.prepare_library(lib, self.table, base)
        keys = [k.key for k in t.KNOBS if k.default]
        result = t.fit(ready, self.table, base, keys)
        self.assertIsNotNone(result.cross_checked)
        stopped = t.fit(ready, self.table, base, keys, cancelled=lambda: True)
        self.assertEqual(len(stopped.parts), len(ready))
        # A hand-set row is kept, not overwritten by the fit.
        (self.folder / t.TUNING_FILE).write_text(
            "key,value,source,notes\ncalibrate_roughing,1.5,manual,mine\n", encoding="utf-8")
        t.save_fit(result, self.folder)
        rows, _errors = t.read_overrides(self.folder)
        self.assertEqual((rows["calibrate_roughing"].value, rows["calibrate_roughing"].source),
                         (1.5, "manual"))


class TestCaptureFiles(Base):
    def test_missing_files_and_a_worker_that_raises(self):
        t = self.t
        missing = self.src / "gone.step"
        out = t.capture_files([missing], MAT, workers=1)
        self.assertEqual(out[missing], "file not found")
        with mock.patch.object(t, "_capture_worker", side_effect=RuntimeError("boom")), \
                mock.patch.object(t.cf, "ProcessPoolExecutor", _InlinePool):
            out = t.capture_files([self.part], MAT, workers=1)
        self.assertIn("RuntimeError: boom", out[self.part])

    def test_cancel_stops_the_pool(self):
        t = self.t
        with mock.patch.object(t.cf, "ProcessPoolExecutor", _SlowPool):
            out = t.capture_files([self.part], MAT, workers=1, cancelled=lambda: True)
        self.assertEqual(out, {})

    def test_kill_survives_a_dead_process(self):
        pool = SimpleNamespace(_processes={1: mock.Mock(terminate=mock.Mock(
            side_effect=OSError("gone")))}, shutdown=mock.Mock())
        self.t._kill(pool)
        pool.shutdown.assert_called_once()

    def test_add_files_reports_read_errors(self):
        t = self.t
        lib = t.Library.load(self.folder)
        with mock.patch.object(t, "capture_files", return_value={self.part: "unreadable"}):
            added, already, errors = t.add_files(lib, [self.part, self.src / "gone.step"], MAT)
        self.assertEqual(added, [])
        self.assertEqual(set(errors.values()), {"unreadable", "file not found"})


class _InlinePool:
    """A ProcessPoolExecutor that runs the job right here."""

    def __init__(self, *a, **k):
        pass

    def submit(self, fn, *args):
        import concurrent.futures as cf

        f = cf.Future()
        try:
            f.set_result(fn(*args))
        except Exception as exc:  # noqa: BLE001
            f.set_exception(exc)
        return f

    def shutdown(self, *a, **k):
        pass


class _SlowPool(_InlinePool):
    def submit(self, fn, *args):
        import concurrent.futures as cf

        return cf.Future()                     # never finishes


class TestCommandLine(Base):
    def run_cli(self, **kw):
        args = SimpleNamespace(dir=str(self.folder), add=None, material="", finish=63.0,
                               operations=1, workers=1, fit=None, save=False)
        for k, v in kw.items():
            setattr(args, k, v)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = self.t.run_cli(args)
        return rc, out.getvalue(), err.getvalue()

    def test_reports(self):
        t = self.t
        self.library(7, extra=[(t.Entry(part="nomat", id="nm", stock_type="plate",
                                        stock_size="1 x 2 x 3", measured_min=5.0), self.snap)])
        rc, out, err = self.run_cli(fit="no_such_key")
        self.assertEqual(rc, 2)
        self.assertIn("not shop-profile settings", err)
        self.assertIn("skipped nomat", out)
        rc, out, _err = self.run_cli(save=True)
        self.assertEqual(rc, 0)
        self.assertIn("Saved to", out)

    def test_nothing_ready_and_adding_with_problems(self):
        t = self.t
        self.library(0, extra=[(t.Entry(part="nomat", id="nm", stock_type="plate",
                                        stock_size="1 x 2 x 3", measured_min=5.0), self.snap)])
        rc, out, _err = self.run_cli()
        self.assertEqual(rc, 1)
        self.assertIn("Nothing to fit yet", out)
        with mock.patch.object(t, "add_files",
                               return_value=([], ["p0"], {"x.step": "unreadable"})):
            rc, out, err = self.run_cli(add=[str(self.part)], material="Unobtainium")
        self.assertEqual(rc, 1)
        self.assertIn("not a material the tool knows", err)
        self.assertIn("already in the library: p0", out)
        self.assertIn("unreadable", err)
        (self.folder / t.LIBRARY_FILE).write_text(
            (self.folder / t.LIBRARY_FILE).read_text(encoding="utf-8-sig")
            + "1,bad,idbad,,auto,63,,1,,cube,1,,5,,,\n", encoding="utf-8-sig")
        _rc, _out, err = self.run_cli()
        self.assertIn("warning:", err)


@unittest.skipUnless(HAVE, "needs the CAD kernel and Qt")
class TestTuningWindow(Base):
    def window(self, lib=None):
        from partestimator.machining.ui.tuning_window import TuningWindow

        lib = lib or self.library(7)
        w = TuningWindow(self.table, defaults=lambda: {"material": MAT, "finish_uin": 63.0,
                                                        "operations": 1},
                         folder=self.folder)
        self.addCleanup(w.close)
        return w

    def spin(self, w, timeout=120):
        t0 = time.time()
        while w.is_busy():
            self.app.processEvents()
            time.sleep(0.02)
            self.assertLess(time.time() - t0, timeout)
        self.app.processEvents()

    def test_odd_rows(self):
        from partestimator.machining.ui import tuning_window as tw

        t = self.t
        lib = self.library(1, extra=[
            (t.Entry(part="custom", id="cm", material="Unobtainium", stock_type="plate",
                     stock_size="1 x 2 x 3", measured_min=5.0), self.snap),
            (t.Entry(part="blank", id="bl", material="", stock_type="plate",
                     stock_size="1 x 2 x 3"), self.snap),
            (t.Entry(part="nosnap", id="ns", material=MAT, stock_type="plate",
                     stock_size="1 x 2 x 3"), None)])
        lib.errors.append("line 9: something")
        with mock.patch.object(t.Library, "load", return_value=lib):
            w = self.window(lib)
        statuses = [w.parts.item(r, tw.C_STATUS).text() for r in range(w.parts.rowCount())]
        self.assertTrue(any("captured geometry missing" in s for s in statuses))
        mats = [w.parts.cellWidget(r, tw.C_MAT) for r in range(w.parts.rowCount())]
        self.assertIn("Unobtainium", [mats[1].itemText(i) for i in range(mats[1].count())])
        self.assertEqual(mats[2].currentIndex(), -1)
        self.assertIn("something", w.lbl_parts.text())
        with mock.patch.object(t, "estimate", side_effect=ValueError("no rates")):
            w._fill_row_estimate(0)
        self.assertIn("no rates", w.parts.item(0, tw.C_STATUS).text())

    def test_every_table_edit(self):
        from partestimator.machining.ui import tuning_window as tw

        w = self.window()
        item = lambda c: w.parts.item(0, c)  # noqa: E731
        item(tw.C_PART).setText("renamed")
        item(tw.C_FINISH).setText("125")
        item(tw.C_FINISH).setText("not a number")            # put back
        item(tw.C_TAPPED).setText("auto")
        item(tw.C_TAPPED).setText("4")
        item(tw.C_MEASURED).setText("-2")                     # not a time
        item(tw.C_MEASURED).setText("")
        item(tw.C_USE).setCheckState(Qt.Unchecked)
        e = w.library.entries[0]
        self.assertEqual((e.part, e.finish_uin, e.tapped_holes, e.measured_min, e.include),
                         ("renamed", 125.0, 4, None, False))
        self.assertEqual(item(tw.C_FINISH).text(), "125")
        w.parts.item(0, tw.C_EST).setText("x")                # not editable data: ignored
        w._set(99, "part", "nobody")                          # no such row
        with mock.patch.object(w.library, "save", side_effect=OSError("excel")), \
                mock.patch.object(QMessageBox, "warning") as warn:
            w._set(0, "operations", 2)
        warn.assert_called_once()

    def test_add_remove_open(self):
        w = self.window()
        with mock.patch.object(QFileDialog, "getOpenFileNames", return_value=([], "")), \
                mock.patch.object(w, "add_paths") as add:
            w.add_files()
            add.assert_not_called()
        with mock.patch.object(QFileDialog, "getOpenFileNames",
                               return_value=([str(self.part)], "")), \
                mock.patch.object(w, "add_paths") as add:
            w.add_files()
            add.assert_called_once()
        w._thread = object()
        w.add_paths([self.part])                              # busy: ignored
        w.run_fit()
        w._thread = None
        w._on_added(([], ["p1", "p2"], {str(self.part): "unreadable"}))
        self.assertIn("could not be read", w.lbl_busy.text())
        w.remove_selected()                                   # nothing selected
        w.parts.selectRow(0)
        with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.No):
            w.remove_selected()
        self.assertEqual(len(w.library.entries), 7)
        with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
            w.remove_selected()
        self.assertEqual(len(w.library.entries), 6)
        w.parts.selectAll()
        w.open_selected()                                     # several: nothing
        w.parts.clearSelection()
        w.parts.selectRow(0)
        got = []
        w.openRequested.connect(lambda e, lib: got.append(e))
        w.open_selected()
        self.assertEqual(len(got), 1)
        with mock.patch("partestimator.machining.ui.tuning_window.QDesktopServices.openUrl") \
                as url, mock.patch.object(w.library, "ensure", side_effect=OSError("ro")):
            w.open_folder()
        url.assert_called_once()
        w.entry_saved()

    def test_fitting_paths(self):
        from partestimator.machining.ui import tuning_window as tw

        w = self.window()
        for r in range(w.knobs.rowCount()):
            w.knobs.item(r, 0).setCheckState(Qt.Unchecked)
        w.run_fit()
        self.assertIn("Tick at least one", w.lbl_summary.text())
        for r, k in enumerate(self.t.KNOBS):
            if k.default:
                w.knobs.item(r, 0).setCheckState(Qt.Checked)
        w.run_fit()
        self.spin(w)
        self.assertIsNotNone(w._fit)
        self.assertIn("did not see", w.lbl_summary.text())
        with mock.patch.object(self.t, "save_fit", side_effect=OSError("locked")), \
                mock.patch.object(QMessageBox, "warning") as warn:
            w.save_fit()
        warn.assert_called_once()
        with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.No):
            w.clear_fit()
        # Nothing ready: the fit says why instead of fitting.
        for e in w.library.entries:
            e.measured_min = None
        w.run_fit()
        self.spin(w)
        self.assertIn("no part is ready", w.lbl_busy.text())
        # A knob the profile switched off, and one set by hand.
        key_off = self.t.KNOBS[0].key
        with mock.patch.object(self.t, "fit_baseline",
                               return_value=SimpleNamespace(get=lambda k, d=0.0: 0.0
                                                            if k == key_off else 1.0)):
            w._fill_knobs()
        self.assertIn("off in the shop profile", w.knobs.item(0, 5).text())
        w.state.manual = {self.t.KNOBS[1].key}
        w._fit = None
        w._fill_knobs()
        self.assertIn("set by hand", w.knobs.item(1, 5).text())
        self.assertTrue(tw._err_colour(1.0))

    def test_cancel_and_close(self):
        w = self.window()
        w._worker = SimpleNamespace(cancel=False)
        w._cancel()
        self.assertTrue(w._worker.cancel)
        w.close()
        self.assertTrue(w._worker.cancel)
        w._worker = None
        from partestimator.machining.ui.tuning_window import _Worker

        failed = []
        worker = _Worker(lambda _w: (_ for _ in ()).throw(ValueError("x")))
        worker.failed.connect(failed.append)
        worker.run()
        self.assertEqual(failed, ["ValueError: x"])


if __name__ == "__main__":
    unittest.main()
