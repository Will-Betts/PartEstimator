"""The sheet metal window's worker (smt/gui/worker.py), run synchronously:
an open whose drawings fail but whose numbers stand, an extruded blank that
also fails, an export that has to build the flat first, cancelling, an
error, and the job handle and relay.

Needs the CAD kernel and Qt.
"""
from __future__ import annotations

import contextlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import OCP  # noqa: F401
    from PySide6.QtWidgets import QApplication

    HAVE = True
except Exception:  # noqa: BLE001
    HAVE = False


@unittest.skipUnless(HAVE, "needs the CAD kernel and Qt")
class TestSheetWorker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests import occparts

        QApplication.instance() or QApplication([])
        cls.tmp = Path(tempfile.mkdtemp())
        with contextlib.redirect_stdout(io.StringIO()):
            cls.path = occparts.l_bracket(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_worker(self, worker):
        got = {"done": [], "failed": [], "progress": []}
        worker.finished.connect(got["done"].append)
        worker.failed.connect(lambda m, tb: got["failed"].append((m, tb)))
        worker.progress.connect(got["progress"].append)
        worker.run()
        return got

    def open_worker(self, stop=False):
        from partestimator.smt.config import UnfoldOptions
        from partestimator.smt.gui.worker import OpenWorker

        return OpenWorker(self.path, UnfoldOptions(), None, {"stop": stop})

    def test_drawings_that_fail_leave_the_numbers(self):
        from partestimator.smt.occ import tess

        with mock.patch.object(tess, "build_scene", side_effect=RuntimeError("no GL")):
            got = self.run_worker(self.open_worker())
        self.assertEqual(got["failed"], [])
        res = got["done"][0]
        self.assertIsNone(res.folded_scene)
        self.assertIsNone(res.flat_scene)
        self.assertGreater(res.flat2d.blank_length, 0)

    def test_no_solid_at_all(self):
        from partestimator.smt.occ import flatten, unfold3d

        with mock.patch.object(unfold3d, "unfold_solid", return_value=None), \
                mock.patch.object(flatten, "build_flat", side_effect=RuntimeError("kernel")):
            got = self.run_worker(self.open_worker())
        res = got["done"][0]
        self.assertTrue(res.extruded)
        self.assertIsNone(res.part.flat_solid)
        self.assertTrue(any("no 3D flat" in p or "extruded" in p for p in got["progress"]))

    def test_cancelled_and_failed(self):
        got = self.run_worker(self.open_worker(stop=True))
        self.assertEqual(got["failed"], [("Cancelled.", "")])
        from partestimator.smt.config import UnfoldOptions
        from partestimator.smt.gui.worker import OpenWorker

        got = self.run_worker(OpenWorker(self.tmp / "missing.step", UnfoldOptions(), None,
                                         {"stop": False}))
        message, tb = got["failed"][0]
        self.assertTrue(message and tb)

    def test_export_builds_the_flat_first(self):
        from partestimator.smt.config import UnfoldOptions
        from partestimator.smt.gui.worker import ExportWorker
        from partestimator.smt.pipeline import unfold_step

        opts = UnfoldOptions()
        part = unfold_step(self.path, opts)
        part.flat = None
        out = self.tmp / "export"
        got = self.run_worker(ExportWorker(part, opts, out, "b",
                                           {"dxf": True, "step": False, "json": False},
                                           {"stop": False}))
        self.assertEqual(set(got["done"][0]), {"report", "dxf"})
        self.assertIsNotNone(part.flat)

    def test_the_base_worker_and_the_job_handle(self):
        from partestimator.smt.gui.worker import JobHandle, _BaseWorker, _Relay

        with self.assertRaises(NotImplementedError):
            _BaseWorker({"stop": False}).work()
        job = JobHandle(thread=None, worker=None)
        self.assertFalse(job.is_cancelled())
        job.cancel()
        self.assertTrue(job.cancelled and job.is_cancelled())
        seen = []
        relay = _Relay(seen.append, lambda m, tb: seen.append((m, tb)), seen.append,
                       lambda: seen.append("released"))
        relay.fail("bad", "tb")
        self.assertEqual(seen, ["released", ("bad", "tb")])


if __name__ == "__main__":
    unittest.main()
