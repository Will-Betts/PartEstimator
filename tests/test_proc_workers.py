"""The worker-process runner (machining/ui/proc_workers.py): its child entry
points called in this process, the in-process fallback when a process cannot
be started, cancelling, and a worker that dies without saying why.

Needs the CAD kernel and Qt; skipped in the light CI environment.
"""
from __future__ import annotations

import contextlib
import io
import os
import queue
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

if os.name != "nt" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    import OCP  # noqa: F401
    from PySide6.QtWidgets import QApplication

    HAVE = True
except Exception:  # noqa: BLE001
    HAVE = False

ROOT = str(Path(__file__).resolve().parents[1])


class _Refusing(queue.Queue):
    """A queue that drops progress messages on the floor, the way a closed
    pipe would: the child must carry on regardless."""

    def put(self, item, *a, **kw):
        if item[0] == "progress":
            raise BrokenPipeError("closed")
        return super().put(item, *a, **kw)


def _drain(q):
    out = []
    while not q.empty():
        out.append(q.get())
    return out


@unittest.skipUnless(HAVE, "needs the CAD kernel and Qt")
class TestChildEntries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from partestimator.machining.geometry import fixture, pipeline
        from tests import occparts

        cls.tmp = Path(tempfile.mkdtemp())
        with contextlib.redirect_stdout(io.StringIO()):
            cls.part = occparts.block(cls.tmp)
        cls.res = pipeline.compute_analysis(str(cls.part))
        cls.plan = fixture.plan_for_results(cls.res, cls.res["stock"], grow_stock=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_child(self, entry, *args, q=None):
        q = q if q is not None else queue.Queue()
        entry(ROOT, *args, q)
        return _drain(q)

    def kinds(self, msgs):
        return [k for k, _p in msgs]

    def test_every_entry_reports_a_result(self):
        from partestimator.machining.ui import proc_workers as pw

        cyl, holes = self.res["cylinder"], self.res["holes"]
        cases = [
            (pw._child_entry, (str(self.part), {})),
            (pw._child_thin_wall, (str(self.part), 1.0, 200.0)),
            (pw._child_turning, (str(self.part), cyl, holes)),
            (pw._child_fixture, (str(self.part), self.plan.to_dict(),
                                 str(self.tmp / "fx.step"))),
        ]
        for entry, args in cases:
            with self.subTest(entry=entry.__name__):
                msgs = self.run_child(entry, *args)
                self.assertIn("result", self.kinds(msgs))
                self.assertIn("progress", self.kinds(msgs))
                # Progress that cannot be delivered is not an error.
                msgs = self.run_child(entry, *args, q=_Refusing())
                self.assertEqual(self.kinds(msgs), ["result"])

    def test_every_entry_reports_an_error(self):
        from partestimator.machining.ui import proc_workers as pw

        bad = str(self.tmp / "missing.step")
        for entry, args in [(pw._child_entry, (bad, {})),
                            (pw._child_thin_wall, (bad, 1.0, 10.0)),
                            (pw._child_turning, (bad, None, [])),
                            (pw._child_fixture, (bad, self.plan.to_dict(), bad))]:
            with self.subTest(entry=entry.__name__):
                msgs = self.run_child(entry, *args)
                self.assertEqual(self.kinds(msgs)[-1], "error")
                name, message, tb = msgs[-1][1]
                self.assertTrue(name and tb)


@unittest.skipUnless(HAVE, "needs the CAD kernel and Qt")
class TestRunner(unittest.TestCase):
    def setUp(self):
        from partestimator.machining.ui import proc_workers as pw

        self.pw = pw
        self.app = QApplication.instance() or QApplication([])
        self.r = pw.AnalysisRunner()
        self.got = {"progress": [], "finished": [], "failed": [], "done": 0}
        self.r.progress.connect(self.got["progress"].append)
        self.r.finished.connect(self.got["finished"].append)
        self.r.failed.connect(lambda t, m: self.got["failed"].append((t, m)))
        self.r.done.connect(lambda: self.got.__setitem__("done", self.got["done"] + 1))

    def no_spawn(self):
        return mock.patch.object(self.pw.mp, "get_context",
                                 side_effect=OSError("no processes here"))

    def test_falls_back_to_running_in_process(self):
        def entry(_root, value, q):
            q.put((self.pw._PROGRESS, "working"))
            q.put((self.pw._RESULT, value * 2))

        with self.no_spawn():
            self.r._start(entry, (21,))
        self.assertEqual(self.got["finished"], [42])
        self.assertIn("working", self.got["progress"])
        self.assertTrue(any("in-process" in p for p in self.got["progress"]))
        self.assertEqual(self.got["done"], 1)

    def test_an_error_in_process(self):
        def entry(_root, q):
            q.put((self.pw._ERROR, ("ValueError", "bad", "tb")))

        with self.no_spawn():
            self.r._start(entry, ())
        self.assertEqual(self.got["failed"][0][0], "ValueError")
        self.assertIn("Traceback:\ntb", self.got["failed"][0][1])

    def test_cancelled_results_and_failures_are_swallowed(self):
        self.r._cancelled = True
        self.r._on_finished("late")
        self.r._on_failed("Late", "error")
        self.assertEqual((self.got["finished"], self.got["failed"]), ([], []))
        self.assertEqual(self.got["done"], 2)

    def test_cancel_terminates_a_live_child(self):
        proc = mock.Mock(is_alive=mock.Mock(return_value=True))
        self.r._process = proc
        self.assertTrue(self.r.is_running())
        self.r.cancel()
        proc.terminate.assert_called()
        self.assertFalse(self.r.is_running())


@unittest.skipUnless(HAVE, "needs the CAD kernel and Qt")
class TestQueuePump(unittest.TestCase):
    def setUp(self):
        from partestimator.machining.ui import proc_workers as pw

        self.pw = pw
        QApplication.instance() or QApplication([])

    def pump(self, q, process):
        p = self.pw._QueuePump(q, process)
        got = {"failed": [], "finished": [], "progress": []}
        p.failed.connect(lambda t, m: got["failed"].append((t, m)))
        p.finished.connect(got["finished"].append)
        p.progress.connect(got["progress"].append)
        return p, got

    def test_a_worker_that_died_is_reported(self):
        dead = mock.Mock(is_alive=mock.Mock(return_value=False), exitcode=-11)
        p, got = self.pump(queue.Queue(), dead)
        p.run()
        self.assertIn("exited unexpectedly", got["failed"][0][1])

    def test_a_worker_that_ended_cleanly_without_a_result_is_not_an_error(self):
        done = mock.Mock(is_alive=mock.Mock(return_value=False), exitcode=0)
        p, got = self.pump(queue.Queue(), done)
        p.run()
        self.assertEqual(got["failed"], [])

    def test_stopping_and_messages(self):
        q = queue.Queue()
        alive = mock.Mock(is_alive=mock.Mock(return_value=True))
        p, got = self.pump(q, alive)
        p.stop()
        p.run()                                          # stopped: returns at once
        q.put((self.pw._PROGRESS, "half way"))
        q.put((self.pw._ERROR, ("OSError", "disk", "tb")))
        p, got = self.pump(q, alive)
        p.run()
        self.assertEqual(got["progress"], ["half way"])
        self.assertEqual(got["failed"][0][0], "OSError")
        q.put((self.pw._RESULT, {"ok": 1}))
        p, got = self.pump(q, alive)
        p.run()
        self.assertEqual(got["finished"], [{"ok": 1}])


if __name__ == "__main__":
    unittest.main()
