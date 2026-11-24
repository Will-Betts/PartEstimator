"""Run the analysis in a separate PROCESS, not just a thread.

Why a process and not a QThread
-------------------------------
OCP is a pybind11 binding, and pybind11 does not release the GIL around
bound C++ calls by default. A single long-running OCCT call --
`STEPControl_Reader.ReadFile()`, `BRepMesh_IncrementalMesh.Perform()` --
therefore holds the GIL for its entire duration. While that one call runs:

  * the Qt UI thread cannot execute, so nothing repaints and no status
    message ever appears;
  * queued progress signals cannot be delivered;
  * a cooperative cancel flag can never be polled.

Moving the work to a QThread does not help, because the thread still
contends for the same GIL. That is exactly the "it freezes and I see no
progress" symptom on complex parts.

A child process has its own interpreter and its own GIL, so:

  * the UI thread keeps running and progress actually renders;
  * cancellation is `terminate()`, which works even in the middle of a
    C++ call that would otherwise be uninterruptible;
  * a hard crash inside OCCT kills the worker, not the whole application.

The parent side spins one QThread whose only job is to block on a queue
and re-emit messages as Qt signals. That thread does no real work, so it
never holds the GIL for long.

Fallback: if a process can't be started (restricted environments, odd
frozen builds), `AnalysisRunner` falls back to running in-process on a
thread, which behaves exactly as before -- correct, just not responsive.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import traceback

from PySide6.QtCore import QObject, QThread, Signal

# Sentinels for the queue protocol.
_PROGRESS = "progress"
_RESULT = "result"
_ERROR = "error"


def _child_entry(project_root: str, path: str, options: dict, queue) -> None:
    """Runs in the child process. Must be a module-level function so it can
    be pickled for the 'spawn' start method used on Windows."""
    try:
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from partestimator.machining.geometry import pipeline

        def progress(msg):
            try:
                queue.put((_PROGRESS, msg))
            except Exception:  # noqa: BLE001 - never let reporting kill the run
                pass

        result = pipeline.compute_analysis(path, options, progress=progress)
        queue.put((_RESULT, result))
    except Exception as exc:  # noqa: BLE001
        queue.put((_ERROR, (type(exc).__name__, str(exc), traceback.format_exc())))


def _child_thin_wall(project_root: str, path: str, threshold_mm: float,
                      max_distance_mm: float, queue) -> None:
    try:
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from partestimator.machining.geometry import pipeline

        def progress(msg):
            try:
                queue.put((_PROGRESS, msg))
            except Exception:  # noqa: BLE001
                pass

        result = pipeline.compute_thin_wall(path, threshold_mm, max_distance_mm, progress=progress)
        queue.put((_RESULT, result))
    except Exception as exc:  # noqa: BLE001
        queue.put((_ERROR, (type(exc).__name__, str(exc), traceback.format_exc())))


def _child_turning(project_root: str, path: str, cylinder, holes, queue) -> None:
    try:
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from partestimator.machining.geometry import pipeline

        def progress(msg):
            try:
                queue.put((_PROGRESS, msg))
            except Exception:  # noqa: BLE001
                pass

        result = pipeline.compute_turning(path, cylinder, holes, progress=progress)
        queue.put((_RESULT, result))
    except Exception as exc:  # noqa: BLE001
        queue.put((_ERROR, (type(exc).__name__, str(exc), traceback.format_exc())))


def _child_fixture(project_root: str, path: str, plan: dict, out_path: str, queue) -> None:
    try:
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from partestimator.machining.geometry import fixture_cad

        def progress(msg):
            try:
                queue.put((_PROGRESS, msg))
            except Exception:  # noqa: BLE001
                pass

        result = fixture_cad.export(path, plan, out_path, progress=progress)
        queue.put((_RESULT, result))
    except Exception as exc:  # noqa: BLE001
        queue.put((_ERROR, (type(exc).__name__, str(exc), traceback.format_exc())))


class _QueuePump(QObject):
    """Blocks on the child's queue and re-emits entries as Qt signals."""

    progress = Signal(str)
    finished = Signal(object)
    failed = Signal(str, str)

    def __init__(self, queue, process):
        super().__init__()
        self._queue = queue
        self._process = process
        self._stopped = False

    def run(self):
        while True:
            try:
                # Timeout so we notice a child that died without reporting.
                kind, payload = self._queue.get(timeout=0.25)
            except Exception:  # noqa: BLE001 - queue.Empty and friends
                if self._stopped:
                    return
                if self._process is not None and not self._process.is_alive():
                    exit_code = self._process.exitcode
                    if exit_code not in (0, None):
                        self.failed.emit(
                            "Analysis stopped",
                            f"The worker process exited unexpectedly (code {exit_code}). "
                            "This usually means OCCT ran out of memory on a very large part — "
                            "try a coarser tessellation or switch some analyses off.",
                        )
                    return
                continue

            if kind == _PROGRESS:
                self.progress.emit(payload)
            elif kind == _RESULT:
                self.finished.emit(payload)
                return
            elif kind == _ERROR:
                name, message, tb = payload
                self.failed.emit(f"{name}", f"{message}\n\nTraceback:\n{tb}")
                return

    def stop(self):
        self._stopped = True


class AnalysisRunner(QObject):
    """Owns a child process plus the pump thread that reports from it."""

    progress = Signal(str)
    finished = Signal(object)
    failed = Signal(str, str)
    done = Signal()  # always emitted, success or not

    def __init__(self, parent=None):
        super().__init__(parent)
        self._process = None
        self._thread = None
        self._pump = None
        self._cancelled = False

    # -------------------------------------------------------------- start
    def start_analysis(self, path: str, options: dict):
        self._start(_child_entry, (path, options))

    def start_thin_wall(self, path: str, threshold_mm: float, max_distance_mm: float):
        self._start(_child_thin_wall, (path, threshold_mm, max_distance_mm))

    def start_turning(self, path: str, cylinder, holes):
        """The turned profile about `cylinder`'s axis (a TurningGeometry)."""
        self._start(_child_turning, (path, cylinder, list(holes or [])))

    def start_fixture_export(self, path: str, plan: dict, out_path: str):
        """The tabbed-fixturing bodies for `plan` (a FixturePlan as a dict),
        written to `out_path`: {"path", "problems", "volumes_in3"}."""
        self._start(_child_fixture, (path, plan, out_path))

    def _start(self, entry, args):
        self._cancelled = False
        # The folder that CONTAINS the `partestimator` package, so the spawned child
        # can import partestimator.machining.geometry however it was launched.
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

        try:
            ctx = mp.get_context("spawn")
            queue = ctx.Queue()
            self._process = ctx.Process(
                target=entry, args=(project_root, *args, queue), daemon=True,
            )
            self._process.start()
        except Exception as exc:  # noqa: BLE001
            # Couldn't spawn -- fall back to in-process execution so the app
            # still works, just without the responsiveness benefit.
            self.progress.emit(f"(running in-process: {exc})")
            self._process = None        # never started: nothing to join
            self._run_inprocess(entry, args, project_root)
            return

        self._pump = _QueuePump(queue, self._process)
        self._thread = QThread()
        self._pump.moveToThread(self._thread)
        self._thread.started.connect(self._pump.run)
        self._pump.progress.connect(self.progress)
        self._pump.finished.connect(self._on_finished)
        self._pump.failed.connect(self._on_failed)
        self._thread.start()

    def _run_inprocess(self, entry, args, project_root):
        import queue as _q
        q = _q.Queue()
        entry(project_root, *args, q)
        while not q.empty():
            kind, payload = q.get()
            if kind == _PROGRESS:
                self.progress.emit(payload)
            elif kind == _RESULT:
                self._on_finished(payload)
            elif kind == _ERROR:
                name, message, tb = payload
                self._on_failed(name, f"{message}\n\nTraceback:\n{tb}")

    # ------------------------------------------------------------- finish
    def _on_finished(self, result):
        if self._cancelled:
            self._cleanup()
            self.done.emit()
            return
        self.finished.emit(result)
        self._cleanup()
        self.done.emit()

    def _on_failed(self, title, message):
        if not self._cancelled:
            self.failed.emit(title, message)
        self._cleanup()
        self.done.emit()

    def cancel(self):
        """Hard-terminates the child. This is the whole point of using a
        process: it interrupts even a C++ call that ignores Python."""
        self._cancelled = True
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
        self._cleanup()
        self.done.emit()

    def is_running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def _cleanup(self):
        if self._pump is not None:
            self._pump.stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
            self._thread = None
        self._pump = None
        if self._process is not None:
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=3)
            self._process = None
