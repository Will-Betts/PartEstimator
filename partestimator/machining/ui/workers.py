"""DEPRECATED -- superseded by ui/proc_workers.py.

The analysis used to run on a QThread here. That turned out not to solve
the real problem: OCP is a pybind11 binding and does not release the GIL
around OCCT calls, so a single long `BRepMesh_IncrementalMesh.Perform()`
or `STEPControl_Reader.ReadFile()` held the GIL for its whole duration and
froze the UI regardless of which thread it ran on -- no repaints, no
progress, and no way to poll a cancel flag.

The work now runs in a child process (`ui/proc_workers.AnalysisRunner`),
which has its own GIL and can be hard-terminated mid-call. The analysis
body itself lives in `geometry/pipeline.py` so it has no Qt dependency and
can run in either context.

This file is intentionally left as a stub rather than deleted so that any
stale import fails loudly with an explanation instead of silently
resurrecting the old behaviour.
"""
from __future__ import annotations

_MESSAGE = (
    "ui.workers has been replaced by ui.proc_workers.AnalysisRunner "
    "(analysis now runs in a separate process; see that module's docstring "
    "for why a thread was not sufficient)."
)


def __getattr__(name):
    raise ImportError(f"{_MESSAGE} Attempted to import '{name}'.")
