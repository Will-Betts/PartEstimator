"""Line coverage for the test run, with nothing but the standard library.

Python imports `sitecustomize` at start-up from the first folder on its path
that has one. `tools/linecov/report.py` puts this folder on PYTHONPATH and
sets PE_LINECOV_DIR, so every Python process of the run -- the test runner
and every worker it spawns (analysis, thin wall, turning, fixture export,
tuning capture) -- records which lines of `partestimator` it ran, and writes
them to PE_LINECOV_DIR when it exits.

Uses sys.monitoring (Python 3.12+): a line is reported once and then switched
off, so after warm-up the run is barely slower than without it. Without
PE_LINECOV_DIR, or on an older Python, this does nothing.
"""
import os
import sys

_DIR = os.environ.get("PE_LINECOV_DIR")


def _start(out_dir: str) -> None:
    import atexit
    import json

    mon = sys.monitoring
    tool = mon.COVERAGE_ID
    try:
        mon.use_tool_id(tool, "pe-linecov")
    except ValueError:
        return                                  # someone else is measuring
    root = os.path.normcase(os.path.abspath(os.environ.get(
        "PE_LINECOV_ROOT", os.path.join(os.path.dirname(__file__), "..", "..",
                                        "partestimator"))))
    hits: dict = {}
    mine: dict = {}

    def on_line(code, line):
        f = code.co_filename
        ok = mine.get(f)
        if ok is None:
            ok = mine[f] = os.path.normcase(os.path.abspath(f)).startswith(root)
        if ok:
            s = hits.get(f)
            if s is None:
                s = hits[f] = set()
            s.add(line)
        return mon.DISABLE

    mon.register_callback(tool, mon.events.LINE, on_line)
    mon.set_events(tool, mon.events.LINE)

    def save():
        try:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"lines-{os.getpid()}.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({f: sorted(s) for f, s in hits.items()}, fh)
        except Exception:  # noqa: BLE001 -- never break the process being measured
            pass

    atexit.register(save)

    # A worker process is often terminated by its parent the moment its
    # result arrives, before atexit could run. So a worker also saves when
    # it hands a result back (at most every quarter second) and when its
    # task returns. Only a worker cancelled mid-job loses its record, which
    # only ever under-counts.
    import time
    import multiprocessing.process as mpp
    import multiprocessing.queues as mpq

    last = [0.0]

    def save_soon():
        now = time.monotonic()
        if now - last[0] >= 0.25:
            last[0] = now
            save()

    put = mpq.Queue.put

    def put_and_save(self, *args, **kwargs):
        if mpp.parent_process() is not None:
            save_soon()
        return put(self, *args, **kwargs)

    mpq.Queue.put = put_and_save
    run = mpp.BaseProcess.run

    def run_and_save(self):
        try:
            return run(self)
        finally:
            save()

    mpp.BaseProcess.run = run_and_save


if _DIR and sys.version_info >= (3, 12):
    _start(_DIR)
