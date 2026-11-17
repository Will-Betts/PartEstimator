"""Run every automated check: `python -m partestimator.selftest`.

1. The machining estimator's analytic self-test (known shapes, hand-computed
   answers).
2. The unit-test suite in tests/: the sheet metal tool's tests, the DXF
   read-back scale test, and the sheet-metal-vs-machined classification test.

Both build their parts in the CAD kernel; no STEP files are needed.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    # A real display is used when there is one (always, on Windows): the
    # window test drives the machining 3D viewer, which needs OpenGL.
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    # Same OpenGL choice as the window, before anything loads Qt or VTK, so the
    # viewer tests also run on a VM or remote desktop without a GPU.
    from partestimator.opengl import choose

    gl = choose("auto")

    print(f" 3D viewer: {gl}")
    print("=" * 70)
    print(" 1/2  Machining estimator self-test")
    print("=" * 70)
    from partestimator.machining import selftest as mach

    rc_mach = mach.run_all()

    print()
    print("=" * 70)
    print(" 2/2  Sheet metal, DXF scale and part classification tests")
    print("=" * 70)
    import logging

    logging.getLogger("smt").setLevel(logging.WARNING)
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    rc_unit = 0 if result.wasSuccessful() else 1

    print()
    print("=" * 70)
    print(f" Machining self-test : {'PASS' if rc_mach == 0 else 'FAIL'}")
    print(f" Unit tests          : {'PASS' if rc_unit == 0 else 'FAIL'}"
          f"  ({result.testsRun} run, {len(result.failures)} failed, "
          f"{len(result.errors)} errors, {len(result.skipped)} skipped)")
    print("=" * 70)
    return rc_mach or rc_unit


if __name__ == "__main__":
    raise SystemExit(main())
