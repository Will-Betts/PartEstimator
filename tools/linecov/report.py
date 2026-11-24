"""Run the full test suite and report how much of `partestimator` it runs.

    python tools/linecov/report.py              run everything, then report
    python tools/linecov/report.py --no-run     report the last run again
    python tools/linecov/report.py --missing machining/geometry/fixture.py
                                                 ... and list a file's unrun lines

Standard library only (see sitecustomize.py beside this). "Executable lines"
are the lines the compiler emits code for, found from the compiled code
objects of every file -- what coverage.py counts too, give or take. Lines
marked `# pragma: no cover` are left out of both counts.

The run is `python -m partestimator selftest`: the machining self-test and
every test in tests/, as `.\\pe.ps1 test -Full` runs them.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PKG = ROOT / "partestimator"
# Not the tool's scratch folder: the sheet metal window empties that when
# it opens, which would throw away what the run has recorded so far.
DATA = Path(tempfile.gettempdir()) / "partestimator-linecov"
PRAGMA = "pragma: no cover"


def executable_lines(path: Path) -> set[int]:
    src = path.read_text(encoding="utf-8")
    code = compile(src, str(path), "exec", dont_inherit=True)
    lines: set[int] = set()
    stack = [code]
    while stack:
        c = stack.pop()
        lines.update(ln for _s, _e, ln in c.co_lines() if ln is not None)
        stack.extend(k for k in c.co_consts if hasattr(k, "co_lines"))
    text = src.splitlines()
    return {ln for ln in lines if 0 < ln <= len(text) and PRAGMA not in text[ln - 1]}


def run_suite() -> int:
    if DATA.exists():
        shutil.rmtree(DATA)
    DATA.mkdir(parents=True)
    env = dict(os.environ)
    env["PE_LINECOV_DIR"] = str(DATA)
    env["PE_LINECOV_ROOT"] = str(PKG)
    env["PYTHONPATH"] = os.pathsep.join([str(HERE), str(ROOT)] + (
        [env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return subprocess.call([sys.executable, "-m", "partestimator", "selftest"],
                           cwd=str(ROOT), env=env)


def collect() -> dict[str, set[int]]:
    hits: dict[str, set[int]] = {}
    for f in DATA.glob("lines-*.json"):
        for name, lines in json.loads(f.read_text(encoding="utf-8")).items():
            key = os.path.normcase(os.path.abspath(name))
            hits.setdefault(key, set()).update(lines)
    return hits


def report(missing_for: list[str]) -> float:
    hits = collect()
    rows, total_exec, total_hit = [], 0, 0
    for path in sorted(PKG.rglob("*.py")):
        ex = executable_lines(path)
        if not ex:
            continue
        got = hits.get(os.path.normcase(str(path.resolve())), set()) & ex
        total_exec += len(ex)
        total_hit += len(got)
        rows.append((path.relative_to(PKG).as_posix(), len(ex), len(got), ex - got))
    width = max(len(r[0]) for r in rows)
    print(f"{'file':<{width}}  {'lines':>6}  {'run':>6}  {'cover':>6}")
    for name, n, k, _miss in sorted(rows, key=lambda r: (r[2] / r[1], r[0])):
        print(f"{name:<{width}}  {n:>6}  {k:>6}  {100.0 * k / n:>5.1f}%")
    pct = 100.0 * total_hit / total_exec if total_exec else 0.0
    print(f"{'TOTAL':<{width}}  {total_exec:>6}  {total_hit:>6}  {pct:>5.1f}%")
    for want in missing_for:
        for name, _n, _k, miss in rows:
            if name == want or name.endswith(want):
                print(f"\n{name}: not run: {_ranges(sorted(miss))}")
    return pct


def _ranges(nums: list[int]) -> str:
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(str(nums[i]) if i == j else f"{nums[i]}-{nums[j]}")
        i = j + 1
    return ", ".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-run", action="store_true", help="report the last run again")
    ap.add_argument("--missing", action="append", default=[],
                    help="list the unrun lines of this file (path under partestimator/)")
    args = ap.parse_args(argv)
    rc = 0
    if not args.no_run:
        rc = run_suite()
    report(args.missing)
    return rc


if __name__ == "__main__":
    sys.exit(main())
