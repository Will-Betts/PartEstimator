"""Fit the shop calibration factors to your own measured cycle times.

A bottom-up estimate built from published cutting data will never match a
specific shop out of the box. Published rates assume a rigid short tool, a
new edge and ideal coolant; your machines, fixtures, programmers and
tooling differ in ways no catalogue can know. The honest way to close
that gap is to measure it, not to guess at it.

USAGE

    python calibrate.py training_table.csv

Point it at the CSV you have been exporting, with `cycle_time_min` filled
in from job records on at least a few rows. It reports the gap and prints
the `calibrate_*` values to paste into data/shop_profile.csv.

WHAT IT FITS

With fewer than FIT_MIN_ROWS usable rows it fits ONE number -- an overall
multiplier -- because that is all a small sample can support. Fitting
four coefficients to five parts would produce numbers that look precise
and mean nothing.

With enough rows it solves a least-squares fit for separate roughing,
finishing and hole multipliers plus a fixed per-part constant:

    measured  ~=  a*roughing + b*finishing + c*holes + d

The constant d absorbs setup, load/unload and anything else that does not
scale with geometry. Coefficients are reported with the spread so you can
see whether they are trustworthy.
"""
from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

# Below this, fit a single overall factor rather than per-term factors.
FIT_MIN_ROWS = 8

ROUGH_COLS = ["est_rough_outside_min", "est_rough_pocket_min"]
FINISH_COLS = ["est_finish_analytic_min", "est_finish_freeform_min"]
HOLE_COLS = ["est_hole_min"]


def _f(row, key, default=0.0):
    try:
        v = row.get(key, "")
        return float(v) if str(v).strip() != "" else default
    except (TypeError, ValueError):
        return default


def load_pairs(path: Path):
    """Rows that have both a measured time and an estimate."""
    with path.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    usable = []
    for r in rows:
        measured = _f(r, "cycle_time_min")
        estimated = _f(r, "estimated_cycle_time_min")
        if measured > 0 and estimated > 0:
            usable.append(r)
    return rows, usable


def report(path: Path) -> int:
    if not path.exists():
        print(f"ERROR: {path} not found.")
        return 1

    all_rows, rows = load_pairs(path)
    print(f"\n{path.name}: {len(all_rows)} row(s), {len(rows)} with a measured time.\n")
    if not rows:
        print("Nothing to fit. Fill in the cycle_time_min column from your job")
        print("records on at least a few rows, then run this again.")
        return 1

    ratios = []
    print(f"  {'part':<26s} {'measured':>9s} {'estimate':>9s} {'ratio':>7s}")
    print("  " + "-" * 56)
    for r in rows:
        m = _f(r, "cycle_time_min")
        e = _f(r, "estimated_cycle_time_min")
        ratios.append(m / e)
        name = (r.get("part_name") or "")[:26]
        print(f"  {name:<26s} {m:9.2f} {e:9.2f} {m / e:7.2f}x")

    median = statistics.median(ratios)
    print("  " + "-" * 56)
    print(f"  median ratio {median:.2f}x", end="")
    if len(ratios) > 1:
        lo, hi = min(ratios), max(ratios)
        spread = hi / lo if lo > 0 else float("inf")
        print(f"   range {lo:.2f}x - {hi:.2f}x  (spread {spread:.1f}x)")
        if spread > 2.5:
            print("\n  WARNING: the ratio varies more than 2.5x across parts. A single")
            print("  multiplier will not fix that -- it means the model is missing")
            print("  something that differs BETWEEN your parts (setup count, a family")
            print("  of features, a material), not something uniformly off. Look at")
            print("  the highest and lowest rows before trusting any factor below.")
    else:
        print()

    print()
    if len(rows) < FIT_MIN_ROWS:
        print(f"  With {len(rows)} row(s), fitting one overall factor only")
        print(f"  (per-term fitting needs {FIT_MIN_ROWS}+ to mean anything).\n")
        print("  Paste into data/shop_profile.csv:")
        print(f"    calibrate_overall,{median:.2f},multiplier,...")
        return 0

    try:
        import numpy as np
    except ImportError:
        print("  numpy not available; reporting the overall factor only.")
        print(f"    calibrate_overall,{median:.2f},multiplier,...")
        return 0

    A, y = [], []
    for r in rows:
        rough = sum(_f(r, c) for c in ROUGH_COLS)
        finish = sum(_f(r, c) for c in FINISH_COLS)
        holes = sum(_f(r, c) for c in HOLE_COLS)
        A.append([rough, finish, holes, 1.0])
        y.append(_f(r, "cycle_time_min"))
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float)

    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b, c, d = (float(x) for x in coef)
    pred = A @ coef
    resid = y - pred
    rms = float(np.sqrt(np.mean(resid ** 2)))
    mape = float(np.mean(np.abs(resid / y))) * 100.0

    print("  Least-squares fit on the per-term breakdown:")
    print(f"    roughing  x {a:6.2f}")
    print(f"    finishing x {b:6.2f}")
    print(f"    holes     x {c:6.2f}")
    print(f"    fixed       {d:6.2f} min per part (setup, handling, everything flat)")
    print(f"\n    RMS error {rms:.2f} min, mean absolute error {mape:.1f}%")

    negative = [n for n, v in (("roughing", a), ("finishing", b), ("holes", c)) if v <= 0]
    if negative:
        print(f"\n  WARNING: {', '.join(negative)} fitted non-positive, which is")
        print("  physically meaningless. The sample is too small or too correlated")
        print("  to separate the terms. Use calibrate_overall instead.")
        print(f"    calibrate_overall,{median:.2f},multiplier,...")
        return 0

    print("\n  Paste into data/shop_profile.csv:")
    print(f"    calibrate_roughing,{a:.2f},multiplier,...")
    print(f"    calibrate_finishing,{b:.2f},multiplier,...")
    print(f"    calibrate_holes,{c:.2f},multiplier,...")
    print("\n  Leave calibrate_overall at 1.00 if you use the per-term factors,")
    print("  otherwise you will apply the correction twice.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nERROR: name the CSV to read, e.g.\n")
        print("    python calibrate.py training_table.csv")
        sys.exit(1)
    sys.exit(report(Path(sys.argv[1])))
