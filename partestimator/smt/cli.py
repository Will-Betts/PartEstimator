"""Command line interface.

    smt-unfold bracket.step --material CRS --k 0.44 --out ./flat
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import UnfoldOptions
from .materials import MATERIALS, load_deduction_table


def build_parser(prog: str = "smt-unfold") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description="Develop the flat pattern of an air-bent sheet metal STEP file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Outputs a flat DXF, an unfolded STEP solid, a bend table and a\n"
            "material summary. Pay attention to the WARNINGS section: the\n"
            "natural air-bend radius check in particular catches a modelled\n"
            "radius the tooling will not actually produce."
        ),
    )
    p.add_argument("step", type=Path, help="input STEP file")
    p.add_argument("-o", "--out", type=Path, default=Path("."),
                   help="output directory (default: current)")

    g = p.add_argument_group("forming")
    g.add_argument("--material", default="CRS",
                   help="sheet material: any name in partestimator/data/materials.csv, "
                        "or an SMTv13 key such as CRS or AL5052 (default: CRS)")
    g.add_argument("--k", type=float, default=None, dest="k_factor",
                   help="fixed K-factor. Omit to interpolate K = f(r/t) from "
                        "the material curve, which is more accurate.")
    g.add_argument("--method", default="air",
                   choices=["air", "bottoming", "coining"],
                   help="forming method (default: air)")
    g.add_argument("--thickness", type=float, default=None,
                   help="override the inferred thickness, in mm")
    g.add_argument("--no-gauge-snap", action="store_true",
                   help="do not snap the inferred thickness to a standard gauge")
    g.add_argument("--v-die", type=float, default=None,
                   help="V-die opening in mm (default: chosen from thickness)")
    g.add_argument("--orient", default="min_area",
                   choices=["min_area", "as_modelled", "grain"],
                   help="how the finished blank is turned. min_area (default): "
                        "rotate until the tightest rectangle is square to the "
                        "axes, long side along X, which is what the material "
                        "estimate is built on. as-modelled: leave it where the "
                        "unfold put it. A --grain setting always wins.")
    g.add_argument("--no-origin", action="store_true",
                   help="do not move the blank into the positive quadrant")
    g.add_argument("--deductions", type=Path, default=None,
                   help="CSV of measured bend deductions; overrides the K-factor "
                        "where a row matches")

    g = p.add_argument_group("nesting and cost")
    g.add_argument("--grain", type=int, choices=[0, 90], default=None,
                   help="constrain the blank's grain direction")
    g.add_argument("--scrap", type=float, default=0.0,
                   help="scrap allowance as a fraction, e.g. 0.1")
    g.add_argument("--cost-per-kg", type=float, default=None,
                   help="material unit cost, for the cost line")
    g.add_argument("--press-tons", type=float, default=None,
                   help="press capacity in tonnes, for the tonnage check")

    g = p.add_argument_group("output")
    g.add_argument("--base-face", type=int, default=None,
                   help="flange id to hold fixed (default: the largest face)")
    g.add_argument("--no-dxf", action="store_true")
    g.add_argument("--no-step", action="store_true")
    g.add_argument("--no-json", action="store_true")
    g.add_argument("-q", "--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None, prog: str = "smt-unfold") -> int:
    args = build_parser(prog).parse_args(argv)

    opts = UnfoldOptions(
        k_factor=args.k_factor,
        method=args.method,
        material=args.material,
        thickness=args.thickness,
        snap_to_gauge=not args.no_gauge_snap,
        base_face=args.base_face,
        grain_deg=args.grain,
        scrap_factor=args.scrap,
        v_die=args.v_die,
        orient=args.orient,
        orient_to_origin=not args.no_origin,
        press_capacity_t=args.press_tons,
        cost_per_kg=args.cost_per_kg,
    )

    table = load_deduction_table(args.deductions) if args.deductions else None

    from .pipeline import unfold_step, write_outputs
    from .report import build_report

    try:
        part = unfold_step(args.step, opts, table)
    except Exception as exc:  # noqa: BLE001 -- CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 1

    written = write_outputs(
        part, opts, args.out, args.step.stem,
        want_dxf=not args.no_dxf,
        want_step=not args.no_step,
        want_json=not args.no_json,
    )

    if not args.quiet:
        print(build_report(part, opts).to_text())
        for kind, path in written.items():
            print(f"  wrote {kind:8s} {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
