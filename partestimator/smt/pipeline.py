"""End-to-end: STEP in, flat pattern out."""

from __future__ import annotations

from pathlib import Path

from .config import UnfoldOptions
from .materials import DeductionTable
from .model import Part
from .unfold import assign_allowances, solve


def unfold_step(
    path: str | Path,
    options: UnfoldOptions | None = None,
    deduction_table: DeductionTable | None = None,
) -> Part:
    """Load a STEP file, detect its bends, and develop the flat pattern.

    Returns a fully populated `Part`. The as-bent solid stays on
    `part.solid_bent` and the flange transforms on `part.transforms`, because
    hydroform form-block generation needs both and throwing them away here
    would make that a rewrite rather than an extension.
    """
    from .occ import require_kernel

    require_kernel()

    from .occ.detect import detect
    from .occ.flatten import build_flat, refold_check
    from .occ.load import load_step

    opts = options or UnfoldOptions()

    shape = load_step(path)
    part: Part = detect(shape, opts)

    assign_allowances(part, opts, deduction_table)

    if part.bends:
        result = solve(part, opts)
        part.base_flange = result.base
        part.transforms = result.transforms
        part.flat = build_flat(part, result, opts)

        # The flat as a real solid, features intact. Falls back to the extruded
        # profile on its own if the booleans do not cooperate, and says so.
        from .model2d import oriented_flat_frame
        from .occ.unfold3d import unfold_solid

        M_frame, _a, _s = oriented_flat_frame(part, result, opts)
        solid = unfold_solid(part, result, opts, M_frame)
        part.flat_solid = solid if solid is not None else part.flat.solid
        part.flat_is_extruded = solid is None

        residual = refold_check(part, result)
        if residual == residual and residual > 0.01:  # NaN-safe
            part.warn(
                f"volume check: the flat pattern implies a volume {residual * 100:.1f}% "
                f"away from the original solid. A bend was probably missed, or a "
                f"formed feature was developed as if it were a bend."
            )
    else:
        # A FLAT BLANK. Still a complete answer: the sheet faces are already in
        # one plane, so the identity transform IS the unfold and every dimension
        # is measured directly rather than developed.
        result = solve(part, opts)
        part.base_flange = result.base
        part.transforms = result.transforms
        part.flat = build_flat(part, result, opts)
        part.note(
            "flat blank: no bends, so the blank is the part outline measured "
            "directly. No bend allowance is involved and nothing is estimated."
        )

    return part


def write_outputs(
    part: Part,
    opts: UnfoldOptions,
    out_dir: str | Path,
    stem: str,
    want_dxf: bool = True,
    want_step: bool = True,
    want_json: bool = True,
) -> dict[str, Path]:
    """Write every requested deliverable, returning what was produced."""
    from .report import build_report

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    report = build_report(part, opts)

    txt = out_dir / f"{stem}_report.txt"
    txt.write_text(report.to_text(), encoding="utf-8")
    written["report"] = txt

    if want_json:
        written["json"] = report.to_json(out_dir / f"{stem}_report.json")

    if want_dxf and part.flat is not None:
        from .dxf import export_dxf
        from .report import resolve_unit

        written["dxf"] = export_dxf(
            part, out_dir / f"{stem}_flat.dxf", unit=resolve_unit(part, opts)
        )

    if want_step:
        # Prefer the SOLID unfold: it was made by moving the real material, so
        # the exported STEP still has its countersinks, chamfers and edge
        # breaks. The extruded blank is dimensionally identical and carries
        # none of them, so it is only the fallback.
        solid = getattr(part, "flat_solid", None)
        if solid is None and part.flat is not None:
            solid = part.flat.solid
        if solid is not None:
            from .occ.load import export_step

            p = out_dir / f"{stem}_flat.step"
            export_step(solid, p)
            written["step"] = p

    return written
