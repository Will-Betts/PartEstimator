"""Bend table, material summary, and manufacturability warnings.

The warnings are arguably the highest-value output here. The flat pattern tells
you what to cut; the warnings tell you whether the part can be made at all, and
whether the flat pattern you just generated is even the right one.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .allowance import bend_allowance, k_factor
from .config import UnfoldOptions
from .materials import (
    air_bend_tonnage,
    get_material,
    min_flange_length,
    natural_air_bend_radius,
    nearest_gauge_number,
    select_v_die,
)
from .model import Part


def resolve_material(part: Part, opts: UnfoldOptions):
    """The options are the authority for material, not the Part.

    `Part.material` records what the part was detected with; `opts.material` is
    what the caller asked for on this run. They normally agree, but a caller
    re-reporting an already-unfolded part with a different alloy must get the
    new alloy's minimum-radius and tonnage numbers, not the stale ones. Reading
    it off the Part here silently ignored the override.
    """
    return get_material(opts.material or part.material)


# ---------------------------------------------------------------------------
# bend table
# ---------------------------------------------------------------------------


@dataclass
class BendRow:
    n: int
    angle_deg: float
    r_inner: float
    direction: str
    length: float
    length_material: float
    k_factor: float
    bend_allowance: float
    bend_deduction: float
    v_die: float
    tonnage: float
    allowance_source: str


def bend_table(part: Part, opts: UnfoldOptions) -> list[BendRow]:
    from .units import standard_dies_mm_for

    mat = resolve_material(part, opts)
    unit = resolve_unit(part, opts)
    dies = standard_dies_mm_for(unit)
    rows = []
    for i, b in enumerate(sorted(part.bends.values(), key=lambda x: x.id), start=1):
        v = opts.v_die or select_v_die(part.thickness, dies)
        rows.append(
            BendRow(
                n=i,
                angle_deg=round(b.angle_deg, 2),
                r_inner=round(b.r_inner, 3),
                direction="UP" if b.direction >= 0 else "DOWN",
                length=round(b.length_envelope, 2),
                length_material=round(b.length_material, 2),
                k_factor=round(b.k_factor, 4),
                bend_allowance=round(b.bend_allowance, 4),
                bend_deduction=(
                    round(b.bend_deduction, 4)
                    if math.isfinite(b.bend_deduction) else None
                ),
                v_die=v,
                # Tonnage uses the MATERIAL length, not the envelope. A bend
                # line that is heavily cut out needs proportionally less force.
                tonnage=round(
                    air_bend_tonnage(
                        b.length_material, part.thickness, v, mat,
                        newtons_per_unit=unit.newtons_per_force_unit,
                    ), 2
                ),
                allowance_source=b.allowance_source,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# material
# ---------------------------------------------------------------------------


@dataclass
class MaterialSummary:
    thickness: float
    gauge: int | None
    material: str
    bbox: tuple[float, float]
    nest_rect: tuple[float, float]
    nest_area: float
    profile_area: float
    scrap_fraction: float
    weight_kg: float
    cut_length: float
    pierce_count: int
    cost: float | None = None


def _gauge_number(part: Part, opts: UnfoldOptions):
    """Sheet gauge number, from the tradition the file was drawn in."""
    from .units import nearest_gauge_number_in

    unit = resolve_unit(part, opts)
    if unit.name == "in":
        return nearest_gauge_number_in(part.thickness)
    return nearest_gauge_number(part.thickness)


def material_summary(part: Part, opts: UnfoldOptions) -> MaterialSummary:
    mat = resolve_material(part, opts)
    flat = part.flat
    if flat is None:
        raise ValueError("no flat pattern")

    bw, bh = flat.bbox
    if opts.grain_deg is None:
        # Grain unconstrained: take the tighter of the axis-aligned box and the
        # minimum-area rectangle, since the blank can be nested at any angle.
        mw, mh, _ang = flat.min_rect
        if mw * mh < bw * bh and mw > 0:
            bw, bh = mw, mh

    f = 1.0 + max(opts.scrap_factor, 0.0)
    nest = (bw * f, bh * f)
    nest_area = nest[0] * nest[1]

    weight = flat.area * part.thickness * mat.density
    scrap = 1.0 - (flat.area / nest_area) if nest_area > 0 else 0.0

    return MaterialSummary(
        thickness=round(part.thickness, 4),
        gauge=_gauge_number(part, opts),
        material=opts.material or part.material,
        bbox=(round(flat.bbox[0], 2), round(flat.bbox[1], 2)),
        nest_rect=(round(nest[0], 2), round(nest[1], 2)),
        nest_area=round(nest_area, 1),
        profile_area=round(flat.area, 1),
        scrap_fraction=round(scrap, 4),
        weight_kg=round(weight, 4),
        cut_length=round(flat.perimeter, 1),
        # Counted from the classified features where they exist, because the
        # flat pattern's hole wires miss any cutout that merged into the outer
        # boundary, and a slot reads as one loop rather than as its two ends.
        pierce_count=(part.pierce_count() or len(flat.holes)),
        cost=(
            round(weight * opts.cost_per_kg, 2)
            if opts.cost_per_kg is not None else None
        ),
    )


# ---------------------------------------------------------------------------
# warnings
# ---------------------------------------------------------------------------


def manufacturability_warnings(part: Part, opts: UnfoldOptions) -> list[str]:
    from .units import standard_dies_mm_for

    mat = resolve_material(part, opts)
    unit = resolve_unit(part, opts)
    dies = standard_dies_mm_for(unit)
    t = part.thickness
    out: list[str] = []

    for b in part.bends.values():
        v = opts.v_die or select_v_die(t, dies)

        # --- minimum inside radius -------------------------------------
        r_min_transverse, r_min_grain = mat.min_radius_t
        if b.r_inner < r_min_transverse * t:
            out.append(
                f"bend {b.id}: inside radius {b.r_inner:.2f} mm is below the minimum "
                f"for {mat.name} ({r_min_transverse * t:.2f} mm = {r_min_transverse}t). "
                f"Expect cracking on the outside of the bend."
            )
        elif b.r_inner < r_min_grain * t:
            out.append(
                f"bend {b.id}: inside radius {b.r_inner:.2f} mm is acceptable across "
                f"the grain but below the {r_min_grain}t minimum with the grain. "
                f"Constrain the blank orientation when nesting."
            )

        # --- natural air-bend radius -----------------------------------
        # THE EXPENSIVE ONE. In air bending the radius is set by the die
        # opening, not the punch nose. If the model disagrees with what the
        # tooling actually produces, every flat length is wrong before a single
        # part is cut.
        r_nat = natural_air_bend_radius(v, mat)
        if r_nat > 0 and abs(b.r_inner - r_nat) / r_nat > 0.25:
            # What the blank length would need to be if the tooling gets its
            # way. K has to be re-interpolated too, because r/t has changed.
            k_nat = k_factor(r_nat, t, mat, opts.method)
            ba_nat = bend_allowance(b.angle, r_nat, t, k_nat)
            error = abs(ba_nat - b.bend_allowance)
            out.append(
                f"bend {b.id}: the model specifies R{b.r_inner:.2f} but a {v:.0f} mm "
                f"V-die naturally air bends {mat.name} to about R{r_nat:.2f}. "
                f"Either change the die, bottom the bend, or re-model the radius -- "
                f"otherwise the blank is developed for a radius the tooling will not "
                f"produce, and every bend lands {error:.3f} mm out "
                f"({error * len(part.bends):.3f} mm over the whole part)."
            )

        # --- tonnage ----------------------------------------------------
        tons = air_bend_tonnage(
            b.length_material, t, v, mat,
            newtons_per_unit=unit.newtons_per_force_unit,
        )
        if opts.press_capacity_t is not None and tons > opts.press_capacity_t:
            out.append(
                f"bend {b.id}: needs about {tons:.1f} t but the press is rated "
                f"{opts.press_capacity_t:.1f} t. Use a wider die or bend in stages."
            )

    # --- minimum flange ------------------------------------------------
    v = opts.v_die or select_v_die(t, dies)
    min_flange = min_flange_length(v)
    for b in part.bends.values():
        for fid in (b.flange_a, b.flange_b):
            fl = part.flanges.get(fid)
            if fl is None or b.length_envelope <= 0:
                continue
            approx = fl.area / b.length_envelope
            if approx < min_flange:
                out.append(
                    f"flange {fid} is roughly {approx:.1f} mm wide, under the "
                    f"{min_flange:.1f} mm minimum for a {v:.0f} mm V-die. It will fall "
                    f"into the die rather than sit on the shoulders."
                )

    # --- holes near bends ----------------------------------------------
    for h in part.holes:
        for b in part.bends.values():
            d = _distance_to_bend_zone(h.centre, b)
            zone = b.r_inner + 1.5 * t
            if d < zone:
                out.append(
                    f"hole at {np.round(h.centre, 1).tolist()} sits {d:.1f} mm from "
                    f"bend {b.id}, inside the {zone:.1f} mm deformation zone. It will "
                    f"distort into an oval. Move it, or pierce after forming."
                )
                break

    return out


def _distance_to_bend_zone(point: np.ndarray, bend) -> float:
    from . import geom

    return abs(
        geom.point_line_distance(point, bend.axis_point, bend.axis_dir) - bend.r_inner
    )


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


@dataclass
class Report:
    material: MaterialSummary
    bends: list[BendRow]
    distinct_radii: list[float]
    total_bend_length: float
    setups_estimate: int
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    #: (label, count) for every classified secondary feature, as it would
    #: appear on a setup sheet: "12 x hole 6.500 mm dia".
    features: list[tuple[str, int]] = field(default_factory=list)
    is_blank: bool = False
    #: Degrees the blank was turned to minimise its envelope, and the fraction
    #: of bounding-box area that saved.
    orient_deg: float = 0.0
    orient_saving: float = 0.0

    def to_dict(self) -> dict:
        return {
            "material": asdict(self.material),
            "bends": [asdict(b) for b in self.bends],
            "distinct_radii": self.distinct_radii,
            "total_bend_length": self.total_bend_length,
            "setups_estimate": self.setups_estimate,
            "features": [{"feature": lbl, "count": n} for lbl, n in self.features],
            "is_blank": self.is_blank,
            "orient_deg": self.orient_deg,
            "orient_saving": self.orient_saving,
            "warnings": self.warnings,
            "notes": self.notes,
        }

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    unit: object = None          # smt.units.Unit; None means millimetres

    def to_text(self) -> str:
        from .units import MM

        u = self.unit or MM
        m = self.material
        L = lambda mm: u.length(mm)              # noqa: E731
        A = lambda mm2: u.area(mm2)              # noqa: E731
        lines = []
        lines.append("=" * 78)
        lines.append("SHEET METAL FLAT PATTERN REPORT")
        lines.append("=" * 78)
        gauge = f" (~{m.gauge} ga)" if m.gauge else ""
        lines.append(
            f"Material        {m.material}   t = {L(m.thickness)}{gauge}"
        )
        lines.append(
            f"Flat blank      {L(m.bbox[0])} x {u.length(m.bbox[1], False)}"
        )
        lines.append(
            f"Nest rectangle  {L(m.nest_rect[0])} x "
            f"{u.length(m.nest_rect[1], False)}   ({A(m.nest_area)})"
        )
        lines.append(
            f"Profile area    {A(m.profile_area)}"
            f"   scrap {m.scrap_fraction * 100:.1f}%"
        )
        lines.append(f"Weight          {m.weight_kg} kg")
        lines.append(
            f"Cut length      {L(m.cut_length)}   pierces {m.pierce_count}"
        )
        if m.cost is not None:
            lines.append(f"Material cost   {m.cost}")
        lines.append("")

        if self.orient_saving > 0.005:
            lines.append(
                f"Blank was turned {abs(self.orient_deg):.2f} deg to square its "
                f"tightest rectangle to the axes, which cut the bounding box "
                f"area by {self.orient_saving * 100:.1f}%."
            )
            lines.append("")

        if self.features:
            lines.append("FEATURES -- cut or machined into the sheet, no effect "
                         "on the blank")
            for label, n in self.features:
                lines.append(f"  {n:>3} x  {label}")
            lines.append("")

        if self.is_blank:
            lines.append(
                "FLAT BLANK -- no bends. Every dimension above is measured "
                "directly from the part; no bend allowance is involved and "
                "nothing is estimated."
            )
            lines.append("")
            radii = ""
        else:
            radii = ", ".join(f"R{u.length(r, False)}" for r in self.distinct_radii)
        if not self.is_blank:
            lines.append(
                f"TOOLING -- distinct inside radii: {radii or 'none'} {u.label}"
            )
            lines.append(f"Total bend length {L(self.total_bend_length)} "
                         f"across {len(self.bends)} bends, "
                         f"~{self.setups_estimate} setup(s)")
            lines.append("")

        if self.bends:
            dp = u.decimals
            lines.append(
                f"{'#':>2}  {'ANGLE':>7} {'R':>8} {'DIR':>5} {'LEN':>9} "
                f"{'K':>6} {'BA':>9} {'BD':>9} {'V':>7} "
                f"{u.force_label.upper():>6}"
            )
            lines.append("-" * 82)
            for b in self.bends:
                conv = u.from_mm
                bd = (f"{conv(b.bend_deduction):9.{dp}f}"
                      if b.bend_deduction is not None else "      n/a")
                lines.append(
                    f"{b.n:>2}  {b.angle_deg:7.2f} {conv(b.r_inner):8.{dp}f} "
                    f"{b.direction:>5} {conv(b.length):9.{dp}f} {b.k_factor:6.3f} "
                    f"{conv(b.bend_allowance):9.{dp}f} {bd} "
                    f"{conv(b.v_die):7.{dp}f} {b.tonnage:6.1f}"
                )
            lines.append("")
            srcs = {b.allowance_source for b in self.bends}
            lines.append(f"Allowance source: {', '.join(sorted(srcs))}")
            lines.append("")

        if self.warnings:
            lines.append("WARNINGS")
            lines.append("-" * 78)
            for w in self.warnings:
                lines.append(f"  ! {w}")
            lines.append("")

        if self.notes:
            lines.append("NOTES")
            lines.append("-" * 78)
            for n in self.notes:
                lines.append(f"  - {n}")
            lines.append("")

        return "\n".join(lines)


def resolve_unit(part: Part, opts: UnfoldOptions):
    from .units import choose_display_unit

    return choose_display_unit(opts.display_units, part.file_units)


def build_report(part: Part, opts: UnfoldOptions) -> Report:
    rows = bend_table(part, opts)
    # Each distinct (radius, direction) combination is a plausible tool change.
    setups = len({(round(b.r_inner, 3), b.direction) for b in rows}) or 1
    unit = resolve_unit(part, opts)

    # Features are reported in whatever unit the rest of the sheet uses. They do
    # not change the blank -- a countersink flattens with the sheet -- but they
    # decide whether the part needs a second operation after forming, which is
    # exactly what you want to know before quoting it.
    feature_lines = [
        (
            g.role.value.replace("_", " ") if not (g.diameter or g.included_angle)
            else _group_label(g, unit),
            g.count,
        )
        for g in getattr(part, "feature_groups", [])
    ]

    return Report(
        unit=unit,
        material=material_summary(part, opts),
        bends=rows,
        distinct_radii=[round(r, 3) for r in part.distinct_radii()],
        total_bend_length=round(part.total_bend_length(), 1),
        setups_estimate=setups,
        features=feature_lines,
        is_blank=bool(getattr(part, "is_blank", False)),
        orient_deg=math.degrees(getattr(part, "flat_rotation", 0.0)),
        orient_saving=float(getattr(part, "flat_rotation_saving", 0.0)),
        warnings=part.warnings + manufacturability_warnings(part, opts),
        notes=part.notes,
    )


def _group_label(group, unit) -> str:
    """One feature group as a line on a setup sheet, in the display unit."""
    from .features import Role

    def n(mm: float) -> str:
        return f"{unit.from_mm(mm):.{unit.decimals}f}"

    role = group.role
    if role is Role.HOLE:
        return f"hole {n(group.diameter)} {unit.label} dia"
    if role is Role.COUNTERSINK:
        ang = f"{group.included_angle:.0f} deg " if group.included_angle else ""
        return f"countersink {ang}on {n(group.diameter)} {unit.label} dia"
    if role is Role.COUNTERBORE:
        return f"counterbore {n(group.diameter)} {unit.label} dia"
    if role is Role.SLOT_END:
        return f"slot end R{n(group.diameter / 2)} {unit.label}"
    if role is Role.CORNER_RADIUS:
        return f"corner R{n(group.diameter / 2)} {unit.label}"
    if role is Role.FILLET:
        return f"edge fillet R{n(group.diameter / 2)} {unit.label}"
    if role is Role.CHAMFER:
        ang = f" at {group.included_angle:.0f} deg" if group.included_angle else ""
        return f"chamfer{ang}"
    return role.value.replace("_", " ")
