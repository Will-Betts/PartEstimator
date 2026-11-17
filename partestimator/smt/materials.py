"""Material properties, K-factor tables, gauge and die tables.

IMPORTANT -- READ THIS BEFORE TRUSTING THE NUMBERS
--------------------------------------------------
The K-factor and bend-deduction values below are *published starting values*,
not shop data. Real K varies with tooling, lubrication, grain direction, coil
supplier and even ambient temperature. The whole point of the table structure is
that you replace these with numbers measured on your own press.

The correct workflow is:
  1. Run test bends on your material and tooling.
  2. Measure the flat blank and the finished outside dimensions.
  3. Back out the actual bend deduction.
  4. Load them via `load_deduction_table()`.

When a measured deduction row exists for (material, thickness, punch radius,
die opening), `allowance.py` uses it and bypasses the K-factor entirely. That is
what your brake operators already trust, and it beats any formula.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# materials
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Material:
    name: str
    density: float          # kg/mm^3
    uts: float              # ultimate tensile strength, MPa
    # Minimum inside bend radius as a multiple of thickness.
    # (transverse to grain, with grain) -- with-grain is the harder direction.
    min_radius_t: tuple[float, float]
    # Natural air-bend radius coefficient: r_natural ~= coeff * V_die.
    air_radius_coeff: float
    # K-factor curve: list of (r/t, K) knots, linearly interpolated.
    k_curve: tuple[tuple[float, float], ...]


# The "20 percent rule" coefficients for natural air-bend radius. Mild steel is
# the classic 0.156; stainless springs back more, aluminium slightly less.
_K_CURVE_STEEL = (
    (0.5, 0.33),
    (1.0, 0.38),
    (1.5, 0.41),
    (2.0, 0.42),
    (3.0, 0.44),
    (4.0, 0.45),
    (10.0, 0.46),
)

_K_CURVE_ALU = (
    (0.5, 0.33),
    (1.0, 0.37),
    (1.5, 0.40),
    (2.0, 0.41),
    (3.0, 0.43),
    (4.0, 0.44),
    (10.0, 0.45),
)

_K_CURVE_SS = (
    (0.5, 0.35),
    (1.0, 0.40),
    (1.5, 0.43),
    (2.0, 0.44),
    (3.0, 0.45),
    (4.0, 0.46),
    (10.0, 0.47),
)


_K_CURVES = {"steel": _K_CURVE_STEEL, "alu": _K_CURVE_ALU, "ss": _K_CURVE_SS}


def _build() -> dict[str, Material]:
    """Every sheet material in the shared library (partestimator/data/materials.csv).

    Keyed by the library name. The SMTv13 short keys ("CRS", "AL5052"...)
    still work through `get_material`, which resolves legacy names, and
    the ten SMTv13 materials keep the exact values they had.
    """
    from partestimator import materials_db as db

    out: dict[str, Material] = {}
    for m in db.load():
        if not m.sheet_metal:
            continue
        out[m.name] = Material(
            m.name, m.density_g_cc * 1e-6, m.uts_mpa,
            (m.min_bend_t_across or 0.0, m.min_bend_t_with or 0.0),
            m.air_radius_coeff or 0.156,
            _K_CURVES.get(m.k_curve, _K_CURVE_STEEL),
        )
    return out


MATERIALS: dict[str, Material] = _build()

#: The default for a fresh session -- what "CRS" always meant.
DEFAULT_MATERIAL = "Steel 1008/1010 CRS"


def get_material(name: str) -> Material:
    from partestimator import materials_db as db

    key = db.canonical(name)
    if key not in MATERIALS:
        raise KeyError(
            f"unknown sheet material {name!r}. known: {', '.join(sorted(MATERIALS))}"
        )
    return MATERIALS[key]


# --------------------------------------------------------------------------
# gauge table
# --------------------------------------------------------------------------

# Sheet thicknesses in mm. Steel gauges plus common metric stock. Used only to
# snap an inferred thickness to a plausible nominal value.
STANDARD_GAUGES_MM: tuple[float, ...] = (
    0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 1.6, 2.0, 2.5, 3.0, 3.2,
    4.0, 5.0, 6.0, 6.35, 8.0, 10.0, 12.0, 12.7,
)

# Steel sheet gauge numbers -> mm, for reporting.
GAUGE_NUMBERS: dict[int, float] = {
    26: 0.454, 24: 0.607, 22: 0.759, 20: 0.912, 18: 1.214,
    16: 1.519, 14: 1.897, 12: 2.657, 11: 3.038, 10: 3.416,
    8: 4.175, 7: 4.554,
}


def snap_to_gauge(t: float) -> tuple[float, float]:
    """Snap a measured thickness to the nearest standard gauge.

    Returns (snapped_thickness, relative_shift).
    """
    best = min(STANDARD_GAUGES_MM, key=lambda g: abs(g - t))
    return best, abs(best - t) / t if t else 0.0


def nearest_gauge_number(t: float) -> int | None:
    """Best-matching steel gauge number for reporting, or None if far off."""
    if not GAUGE_NUMBERS:
        return None
    n = min(GAUGE_NUMBERS, key=lambda k: abs(GAUGE_NUMBERS[k] - t))
    return n if abs(GAUGE_NUMBERS[n] - t) / t < 0.05 else None


# --------------------------------------------------------------------------
# die selection
# --------------------------------------------------------------------------

# Common V-die openings (mm) found on a typical press brake.
STANDARD_V_DIES: tuple[float, ...] = (
    4, 6, 8, 10, 12, 16, 20, 25, 30, 35, 40, 50, 60, 80, 100, 125, 160, 200,
)


def select_v_die(thickness: float, available=STANDARD_V_DIES) -> float:
    """Pick a V-die opening for air bending.

    Rule of thumb by thickness band:
      t < 3 mm    ->  V ~= 8 t
      3-10 mm     ->  V ~= 10 t
      t > 10 mm   ->  V ~= 12 t
    Snapped up to the nearest die actually on the rack.
    """
    if thickness < 3.0:
        target = 8.0 * thickness
    elif thickness <= 10.0:
        target = 10.0 * thickness
    else:
        target = 12.0 * thickness
    bigger = [v for v in available if v >= target]
    return float(bigger[0]) if bigger else float(max(available))


def natural_air_bend_radius(v_die: float, material: Material) -> float:
    """Inside radius the tooling naturally produces when air bending.

    This is the "20 percent rule": the part takes on a radius set by the die
    opening, NOT by the punch nose. If the CAD model specifies a radius that
    disagrees with this, the flat pattern is wrong before anything is cut --
    which is exactly the check in report.warnings.
    """
    return material.air_radius_coeff * v_die


def min_flange_length(v_die: float) -> float:
    """Shortest flange that will still sit on the die shoulders."""
    return 0.65 * v_die


def air_bend_tonnage(
    length_mm: float, thickness: float, v_die: float, material: Material,
    coefficient: float = 1.33, newtons_per_unit: float = 9806.65,
) -> float:
    """Estimated air bending force.

        F = C * UTS * L * t^2 / V

    C ~= 1.33 for air bending. Lengths in mm, UTS in MPa, result in whatever
    force unit `newtons_per_unit` describes -- 9806.65 for metric tonnes,
    8896.44 for US short tons, which is what an imperial shop quotes.

    This is an ESTIMATE. Check it against your press manufacturer's chart before
    making a capacity decision on it.
    """
    if v_die <= 0:
        return float("nan")
    force_n = coefficient * material.uts * length_mm * thickness ** 2 / v_die
    return force_n / newtons_per_unit


# --------------------------------------------------------------------------
# measured deduction tables
# --------------------------------------------------------------------------


def _same_material(a: str, b: str) -> bool:
    """Legacy short keys and library names compare equal ("CRS" == "Steel 1008/1010 CRS")."""
    from partestimator import materials_db as db

    ca, cb = db.canonical(a), db.canonical(b)
    if ca is not None and cb is not None:
        return ca == cb
    return a.strip().upper() == b.strip().upper()


@dataclass
class DeductionTable:
    """Measured bend deductions keyed by (material, thickness, radius, die).

    Populate from your own test bends. When a row matches within tolerance it
    overrides the K-factor calculation entirely.
    """

    rows: list[dict] = field(default_factory=list)

    def lookup(
        self, material: str, thickness: float, radius: float,
        v_die: float | None = None, rel_tol: float = 0.02,
    ) -> float | None:
        def close(a, b):
            return abs(a - b) <= rel_tol * max(abs(a), abs(b), 1e-9)

        best = None
        for r in self.rows:
            if not _same_material(r["material"], material):
                continue
            if not close(r["thickness"], thickness):
                continue
            if not close(r["radius"], radius):
                continue
            if v_die is not None and r.get("v_die") and not close(r["v_die"], v_die):
                continue
            best = r["deduction"]
        return best

    @classmethod
    def from_csv(cls, path: str | Path) -> DeductionTable:
        """Load from CSV with columns:
        material,thickness,radius,v_die,angle_deg,deduction
        """
        rows = []
        with open(path, newline="", encoding="utf-8") as fh:
            for rec in csv.DictReader(fh):
                rows.append(
                    {
                        "material": rec["material"],
                        "thickness": float(rec["thickness"]),
                        "radius": float(rec["radius"]),
                        "v_die": float(rec["v_die"]) if rec.get("v_die") else None,
                        "angle_deg": float(rec.get("angle_deg") or 90.0),
                        "deduction": float(rec["deduction"]),
                    }
                )
        return cls(rows)


def load_deduction_table(path: str | Path) -> DeductionTable:
    return DeductionTable.from_csv(path)
