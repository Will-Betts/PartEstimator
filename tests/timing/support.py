"""Hand-built inputs for the timing-model tests.

The shipped CSVs are tuned over time, so a test that pins an exact number
against them breaks every time someone recalibrates. The arithmetic tests
therefore run against a small synthetic rate table and shop profile whose
values are fixed here; the shipped data files are exercised separately for
validity and for qualitative behaviour (ordering, monotonicity).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from partestimator.machining.geometry.rates import RateRow, RateTable
from partestimator.machining.geometry.shop import ShopProfile
from partestimator.machining.geometry.turning import TurningGeometry

MAT = "Testium"  # not a real material: canonical() passes it through unchanged

MILL_RATES = {
    "unit_power": 0.30,
    "rough_open_sfm": 800.0,
    "rough_open_ipt": 0.0048,
    "drill_sfm": 680.0,
    "drill_ipr_per_inch_dia": 0.016,
}

LATHE_RATES = {
    "turn_rough_sfm": 900.0,
    "turn_rough_ipr": 0.012,
    "turn_rough_doc_in": 0.100,
    "turn_finish_sfm": 1100.0,
    "turn_finish_ipr_max": 0.006,
}


def make_table(mill=None, lathe=None, material: str = MAT) -> RateTable:
    """RateTable from {operation: effective_value} dicts."""
    mill = MILL_RATES if mill is None else mill
    lathe = LATHE_RATES if lathe is None else lathe
    rows = []
    for removal, ops in (("mill", mill), ("lathe", lathe)):
        for op, val in ops.items():
            rows.append(RateRow(
                material=material, iso_group="N", removal_type=removal, operation=op,
                published_value=val, units="", derate=1.0, effective_value=val,
                notes="", source="test", source_url="https://example.invalid",
                retrieved="2026-01-01"))
    return RateTable(rows)


def make_profile(**values) -> ShopProfile:
    return ShopProfile(values=dict(values))


@dataclass
class FakeHole:
    """The attributes of a detected hole that the timing model reads."""
    nominal_diameter_mm: float
    depth_mm: float
    axis_direction: tuple = (0.0, 0.0, 1.0)
    axis_point: tuple = (0.0, 0.0, 0.0)
    internal_groove_count: int = 0

    @classmethod
    def inches(cls, dia_in: float, depth_in: float, **kw) -> "FakeHole":
        return cls(dia_in * 25.4, depth_in * 25.4, **kw)


def prismatic_features(**overrides) -> dict:
    """A plain milled block: 10 in^3 part, 20 in^3 hull, 30 in^3 stock."""
    f = {
        "volume_in3": 10.0,
        "stock_volume_in3": 30.0,
        "solidity": 0.5,
        "area_analytic_in2": 40.0,
        "area_freeform_in2": 0.0,
        "material_density_lb_in3": 0.098,
    }
    f.update(overrides)
    return f


def shaft(diameter_in=1.5, length_in=3.0, step_diameter_in=None, bins=60,
          axial_holes=(), other_area_in2=0.0, turned_od_area_in2=None,
          total_area_in2=None) -> TurningGeometry:
    """A synthetic turned shaft along +Z: plain or with one step down."""
    mm = 25.4
    r = np.full(bins, diameter_in / 2.0 * mm)
    if step_diameter_in is not None:
        r[bins // 2:] = step_diameter_in / 2.0 * mm
    z = np.linspace(0.0, length_in * mm, bins)
    od = turned_od_area_in2
    if od is None:
        od = 3.14159 * diameter_in * length_in
    total = total_area_in2 if total_area_in2 is not None else od + other_area_in2
    return TurningGeometry(
        axis_point_mm=np.zeros(3),
        axis_direction=np.array([0.0, 0.0, 1.0]),
        length_mm=length_in * mm,
        max_diameter_mm=diameter_in * mm,
        envelope_volume_mm3=float(np.sum(np.pi * r ** 2) * (length_in * mm / bins)),
        mean_od_removal_diameter_mm=diameter_in * mm,
        turned_od_area_mm2=od * mm ** 2,
        other_area_mm2=other_area_in2 * mm ** 2,
        total_area_mm2=total * mm ** 2,
        axial_holes=list(axial_holes),
        profile_z_mm=z,
        profile_r_mm=r,
    )
