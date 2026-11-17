"""Material densities, for converting part volume into weight.

Densities are nominal handbook values in g/cm^3 and are converted to
lb/in^3 on use. They're good to roughly 1% for the wrought alloys listed;
cast and filled grades vary more, so treat the resulting weight as an
estimate for quoting rather than a certified figure.
"""
from __future__ import annotations

# 1 g/cm^3 in lb/in^3
G_PER_CM3_TO_LB_PER_IN3 = 0.0361273
MM3_PER_IN3 = 25.4 ** 3

# Densities come from the shared library (partestimator/data/materials.csv),
# so the dropdown, the weight and the sheet metal tool all agree.
from partestimator import materials_db as _db

DENSITY_G_PER_CM3 = {m.name: m.density_g_cc for m in _db.load() if m.machining}

MATERIALS = _db.machining_names() + ["Other"]


def density_lb_per_in3(material: str) -> float | None:
    """Density in lb/in^3, or None for an unknown material."""
    rec = _db.get(material)
    if rec is None:
        return None
    g_cm3 = rec.density_g_cc
    return g_cm3 * G_PER_CM3_TO_LB_PER_IN3


def weight_lb(volume_mm3: float, material: str) -> float | None:
    """Part weight in pounds, or None if the material has no density."""
    density = density_lb_per_in3(material)
    if density is None:
        return None
    return (volume_mm3 / MM3_PER_IN3) * density
