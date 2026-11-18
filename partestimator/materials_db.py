"""The one material library both tools read.

`data/materials.csv` holds every material with the properties each tool
needs: density and family for both; UTS, minimum bend radius, natural air-bend
radius coefficient and K-factor curve for the ones that are formed as sheet.
Machining speeds and feeds live beside the machining tool in
`machining/data/machining_rates.csv`, keyed by the SAME names.

Names changed when tempers were added ("Aluminum 6061" became
"Aluminum 6061-T6"), and the sheet metal tool used short keys ("CRS",
"AL5052"). `canonical()` resolves every old name, so old CSV exports, the
deductions file and command-line flags keep working.

Kernel-free and Qt-free: safe to import from anywhere.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "data" / "materials.csv"

#: Names used by earlier versions -> current names.
LEGACY_NAMES = {
    "Aluminum 6061": "Aluminum 6061-T6",
    "Aluminum 7075": "Aluminum 7075-T6",
    "Aluminum 2024": "Aluminum 2024-T3",
    "Steel 4140": "Steel 4140 (annealed)",
    "Nylon": "Nylon 6/6",
}


@dataclass(frozen=True)
class MaterialRecord:
    name: str
    family: str
    iso_group: str
    machining: bool
    sheet_metal: bool
    density_g_cc: float
    uts_mpa: float
    min_bend_t_across: float | None
    min_bend_t_with: float | None
    air_radius_coeff: float | None
    k_curve: str
    legacy_key: str
    property_source: str
    property_url: str
    notes: str


def _f(v):
    v = (v or "").strip()
    return float(v) if v else None


@lru_cache(maxsize=1)
def load(path: str | None = None) -> tuple[MaterialRecord, ...]:
    p = Path(path) if path else DATA_PATH
    out = []
    with p.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.append(MaterialRecord(
                name=row["name"].strip(), family=row["family"].strip(),
                iso_group=row["iso_group"].strip(),
                machining=row["machining"].strip() == "1",
                sheet_metal=row["sheet_metal"].strip() == "1",
                density_g_cc=float(row["density_g_cc"]), uts_mpa=float(row["uts_mpa"]),
                min_bend_t_across=_f(row["min_bend_t_across"]),
                min_bend_t_with=_f(row["min_bend_t_with"]),
                air_radius_coeff=_f(row["air_radius_coeff"]),
                k_curve=row["k_curve"].strip(), legacy_key=row["legacy_key"].strip(),
                property_source=row["property_source"].strip(),
                property_url=row["property_url"].strip(), notes=row["notes"].strip(),
            ))
    names = [m.name for m in out]
    if len(set(names)) != len(names):
        raise ValueError("materials.csv has duplicate names")
    return tuple(out)


def _squash(s: str) -> str:
    return "".join(ch for ch in s.upper() if ch.isalnum())


@lru_cache(maxsize=None)
def _index() -> dict:
    idx = {}
    for m in load():
        idx[_squash(m.name)] = m.name
        if m.legacy_key:
            idx[_squash(m.legacy_key)] = m.name
    for old, new in LEGACY_NAMES.items():
        idx[_squash(old)] = new
    return idx


def canonical(name: str) -> str | None:
    """Current name for any current or legacy name, or None if unknown."""
    if not name:
        return None
    return _index().get(_squash(name))


def get(name: str) -> MaterialRecord | None:
    n = canonical(name)
    if n is None:
        return None
    return next(m for m in load() if m.name == n)


def machining_names() -> list[str]:
    return [m.name for m in load() if m.machining]


def sheet_names() -> list[str]:
    return [m.name for m in load() if m.sheet_metal]


def families() -> list[str]:
    seen = []
    for m in load():
        if m.family not in seen:
            seen.append(m.family)
    return seen


def token_match(query: str, text: str) -> bool:
    """True when every word of `query` appears in `text`, ignoring case,
    spaces and punctuation: "al 7075" matches "Aluminum 7075-T6"."""
    hay = _squash(text)
    words = [_squash(w) for w in query.replace("/", " ").replace("-", " ").split()]
    return all(w in hay for w in words if w)


def search_text(name: str) -> str:
    """What a material is searchable by: its name, family, ISO group and old key."""
    m = get(name)
    if m is None:
        return name
    return " ".join((m.name, m.family, m.iso_group, m.legacy_key))


def search(query: str, names: list[str] | None = None) -> list[str]:
    """Materials whose name, family, ISO group or legacy key match every
    word of `query`, in library order."""
    pool = names if names is not None else [m.name for m in load()]
    return [n for n in pool if token_match(query, search_text(n))]
