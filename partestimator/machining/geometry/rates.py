"""Loader for the cited machining-rate table.

The rates live in `data/machining_rates.csv` rather than in code so they
can be tuned to your shop without editing Python, and so every number
carries its source with it. Each row records:

    published_value   what the manufacturer or reference actually says
    derate            the safety factor applied (0.80 by default)
    effective_value   published x derate -- what the estimator uses
    source / url      where the number came from
    retrieved         when it was read

Two kinds of row, and the distinction matters:

  * `unit_power` is a PHYSICAL CONSTANT (hp per in^3/min) for the
    material. It is never derated, because derating a constant would be
    meaningless -- the 80% buffer belongs on the rate it produces, not on
    the physics. It bounds MRR by the spindle's horsepower.

  * everything else is a RECOMMENDED CUTTING PARAMETER (sfm, in/tooth,
    ipr). These are derated to 80%, which is where the buffer lives.
    They bound MRR by what the tool will actually take.

The estimator takes the MINIMUM of the two bounds. That matters more
than it sounds: on a 25 hp 40-taper cutting aluminium, the power bound
works out near 70 in^3/min while the tool bound is near 20, so the
machine is nowhere near its power limit and quoting the power figure
would overstate the rate three-fold. In titanium the two converge.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "machining_rates.csv"

# Applied to recommended cutting parameters, not to unit power.
DEFAULT_DERATE = 0.80


@dataclass(frozen=True)
class RateRow:
    material: str
    iso_group: str
    removal_type: str
    operation: str
    published_value: float
    units: str
    derate: float
    effective_value: float
    notes: str
    source: str
    source_url: str
    retrieved: str

    @property
    def is_overridden(self) -> bool:
        """True when effective_value does not equal published x derate --
        i.e. somebody tuned this row by hand. Worth surfacing, because a
        hand-tuned row is no longer traceable to its citation."""
        expected = self.published_value * self.derate
        return abs(self.effective_value - expected) > max(1e-6, abs(expected) * 1e-4)


class RateTable:
    """Keyed lookup over the CSV, with the sources kept attached."""

    def __init__(self, rows: list[RateRow]):
        self.rows = rows
        self._index: dict[tuple[str, str, str], RateRow] = {}
        for r in rows:
            self._index[(r.material, r.removal_type, r.operation)] = r

    @property
    def materials(self) -> list[str]:
        seen: list[str] = []
        for r in self.rows:
            if r.material not in seen:
                seen.append(r.material)
        return seen

    def get(self, material: str, removal_type: str, operation: str) -> RateRow | None:
        # Old names ("Aluminum 6061", "Nylon") resolve to current ones.
        from partestimator.materials_db import canonical

        material = canonical(material) or material
        row = self._index.get((material, removal_type, operation))
        if row is not None:
            return row
        # Milling parameters are the fallback for turning: a turn-mill's
        # live tooling is doing exactly the milling operation described,
        # and the turned-surface terms are handled separately.
        if removal_type != "mill":
            return self._index.get((material, "mill", operation))
        return None

    def value(self, material: str, removal_type: str, operation: str,
               default: float | None = None) -> float | None:
        row = self.get(material, removal_type, operation)
        return row.effective_value if row is not None else default

    def iso_group(self, material: str) -> str:
        from partestimator.materials_db import canonical

        material = canonical(material) or material
        for r in self.rows:
            if r.material == material:
                return r.iso_group
        return "?"

    def sources_for(self, material: str, removal_type: str) -> list[RateRow]:
        """Every row the estimate for this selection rests on."""
        from partestimator.materials_db import canonical

        material = canonical(material) or material
        out = [r for r in self.rows
               if r.material == material and r.removal_type in (removal_type, "mill")]
        out.sort(key=lambda r: r.operation)
        return out


def _to_float(value: str, field: str, line: int) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{DATA_PATH.name} line {line}: {field} is not a number ({value!r})"
        ) from None


def load_rates(path: str | Path | None = None) -> RateTable:
    """Reads and validates the rate table.

    Validation is deliberately strict and loud. A silently mis-parsed rate
    would produce a plausible-looking cycle time that is quietly wrong,
    which is far worse than refusing to load.
    """
    path = Path(path) if path is not None else DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Rate table not found at {path}. It ships in the data/ folder "
            f"beside the application."
        )

    rows: list[RateRow] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"material", "iso_group", "removal_type", "operation",
                    "published_value", "units", "derate", "effective_value",
                    "notes", "source", "source_url", "retrieved"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

        for line, raw in enumerate(reader, start=2):
            if any(v is None for v in raw.values()):
                raise ValueError(f"{path.name} line {line}: wrong number of fields")
            published = _to_float(raw["published_value"], "published_value", line)
            derate = _to_float(raw["derate"], "derate", line)
            effective = _to_float(raw["effective_value"], "effective_value", line)
            if not 0.0 < derate <= 1.0:
                raise ValueError(
                    f"{path.name} line {line}: derate must be in (0, 1], got {derate}")
            if effective <= 0 or published <= 0:
                raise ValueError(
                    f"{path.name} line {line}: rates must be positive")
            if not raw["source_url"].startswith("http"):
                raise ValueError(
                    f"{path.name} line {line}: source_url must be a URL, so every "
                    f"rate stays traceable (got {raw['source_url']!r})")
            rows.append(RateRow(
                material=raw["material"].strip(),
                iso_group=raw["iso_group"].strip(),
                removal_type=raw["removal_type"].strip(),
                operation=raw["operation"].strip(),
                published_value=published,
                units=raw["units"].strip(),
                derate=derate,
                effective_value=effective,
                notes=raw["notes"].strip(),
                source=raw["source"].strip(),
                source_url=raw["source_url"].strip(),
                retrieved=raw["retrieved"].strip(),
            ))

    if not rows:
        raise ValueError(f"{path.name} has no rate rows")
    return RateTable(rows)
