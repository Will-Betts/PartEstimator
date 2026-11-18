"""Feature export for building an ML training set.

Writes one row per part. CSV is append-friendly on purpose: point it at the
same file for every part you have a measured cycle time for, and it
accumulates into a training table you can join against your historicals.

Column order is stable and the header is written from the union of keys, so
rows produced by different versions of the tool still line up -- missing
values come out blank rather than shifting columns.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

# Columns that identify the row or qualify how much to trust it, rather
# than describing the geometry. Kept first for readability and excluded
# from any "feature" view.
#
# The quality flags belong here, not among the features: they say how
# reliable the row is, so they are the right thing to FILTER on and the
# wrong thing to FIT on. `hull_is_approximate` marks a row whose solidity
# came from a reduced convex hull; a high `rejected_hole_count` means
# several cylindrical features failed the enclosure test.
#
# The blank input columns are for values the geometry cannot supply.
# `cycle_time_min` is the training target. `tightest_tolerance_in` is the
# single biggest non-geometric driver -- a 0.0005" bore needs boring and
# reaming where a 0.010" one needs a drill -- and STEP rarely carries PMI,
# so it has to be entered.
METADATA_COLUMNS = [
    "part_name", "source_file", "analyzed_at", "units_note",
    "material", "target_finish", "target_finish_units",
    "tightest_tolerance_in",
    "removal_type",
    # The measured target, left blank for you to fill from job records.
    "cycle_time_min",
    # The bottom-up handbook estimate, for comparison against it. Kept as
    # metadata rather than as a feature: fitting a measured time against
    # an estimate built from the same geometry would just relearn the
    # estimator's own coefficients. The ratio of the two is the useful
    # diagnostic -- it says where the published rates miss.
    "estimated_cycle_time_min",
    "est_machine_cycle_min",
    "est_cut_min",
    "est_rough_outside_min",
    "est_rough_pocket_min",
    "est_finish_analytic_min",
    "est_finish_freeform_min",
    "est_hole_min",
    "est_turn_rough_min",
    "est_turn_finish_min",
    "est_live_tool_min",
    "est_load_unload_min",
    # --- row quality / provenance ---
    "hull_is_approximate", "rejected_hole_count",
    "holes_unprobed", "probe_timed_out",
    "wall_samples_used", "wall_samples_rejected",
    "face_count", "tessellation_deflection_mm",
]


TERM_COLUMNS = {
    "Outside roughing": "est_rough_outside_min",
    "Pocket roughing": "est_rough_pocket_min",
    "Finishing, analytic": "est_finish_analytic_min",
    "Finishing, freeform": "est_finish_freeform_min",
    "Holes": "est_hole_min",
    "OD roughing (turn)": "est_turn_rough_min",
    "Finish turning": "est_turn_finish_min",
    "Live-tool milling": "est_live_tool_min",
}


def estimate_columns(estimate) -> dict:
    """Flattens a CycleEstimate into its export columns.

    The per-term breakdown is exported, not just the total, because when
    the estimate disagrees with a measured time the useful question is
    *which term* was wrong -- a roughing miss means the MRR is off, a
    finishing miss means the scallop assumption is off.
    """
    if estimate is None:
        return {}
    out = {
        "estimated_cycle_time_min": round(estimate.door_to_door_minutes, 3),
        "est_machine_cycle_min": round(estimate.machine_cycle_minutes, 3),
        "est_cut_min": round(estimate.cut_minutes, 3),
        "est_load_unload_min": round(estimate.load_unload_minutes, 3),
    }
    for column in TERM_COLUMNS.values():
        out[column] = 0.0
    for term in estimate.terms:
        column = TERM_COLUMNS.get(term.name)
        if column:
            out[column] = round(term.minutes, 3)
    return out


def build_row(results: dict, derived_features: dict, material: str = "",
               finish_value: float | None = None, finish_units: str = "",
               tolerance_in: float | None = None,
               quality: dict | None = None,
               removal_type: str = "",
               estimate=None) -> dict:
    """Combines identification metadata, quality flags and the feature vector."""
    path = results.get("path", "")
    row = {
        "part_name": Path(path).stem if path else "",
        "source_file": path,
        "analyzed_at": datetime.now().isoformat(timespec="seconds"),
        "units_note": "all lengths in, areas in^2, volumes in^3, angles deg",
        "material": material,
        "target_finish": "" if finish_value is None else finish_value,
        "target_finish_units": finish_units,
        "tightest_tolerance_in": "" if tolerance_in is None else tolerance_in,
        "removal_type": removal_type,
        "cycle_time_min": "",  # <- fill from your job records before training
    }
    row.update(estimate_columns(estimate))
    stock = results.get("stock")
    if stock is not None:
        # Which stock the row was priced on: the rules, the bare part
        # envelope, a size typed in, or a named extrusion profile.
        row["stock_basis"] = stock.basis
        row["stock_profile"] = stock.profile.describe() if stock.profile is not None else ""
    if quality:
        row.update(quality)
    row.update(derived_features)
    return row


def write_json(path: str | Path, row: dict) -> None:
    Path(path).write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")


def append_csv(path: str | Path, row: dict) -> None:
    """Appends a row, rewriting the file if this row introduces new columns.

    Feature sets grow over time; silently dropping new columns (or worse,
    writing them in a different order) would quietly corrupt a training
    table, so a schema change triggers a rewrite that preserves old rows.
    """
    path = Path(path)
    existing: list[dict] = []
    existing_cols: list[str] = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            existing_cols = list(reader.fieldnames or [])
            existing = list(reader)

    new_cols = [c for c in row.keys() if c not in existing_cols]
    if existing_cols and not new_cols:
        with path.open("a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=existing_cols).writerow(
                {k: row.get(k, "") for k in existing_cols}
            )
        return

    # Build a stable column order: metadata first, then features sorted.
    all_cols = set(existing_cols) | set(row.keys())
    feature_cols = sorted(c for c in all_cols if c not in METADATA_COLUMNS)
    ordered = [c for c in METADATA_COLUMNS if c in all_cols] + feature_cols

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ordered)
        writer.writeheader()
        for old in existing:
            writer.writerow({k: old.get(k, "") for k in ordered})
        writer.writerow({k: row.get(k, "") for k in ordered})
