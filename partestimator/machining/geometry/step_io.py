"""STEP file loading.

Design note on units: instead of parsing the STEP header text ourselves to
guess the file's length unit, we let OpenCASCADE's STEP translator do the
(much more reliable) unit conversion, forcing every imported shape into
millimeters internally via the "xstep.cascade.unit" static parameter. This
matches how most CAD systems normalize STEP import and avoids the failure
modes of hand-rolled header parsing (comments, unusual unit definitions,
etc.). The UI's mm/inch toggle is then purely a *display* conversion on
already-consistent internal (mm) values -- it never touches the geometry.
"""
from __future__ import annotations

from pathlib import Path

from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.STEPControl import STEPControl_Reader
from OCP.TopoDS import TopoDS_Shape

MM_PER_INCH = 25.4


class StepLoadError(RuntimeError):
    """Raised when a STEP file cannot be read or contains no geometry."""


def load_step_file(path: str | Path) -> TopoDS_Shape:
    """Read a STEP file and return a single merged TopoDS_Shape, in mm.

    Raises StepLoadError on any failure (missing file, bad extension,
    unparseable STEP data, or empty geometry).
    """
    path = Path(path)
    if not path.exists():
        raise StepLoadError(f"File not found: {path}")
    if path.suffix.lower() not in (".step", ".stp"):
        raise StepLoadError(f"Not a STEP file (expected .step/.stp): {path}")

    # Force the STEP translator's target unit to millimeters regardless of
    # what unit the file itself was authored in. (OCP's binding generator
    # suffixes static C++ methods with "_s" -- SetCVal_s here, vs.
    # pythonocc-core's plain SetCVal.)
    Interface_Static.SetCVal_s("xstep.cascade.unit", "MM")

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))
    if status != IFSelect_RetDone:
        raise StepLoadError(f"Failed to parse STEP file: {path}")

    num_transferred = reader.TransferRoots()
    if num_transferred < 1:
        raise StepLoadError(f"STEP file contained no transferable roots: {path}")

    shape = reader.OneShape()
    if shape is None or shape.IsNull():
        raise StepLoadError(f"STEP file produced no geometry: {path}")
    return shape


def mm_to_display(value_mm: float, unit: str) -> float:
    """Convert an already-mm value to the requested display unit ('MM' or 'IN')."""
    if unit.upper() == "IN":
        return value_mm / MM_PER_INCH
    return value_mm


def display_to_mm(value_display: float, unit: str) -> float:
    """Inverse of mm_to_display -- converts a value entered in the display
    unit back to mm for internal computation."""
    if unit.upper() == "IN":
        return value_display * MM_PER_INCH
    return value_display


def mm2_to_display(value_mm2: float, unit: str) -> float:
    if unit.upper() == "IN":
        return value_mm2 / (MM_PER_INCH ** 2)
    return value_mm2


def mm3_to_display(value_mm3: float, unit: str) -> float:
    if unit.upper() == "IN":
        return value_mm3 / (MM_PER_INCH ** 3)
    return value_mm3


def mm5_to_display(value_mm5: float, unit: str) -> float:
    """For the geometric moment of inertia (unit density), which has units
    of length^5 (a volume integral of r^2 dV, not a 2D area moment)."""
    if unit.upper() == "IN":
        return value_mm5 / (MM_PER_INCH ** 5)
    return value_mm5
