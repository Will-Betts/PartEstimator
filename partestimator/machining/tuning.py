"""Shop tuning: a library of parts your shop has run, and the cycle-time
model fitted to them.

WHY

The estimate is built bottom-up from published cutting data. No catalogue
knows your machines, fixtures, tooling or programmers, so out of the box it
is a physically sensible starting point, not your shop's number. The way to
close that gap is to measure it: keep a library of parts you have actually
run, each with the inputs it was run with and the door-to-door minutes it
took, and fit a few of the shop-profile settings until the estimates match.

WHAT IS KEPT

Not the STEP files. When a part is added, its geometry is analysed once and
what the estimate reads is CAPTURED -- the derived feature vector (volumes,
areas by surface type, edge lengths, radii, thin-wall share...), every hole
(diameter, depth, axis) and, for round parts, the turning profile. That is a
few kilobytes of JSON per part, and it is all the estimate ever needs, so the
library never goes back to the model. The STEP file's path is noted only so
"Open in Machining tab" can show the part again while it is still there.

What can change after capture is the part's INPUTS -- material, process,
finish, operations, tapped holes and the stock -- and those are re-applied
to the captured geometry on every estimate.

WHERE

    <the tool's folder>\\tuning\\        (or the folder in PE_TUNING_DIR)
        parts.csv          the library: one row per part; edit it in Excel too
        captured\\         <id>.json: each part's captured geometry
        shop_tuning.csv    the adjustments, laid over data/shop_profile.csv
        last_fit.json      a summary of the fit that is in use

It is in .gitignore: your shop's data never goes into the repository, and a
new shop starts from the published values and builds its own. Point
PE_TUNING_DIR at a shared folder to share one library and one tuning between
the people in a shop.

shop_tuning.csv can hold ANY shop_profile.csv key -- spindle_hp, operations,
deburr_enabled. Rows marked `fit` are written by the fit; any other row is
yours and the fit leaves it alone (and fits on top of it).

THE FIT

Each part's estimate is a function of the chosen settings. The fit finds the
values that make estimate/measured closest to 1 across the library:

    minimise  sum over parts   rho( ln(estimate / measured) / 0.25 )^2
            + sum over settings     ( ln(value / shipped)     / ln 3 )^2

  * Errors are RATIOS (log), so a 2-minute part counts as much as a 2-hour
    one.
  * rho is a soft (Huber-like) loss: one bad record -- a measured time that
    included setup, a wrong operation count -- cannot drag the fit far.
  * The second sum keeps each setting near its shipped value unless the
    parts give a reason to move it. A setting the parts say nothing about
    (lathe settings with no lathe parts) simply stays put, and the report
    says the data did not pin it.

It is solved by Levenberg-Marquardt in numpy alone (no SciPy), on the log of
each setting, inside a factor of 10 either way. With 6+ parts it also
reports a cross-validated error: each part estimated by a fit that never saw
it -- the honest figure for what to expect on a new part.
"""
from __future__ import annotations

import concurrent.futures as cf
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import sys
import time
from dataclasses import dataclass, field, fields
from datetime import date, datetime
from pathlib import Path

import numpy as np

from partestimator import __version__

from .geometry import cycle_time, materials, shop
from .geometry import stock as st

ENV_DIR = "PE_TUNING_DIR"
FOLDER_NAME = "tuning"
LIBRARY_FILE = "parts.csv"
TUNING_FILE = "shop_tuning.csv"
SUMMARY_FILE = "last_fit.json"
CAPTURED_DIR = "captured"
# Bump if the captured record changes shape. Records are never thrown away
# for being old -- the STEP file may be long gone -- so readers must cope
# with every format written so far.
CAPTURE_FORMAT = 1

# The analysis the Machining window runs on every file. Kept here, the one
# place both it and the library read, so a captured part is always what the
# window would have seen.
ANALYSIS_OPTIONS = {
    "oriented_box": True,      # tightest stock size
    "cylinder": True,          # round-stock fit
    "undercut": False,         # expensive, coarse heuristic
    "edges": True,             # model edges for display
    "robust_concavity": False, # slow global point-in-solid fallback
    "edge_convexity": True,    # concave-edge complexity metric
    "extrusion_fit": True,     # open on an extrusion when one beats plate
}

PROCESSES = ("auto", "mill", "lathe")
STOCK_TYPES = ("plate", "round", "extrusion")

COLUMNS = ["include", "part", "id", "material", "process", "finish_uin",
           "tolerance_in", "operations", "tapped_holes", "stock_type", "stock_size",
           "stock_profile", "measured_min", "notes", "source_file", "added"]

README = """\
PartEstimator shop tuning
=========================

This folder is your shop's own data. It is never committed to the
repository.

The STEP files are not kept. Each part's geometry was analysed when it was
added and what the cycle-time estimate needs is in captured\\<id>.json.

parts.csv        One row per part. Edit it here in Excel (keep the header
                 row, save as CSV) or in the Shop tuning window.
  include        1 to use the part in the fit, 0 to leave it out
  part           a name for it
  id             links the row to captured\\<id>.json -- do not edit
  material       as named in the tool, e.g. Aluminum 6061-T6
  process        mill, lathe or auto
  finish_uin     target surface finish, micro-inches Ra
  tolerance_in   tightest tolerance, inches (recorded, not yet used)
  operations     machine operations (setups) per part
  tapped_holes   blank = detect from tap-drill sizes
  stock_type     plate, round or extrusion
  stock_size     plate: T x W x L     round: D x L     extrusion: length
                 inches; fractions are fine (1-1/4)
  stock_profile  extrusion only: AND10133-2004, or e.g. Angle 2 x 2 x 1/4
  measured_min   MEASURED door-to-door minutes PER PART, every operation,
                 load/unload and deburr at the machine included. NOT the
                 batch setup: divide that out or leave it off.
  notes          anything
  source_file    where the STEP file was when it was added (for reference)

shop_tuning.csv  The adjustments in use, over the tool's shop_profile.csv.
                 Rows with source "fit" come from the last saved fit. Add
                 your own rows (source "manual") for any shop_profile key,
                 e.g. spindle_hp -- they are kept and fitted on top of.

Delete shop_tuning.csv (or press "Back to shipped values") to go back to the
published values.
"""


# ---------------------------------------------------------------------------
# where
# ---------------------------------------------------------------------------


def tool_dir() -> Path:
    """The PartEstimator folder (the one holding the launcher)."""
    return Path(__file__).resolve().parents[2]


def tuning_dir() -> Path:
    env = os.environ.get(ENV_DIR, "").strip()
    if env:
        return Path(env).expanduser()
    return tool_dir() / FOLDER_NAME


# ---------------------------------------------------------------------------
# numbers as people type them
# ---------------------------------------------------------------------------

_MIXED = re.compile(r"^(\d+(?:\.\d*)?)?\s*[- ]?\s*(?:(\d+)\s*/\s*(\d+))?$")


def parse_length(text: str) -> float:
    """'1.25', '1-1/4', '1 1/4', '5/8', '2"' and '2 in' -> inches."""
    s = str(text).strip().lower().replace('"', "").replace("in", "").strip()
    if not s:
        raise ValueError("empty")
    try:
        return float(s)
    except ValueError:
        pass
    m = _MIXED.match(s)
    if not m or not (m.group(1) or m.group(2)):
        raise ValueError(f"not a length: {text!r}")
    whole = float(m.group(1) or 0.0)
    frac = float(m.group(2)) / float(m.group(3)) if m.group(2) else 0.0
    return whole + frac


def parse_sizes(text: str) -> list[float]:
    """'1.25 x 4 x 6' -> [1.25, 4.0, 6.0]. 'x', 'X', '*' and the times sign
    all separate."""
    parts = [p for p in re.split(r"\s*[x×*]\s*", str(text).strip(), flags=re.IGNORECASE) if p]
    return [parse_length(p) for p in parts]


def _num(text, default=None, cast=float):
    s = str(text if text is not None else "").strip()
    if s == "":
        return default
    try:
        return cast(float(s)) if cast is int else cast(s)
    except ValueError:
        return default


def _fmt(v: float) -> str:
    return f"{v:.4f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------------------
# the library
# ---------------------------------------------------------------------------


@dataclass
class Entry:
    part: str
    id: str
    material: str = ""
    process: str = "mill"
    finish_uin: float = 63.0
    tolerance_in: float | None = None
    operations: int = 1
    tapped_holes: int | None = None
    stock_type: str = "plate"
    stock_size: str = ""
    stock_profile: str = ""
    measured_min: float | None = None
    include: bool = True
    notes: str = ""
    source_file: str = ""
    added: str = ""

    def to_row(self) -> dict:
        return {
            "include": "1" if self.include else "0",
            "part": self.part,
            "id": self.id,
            "material": self.material,
            "process": self.process,
            "finish_uin": _fmt(self.finish_uin),
            "tolerance_in": "" if self.tolerance_in is None else _fmt(self.tolerance_in),
            "operations": str(self.operations),
            "tapped_holes": "" if self.tapped_holes is None else str(self.tapped_holes),
            "stock_type": self.stock_type,
            "stock_size": self.stock_size,
            "stock_profile": self.stock_profile,
            "measured_min": "" if self.measured_min is None else _fmt(self.measured_min),
            "notes": self.notes,
            "source_file": self.source_file,
            "added": self.added,
        }

    @classmethod
    def from_row(cls, row: dict) -> "Entry":
        g = {k: (row.get(k) or "").strip() for k in COLUMNS}
        if not g["id"]:
            raise ValueError("no id (rows are added from the tool, which captures "
                             "the part's geometry)")
        process = g["process"].lower() or "auto"
        if process not in PROCESSES:
            raise ValueError(f"process {g['process']!r} is not mill, lathe or auto")
        stock_type = g["stock_type"].lower() or "plate"
        if stock_type not in STOCK_TYPES:
            raise ValueError(f"stock_type {g['stock_type']!r} is not one of "
                             f"{', '.join(STOCK_TYPES)}")
        measured = _num(g["measured_min"])
        if measured is not None and measured < 0:
            raise ValueError("measured_min is negative")
        return cls(
            part=g["part"] or g["id"],
            id=g["id"],
            material=g["material"],
            process=process,
            finish_uin=_num(g["finish_uin"], 63.0),
            tolerance_in=_num(g["tolerance_in"]),
            operations=max(_num(g["operations"], 1, int) or 1, 1),
            tapped_holes=_num(g["tapped_holes"], None, int),
            stock_type=stock_type,
            stock_size=g["stock_size"],
            stock_profile=g["stock_profile"],
            measured_min=measured,
            include=g["include"].lower() not in ("0", "no", "n", "false", "x"),
            notes=g["notes"],
            source_file=g["source_file"],
            added=g["added"],
        )


def file_id(path: Path) -> str:
    """A part's id: the start of its STEP file's SHA-1, so adding the same
    file twice finds the first entry."""
    h = hashlib.sha1()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class Library:
    """parts.csv and the captured geometry beside it."""

    def __init__(self, folder: Path | None = None):
        self.folder = Path(folder) if folder is not None else tuning_dir()
        self.entries: list[Entry] = []
        self.errors: list[str] = []      # rows of parts.csv that could not be read

    @property
    def csv_path(self) -> Path:
        return self.folder / LIBRARY_FILE

    @property
    def captured_dir(self) -> Path:
        return self.folder / CAPTURED_DIR

    def ensure(self) -> None:
        """Create the folder, its README and an empty library."""
        self.captured_dir.mkdir(parents=True, exist_ok=True)
        readme = self.folder / "README.txt"
        if not readme.exists():
            readme.write_text(README, encoding="utf-8")
        if not self.csv_path.exists():
            self.save()

    # -- parts.csv -------------------------------------------------------------
    @classmethod
    def load(cls, folder: Path | None = None) -> "Library":
        lib = cls(folder)
        if not lib.csv_path.exists():
            return lib
        with lib.csv_path.open("r", newline="", encoding="utf-8-sig") as fh:
            for line, row in enumerate(csv.DictReader(fh), start=2):
                if not any((v or "").strip() for v in row.values() if isinstance(v, str)):
                    continue
                try:
                    lib.entries.append(Entry.from_row(row))
                except ValueError as exc:
                    lib.errors.append(f"line {line}: {exc}")
        return lib

    def save(self) -> None:
        """Write parts.csv. A PermissionError usually means it is open in
        Excel."""
        self.folder.mkdir(parents=True, exist_ok=True)
        tmp = self.csv_path.with_suffix(".tmp")
        with tmp.open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader()
            for e in self.entries:
                w.writerow(e.to_row())
        os.replace(tmp, self.csv_path)

    # -- entries and their captured geometry -----------------------------------
    def find(self, part_id: str) -> Entry | None:
        return next((e for e in self.entries if e.id == part_id), None)

    def snapshot_path(self, e: Entry) -> Path:
        return self.captured_dir / f"{e.id}.json"

    def has_snapshot(self, e: Entry) -> bool:
        return self.snapshot_path(e).is_file()

    def load_snapshot(self, e: Entry) -> dict | None:
        try:
            snap = json.loads(self.snapshot_path(e).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return snap if isinstance(snap, dict) and "features" in snap else None

    def save_snapshot(self, e: Entry, snap: dict) -> None:
        self.ensure()
        path = self.snapshot_path(e)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(snap, indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    def add(self, entry: Entry, snap: dict) -> Entry:
        """Add (or, for an id already here, replace the geometry of) a part."""
        existing = self.find(entry.id)
        self.save_snapshot(entry, snap)
        if existing is None:
            if not entry.added:
                entry.added = date.today().isoformat()
            self.entries.append(entry)
            return entry
        return existing

    def remove(self, entry: Entry) -> None:
        self.entries.remove(entry)
        if self.find(entry.id) is None:
            try:
                self.snapshot_path(entry).unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# capturing a part: analysis results -> a small JSON record
# ---------------------------------------------------------------------------


def _plain(v):
    """JSON-safe: numpy scalars and arrays to Python numbers and lists."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return _plain(v.tolist())
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, (int, str)) or v is None:
        return v
    return str(v)


def capture(results: dict, stock, material: str = "") -> dict:
    """Everything the cycle-time estimate reads about a part's geometry.

    `results` is the analysis (pipeline.compute_analysis, as the Machining
    window holds it), `stock` the stock it is cut from. Stock and material
    are re-applied from the library row on every estimate; they are used
    here only so the captured features are complete as of today.
    """
    from .geometry import derived

    res = dict(results)
    res["stock"] = stock
    feats = derived.compute_derived_features(res, material=material)
    holes = [{
        "nominal_diameter_mm": float(getattr(h, "nominal_diameter_mm", 0.0)),
        "depth_mm": float(getattr(h, "depth_mm", 0.0)),
        "axis_direction": _plain(np.asarray(getattr(h, "axis_direction", (0, 0, 1)), float)),
        "axis_point": _plain(np.asarray(getattr(h, "axis_point", (0, 0, 0)), float)),
        "internal_groove_count": int(getattr(h, "internal_groove_count", 0) or 0),
    } for h in (results.get("holes") or [])]
    tg = results.get("turning")
    turning = ({f.name: _plain(getattr(tg, f.name)) for f in fields(tg)}
               if tg is not None else None)
    return {
        "format": CAPTURE_FORMAT,
        "tool_version": __version__,
        "captured": datetime.now().isoformat(timespec="seconds"),
        "features": _plain(dict(feats)),
        "holes": holes,
        "turning": turning,
    }


@dataclass
class CapturedHole:
    """The attributes of a detected hole that the timing model reads."""
    nominal_diameter_mm: float
    depth_mm: float
    axis_direction: np.ndarray
    axis_point: np.ndarray
    internal_groove_count: int = 0


def _holes(snap: dict) -> list[CapturedHole]:
    return [CapturedHole(float(h["nominal_diameter_mm"]), float(h["depth_mm"]),
                         np.asarray(h.get("axis_direction", (0, 0, 1)), float),
                         np.asarray(h.get("axis_point", (0, 0, 0)), float),
                         int(h.get("internal_groove_count", 0) or 0))
            for h in snap.get("holes") or []]


def _turning(snap: dict):
    data = snap.get("turning")
    if not data:
        return None
    from .geometry.turning import TurningGeometry

    known = {f.name for f in fields(TurningGeometry)}
    kw = {k: v for k, v in data.items() if k in known}
    for k in ("axis_point_mm", "axis_direction", "profile_z_mm", "profile_r_mm"):
        if kw.get(k) is not None:
            kw[k] = np.asarray(kw[k], dtype=float)
    kw["axial_holes"] = [tuple(h) for h in kw.get("axial_holes") or []]
    return TurningGeometry(**kw)


# ---------------------------------------------------------------------------
# stock, written as a person would and rebuilt without the geometry
# ---------------------------------------------------------------------------


def find_profile(text: str):
    """An extrusion profile from what a person would write: a part number
    (AND10133-2004 / AMS...), '<Shape> <name>' (Angle 2 x 2 x 1/4), 'shape|name',
    or a bare catalog name if only one shape has it."""
    from .geometry import extrusions as ex

    s = (text or "").strip()
    if not s:
        return None
    if "|" in s:
        return ex.find(*[t.strip() for t in s.split("|", 1)])
    hit = ex.find_part_number(s)
    if hit is not None:
        return hit
    low = s.lower()
    profiles = ex.load_profiles()
    for p in profiles:
        if low in (f"{p.shape_label} {p.name}".lower(), f"{p.shape} {p.name}".lower()):
            return p
    named = [p for p in profiles if p.name.lower() == low]
    return named[0] if len(named) == 1 else None


def profile_text(p) -> str:
    """How an extrusion profile is written into parts.csv."""
    return p.name if (p.is_standard or p.custom) else f"{p.shape_label} {p.name}"


def stock_fields(stock) -> tuple[str, str, str]:
    """(stock_type, stock_size, stock_profile) for a StockSize."""
    if stock is None:
        raise ValueError("no stock")
    if stock.is_extrusion:
        return "extrusion", _fmt(stock.length_in), profile_text(stock.profile)
    if stock.is_round:
        return "round", f"{_fmt(stock.thickness_or_diameter_in)} x {_fmt(stock.length_in)}", ""
    return ("plate", f"{_fmt(stock.thickness_or_diameter_in)} x {_fmt(stock.width_in or 0.0)}"
                     f" x {_fmt(stock.length_in)}", "")


def stock_text(e: Entry) -> str:
    """One line for a table, readable back by parse_stock_text."""
    if e.stock_type == "extrusion":
        return f"extrusion {e.stock_profile} @ {e.stock_size}"
    return f"{e.stock_type} {e.stock_size}"


def parse_stock_text(text: str) -> tuple[str, str, str]:
    """'plate 1 x 4 x 6', 'round 2 x 6', 'extrusion AND10133-2004 @ 6.5'
    -> (stock_type, stock_size, stock_profile). Checked as it is parsed."""
    s = (text or "").strip()
    kind, _sp, rest = s.partition(" ")
    kind = kind.lower()
    if kind == "extrusion":
        profile, at, length = rest.partition("@")
        if not at:
            raise ValueError("write an extrusion as: extrusion <profile> @ <length>")
        fields_ = ("extrusion", length.strip(), profile.strip())
    elif kind in ("plate", "round"):
        fields_ = (kind, rest.strip(), "")
    else:
        raise ValueError("start with plate, round or extrusion")
    build_stock(Entry(part="", id="", stock_type=fields_[0], stock_size=fields_[1],
                      stock_profile=fields_[2]))
    return fields_


def build_stock(e: Entry):
    """The StockSize an entry was cut from, from its stock columns alone.
    Raises ValueError for a stock description that does not make sense."""
    try:
        size = parse_sizes(e.stock_size) if e.stock_size.strip() else []
    except ValueError as exc:
        raise ValueError(f"stock size {e.stock_size!r}: {exc}") from None
    if any(v <= 0 for v in size):
        raise ValueError(f"stock size {e.stock_size!r} has a zero")
    if e.stock_type == "plate":
        if len(size) != 3:
            raise ValueError(f"plate size needs T x W x L, got {e.stock_size!r}")
        return st.custom_rectilinear(*size)
    if e.stock_type == "round":
        if len(size) != 2:
            raise ValueError(f"round size needs D x L, got {e.stock_size!r}")
        return st.custom_cylindrical(*size)
    if e.stock_type != "extrusion":
        raise ValueError(f"stock type {e.stock_type!r}")
    prof = find_profile(e.stock_profile)
    if prof is None:
        raise ValueError(f"no extrusion profile called {e.stock_profile!r}")
    if len(size) != 1:
        raise ValueError(f"extrusion size is its length, got {e.stock_size!r}")
    from .geometry import extrusions as ex

    ex_in, ey_in = prof.envelope_in
    return st.StockSize(
        kind="extrusion", thickness_or_diameter_in=ex_in,
        width_in=None if prof.shape in ex.ROUND_SHAPES else ey_in,
        length_in=size[0], volume_in3=prof.area_in2 * size[0], basis=st.EXTRUSION,
        profile=prof, min_length_in=size[0], recommended_length_in=size[0])


def auto_stock(results: dict):
    """The stock the Machining window opens a part on: an extrusion when the
    analysis found one beats plate, else the recommended plate or bar."""
    from .geometry import extrusions as ex

    rect, cyl = results.get("rect_box"), results.get("cylinder")
    primary = results.get("primary")
    found = results.get("extrusion")
    if found and not (primary == "cylindrical" and cyl is not None):
        prof = ex.find(*found)
        if prof is not None:
            sp = results.get("shape_props")
            pts, tris = results.get("render_points"), results.get("render_triangles")
            samples = (ex._sample(ex.surface_samples(pts, tris))
                       if pts is not None and tris is not None and len(pts) else None)
            s = ex.extrusion_stock(
                prof, rect, cyl, points_mm=samples,
                part_volume_in3=sp.volume_mm3 / st.MM_PER_INCH ** 3 if sp is not None else None)
            if s is not None:
                return s
    return results.get("stock") or st.compute_stock(primary, rect, cyl)


# ---------------------------------------------------------------------------
# adding STEP files: analysed in worker processes, captured, not kept
# ---------------------------------------------------------------------------


def default_workers() -> int:
    return max(1, min(3, (os.cpu_count() or 2) // 2))


def _capture_worker(step_path: str, material: str, allowances: dict | None = None) -> dict:
    """Runs in a child process: analyse one file and capture it, with the
    parent's stock allowances (a new process starts on the shipped ones)."""
    from .geometry import pipeline

    results = pipeline.compute_analysis(
        step_path, options={**ANALYSIS_OPTIONS, "stock_allowances": allowances})
    stock = auto_stock(results)
    return {"snapshot": capture(results, stock, material), "stock": stock_fields(stock)}


def _kill(pool: cf.ProcessPoolExecutor) -> None:
    procs = list(getattr(pool, "_processes", {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    for p in procs:
        try:
            p.terminate()
        except Exception:  # noqa: BLE001
            pass


def capture_files(paths, material: str = "", workers: int | None = None,
                  progress=None, cancelled=None) -> dict:
    """{path: captured or error string} for each STEP file.

    Runs `workers` files at a time, each in its own process, so a file that
    crashes the CAD kernel takes only itself down: the pool is rebuilt and
    the files that were in flight are retried one at a time.
    """
    paths = [Path(p) for p in paths]
    out: dict = {}
    total, done = len(paths), 0

    def report(p, value):
        nonlocal done
        done += 1
        out[p] = value
        if progress is not None:
            ok = isinstance(value, dict)
            progress(done, total, f"{p.stem}: {'captured' if ok else value}")

    def run(batch, n):
        orphans = []
        if not batch:
            return orphans
        pool = cf.ProcessPoolExecutor(max_workers=max(1, min(n, len(batch))),
                                      mp_context=mp.get_context("spawn"))
        allowances = st.allowances()
        futs = {pool.submit(_capture_worker, str(p), material, allowances): p for p in batch}
        pending = set(futs)
        try:
            while pending:
                if cancelled is not None and cancelled():
                    _kill(pool)
                    raise KeyboardInterrupt
                finished, pending = cf.wait(pending, timeout=0.25,
                                            return_when=cf.FIRST_COMPLETED)
                for f in finished:
                    try:
                        report(futs[f], f.result())
                    except cf.process.BrokenProcessPool:
                        orphans.append(futs[f])
                    except Exception as exc:  # noqa: BLE001
                        report(futs[f], f"{type(exc).__name__}: {exc}")
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        return orphans

    todo = []
    for p in paths:
        if p.is_file():
            todo.append(p)
        else:
            report(p, "file not found")
    try:
        # A crash breaks the whole pool, so which file did it is unknown.
        # One at a time, the culprit only takes itself down.
        for p in run(todo, workers or default_workers()):
            for again in run([p], 1):
                report(again, "the CAD kernel crashed reading this file")
    except KeyboardInterrupt:
        pass
    return out


def add_files(library: Library, paths, material: str = "", finish_uin: float = 63.0,
              operations: int = 1, profile=None, workers=None, progress=None,
              cancelled=None) -> tuple[list[Entry], list[str], dict]:
    """Analyse, capture and add STEP files. Returns (added, already in the
    library, {path: error}). The files themselves are not kept."""
    already, fresh, ids = [], [], {}
    for p in map(Path, paths):
        if not p.is_file():
            continue
        pid = file_id(p)
        hit = library.find(pid)
        if hit is not None or pid in ids.values():
            already.append(hit.part if hit is not None else p.stem)
            continue
        ids[p] = pid
        fresh.append(p)
    errors = {str(p): "file not found" for p in map(Path, paths) if not p.is_file()}
    added = []
    for p, value in capture_files(fresh, material, workers, progress, cancelled).items():
        if not isinstance(value, dict):
            errors[str(p)] = value
            continue
        stock_type, stock_size, stock_profile = value["stock"]
        e = Entry(part=p.stem, id=ids[p], material=material, finish_uin=finish_uin,
                  operations=operations, stock_type=stock_type, stock_size=stock_size,
                  stock_profile=stock_profile, source_file=str(p.resolve()))
        snap = value["snapshot"]
        e.process, _why = cycle_time.auto_removal_type(
            {"derived": snap["features"], "stock": build_stock(e)}, profile)
        added.append(library.add(e, snap))
    return added, already, errors


# ---------------------------------------------------------------------------
# turning an entry into an estimate
# ---------------------------------------------------------------------------


@dataclass
class Prepared:
    """Everything an estimate needs that does not depend on the shop settings,
    worked out once per part so the fit only re-runs the arithmetic."""
    entry: Entry
    features: dict
    results: dict
    material: str
    finish_uin: float
    removal: str
    operations: int
    tapped: int | None
    measured: float | None


def prepare(e: Entry, snap: dict, profile=None) -> Prepared:
    """The captured geometry with this entry's inputs applied."""
    stock = build_stock(e)
    feats = dict(snap["features"])
    feats.update(st.stock_features(stock, float(feats.get("volume_in3") or 0.0)))
    density = materials.density_lb_per_in3(e.material) if e.material else None
    if density is not None:
        feats["material_density_lb_in3"] = float(density)
    else:
        feats.pop("material_density_lb_in3", None)
    results = {"holes": _holes(snap), "turning": _turning(snap), "stock": stock,
               "derived": feats}
    removal = e.process
    if removal == "auto":
        removal, _why = cycle_time.auto_removal_type(results, profile)
    return Prepared(entry=e, features=feats, results=results, material=e.material,
                    finish_uin=float(e.finish_uin), removal=removal,
                    operations=int(e.operations), tapped=e.tapped_holes,
                    measured=e.measured_min)


def estimate(p: Prepared, table, profile):
    return cycle_time.estimate_cycle_time(
        p.features, p.results, table, material=p.material, finish_uin=p.finish_uin,
        removal_type=p.removal, profile=profile, operations=p.operations,
        tapped_override=p.tapped)


def problems(e: Entry, table=None) -> list[str]:
    """What stops this entry being fitted, apart from its captured geometry."""
    out = []
    if not e.material:
        out.append("no material")
    elif table is not None and table.get(e.material, "mill", "unit_power") is None:
        out.append(f"no rates for {e.material}")
    if not e.measured_min or e.measured_min <= 0:
        out.append("no measured time")
    try:
        build_stock(e)
    except ValueError as exc:
        out.append(str(exc))
    return out


# ---------------------------------------------------------------------------
# the shop overrides (shop_tuning.csv)
# ---------------------------------------------------------------------------

OVERRIDE_COLUMNS = ["key", "value", "source", "notes"]


@dataclass
class Override:
    key: str
    value: float
    source: str = "manual"
    notes: str = ""


def read_overrides(folder: Path | None = None) -> tuple[dict[str, Override], list[str]]:
    """({key: Override}, problems). A missing file is no overrides."""
    path = (Path(folder) if folder is not None else tuning_dir()) / TUNING_FILE
    out: dict[str, Override] = {}
    errors: list[str] = []
    if not path.is_file():
        return out, errors
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        for line, row in enumerate(csv.DictReader(fh), start=2):
            key = (row.get("key") or "").strip()
            if not key or key.startswith("#"):
                continue
            try:
                value = float((row.get("value") or "").strip())
            except ValueError:
                errors.append(f"{TUNING_FILE} line {line}: {key} is not a number")
                continue
            source = (row.get("source") or "").strip().lower() or "manual"
            out[key] = Override(key, value, source, (row.get("notes") or "").strip())
    return out, errors


def write_overrides(overrides: dict[str, Override], folder: Path | None = None) -> Path:
    folder = Path(folder) if folder is not None else tuning_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / TUNING_FILE
    tmp = path.with_suffix(".tmp")
    rows = sorted(overrides.values(), key=lambda o: (o.source == "fit", o.key))
    with tmp.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=OVERRIDE_COLUMNS)
        w.writeheader()
        for o in rows:
            w.writerow({"key": o.key, "value": f"{o.value:.6g}", "source": o.source,
                        "notes": o.notes})
    os.replace(tmp, path)
    return path


@dataclass
class TuningState:
    """What the window says about the tuning in use."""
    folder: Path
    fitted: dict = field(default_factory=dict)     # key -> value from the fit
    manual: dict = field(default_factory=dict)     # key -> value set by hand
    summary: dict | None = None                    # last_fit.json
    errors: list = field(default_factory=list)

    @property
    def active(self) -> bool:
        return bool(self.fitted or self.manual)

    def describe(self) -> str:
        """One line, or "" for none."""
        if not self.active:
            return ""
        bits = []
        if self.fitted:
            s = self.summary or {}
            n, when = s.get("parts"), s.get("date", "")
            err = s.get("median_error_pct")
            what = f"tuned to {n} part{'s' if n != 1 else ''}" if n else "tuned"
            if when:
                what += f" on {when}"
            if err is not None:
                what += f", typical error {err:.0f}%"
            bits.append(what)
        if self.manual:
            bits.append(f"{len(self.manual)} setting{'s' if len(self.manual) != 1 else ''} "
                        f"set by hand")
        return "Shop tuning: " + "; ".join(bits) + "."


def load_tuned_profile(folder: Path | None = None, base_path=None):
    """(ShopProfile, TuningState): the shipped profile with this shop's
    overrides laid over it. A broken override is reported, not fatal: the
    shipped profile still loads."""
    folder = Path(folder) if folder is not None else tuning_dir()
    state = TuningState(folder=folder)
    overrides, state.errors = read_overrides(folder)
    for o in overrides.values():
        (state.fitted if o.source == "fit" else state.manual)[o.key] = o.value
    try:
        profile = shop.load_profile(base_path, overrides={k: o.value for k, o in overrides.items()})
    except ValueError as exc:
        state.errors.append(f"{TUNING_FILE}: {exc} -- using the shipped values")
        state.fitted, state.manual = {}, {}
        profile = shop.load_profile(base_path)
    summary = folder / SUMMARY_FILE
    if state.fitted and summary.is_file():
        try:
            state.summary = json.loads(summary.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state.summary = None
    return profile, state


def fit_baseline(folder: Path | None = None, base_path=None):
    """The profile a fit starts from: shipped values plus the hand-set rows,
    without any earlier fit. Fitting from here every time makes re-fitting
    give the same answer, rather than compounding on the last one."""
    overrides, _errors = read_overrides(folder)
    manual = {k: o.value for k, o in overrides.items() if o.source != "fit"}
    return shop.load_profile(base_path, overrides=manual)


# ---------------------------------------------------------------------------
# the fit
# ---------------------------------------------------------------------------

SIGMA_PART = 0.25            # ln units: ~25% is ordinary scatter for one part
SIGMA_PRIOR = math.log(3.0)  # a setting moving 3x from shipped is one "sigma"
HUBER = 1.5                  # residuals past this many sigma count linearly
BOUND = 10.0                 # no setting moves more than 10x either way


@dataclass(frozen=True)
class Knob:
    key: str
    label: str
    tip: str
    default: bool = True


KNOBS = (
    Knob("calibrate_roughing", "Roughing time ×",
         "Multiplies every roughing term, milled and turned.\n"
         "Above 1: your roughing takes longer than the published rates allow."),
    Knob("calibrate_finishing", "Finishing time ×",
         "Multiplies the finishing terms (semi-finish, finish, spring pass)."),
    Knob("calibrate_holes", "Hole and tapping time ×",
         "Multiplies drilling and tapping, cycle overheads included."),
    Knob("tool_change_s", "Tool change (s)",
         "Seconds per tool change, chip to chip. Repeats per operation."),
    Knob("load_unload_base_min", "Load / unload, mill (min)",
         "Minutes to load and unload a part, per operation, before the\n"
         "allowance for its weight."),
    Knob("lathe_load_unload_min", "Load / unload, lathe (min)",
         "Minutes per lathe operation to load and unload. Only lathe parts\n"
         "can move it."),
    Knob("deburr_rate_ipm", "Hand deburr rate (in/min)",
         "Inches of edge broken per minute at the machine. Higher = faster.", False),
    Knob("turret_index_s", "Turret index, lathe (s)",
         "Seconds per lathe tool change.", False),
    Knob("index_time_s", "5-axis index (s)",
         "Seconds per rotary index between tool orientations.", False),
    Knob("finish_ra_factor_63uin", "63 µin finish ×",
         "Extra cost of a 63 µin finish on flat-tool surfaces, against 125.", False),
    Knob("finish_ra_factor_32uin", "32 µin finish ×",
         "Extra cost of a 32 µin finish on flat-tool surfaces, against 125.", False),
    Knob("finish_ra_factor_16uin", "16 µin finish ×",
         "Extra cost of a 16 µin finish on flat-tool surfaces, against 125.", False),
)
KNOB_BY_KEY = {k.key: k for k in KNOBS}


@dataclass
class Stats:
    n: int = 0
    median_error_pct: float = 0.0     # median |estimate/measured - 1|
    bias_pct: float = 0.0             # geometric-mean estimate/measured - 1
    within_20_pct: float = 0.0        # share of parts within 20%

    @classmethod
    def of(cls, ratios) -> "Stats":
        r = np.asarray([x for x in ratios if x and x > 0 and math.isfinite(x)], dtype=float)
        if not len(r):
            return cls()
        return cls(n=len(r),
                   median_error_pct=float(np.median(np.abs(r - 1.0)) * 100.0),
                   bias_pct=float((np.exp(np.mean(np.log(r))) - 1.0) * 100.0),
                   within_20_pct=float(np.mean(np.abs(r - 1.0) <= 0.20) * 100.0))

    def text(self) -> str:
        if not self.n:
            return "—"
        side = "high" if self.bias_pct > 0 else "low"
        return (f"typical error {self.median_error_pct:.0f}%, "
                f"{self.within_20_pct:.0f}% of parts within ±20%, "
                f"{abs(self.bias_pct):.0f}% {side} on average")


@dataclass
class KnobResult:
    key: str
    label: str
    baseline: float
    fitted: float
    spread: float          # 1-sigma factor, e.g. 1.15 = about ±15%
    pinned: bool           # did the parts decide it (vs. the shipped value)?
    used: float            # mean share of a part's estimate it touches


@dataclass
class PartResult:
    entry: Entry
    measured: float
    before: float
    after: float
    held_out: float | None = None    # estimate from a fit that did not see it

    @property
    def ratio_before(self) -> float:
        return self.before / self.measured

    @property
    def ratio_after(self) -> float:
        return self.after / self.measured


@dataclass
class Fit:
    knobs: list[KnobResult]
    parts: list[PartResult]
    before: Stats
    after: Stats
    cross_checked: Stats | None
    warnings: list[str]
    when: str = ""

    def values(self) -> dict:
        return {k.key: k.fitted for k in self.knobs}


def _profile_with(base, values: dict):
    return shop.ShopProfile(values={**base.values, **values}, notes=dict(base.notes))


def usable_knobs(base, keys) -> list[str]:
    """Settings that can be fitted: present and above zero in the profile
    (index_time_s = 0 means a 3-axis machine, which a ratio cannot move)."""
    return [k for k in keys if base.get(k, 0.0) > 0.0]


def _estimates(parts, table, base, keys, u) -> np.ndarray:
    prof = _profile_with(base, {k: base.get(k) * math.exp(x) for k, x in zip(keys, u)})
    return np.array([max(estimate(p, table, prof).door_to_door_minutes, 1e-6)
                     for p in parts])


def _solve(parts, table, base, keys, max_iter: int = 40, cancelled=None):
    """Levenberg-Marquardt on u = ln(value / baseline), with Huber weights
    (iteratively reweighted). Returns (u, J_part, residual_part)."""
    n, k = len(parts), len(keys)
    y = np.log([p.measured for p in parts])
    lo, hi = -math.log(BOUND), math.log(BOUND)
    step = 1e-4

    def part_res(u):
        return (np.log(_estimates(parts, table, base, keys, u)) - y) / SIGMA_PART

    def jac(u, r0):
        J = np.empty((n, k))
        for j in range(k):
            du = np.array(u, dtype=float)
            h = step if du[j] + step <= hi else -step
            du[j] += h
            J[:, j] = (part_res(du) - r0) / h
        return J

    def weights(r):
        a = np.abs(r)
        return np.where(a <= HUBER, 1.0, HUBER / np.maximum(a, 1e-12))

    def cost(u, r):
        a = np.abs(r)
        rho = np.where(a <= HUBER, 0.5 * a * a, HUBER * a - 0.5 * HUBER * HUBER)
        return float(np.sum(rho) + 0.5 * np.sum((u / SIGMA_PRIOR) ** 2))

    u = np.zeros(k)
    r = part_res(u)
    c = cost(u, r)
    lam = 1e-2
    J = jac(u, r) if k else np.zeros((n, 0))
    for _ in range(max_iter):
        if k == 0 or (cancelled is not None and cancelled()):
            break
        w = weights(r)
        A = J.T @ (J * w[:, None]) + np.eye(k) / SIGMA_PRIOR ** 2
        g = J.T @ (w * r) + u / SIGMA_PRIOR ** 2
        improved = False
        for _try in range(8):
            delta = np.linalg.solve(A + lam * np.diag(np.diag(A)), -g)
            u_new = np.clip(u + delta, lo, hi)
            r_new = part_res(u_new)
            c_new = cost(u_new, r_new)
            if c_new < c - 1e-12:
                improved = True
                break
            lam *= 4.0
        if not improved:
            break
        moved = float(np.max(np.abs(u_new - u)))
        u, r, c = u_new, r_new, c_new
        lam = max(lam / 3.0, 1e-7)
        if moved < 1e-5:
            break
        J = jac(u, r)
    return u, J, r


def fit(parts: list[Prepared], table, base, keys, cross_check: bool = True,
        progress=None, cancelled=None) -> Fit:
    """Fit `keys` (shop-profile settings) to the measured times of `parts`.
    `base` is the profile to start from (see fit_baseline)."""
    parts = [p for p in parts if p.measured and p.measured > 0]
    if not parts:
        raise ValueError("no parts with a measured time to fit to")
    keys = usable_knobs(base, keys)
    if progress:
        progress("Fitting…")
    u, J, r = _solve(parts, table, base, keys, cancelled=cancelled)
    n, k = len(parts), len(keys)

    # Uncertainty from the curvature at the answer, scaled by how scattered
    # the parts really are (not just the assumed 25%).
    w = np.where(np.abs(r) <= HUBER, 1.0, HUBER / np.maximum(np.abs(r), 1e-12))
    A = J.T @ (J * w[:, None]) + np.eye(k) / SIGMA_PRIOR ** 2
    scatter = float(np.sum(w * r * r) / max(n - min(k, n - 1), 1))
    cov = np.linalg.inv(A) * max(scatter, 1.0) if k else np.zeros((0, 0))
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, None)) if k else np.zeros(0)
    used = (np.mean(np.abs(J), axis=0) * SIGMA_PART) if k else np.zeros(0)

    knobs = []
    for j, key in enumerate(keys):
        knob = KNOB_BY_KEY.get(key)
        b = base.get(key)
        knobs.append(KnobResult(
            key=key, label=knob.label if knob else key, baseline=b,
            fitted=b * math.exp(u[j]), spread=math.exp(sd[j]),
            pinned=bool(sd[j] < 0.5 * SIGMA_PRIOR and used[j] > 0.01),
            used=float(used[j])))

    before = _estimates(parts, table, base, keys, np.zeros(k))
    after = _estimates(parts, table, base, keys, u)
    results = [PartResult(p.entry, p.measured, float(b), float(a))
               for p, b, a in zip(parts, before, after)]

    cv = None
    if cross_check and n >= 6 and k:
        folds = n if n <= 12 else 5
        order = np.arange(n)
        for f in range(folds):
            if cancelled is not None and cancelled():
                break
            if progress:
                progress(f"Cross-checking… {f + 1} / {folds}")
            test = order[f::folds]
            train = [parts[i] for i in order if i not in set(test)]
            u_f, _J, _r = _solve(train, table, base, keys, max_iter=25, cancelled=cancelled)
            est = _estimates([parts[i] for i in test], table, base, keys, u_f)
            for i, e in zip(test, est):
                results[i].held_out = float(e)
        held = [p.held_out / p.measured for p in results if p.held_out]
        cv = Stats.of(held) if len(held) == n else None

    stats_before = Stats.of([p.ratio_before for p in results])
    stats_after = Stats.of([p.ratio_after for p in results])
    warnings = _warnings(results, knobs, stats_before, stats_after, cv)
    return Fit(knobs=knobs, parts=results, before=stats_before, after=stats_after,
               cross_checked=cv, warnings=warnings,
               when=datetime.now().isoformat(timespec="seconds"))


def _warnings(parts, knobs, before, after, cv) -> list[str]:
    out = []
    n = len(parts)
    if n < 10:
        out.append(f"Only {n} part{'s' if n != 1 else ''}: treat this fit as provisional. "
                   f"Aim for 30 or more, spread across your usual work.")
    logs = np.log([p.ratio_after for p in parts])
    mad = float(np.median(np.abs(logs - np.median(logs)))) * 1.4826
    limit = max(math.log(1.6), 3.0 * mad)
    odd = [p for p, x in zip(parts, logs) if abs(x) > limit]
    if odd:
        names = ", ".join(f"{p.entry.part} ({p.ratio_after - 1:+.0%})" for p in odd[:6])
        out.append(f"Still far off after tuning: {names}"
                   f"{' and more' if len(odd) > 6 else ''}. Check their inputs "
                   f"(operations? stock? does the measured time include setup?) or "
                   f"leave them out.")
    loose = [k.label for k in knobs if not k.pinned]
    if loose:
        out.append("Loosely set by these parts (the prior holds them near the shipped "
                   "values; more varied parts will settle them): " + ", ".join(loose) + ".")
    if after.n and after.median_error_pct >= before.median_error_pct - 0.5:
        out.append("The fit barely improves on the shipped values. The misses are part "
                   "to part, not a shop-wide offset: look at the parts furthest off.")
    if cv is not None and cv.median_error_pct > 1.5 * max(after.median_error_pct, 1.0) + 5:
        out.append(f"On parts it has not seen the fit does worse "
                   f"({cv.median_error_pct:.0f}% vs {after.median_error_pct:.0f}%): it is "
                   f"fitting noise. Fit fewer settings or add parts.")
    return out


def save_fit(result: Fit, folder: Path | None = None) -> Path:
    """Write the fitted values into shop_tuning.csv (replacing any earlier
    fit, keeping hand-set rows) and the summary beside it."""
    folder = Path(folder) if folder is not None else tuning_dir()
    overrides, _errors = read_overrides(folder)
    keep = {k: o for k, o in overrides.items() if o.source != "fit"}
    n = len(result.parts)
    day = result.when[:10] or date.today().isoformat()
    for k in result.knobs:
        if k.key in keep:       # a hand-set row wins; it was the fit's baseline
            continue
        keep[k.key] = Override(k.key, k.fitted, "fit",
                               f"fitted {day} to {n} parts; shipped {k.baseline:.4g}")
    path = write_overrides(keep, folder)
    summary = {
        "date": day, "parts": n,
        "median_error_pct": round(result.after.median_error_pct, 1),
        "median_error_before_pct": round(result.before.median_error_pct, 1),
        "cross_checked_error_pct": (round(result.cross_checked.median_error_pct, 1)
                                    if result.cross_checked else None),
        "values": {k.key: k.fitted for k in result.knobs},
        "tool_version": __version__,
    }
    (folder / SUMMARY_FILE).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return path


def clear_fit(folder: Path | None = None) -> None:
    """Back to the shipped values: drop the fitted rows, keep hand-set ones."""
    folder = Path(folder) if folder is not None else tuning_dir()
    overrides, _errors = read_overrides(folder)
    keep = {k: o for k, o in overrides.items() if o.source != "fit"}
    path = folder / TUNING_FILE
    if keep:
        write_overrides(keep, folder)
    elif path.exists():
        path.unlink()
    summary = folder / SUMMARY_FILE
    if summary.exists():
        summary.unlink()


# ---------------------------------------------------------------------------
# the whole round trip, for the command line
# ---------------------------------------------------------------------------


def prepare_library(library: Library, table, base, entries=None):
    """([Prepared], {id(entry): problem}) for the included entries that are
    captured and complete."""
    ready, skipped = [], {}
    for e in (entries if entries is not None else library.entries):
        if not e.include:
            continue
        issues = problems(e, table)
        if issues:
            skipped[id(e)] = ", ".join(issues)
            continue
        snap = library.load_snapshot(e)
        if snap is None:
            skipped[id(e)] = "captured geometry missing"
            continue
        try:
            ready.append(prepare(e, snap, base))
        except Exception as exc:  # noqa: BLE001 -- one bad row must not stop the fit
            skipped[id(e)] = f"{type(exc).__name__}: {exc}"
    return ready, skipped


def format_report(result: Fit) -> str:
    lines = [f"Fitted to {len(result.parts)} part(s).", ""]
    lines.append(f"  Shipped values: {result.before.text()}")
    lines.append(f"  Tuned values:   {result.after.text()}")
    if result.cross_checked:
        lines.append(f"  On parts the fit did not see: {result.cross_checked.text()}")
    lines += ["", f"  {'setting':<30s} {'shipped':>10s} {'tuned':>10s}  {'± (1 sigma)':>12s}"]
    for k in result.knobs:
        note = "" if k.pinned else "   (loosely set)"
        lines.append(f"  {k.label:<30s} {k.baseline:10.4g} {k.fitted:10.4g}  "
                     f"{(k.spread - 1) * 100:11.0f}%{note}")
    lines += ["", f"  {'part':<28s} {'measured':>9s} {'shipped':>9s} {'tuned':>9s} {'error':>7s}"]
    for p in sorted(result.parts, key=lambda p: -abs(math.log(p.ratio_after))):
        lines.append(f"  {p.entry.part[:28]:<28s} {p.measured:9.2f} {p.before:9.2f} "
                     f"{p.after:9.2f} {p.ratio_after - 1:+7.0%}")
    if result.warnings:
        lines.append("")
        lines += [f"  ! {w}" for w in result.warnings]
    return "\n".join(lines)


def run_cli(args) -> int:
    """`part-estimator tune ...`"""
    from .geometry import rates

    folder = Path(args.dir) if getattr(args, "dir", None) else tuning_dir()
    lib = Library.load(folder)
    for err in lib.errors:
        print(f"warning: {LIBRARY_FILE} {err}", file=sys.stderr)

    if args.add:
        from partestimator.__main__ import _expand

        if args.material and materials.density_lb_per_in3(args.material) is None:
            print(f"warning: {args.material!r} is not a material the tool knows", file=sys.stderr)
        paths = _expand(args.add)
        print(f"Capturing {len(paths)} file(s), {args.workers or default_workers()} at a time…")
        t0 = time.perf_counter()
        added, already, errors = add_files(
            lib, paths, material=args.material or "", finish_uin=args.finish,
            operations=args.operations, profile=fit_baseline(folder), workers=args.workers,
            progress=lambda d, n, m: print(f"  [{d}/{n}] {m}"))
        lib.save()
        for name in already:
            print(f"  already in the library: {name}")
        for path, err in errors.items():
            print(f"error: {path}: {err}", file=sys.stderr)
        print(f"\n{len(added)} part(s) added to {lib.csv_path} in "
              f"{time.perf_counter() - t0:.0f} s. The STEP files were not copied.\n"
              f"Fill in measured_min (and anything else that differs), then run "
              f"`tune` again.")
        return 1 if errors else 0

    if not lib.entries:
        lib.ensure()
        print(f"The library is empty: {lib.csv_path}\n"
              f"Add parts with `tune --add part.step ...` or from the Shop tuning window.")
        return 1

    table = rates.load_rates()
    base = fit_baseline(folder)
    ready, skipped = prepare_library(lib, table, base)
    for e in lib.entries:
        if id(e) in skipped:
            print(f"  skipped {e.part}: {skipped[id(e)]}")
    if not ready:
        print("\nNothing to fit yet.")
        return 1

    keys = args.fit.split(",") if args.fit else [k.key for k in KNOBS if k.default]
    unknown = [k for k in keys if k not in base.values]
    if unknown:
        print(f"error: not shop-profile settings: {', '.join(unknown)}", file=sys.stderr)
        return 2
    result = fit(ready, table, base, keys,
                 progress=lambda m: print(m, file=sys.stderr))
    print()
    print(format_report(result))
    if args.save:
        path = save_fit(result, folder)
        print(f"\nSaved to {path}. The window uses it the next time it starts.")
    else:
        print("\nNot saved. Run again with --save to use these values.")
    return 0
