"""DXF export of the flat pattern -- ALWAYS INCHES, geometry only.

What is in the file, and nothing else:

    OUTLINE   outer profile of the blank        (continuous)
    HOLES     every internal cutout             (continuous)
    BEND      every bend centreline, one layer:
                  bend UP   -> linetype DASHED
                  bend DOWN -> linetype PHANTOM

No text, no dimensions, no title block, no blocks. The bend direction is
carried by the linetype alone, so a nesting package that strips text still
gets it.

Why this was rewritten (v13 -> unified tool)
--------------------------------------------
The v13 file was reported opening far too big in SolidWorks and eDrawings.
Its coordinates were in fact right -- an L-bracket's outline read back at
exactly the developed blank size -- but the file around them was not a clean
inch drawing:

  * `ezdxf.new(setup=True)` with ezdxf's default `units=6` (METRES) built the
    whole template -- text styles, dimension styles, linetype patterns, the
    paper-space layout -- for a metric drawing, and only afterwards was
    $INSUNITS overwritten. The file declared one unit in its header and was
    furnished for another.
  * $EXTMIN / $EXTMAX were left at the +/-1e20 placeholders, so a viewer that
    trusts the header extents had nothing sane to fit to.
  * A TEXT note sat on every bend line.

Every one of those is gone. The document is now created as inches from the
start (`units=1`), with nothing set up but the two linetypes it uses, defined
here with standard inch patterns; the model-space block record is stamped
inches too; and the real extents are written into the header and the active
viewport. `tests/test_dxf_scale.py` reads each exported file back and checks
its extents, hole sizes and bend positions against the developed blank.

`ezdxf` is imported lazily so the rest of the package works without it.
"""

from __future__ import annotations

import math
from pathlib import Path

from .model import Part
from .units import INCH, MM_PER_INCH

LAYER_COLOURS = {
    "OUTLINE": 7,   # white/black
    "HOLES": 5,     # blue
    "BEND": 1,      # red
}

# Standard acad.lin inch patterns. Positive = dash, negative = gap, 0 = dot.
LINETYPES = {
    "DASHED": ("Dashed __ __ __ __", [0.75, 0.5, -0.25]),
    "PHANTOM": (
        "Phantom ______  __  __  ______",
        [2.0, 1.25, -0.25, 0.25, -0.25],
    ),
}
BEND_UP_LINETYPE = "DASHED"
BEND_DOWN_LINETYPE = "PHANTOM"

DXF_INSUNITS_INCHES = 1


def _confine_ezdxf(ezdxf) -> None:
    """Stop ezdxf writing a font cache into the user profile.

    ezdxf can build a cache of system fonts, by default in a per-user folder --
    outside this tool's directory. This file contains no text, so it should
    never happen, but the cache directory is pointed at the tool's own scratch
    folder anyway so "delete the folder to uninstall" stays true.
    """
    from .diag import LOGGER, scratch_dir

    try:
        options = ezdxf.options
    except AttributeError:  # pragma: no cover -- very old ezdxf
        return

    target = str(scratch_dir())
    for attr in ("font_cache_directory", "test_files"):
        try:
            if hasattr(options, attr):
                setattr(options, attr, target)
        except Exception:  # noqa: BLE001 -- never let this stop an export
            LOGGER.debug("could not redirect ezdxf option %s", attr)
    try:
        if hasattr(options, "auto_load_fonts"):
            options.auto_load_fonts = False
    except Exception:  # noqa: BLE001
        pass


def _require_ezdxf():
    try:
        import ezdxf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "DXF export needs ezdxf:\n\n    pip install ezdxf\n"
        ) from exc
    _confine_ezdxf(ezdxf)
    return ezdxf


def new_inch_document(ezdxf):
    """An empty R2010 drawing that is inches through and through."""
    doc = ezdxf.new("R2010", setup=False, units=DXF_INSUNITS_INCHES)
    hdr = doc.header
    hdr["$INSUNITS"] = DXF_INSUNITS_INCHES
    hdr["$MEASUREMENT"] = 0          # imperial
    hdr["$LUNITS"] = 2               # decimal
    hdr["$LUPREC"] = 4
    hdr["$LTSCALE"] = 1.0
    hdr["$CELTSCALE"] = 1.0
    hdr["$PSLTSCALE"] = 1
    hdr["$TILEMODE"] = 1             # open in model space
    try:
        doc.modelspace().block_record.dxf.units = DXF_INSUNITS_INCHES
    except Exception:  # noqa: BLE001 -- attribute exists from R2000 on
        pass

    for name, (desc, pattern) in LINETYPES.items():
        if name not in doc.linetypes:
            doc.linetypes.add(name, pattern=pattern, description=desc)
    for name, colour in LAYER_COLOURS.items():
        if name not in doc.layers:
            doc.layers.add(name, color=colour)
    return doc


def export_dxf(part: Part, path: str | Path, exact: bool = True, unit=None) -> Path:
    """Write the flat pattern to DXF, in inches.

    `unit` is accepted for compatibility with older callers and IGNORED: the
    DXF is always inches. Geometry is millimetres internally and is divided by
    25.4 exactly once, here, on the way out.
    """
    del unit  # always inches -- see module docstring
    ezdxf = _require_ezdxf()
    if part.flat is None:
        raise ValueError("part has no flat pattern; run the unfold first")

    scale = INCH.per_mm  # 1 / 25.4
    doc = new_inch_document(ezdxf)
    msp = doc.modelspace()

    wrote_exact = False
    if exact and part.flat.faces:
        try:
            # Write into a scratch list first so a failure half-way through
            # cannot leave a partial exact profile AND the polyline fallback
            # on top of each other in the same file.
            ops: list = []
            for profile in part.flat.faces:
                _collect_face_exact(ops, profile, scale)
            for op in ops:
                op(msp)
            wrote_exact = True
        except Exception:  # noqa: BLE001 -- fall back rather than fail the run
            wrote_exact = False

    if not wrote_exact:
        _write_polylines(msp, part, scale)

    _write_bend_lines(msp, part, scale)
    _set_extents(doc, msp)

    path = Path(path)
    doc.saveas(path)
    return path


# ---------------------------------------------------------------------------


def _write_polylines(msp, part: Part, scale: float) -> None:
    for layer, polys in (("OUTLINE", part.flat.outline), ("HOLES", part.flat.holes)):
        for poly in polys:
            if len(poly) >= 2:
                msp.add_lwpolyline(
                    [(float(x) * scale, float(y) * scale) for x, y in poly[:, :2]],
                    close=True, dxfattribs={"layer": layer},
                )


def _collect_face_exact(ops: list, profile, scale: float) -> None:
    """Queue true LINE / ARC / CIRCLE entities straight off the kernel face."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepTools import BRepTools, BRepTools_WireExplorer
    from OCP.GeomAbs import GeomAbs_Circle, GeomAbs_Line
    from OCP.TopAbs import TopAbs_WIRE
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedMapOfShape

    from .occ.topo import discretise_edge, edge_has_curve

    outer = BRepTools.OuterWire_s(profile)
    wires = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(profile, TopAbs_WIRE, wires)

    for i in range(1, wires.Extent() + 1):
        w = TopoDS.Wire_s(wires.FindKey(i))
        layer = "OUTLINE" if w.IsSame(outer) else "HOLES"
        attrs = {"layer": layer}

        exp = BRepTools_WireExplorer(w, profile)
        while exp.More():
            edge = exp.Current()
            # v13 did `continue` here WITHOUT advancing the explorer, which
            # loops forever on an edge with no 3D curve.
            exp.Next()
            if not edge_has_curve(edge):
                continue
            curve = BRepAdaptor_Curve(edge)
            t = curve.GetType()
            p0 = curve.Value(curve.FirstParameter())
            p1 = curve.Value(curve.LastParameter())

            if t == GeomAbs_Line:
                a = (p0.X() * scale, p0.Y() * scale)
                b = (p1.X() * scale, p1.Y() * scale)
                ops.append(lambda m, a=a, b=b, d=attrs: m.add_line(a, b, dxfattribs=d))
            elif t == GeomAbs_Circle:
                circ = curve.Circle()
                c = circ.Location()
                r = circ.Radius()
                centre = (c.X() * scale, c.Y() * scale)
                if math.hypot(p1.X() - p0.X(), p1.Y() - p0.Y()) < 1e-9:
                    ops.append(lambda m, c=centre, r=r * scale, d=attrs:
                               m.add_circle(c, r, dxfattribs=d))
                else:
                    a0 = math.degrees(math.atan2(p0.Y() - c.Y(), p0.X() - c.X()))
                    a1 = math.degrees(math.atan2(p1.Y() - c.Y(), p1.X() - c.X()))
                    # DXF arcs run counter-clockwise; a circle whose axis points
                    # down runs the other way, so swap the end angles.
                    if circ.Axis().Direction().Z() < 0:
                        a0, a1 = a1, a0
                    ops.append(lambda m, c=centre, r=r * scale, a0=a0, a1=a1, d=attrs:
                               m.add_arc(c, r, a0, a1, dxfattribs=d))
            else:
                pts = [(float(p[0]) * scale, float(p[1]) * scale)
                       for p in discretise_edge(edge)]
                if len(pts) >= 2:
                    ops.append(lambda m, pts=pts, d=attrs:
                               m.add_lwpolyline(pts, dxfattribs=d))


def _write_bend_lines(msp, part: Part, scale: float) -> None:
    for a, b, bid in part.flat.bend_lines:
        bend = part.bends.get(bid)
        up = bend is None or bend.direction >= 0
        msp.add_line(
            (float(a[0]) * scale, float(a[1]) * scale),
            (float(b[0]) * scale, float(b[1]) * scale),
            dxfattribs={
                "layer": "BEND",
                "linetype": BEND_UP_LINETYPE if up else BEND_DOWN_LINETYPE,
            },
        )


def _set_extents(doc, msp) -> None:
    """Write the real drawing extents into the header and the active view."""
    try:
        from ezdxf import bbox

        box = bbox.extents(msp)
        if not box.has_data:
            return
        lo = (box.extmin.x, box.extmin.y, 0.0)
        hi = (box.extmax.x, box.extmax.y, 0.0)
        # ezdxf rewrites the header extents from the model-space LAYOUT on
        # save, so they must be set there; setting $EXTMIN alone is undone.
        msp.dxf.extmin = lo
        msp.dxf.extmax = hi
        msp.dxf.limmin = lo[:2]
        msp.dxf.limmax = hi[:2]
        doc.header["$EXTMIN"] = lo
        doc.header["$EXTMAX"] = hi
        from ezdxf import zoom

        zoom.extents(msp, factor=1.1)
    except Exception:  # noqa: BLE001 -- extents are a courtesy, not geometry
        pass


def flat_size_inches(part: Part) -> tuple[float, float]:
    """The developed blank (length, width) in inches -- for tests and checks."""
    w, h = part.flat.bbox
    return w / MM_PER_INCH, h / MM_PER_INCH
