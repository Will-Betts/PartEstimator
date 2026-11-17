"""Real OpenCASCADE test parts with KNOWN dimensions (inches), written to STEP.

Used by the DXF scale test and the part-type detection test. Each builder
returns the path of a STEP file in `out_dir`. Everything is built in
millimetres (the kernel's unit) from inch dimensions, so the expected answers
below are exact.
"""
from __future__ import annotations

import math
from pathlib import Path

IN = 25.4


def _occ():
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge,
                                    BRepBuilderAPI_MakeWire)
    from OCP.BRepPrimAPI import (BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder,
                                 BRepPrimAPI_MakePrism)
    from OCP.GC import GC_MakeArcOfCircle
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Vec
    return locals()


def _write(shape, path: Path) -> Path:
    from OCP.Interface import Interface_Static
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    Interface_Static.SetCVal_s("write.step.unit", "MM")
    w = STEPControl_Writer()
    w.Transfer(shape, STEPControl_AsIs)
    if w.Write(str(path)) != 1:
        raise RuntimeError(f"could not write {path}")
    return path


def _drill(shape, x, y, d, length=400.0):
    o = _occ()
    ax = o["gp_Ax2"](o["gp_Pnt"](x, y, -length / 2), o["gp_Dir"](0, 0, 1))
    tool = o["BRepPrimAPI_MakeCylinder"](ax, d / 2, length).Shape()
    return o["BRepAlgoAPI_Cut"](shape, tool).Shape()


def section(points_in, r_inner_in, t_in, width_in):
    """A formed sheet section.

    `points_in` is the MID-surface polyline in the XZ plane (inches); every
    corner is rolled at mid radius r_inner + t/2; the result is extruded
    `width_in` along +Y and thickened t/2 each side.
    """
    import numpy as np
    from OCP.BRepLib import BRepLib
    from OCP.BRepOffsetAPI import (BRepOffsetAPI_MakeOffsetShape,
                                   BRepOffsetAPI_MakeThickSolid)
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    o = _occ()
    t = t_in * IN
    rmid = (r_inner_in + t_in / 2) * IN

    def P(p):
        return o["gp_Pnt"](float(p[0]), 0.0, float(p[1]))

    def line(a, b):
        return o["BRepBuilderAPI_MakeEdge"](P(a), P(b)).Edge()

    def arc(a, m, b):
        return o["BRepBuilderAPI_MakeEdge"](
            o["GC_MakeArcOfCircle"](P(a), P(m), P(b)).Value()).Edge()

    pts = [np.array(p, float) * IN for p in points_in]
    wire = o["BRepBuilderAPI_MakeWire"]()
    cur = pts[0]
    for i in range(1, len(pts) - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        d1 = (b - a) / np.linalg.norm(b - a)
        d2 = (c - b) / np.linalg.norm(c - b)
        ang = math.acos(max(-1.0, min(1.0, float(d1 @ d2))))
        k = rmid * math.tan(ang / 2)
        s, e = b - d1 * k, b + d2 * k
        bis = (d2 - d1) / np.linalg.norm(d2 - d1)
        ctr = b + bis * (rmid / math.cos(ang / 2))
        m = ctr + (b - ctr) / np.linalg.norm(b - ctr) * rmid
        wire.Add(line(cur, s))
        wire.Add(arc(s, m, e))
        cur = e
    wire.Add(line(cur, pts[-1]))
    sheet = o["BRepPrimAPI_MakePrism"](wire.Wire(), o["gp_Vec"](0, width_in * IN, 0)).Shape()
    mid = BRepOffsetAPI_MakeOffsetShape()
    mid.PerformBySimple(sheet, -t / 2)
    thick = BRepOffsetAPI_MakeThickSolid()
    thick.MakeThickSolidBySimple(mid.Shape(), t)
    solid = TopoDS.Solid_s(TopExp_Explorer(thick.Shape(), TopAbs_SOLID).Current())
    BRepLib.OrientClosedSolid_s(solid)
    return solid


# --- the parts ---------------------------------------------------------------
# Sheet: 0.060" thick, 0.060" inside radius throughout.
T, R = 0.060, 0.060


def l_bracket(out_dir: Path) -> Path:
    """Legs 3.000 (base, along X) and 2.000 (up) OUTSIDE, 4.000 wide.
    One 0.250 hole in the base, 1.000 from the outside of the bend line's
    far end... specifically centred at mid-surface x = 2.000, y = 2.000."""
    # Mid-surface polyline: outside dims less t/2.
    h = T / 2
    pts = [(3.0, h), (h, h), (h, 2.0)]
    s = section(pts, R, T, 4.0)
    s = _drill(s, 2.0 * IN, 2.0 * IN, 0.250 * IN)
    return _write(s, out_dir / "l_bracket.step")


def z_section(out_dir: Path) -> Path:
    """One bend up, one bend down. 3.000 wide."""
    s = section([(0, 0), (2.0, 0), (2.0, 1.5), (4.0, 1.5)], R, T, 3.0)
    return _write(s, out_dir / "z_section.step")


def flat_blank(out_dir: Path, thickness_in: float = 0.090, name="flat_blank") -> Path:
    """6.000 x 4.000 blank with a 0.375 and a 0.500 hole."""
    o = _occ()
    s = o["BRepPrimAPI_MakeBox"](6 * IN, 4 * IN, thickness_in * IN).Shape()
    s = _drill(s, 1 * IN, 1 * IN, 0.375 * IN)
    s = _drill(s, 5 * IN, 3 * IN, 0.500 * IN)
    return _write(s, out_dir / f"{name}.step")


def machined_plate(out_dir: Path) -> Path:
    """0.500 plate, 6 x 4, with a 0.250-deep pocket and two holes."""
    o = _occ()
    s = o["BRepPrimAPI_MakeBox"](6 * IN, 4 * IN, 0.5 * IN).Shape()
    pocket = o["BRepPrimAPI_MakeBox"](o["gp_Pnt"](1 * IN, 1 * IN, 0.25 * IN),
                                       3 * IN, 2 * IN, 1 * IN).Shape()
    s = o["BRepAlgoAPI_Cut"](s, pocket).Shape()
    s = _drill(s, 5 * IN, 1 * IN, 0.25 * IN)
    s = _drill(s, 5 * IN, 3 * IN, 0.25 * IN)
    return _write(s, out_dir / "machined_plate.step")


def thin_pocketed_plate(out_dir: Path) -> Path:
    """0.125 plate with a 0.060-deep pocket: thin, but NOT uniform thickness."""
    o = _occ()
    s = o["BRepPrimAPI_MakeBox"](4 * IN, 3 * IN, 0.125 * IN).Shape()
    pocket = o["BRepPrimAPI_MakeBox"](o["gp_Pnt"](1 * IN, 1 * IN, 0.065 * IN),
                                       2 * IN, 1 * IN, 1 * IN).Shape()
    return _write(o["BRepAlgoAPI_Cut"](s, pocket).Shape(), out_dir / "thin_pocketed.step")


def block(out_dir: Path) -> Path:
    o = _occ()
    return _write(o["BRepPrimAPI_MakeBox"](3 * IN, 2 * IN, 1.5 * IN).Shape(),
                  out_dir / "block.step")


def turned_part(out_dir: Path) -> Path:
    """A plain 1.000 dia x 3.000 cylinder."""
    o = _occ()
    ax = o["gp_Ax2"](o["gp_Pnt"](0, 0, 0), o["gp_Dir"](0, 0, 1))
    return _write(o["BRepPrimAPI_MakeCylinder"](ax, 0.5 * IN, 3 * IN).Shape(),
                  out_dir / "turned.step")


def countersunk_l_bracket(out_dir: Path) -> Path:
    """The L bracket with a 100-degree countersink on its base hole."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCone

    o = _occ()
    h = T / 2
    s = section([(3.0, h), (h, h), (h, 2.0)], R, T, 4.0)
    s = _drill(s, 2.0 * IN, 2.0 * IN, 0.190 * IN)
    # Cone opening upward from the top skin (z = T): 0.380 dia at the surface.
    r_top = 0.190 * IN
    depth = (r_top - 0.095 * IN) / math.tan(math.radians(50))
    ax = o["gp_Ax2"](o["gp_Pnt"](2.0 * IN, 2.0 * IN, T * IN - depth), o["gp_Dir"](0, 0, 1))
    cone = BRepPrimAPI_MakeCone(ax, 0.095 * IN, r_top + 0.2 * IN,
                                depth + 0.2 * IN / math.tan(math.radians(50))).Shape()
    s = o["BRepAlgoAPI_Cut"](s, cone).Shape()
    return _write(s, out_dir / "csk_l_bracket.step")


def plate_with_spotface(out_dir: Path) -> Path:
    """0.125 plate, through holes, and one 0.75 dia x 0.030 deep spot face."""
    o = _occ()
    s = o["BRepPrimAPI_MakeBox"](4 * IN, 3 * IN, 0.125 * IN).Shape()
    s = _drill(s, 1 * IN, 1 * IN, 0.25 * IN)
    ax = o["gp_Ax2"](o["gp_Pnt"](3 * IN, 2 * IN, 0.095 * IN), o["gp_Dir"](0, 0, 1))
    spot = o["BRepPrimAPI_MakeCylinder"](ax, 0.375 * IN, 1 * IN).Shape()
    s = o["BRepAlgoAPI_Cut"](s, spot).Shape()
    return _write(s, out_dir / "spotface_plate.step")


def stepped_shaft(out_dir: Path, flat=True, cross_hole=True) -> Path:
    """Turned part along Z: 2.000 dia x 1.500, then 1.250 dia x 2.000;
    0.500 axial through bore; optional 0.250 cross hole and a 0.100 deep,
    1.000 long flat on the small diameter (both live-tool work)."""
    o = _occ()
    ax = o["gp_Ax2"](o["gp_Pnt"](0, 0, 0), o["gp_Dir"](0, 0, 1))
    big = o["BRepPrimAPI_MakeCylinder"](ax, 1.0 * IN, 1.5 * IN).Shape()
    ax2 = o["gp_Ax2"](o["gp_Pnt"](0, 0, 1.5 * IN), o["gp_Dir"](0, 0, 1))
    small = o["BRepPrimAPI_MakeCylinder"](ax2, 0.625 * IN, 2.0 * IN).Shape()
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    s = BRepAlgoAPI_Fuse(big, small).Shape()
    s = _drill(s, 0.0, 0.0, 0.5 * IN)
    if cross_hole:
        axx = o["gp_Ax2"](o["gp_Pnt"](-2 * IN, 0, 0.75 * IN), o["gp_Dir"](1, 0, 0))
        s = o["BRepAlgoAPI_Cut"](s, o["BRepPrimAPI_MakeCylinder"](axx, 0.125 * IN, 4 * IN).Shape()).Shape()
    if flat:
        cutter = o["BRepPrimAPI_MakeBox"](o["gp_Pnt"](0.525 * IN, -1 * IN, 2.25 * IN),
                                          1 * IN, 2 * IN, 1.0 * IN).Shape()
        s = o["BRepAlgoAPI_Cut"](s, cutter).Shape()
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    u = ShapeUpgrade_UnifySameDomain(s, True, True, False)
    u.Build()
    name = "stepped_shaft" + ("_flat" if flat else "") + ("_xhole" if cross_hole else "")
    return _write(u.Shape(), out_dir / f"{name}.step")
