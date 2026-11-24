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


def edge_drilled_plate(out_dir: Path) -> Path:
    """6 x 4 x 1.000 plate with blind 0.250 holes, 0.75 deep, drilled into its
    edges at mid-thickness: every 0.75 along the long edges (x = 0.75..5.25)
    and at y = 1, 2, 3 on the short ones."""
    o = _occ()
    s = o["BRepPrimAPI_MakeBox"](6 * IN, 4 * IN, 1 * IN).Shape()
    r, depth, z = 0.125 * IN, 0.75 * IN, 0.5 * IN

    def hole(p, d):
        ax = o["gp_Ax2"](o["gp_Pnt"](*p), o["gp_Dir"](*d))
        return o["BRepPrimAPI_MakeCylinder"](ax, r, depth + 0.1 * IN).Shape()

    tools = []
    for k in range(1, 8):
        x = 0.75 * k * IN
        tools.append(hole((x, -0.1 * IN, z), (0, 1, 0)))
        tools.append(hole((x, 4.1 * IN, z), (0, -1, 0)))
    for y in (1, 2, 3):
        tools.append(hole((-0.1 * IN, y * IN, z), (1, 0, 0)))
        tools.append(hole((6.1 * IN, y * IN, z), (-1, 0, 0)))
    for t in tools:
        s = o["BRepAlgoAPI_Cut"](s, t).Shape()
    return _write(s, out_dir / "edge_drilled_plate.step")


def fixture_torture_plate(out_dir: Path) -> Path:
    """A plate that tests every way a tab can go wrong. 7 x 4 x 1.250:

    - one end rounded (radius 2.0 about x = 5, y = 2): curved edge;
    - the top outer edge filleted 0.400: the wall reaches the outline only
      from the bottom to 0.850 up;
    - a groove 0.150 deep along the y = 0 edge, x 0.5..4.5, z 0.5..0.7;
    - a pocket 0.6 deep opening into the y = 4 edge, x 1.0..2.5, z 0.4..0.8;
    - blind 0.250 holes, 0.75 deep, into the x = 0 end at y = 1 and 3, and
      radially into the rounded end along +x, all at z = 0.6;
    - a top pocket 2.0 x 1.5 x 0.4 deep at (1.0, 1.25) and a 1.0 dia through
      window at (5, 2).
    """
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.BRep import BRep_Tool

    o = _occ()
    H = 1.25 * IN
    s = o["BRepPrimAPI_MakeBox"](5 * IN, 4 * IN, H).Shape()
    end = o["BRepPrimAPI_MakeCylinder"](
        o["gp_Ax2"](o["gp_Pnt"](5 * IN, 2 * IN, 0), o["gp_Dir"](0, 0, 1)), 2 * IN, H).Shape()
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain

    u = ShapeUpgrade_UnifySameDomain(BRepAlgoAPI_Fuse(s, end).Shape(), True, True, False)
    u.Build()
    s = u.Shape()

    # Fillet every edge lying wholly on the top face's outer boundary.
    fil = BRepFilletAPI_MakeFillet(s)
    ex = TopExp_Explorer(s, TopAbs_EDGE)
    seen = []
    while ex.More():
        e = TopoDS.Edge_s(ex.Current())
        zs = []
        vx = TopExp_Explorer(e, TopAbs_VERTEX)
        while vx.More():
            zs.append(BRep_Tool.Pnt_s(TopoDS.Vertex_s(vx.Current())).Z())
            vx.Next()
        if zs and all(abs(z - H) < 1e-6 for z in zs) and not any(e.IsSame(k) for k in seen):
            seen.append(e)
            fil.Add(0.4 * IN, e)
        ex.Next()
    s = fil.Shape()

    def box(x0, y0, z0, x1, y1, z1):
        return o["BRepPrimAPI_MakeBox"](o["gp_Pnt"](x0 * IN, y0 * IN, z0 * IN),
                                        o["gp_Pnt"](x1 * IN, y1 * IN, z1 * IN)).Shape()

    def hole(p, d, depth=0.75):
        ax = o["gp_Ax2"](o["gp_Pnt"](*(c * IN for c in p)), o["gp_Dir"](*d))
        return o["BRepPrimAPI_MakeCylinder"](ax, 0.125 * IN, (depth + 0.1) * IN).Shape()

    tools = [
        box(0.5, -0.1, 0.5, 4.5, 0.15, 0.7),            # groove
        box(1.0, 3.4, 0.4, 2.5, 4.1, 0.8),               # side pocket
        hole((-0.1, 1.0, 0.6), (1, 0, 0)),
        hole((-0.1, 3.0, 0.6), (1, 0, 0)),
        hole((7.1, 2.0, 0.6), (-1, 0, 0)),
        box(1.0, 1.25, 0.85, 3.0, 2.75, 2.0),           # top pocket
    ]
    window = o["BRepPrimAPI_MakeCylinder"](
        o["gp_Ax2"](o["gp_Pnt"](5 * IN, 2 * IN, -IN), o["gp_Dir"](0, 0, 1)), 0.5 * IN, 4 * IN).Shape()
    for t in tools + [window]:
        s = o["BRepAlgoAPI_Cut"](s, t).Shape()
    return _write(s, out_dir / "fixture_torture_plate.step")



def hole_zoo(out_dir: Path) -> Path:
    """A 6 x 4 x 1.000 plate with one of every kind of hole, and a boss.

    Along y = 1, left to right (all 0.250 bores unless noted):
      x = 0.75  through
      x = 1.75  blind 0.6 deep, 118 deg drill point
      x = 2.75  through, counterbored 0.438 x 0.25 deep from the top
      x = 3.75  through, 82 deg countersink to 0.500 at the top
      x = 4.75  through, counterbored 0.438 x 0.2 AND countersunk below it
    Along y = 3:
      x = 1.0   through 0.375 with an internal groove 0.5 dia, z 0.45..0.55
      x = 2.5   blind 0.5 deep drilled up from the bottom, flat bottomed
    Boss: 1.0 dia x 0.5 tall on the top at (4.5, 3.0), its base filleted
    0.125 (a concave torus) and its top edge rounded 0.1 (a convex one).
    """
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCone
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GeomAbs import GeomAbs_Circle

    o = _occ()
    T = 1.0 * IN

    def cyl(x, y, z0, r, h, down=False):
        d = o["gp_Dir"](0, 0, -1 if down else 1)
        return o["BRepPrimAPI_MakeCylinder"](
            o["gp_Ax2"](o["gp_Pnt"](x * IN, y * IN, z0 * IN), d), r * IN, h * IN).Shape()

    def cone(x, y, z0, r1, r2, h, down=False):
        d = o["gp_Dir"](0, 0, -1 if down else 1)
        return BRepPrimAPI_MakeCone(
            o["gp_Ax2"](o["gp_Pnt"](x * IN, y * IN, z0 * IN), d), r1 * IN, r2 * IN, h * IN).Shape()

    s = o["BRepPrimAPI_MakeBox"](6 * IN, 4 * IN, T).Shape()
    boss = cyl(4.5, 3.0, 1.0, 0.5, 0.5)
    s = BRepAlgoAPI_Fuse(s, boss).Shape()

    # Fillet the boss: its base circle (z = 1.0) and its top circle (z = 1.5).
    fil = BRepFilletAPI_MakeFillet(s)
    ex = TopExp_Explorer(s, TopAbs_EDGE)
    seen = []
    while ex.More():
        e = TopoDS.Edge_s(ex.Current())
        c = BRepAdaptor_Curve(e)
        if c.GetType() == GeomAbs_Circle and abs(c.Circle().Radius() - 0.5 * IN) < 1e-6 \
                and not any(e.IsSame(k) for k in seen):
            z = c.Circle().Location().Z()
            seen.append(e)
            fil.Add((0.125 if abs(z - T) < 1e-6 else 0.1) * IN, e)
        ex.Next()
    s = fil.Shape()

    r = 0.125
    tip = r / 0.5543                     # 118 deg point: h = r / tan(59 deg)
    tools = [
        cyl(0.75, 1, -0.1, r, 1.2),
        cyl(1.75, 1, 1.1, r, 0.7, down=True),
        cone(1.75, 1, 0.4, r, 0.0, r * 0.6, down=True),
        cyl(2.75, 1, -0.1, r, 1.2), cyl(2.75, 1, 0.75, 0.219, 0.35),
        cyl(3.75, 1, -0.1, r, 1.2), cone(3.75, 1, 1.0 - 0.1306, r, 0.25, 0.1306 + 0.01),
        cyl(4.75, 1, -0.1, r, 1.2), cyl(4.75, 1, 0.8, 0.219, 0.3),
        cone(4.75, 1, 0.8 - 0.0819, r, 0.2, 0.0819),
        cyl(1.0, 3, -0.1, 0.1875, 1.2), cyl(1.0, 3, 0.45, 0.25, 0.1),
        cyl(2.5, 3, -0.1, r, 0.6),
    ]
    del tip
    for t in tools:
        s = o["BRepAlgoAPI_Cut"](s, t).Shape()
    return _write(s, out_dir / "hole_zoo.step")


def many_holes_plate(out_dir: Path, n: int = 17) -> Path:
    """A plate with an n x n grid of small through holes: plenty of faces."""
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound

    o = _occ()
    s = o["BRepPrimAPI_MakeBox"](n * 0.3 * IN, n * 0.3 * IN, 0.25 * IN).Shape()
    comp, b = TopoDS_Compound(), BRep_Builder()
    b.MakeCompound(comp)
    for i in range(n):
        for j in range(n):
            b.Add(comp, o["BRepPrimAPI_MakeCylinder"](
                o["gp_Ax2"](o["gp_Pnt"]((i + 0.5) * 0.3 * IN, (j + 0.5) * 0.3 * IN, -1),
                            o["gp_Dir"](0, 0, 1)), 0.05 * IN, 0.25 * IN + 2).Shape())
    s = o["BRepAlgoAPI_Cut"](s, comp).Shape()
    return _write(s, out_dir / "many_holes.step")



def sheet_feature_bracket(out_dir: Path) -> Path:
    """An L of 0.125 sheet, 0.125 inside radius, base 3.0 along X and 2.0 up,
    4.0 wide -- with one of every sheet feature on the base flange:

      (2.0, 0.8)  countersink 82 deg, 0.200 hole, 0.400 at the top
      (2.0, 1.8)  counterbore 0.350 x 0.060 deep on a 0.200 hole
      (1.6..2.4, 3.0)  slot 0.200 wide
      (0.45, 2.0) 0.150 hole inside the bend's deformation zone
      the outer corner at (3, 0) rounded R0.25 (a corner radius)
      the top edge along x = 3 chamfered 0.040
    """
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCone
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    o = _occ()
    t = 0.125
    h = t / 2
    s = section([(3.0, h), (h, h), (h, 2.0)], 0.125, t, 4.0)

    # Cut, not filleted: the kernel's fillet and chamfer tools leave this
    # solid invalid, and it then reads back from STEP as an open shell.
    # Corner R0.25: the corner block less a cylinder.
    corner = o["BRepAlgoAPI_Cut"](
        o["BRepPrimAPI_MakeBox"](o["gp_Pnt"](2.75 * IN, -0.1 * IN, -IN), 0.35 * IN,
                                 0.35 * IN, 2 * IN).Shape(),
        o["BRepPrimAPI_MakeCylinder"](
            o["gp_Ax2"](o["gp_Pnt"](2.75 * IN, 0.25 * IN, -IN), o["gp_Dir"](0, 0, 1)),
            0.25 * IN, 2 * IN).Shape()).Shape()
    # Chamfer 0.040 x 45 deg on the top edge along x = 3: a triangular prism.
    tri = BRepBuilderAPI_MakePolygon()
    for x, z in ((2.96, t), (3.1, t), (3.1, t - 0.14)):
        tri.Add(o["gp_Pnt"](x * IN, -IN, z * IN))
    tri.Close()
    wedge = o["BRepPrimAPI_MakePrism"](BRepBuilderAPI_MakeFace(tri.Wire()).Face(),
                                       o["gp_Vec"](0, 6 * IN, 0)).Shape()
    s = o["BRepAlgoAPI_Cut"](s, corner).Shape()
    s = o["BRepAlgoAPI_Cut"](s, wedge).Shape()

    def down(x, y, r, depth, z0=t + 0.1):
        return o["BRepPrimAPI_MakeCylinder"](
            o["gp_Ax2"](o["gp_Pnt"](x * IN, y * IN, z0 * IN), o["gp_Dir"](0, 0, -1)),
            r * IN, depth * IN).Shape()

    csk_h = 0.1 / 1.1504                     # tan(41 deg)
    tools = [
        down(2.0, 0.8, 0.1, 1.0),
        BRepPrimAPI_MakeCone(o["gp_Ax2"](o["gp_Pnt"](2.0 * IN, 0.8 * IN, (t - csk_h) * IN),
                                         o["gp_Dir"](0, 0, 1)),
                             0.1 * IN, 0.2 * IN, csk_h * IN).Shape(),
        o["BRepPrimAPI_MakeCylinder"](
            o["gp_Ax2"](o["gp_Pnt"](2.0 * IN, 0.8 * IN, t * IN), o["gp_Dir"](0, 0, 1)),
            0.2 * IN, 0.1 * IN).Shape(),
        down(2.0, 1.8, 0.1, 1.0),
        down(2.0, 1.8, 0.175, 0.16),
        down(1.6, 3.0, 0.1, 1.0), down(2.4, 3.0, 0.1, 1.0),
        o["BRepPrimAPI_MakeBox"](o["gp_Pnt"](1.6 * IN, 2.9 * IN, -IN),
                                 0.8 * IN, 0.2 * IN, 2 * IN).Shape(),
        down(0.45, 2.0, 0.075, 1.0),
    ]
    for tool in tools:
        s = o["BRepAlgoAPI_Cut"](s, tool).Shape()
    # The booleans hand back a compound holding the solid; written as it is,
    # it reads back as an open shell. Write the solid itself.
    from OCP.TopAbs import TopAbs_SOLID

    s = TopoDS.Solid_s(TopExp_Explorer(s, TopAbs_SOLID).Current())
    return _write(s, out_dir / "sheet_feature_bracket.step")
