"""The tabbed-fixturing bodies as solids, written to one STEP file.

From a FixturePlan (geometry/fixture.py) and the part's own solid:

  Stock            the blank, as bought
  After side 1     vertical: side 1 faced, channel cut to the middle, the
                   side-1 half of the part machined
  After op 1       flat: top faced, channel cut to the part's bottom, the
                   part machined from the top, floor still under it
  After side 2 /   the part held in the frame by its tabs, both faces
  After op 2       faced: what comes off the machine
  Part             the finished part

All four are overlaid in the plan frame (fixture.py), in millimetres, each
named and coloured, so a CAM system can use them as the stock and the
in-process stock of each setup. Runs in a worker process: the part's solid
cannot cross processes, so it is read again from its STEP file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import fixture as fx_mod

MM = fx_mod.MM
FUZZY_MM = 1e-4

COLOURS = {
    "Stock": (0.80, 0.80, 0.82),
    "After side 1": (0.55, 0.70, 0.90),
    "After op 1": (0.55, 0.70, 0.90),
    "After side 2": (0.95, 0.70, 0.35),
    "After op 2": (0.95, 0.70, 0.35),
    "Part": (0.30, 0.75, 0.40),
}


def _occ():
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
    from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon,
                                    BRepBuilderAPI_Transform)
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakePrism
    from OCP.gp import gp_Pnt, gp_Trsf, gp_Vec
    from OCP.TopTools import TopTools_ListOfShape

    return locals()


def to_plan(shape, plan):
    """`shape` moved from the file's coordinates into the plan frame."""
    o = _occ()
    r = np.asarray(plan.frame, dtype=float)
    t = -r @ np.asarray(plan.origin_mm, dtype=float)
    trsf = o["gp_Trsf"]()
    trsf.SetValues(r[0, 0], r[0, 1], r[0, 2], t[0],
                   r[1, 0], r[1, 1], r[1, 2], t[1],
                   r[2, 0], r[2, 1], r[2, 2], t[2])
    return o["BRepBuilderAPI_Transform"](shape, trsf, True).Shape()


def box(x0, y0, z0, x1, y1, z1):
    """A box between two corners given in inches."""
    o = _occ()
    return o["BRepPrimAPI_MakeBox"](o["gp_Pnt"](x0 * MM, y0 * MM, z0 * MM),
                                    o["gp_Pnt"](x1 * MM, y1 * MM, z1 * MM)).Shape()


def prism(loop_xy, z0, z1):
    """The closed loop (inches, plan XY) at z0, run up to z1."""
    o = _occ()
    poly = o["BRepBuilderAPI_MakePolygon"]()
    for x, y in loop_xy:
        poly.Add(o["gp_Pnt"](float(x) * MM, float(y) * MM, z0 * MM))
    poly.Close()
    face = o["BRepBuilderAPI_MakeFace"](poly.Wire(), True).Face()
    return o["BRepPrimAPI_MakePrism"](face, o["gp_Vec"](0.0, 0.0, (z1 - z0) * MM)).Shape()


def _list(shapes):
    o = _occ()
    lst = o["TopTools_ListOfShape"]()
    for s in shapes:
        lst.Append(s)
    return lst


def _boolean(op, a, tools):
    o = _occ()
    alg = o[op]()
    alg.SetArguments(_list([a]))
    alg.SetTools(_list(tools))
    alg.SetFuzzyValue(FUZZY_MM)
    alg.SetRunParallel(True)
    alg.Build()
    if not alg.IsDone():
        raise RuntimeError(f"{op} failed")
    return _clean(alg.Shape())


def _clean(shape):
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain

    u = ShapeUpgrade_UnifySameDomain(shape, True, True, False)
    u.Build()
    return u.Shape()


def fuse(a, *others):
    return _boolean("BRepAlgoAPI_Fuse", a, list(others)) if others else a


def cut(a, *others):
    return _boolean("BRepAlgoAPI_Cut", a, list(others)) if others else a


def common(a, b):
    return _boolean("BRepAlgoAPI_Common", a, [b])


def solid_count(shape) -> int:
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    n, ex = 0, TopExp_Explorer(shape, TopAbs_SOLID)
    while ex.More():
        n += 1
        ex.Next()
    return n


def volume_in3(shape) -> float:
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return props.Mass() / MM ** 3


def build_bodies(part_shape, plan) -> tuple[list[tuple[str, object]], list[str]]:
    """[(name, shape)] in the plan frame, and any problems found building them."""
    if isinstance(plan, dict):
        plan = fx_mod.FixturePlan.from_dict(plan)
    sx, sy, sz = plan.stock_in
    zb, zt = plan.part_lo_in[2], plan.part_hi_in[2]
    part = to_plan(part_shape, plan)
    stock = box(0, 0, 0, sx, sy, sz)
    tabs = [prism(t.corners(), t.z0, t.z1) for t in plan.tabs]

    def channel(z0, z1):
        shapes = [prism(loop, z0, z1) for loop in plan.channel]
        return fuse(shapes[0], *shapes[1:]) if len(shapes) > 1 else shapes[0]

    # Held in the frame by the tabs, both faces down to the part.
    held = fuse(cut(box(0, 0, zb, sx, sy, zt), channel(zb, zt)), part, *tabs)
    if plan.procedure == fx_mod.VERTICAL:
        zm = (zb + zt) / 2.0
        first = fuse(cut(box(0, 0, 0, sx, sy, zt), channel(zm, zt)),
                     common(part, box(0, 0, zm, sx, sy, zt)), *tabs)
        names = ("After side 1", "After side 2")
    elif plan.operations == 1:
        # Sheet at the part's thickness: no floor, so op 1 already leaves the
        # part in its frame. Three bodies, not a copy.
        first, names = None, (None, "After op 1")
    else:
        first = fuse(cut(box(0, 0, 0, sx, sy, zt), channel(zb, zt)), part, *tabs)
        names = ("After op 1", "After op 2")

    problems = []
    n = solid_count(held)
    if n != 1:
        problems.append(f"The part and the frame came out as {n} separate solids: a tab "
                        f"does not reach both (check the tab positions).")
    for k, t in enumerate(plan.tabs, start=1):
        if solid_count(fuse(part, tabs[k - 1])) != 1:
            problems.append(f"Tab {k} does not touch the part at its height.")
    bodies = [("Stock", stock), (names[0], first), (names[1], held), ("Part", part)]
    return [(n, s) for n, s in bodies if s is not None], problems


def write_step(bodies, path: str | Path) -> None:
    """One STEP file holding every (name, shape), named and coloured."""
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.Quantity import Quantity_Color, Quantity_TOC_RGB
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.STEPControl import STEPControl_AsIs
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDocStd import TDocStd_Document
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.XCAFDoc import XCAFDoc_ColorGen, XCAFDoc_DocumentTool

    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    app.InitDocument(doc)
    shapes = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    colours = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())
    for name, shape in bodies:
        label = shapes.AddShape(shape, False)
        TDataStd_Name.Set_s(label, TCollection_ExtendedString(name))
        r, g, b = COLOURS.get(name, (0.7, 0.7, 0.7))
        colours.SetColor(label, Quantity_Color(r, g, b, Quantity_TOC_RGB), XCAFDoc_ColorGen)
    writer = STEPCAFControl_Writer()
    writer.SetNameMode(True)
    writer.SetColorMode(True)
    if not writer.Transfer(doc, STEPControl_AsIs):
        raise RuntimeError("Could not convert the bodies to STEP")
    if writer.Write(str(path)) != IFSelect_RetDone:
        raise RuntimeError(f"Could not write {path}")


def export(step_path: str, plan, out_path: str, progress=None) -> dict:
    """Read the part, build the four bodies for `plan` and write them to
    `out_path`. Returns {"path", "problems", "volumes_in3"}."""
    from . import mesh_utils, step_io

    if progress is not None:
        progress(f"Reading {Path(step_path).name}…")
    shape = step_io.load_step_file(step_path)
    if progress is not None:
        progress("Building the stock, in-process and tabbed bodies…")
    bodies, problems = build_bodies(shape, plan)
    if progress is not None:
        progress(f"Writing {Path(out_path).name}…")
    write_step(bodies, out_path)
    del mesh_utils
    return {"path": str(out_path), "problems": problems,
            "volumes_in3": {name: volume_in3(s) for name, s in bodies}}
