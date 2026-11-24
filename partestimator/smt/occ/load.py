"""STEP import and export."""

from __future__ import annotations

from pathlib import Path

from ..diag import LOGGER, name_only

from OCP.BRep import BRep_Tool
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing
from OCP.BRepLib import BRepLib
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
from OCP.TopAbs import TopAbs_SHELL, TopAbs_SOLID
from OCP.TopExp import TopExp
from OCP.TopoDS import TopoDS, TopoDS_Shape
from OCP.TopTools import TopTools_IndexedMapOfShape


class LoadError(Exception):
    pass


def load_step(path: str | Path) -> TopoDS_Shape:
    """Read a STEP file and return a single solid, in millimetres.

    STEP carries its own unit declaration and OpenCASCADE honours it, so an inch
    model arrives correctly converted. Geometry is millimetres from here on --
    see `smt.units` for why that rule is worth defending.
    """
    path = Path(path)
    if not path.exists():
        raise LoadError(f"no such file: {name_only(path)}")


    from ..units import sniff_step_units

    declared = sniff_step_units(path)
    LOGGER.info("STEP declares units: %s", declared or "undeclared")

    Interface_Static.SetCVal_s("xstep.cascade.unit", "MM")

    LOGGER.debug("opening the STEP reader")
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise LoadError(f"could not read STEP file: {name_only(path)}")
    LOGGER.debug("transferring roots")
    reader.TransferRoots()
    LOGGER.debug("transfer complete")

    n = reader.NbShapes()
    if n < 1:
        raise LoadError(f"STEP file contains no shapes: {name_only(path)}")
    shape = reader.OneShape()

    solids = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_SOLID, solids)

    if solids.Extent() == 0:
        # A surface model. Try to sew it into a solid; sheet metal exported as
        # open shells is common enough to be worth attempting.
        sew = BRepBuilderAPI_Sewing(1e-6)
        sew.Add(shape)
        sew.Perform()
        shape = sew.SewedShape()
        TopExp.MapShapes_s(shape, TopAbs_SOLID, solids)
        # Sewing gives back SHELLS, never solids: a closed one has to be made
        # into a solid explicitly, or a perfectly good surface export is
        # refused below as having no body.
        shells = TopTools_IndexedMapOfShape()
        TopExp.MapShapes_s(shape, TopAbs_SHELL, shells)
        for i in range(1, shells.Extent() + 1):
            shell = TopoDS.Shell_s(shells.FindKey(i))
            if solids.Extent() or not BRep_Tool.IsClosed_s(shell):
                continue
            made = BRepBuilderAPI_MakeSolid(shell)
            if made.IsDone():
                solid = made.Solid()
                BRepLib.OrientClosedSolid_s(solid)      # outward normals
                solids.Add(solid)
        if solids.Extent() == 0:
            raise LoadError(
                f"{path.name} contains no solid body. Sheet metal recognition needs "
                f"a closed solid -- a surface or wireframe model has no thickness to "
                f"measure."
            )

    if solids.Extent() > 1:
        # v8 REFUSED here, and that was wrong. A STEP export routinely carries
        # more than one solid without being an assembly: a suppressed body left
        # in the file, a construction body, a sliver left by a failed boolean in
        # the originating CAD system. Refusing outright meant a perfectly
        # ordinary single-part file would not open at all.
        #
        # Take the largest by volume, which is the part in every one of those
        # cases, and say what was dropped.
        volumes = []
        for i in range(1, solids.Extent() + 1):
            body = TopoDS.Solid_s(solids.FindKey(i))
            try:
                props = GProp_GProps()
                BRepGProp.VolumeProperties_s(body, props)
                volumes.append((abs(props.Mass()), i, body))
            except Exception:  # noqa: BLE001 -- a body we cannot measure
                volumes.append((0.0, i, body))

        volumes.sort(key=lambda v: -v[0])
        biggest = volumes[0]
        others = sum(v[0] for v in volumes[1:])

        LOGGER.warning(
            "%d solid bodies; using the largest (%.1f mm3 of %.1f mm3 total)",
            solids.Extent(), biggest[0], biggest[0] + others,
        )
        solid = biggest[2]

        if others > 0.05 * biggest[0]:
            # Big enough that it might genuinely be an assembly rather than
            # stray geometry, which is worth saying plainly.
            LOGGER.warning(
                "the discarded bodies are %.0f%% of the largest -- this may be "
                "an assembly rather than one part",
                100.0 * others / max(biggest[0], 1e-9),
            )
    else:
        solid = TopoDS.Solid_s(solids.FindKey(1))

    try:
        if not BRepCheck_Analyzer(solid).IsValid():
            # Not fatal. Translated STEP is frequently 'invalid' by OCC's strict
            # standard while still being perfectly usable for this analysis --
            # so this is recorded rather than acted on.
            LOGGER.info("the solid does not pass a strict validity check")
    except Exception:  # noqa: BLE001 -- the checker itself can fail
        LOGGER.debug("validity check could not be run")

    return solid


def export_step(shape: TopoDS_Shape, path: str | Path) -> None:
    writer = STEPControl_Writer()
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    writer.Transfer(shape, STEPControl_AsIs)
    if writer.Write(str(path)) != IFSelect_RetDone:
        raise LoadError(f"could not write STEP file: {name_only(path)}")
