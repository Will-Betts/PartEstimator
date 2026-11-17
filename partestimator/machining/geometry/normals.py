"""Shared helper: outward-facing surface normal at a (u, v) parameter,
accounting for TopAbs_REVERSED face orientation. Used by the hole/fillet
concavity tests and the undercut/thin-wall feature heuristics.

Note on OCP vs. pythonocc-core: GeomLProp_SLProps's constructor in OCP only
accepts a genuine Geom_Surface, not a BRepAdaptor_Surface/Adaptor3d_Surface
directly (pythonocc-core exposed an overload that did). So we pull the
underlying Geom_Surface out of the adaptor first via
`adaptor.Surface().Surface()` -- BRepAdaptor_Surface.Surface() returns a
GeomAdaptor_Surface, and GeomAdaptor_Surface.Surface() returns the actual
Geom_Surface handle.
"""
from __future__ import annotations

import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepLProp import BRepLProp_SLProps
from OCP.TopAbs import TopAbs_REVERSED
from OCP.TopoDS import TopoDS_Face


def sampled_outward_normal(face: TopoDS_Face, adaptor: BRepAdaptor_Surface, u: float, v: float) -> np.ndarray | None:
    """Returns the unit outward-facing normal (material-to-void direction)
    at parameter (u, v), or None if the surface has no well-defined normal
    there (degenerate/seam point)."""
    # BRepLProp on the BRep adaptor, NOT GeomLProp on the bare Geom_Surface:
    # the adaptor carries the face's TopLoc_Location, the bare surface does
    # not. v2.5 used the bare surface, so any located face (instanced bodies
    # and many STEP exports carry one) returned an unrotated normal.
    sl_props = BRepLProp_SLProps(adaptor, u, v, 1, 1e-6)
    if not sl_props.IsNormalDefined():
        return None
    n = sl_props.Normal()
    vec = np.array([n.X(), n.Y(), n.Z()])
    if face.Orientation() == TopAbs_REVERSED:
        vec = -vec
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return None
    return vec / norm
