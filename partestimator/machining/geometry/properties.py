"""Mass/geometric property extraction from a TopoDS_Shape.

Note on the inertia tensor: this is the *geometric* moment of inertia --
the volume integral of r^2 dV assuming unit density (units: length^5, e.g.
mm^5) -- not the true mass moment of inertia (which would additionally
require multiplying by a material density) and not the 2D cross-sectional
"area moment of inertia" (units length^4) used in beam-bending formulas.
Per the user's request, no material density lookup is used here; multiply
this value by a material's density to get the true mass moment of
inertia if that's needed later.

BRepGProp's VolumeProperties gives the inertia matrix about the part's
centre of mass directly (GProp_GProps.MatrixOfInertia), so no
parallel-axis shift is applied.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopoDS import TopoDS_Shape


@dataclass
class ShapeProperties:
    volume_mm3: float
    surface_area_mm2: float
    total_edge_length_mm: float
    centroid_mm: np.ndarray  # (3,)
    inertia_tensor_about_centroid: np.ndarray  # (3,3), geometric moment of inertia (unit density), mm^5


def compute_shape_properties(shape: TopoDS_Shape) -> ShapeProperties:
    vol_props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, vol_props)
    volume = vol_props.Mass()
    com = vol_props.CentreOfMass()
    centroid = np.array([com.X(), com.Y(), com.Z()])

    mat = vol_props.MatrixOfInertia()
    i_origin = np.array([[mat.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)])

    # GProp_GProps.MatrixOfInertia() is ALREADY about the centre of mass
    # (OCCT documents it so, and it is verified in selftest by moving a box
    # away from the origin and getting the same tensor). v2.5 applied the
    # parallel-axis shift a second time, which drove the polar MOI negative
    # for any part not centred on the origin.
    i_centroid = i_origin

    surf_props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, surf_props)
    area = surf_props.Mass()

    lin_props = GProp_GProps()
    BRepGProp.LinearProperties_s(shape, lin_props)
    edge_len = lin_props.Mass()

    return ShapeProperties(
        volume_mm3=volume,
        surface_area_mm2=area,
        total_edge_length_mm=edge_len,
        centroid_mm=centroid,
        inertia_tensor_about_centroid=i_centroid,
    )


def spherical_moment_of_inertia(inertia_tensor_about_centroid: np.ndarray) -> float:
    """Spherical (polar) geometric moment of inertia about the center of
    mass, mm^5 -- the integral of r^2 dV where r is the straight-line
    distance from the centroid to each volume element, i.e. distance from a
    *point* rather than from an axis.

    This is direction-independent: unlike an axial moment, there's no axis
    to choose, so it characterizes the part's overall mass distribution
    about its centroid as a single number.

    Derivation: the inertia tensor's diagonal entries are
        Ixx = integral(y^2 + z^2)dV
        Iyy = integral(x^2 + z^2)dV
        Izz = integral(x^2 + y^2)dV
    Summing gives trace(I) = 2 * integral(x^2 + y^2 + z^2)dV = 2 * J,
    so the polar/spherical moment is J = trace(I) / 2. Because the trace is
    rotation-invariant, this holds in any orientation of the tensor.
    """
    return float(np.trace(inertia_tensor_about_centroid) / 2.0)


def radius_of_gyration_spherical(spherical_moment_mm5: float, volume_mm3: float) -> float:
    """The radius of gyration corresponding to the spherical moment:
    k = sqrt(J / V), in mm. A single intuitive length -- the radius of a
    thin spherical shell of the same volume with the same J."""
    if volume_mm3 <= 0:
        raise ValueError("volume must be positive")
    return float(np.sqrt(spherical_moment_mm5 / volume_mm3))
