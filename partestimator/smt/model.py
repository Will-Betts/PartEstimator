"""Core data model.

Deliberately holds BOTH geometric states -- the as-bent solid and the flat --
plus the rigid transform that maps between them. Two reasons:

  1. The re-fold validation test (apply the inverse transforms to the flat and
     compare against the original) is the strongest correctness check available,
     and it is nearly free once the transforms exist.
  2. Hydroform form-block generation consumes the *bent* solid and the feature
     classification. Discarding them after unfolding would make that work a
     rewrite rather than an extension.

The geometry fields are plain numpy arrays so this module imports without a CAD
kernel present; the raw kernel handles hang off `.face` / `.solid` as opaque
objects that only the `occ` subpackage touches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class BendKind(str, Enum):
    AIR = "air_bend"
    # Reserved for later phases -- detected and reported, not unfolded, in v1.
    TEARDROP_HEM = "teardrop_hem"
    CLOSED_HEM = "closed_hem"
    CURL = "curl"


class FaceKind(str, Enum):
    FLANGE = "flange"
    BEND_INNER = "bend_inner"
    BEND_OUTER = "bend_outer"
    HOLE = "hole"
    CORNER_FILLET = "corner_fillet"
    WALL = "wall"          # the thickness edge around the perimeter
    UNKNOWN = "unknown"


@dataclass
class Flange:
    """One flat region of the sheet: a pair of parallel planar faces.

    Both faces are carried because the unfold solver has to pick a consistent
    *side* of the sheet as it walks the part. On a part mixing up-bends and
    down-bends there is no global "top" -- the side has to be propagated across
    each bend, and that needs both candidate faces at every node. See
    `unfold.choose_side`.
    """

    id: int
    normal: np.ndarray            # unit outward normal of face A
    origin: np.ndarray            # a point on face A's plane
    centroid: np.ndarray
    area: float = 0.0

    # Face B: the opposite side of the sheet. Its normal is -normal, and its
    # plane sits one thickness away.
    origin_b: np.ndarray | None = None

    # Discretised boundary loops of each face, in 3D world coordinates, outer
    # wire first. Extracted ONCE at detection time and cached here.
    #
    # This is what makes the live K-factor slider possible. Detection does not
    # depend on K at all -- only the allowances and the unfold transforms do.
    # With the boundaries cached, changing K is a numpy transform of these
    # points, which is instant. Re-running the kernel's boolean fuse on every
    # slider frame would not be.
    loops_a: list[np.ndarray] | None = None
    loops_b: list[np.ndarray] | None = None

    # Opaque kernel handles. `face` / `partner_face` are the LARGEST on each
    # side; `faces_a` / `faces_b` are all of them.
    #
    # A flange is not necessarily one face per side. Two tabs bent up from the
    # same bend line are coplanar, on the same side of the same bend, and
    # therefore take the same rigid transform -- so they are one flange made of
    # two disconnected regions. Keeping only the largest is how the second tab
    # came to be missing from the flat pattern entirely.
    face: Any = None
    partner_face: Any = None
    faces_a: list = field(default_factory=list)
    faces_b: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.normal = np.asarray(self.normal, dtype=float)
        self.origin = np.asarray(self.origin, dtype=float)
        self.centroid = np.asarray(self.centroid, dtype=float)
        if self.origin_b is not None:
            self.origin_b = np.asarray(self.origin_b, dtype=float)

    def sides(self) -> list[tuple[np.ndarray, np.ndarray, Any, int]]:
        """Both candidates as (outward_normal, plane_point, face, side_index).

        The side INDEX travels with the choice. An earlier version identified
        the chosen side by comparing kernel face handles with `is`, which is
        never true: two Python wrappers around the same TopoDS_Face are
        different objects, so it always fell through to side 0.
        """
        out = [(self.normal, self.origin, self.face, 0)]
        if self.origin_b is not None:
            out.append((-self.normal, self.origin_b, self.partner_face, 1))
        return out

    def regions_for_side(self, side: int) -> list[list[np.ndarray]]:
        """Boundary loops grouped BY REGION, each as [outer wire, *holes].

        The grouping matters. Flattening every loop into one list makes the
        second region's outer wire indistinguishable from a hole in the first,
        so a part with two tabs on one bend would draw the second tab as a
        cutout in the first. Which is worse than losing it.

        Tolerates the older flat form -- a bare list of loops -- so fixtures and
        any caller that predates regions keep working, treated as a single
        region.
        """
        raw = self.loops_b if side == 1 else self.loops_a
        if not raw:
            raw = self.loops_a if side == 1 else self.loops_b
        if not raw:
            return []
        if isinstance(raw[0], np.ndarray):
            return [list(raw)]
        return [list(region) for region in raw if len(region)]

    def faces_for_side(self, side: int) -> list:
        """Every kernel face handle on the referenced side, largest first."""
        primary = self.faces_b if side == 1 else self.faces_a
        if primary:
            return list(primary)
        single = self.partner_face if side == 1 else self.face
        return [single] if single is not None else []

    def loops_for_side(self, side: int) -> list[np.ndarray]:
        """Every loop on the referenced side, flattened.

        For callers that only want points and do not care which wire is which
        -- bounding boxes, orientation. Anything that draws must use
        `regions_for_side`.
        """
        return [loop for region in self.regions_for_side(side) for loop in region]

    def has_outline(self) -> bool:
        return bool(self.loops_a or self.loops_b)

    @property
    def region_count(self) -> int:
        """How many disconnected pieces of sheet this flange is."""
        return len(self.regions_for_side(0)) or 1


@dataclass
class Bend:
    """A brake bend: a coaxial pair of cylindrical faces."""

    id: int
    axis_point: np.ndarray
    axis_dir: np.ndarray
    r_inner: float
    thickness: float
    angle: float                  # sweep, radians
    # Envelope length drives the flat pattern.
    length_envelope: float
    # Actual material along the bend line, net of cutouts. Drives TONNAGE --
    # a bend line that is 40 percent cut out needs materially less force.
    length_material: float
    flange_a: int
    flange_b: int
    # Axial extent of the bend, as signed distances along axis_dir measured
    # from axis_point. Gives the endpoints of the bend line.
    v_min: float = 0.0
    v_max: float = 0.0
    direction: int = 1            # +1 up, -1 down, relative to the base flange
    kind: BendKind = BendKind.AIR

    # Filled in by the allowance pass.
    k_factor: float = 0.0
    bend_allowance: float = 0.0
    bend_deduction: float = 0.0
    allowance_source: str = ""

    inner_face: Any = None
    outer_face: Any = None

    # The cylinder's own parametrisation, and the real trimmed boundary of its
    # inner face. Together these let the bend zone be DEVELOPED rather than
    # approximated by a rectangle -- see geom.develop_cylinder.
    axis_x_dir: np.ndarray | None = None
    u_min: float = 0.0
    u_max: float = 0.0
    inner_loops: list[np.ndarray] | None = None

    @property
    def u_mid(self) -> float:
        return 0.5 * (self.u_min + self.u_max)

    def neutral_radius(self) -> float:
        """rho = R + K*t. The layer that neither stretches nor compresses."""
        return self.r_inner + self.k_factor * self.thickness

    def __post_init__(self) -> None:
        self.axis_point = np.asarray(self.axis_point, dtype=float)
        self.axis_dir = np.asarray(self.axis_dir, dtype=float)

    @property
    def angle_deg(self) -> float:
        return math.degrees(self.angle)

    @property
    def r_outer(self) -> float:
        return self.r_inner + self.thickness

    def other(self, flange_id: int) -> int:
        if flange_id == self.flange_a:
            return self.flange_b
        if flange_id == self.flange_b:
            return self.flange_a
        raise KeyError(f"bend {self.id} does not touch flange {flange_id}")


@dataclass
class Hole:
    """A through feature, reported for pierce count and cut length."""

    id: int
    centre: np.ndarray
    diameter: float | None        # None for non-circular cutouts
    perimeter: float
    flange_id: int | None = None

    def __post_init__(self) -> None:
        self.centre = np.asarray(self.centre, dtype=float)


@dataclass
class FlatPattern:
    """The developed blank."""

    # Outer boundary and hole boundaries as 2D polylines (already discretised
    # for reporting; the exact geometry lives on `face`).
    outline: list[np.ndarray] = field(default_factory=list)
    holes: list[np.ndarray] = field(default_factory=list)
    # Bend lines in flat space: (start_xy, end_xy, bend_id).
    bend_lines: list[tuple[np.ndarray, np.ndarray, int]] = field(default_factory=list)

    area: float = 0.0             # true profile area, net of holes
    perimeter: float = 0.0        # total cut length including holes
    bbox: tuple[float, float] = (0.0, 0.0)
    # Minimum-area rectangle, used when grain direction is unconstrained.
    min_rect: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # EVERY region of the profile. A flat pattern that fuses into more than one
    # face is unusual but legitimate, and an earlier version kept only the first
    # one -- so the DXF silently lost whole flanges the preview was showing.
    faces: list = field(default_factory=list)
    solid: Any = None             # the extruded flat solid

    @property
    def face(self):
        """The largest region, for callers that want a single profile."""
        return self.faces[0] if self.faces else None


@dataclass
class Part:
    """Everything known about one sheet metal part."""

    thickness: float
    material: str = "CRS"

    # Thickness as MEASURED from the geometry. This is what detection and the
    # bend allowances use. `nominal_thickness` is the nearest catalogue gauge and
    # is for reporting only -- it must never be fed back into a geometric test.
    measured_thickness: float = 0.0
    nominal_thickness: float | None = None

    # What unit the source file declared: "mm", "in" or None if undeclared.
    # Geometry is always stored in millimetres regardless.
    file_units: str | None = None

    # Why detection accepted or rejected each candidate. See occ.detect.Funnel.
    funnel: object = None

    # Face index -> `smt.features.Feature`. Every face on the part gets a role,
    # which is what lets a countersink, a chamfer and a slot be reported rather
    # than mistaken for something that breaks the flat pattern.
    features: dict = field(default_factory=dict)

    # The same thing collapsed for a setup sheet: "12 x hole 6.5" rather than
    # twelve separate lines.
    feature_groups: list = field(default_factory=list)

    # True when the part has no bends at all. Not a failure -- a flat blank is
    # an ordinary thing to open, and every dimension is then measured directly.
    is_blank: bool = False

    #: B-rep face index the user nominated as the sheet surface, if any.
    sheet_face: int | None = None

    # How far the flat pattern was turned to minimise its bounding box, in
    # radians. Recorded so the number on screen can be explained, and so the
    # rotation can be undone if you need the blank in its modelled orientation.
    flat_rotation: float = 0.0
    #: Fraction of bounding-box area that rotation saved.
    flat_rotation_saving: float = 0.0

    flanges: dict[int, Flange] = field(default_factory=dict)
    bends: dict[int, Bend] = field(default_factory=dict)
    holes: list[Hole] = field(default_factory=list)

    base_flange: int | None = None
    # flange id -> 4x4 rigid transform, bent frame to flat frame.
    transforms: dict[int, np.ndarray] = field(default_factory=dict)

    flat: FlatPattern | None = None

    # KEEP these. Hydroform form-block generation needs the bent geometry.
    solid_bent: Any = None

    # `describe_all` output for the bent solid, cached so the 3D view does not
    # repeat the most expensive read in the pipeline.
    face_infos: Any = None

    # Face index -> boundary loops, shared with the 3D view so the most
    # expensive read in the pipeline happens once rather than twice.
    loop_cache: dict = field(default_factory=dict)

    # The flat pattern as a real SOLID, with countersinks, chamfers and every
    # other machined feature still on it -- built by moving the material rather
    # than redrawing the outline. None when that could not be done and the
    # extruded profile was used instead; `flat_is_extruded` says which.
    flat_solid: Any = None
    flat_is_extruded: bool = False

    volume: float = 0.0
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def warn(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)

    def note(self, msg: str) -> None:
        if msg not in self.notes:
            self.notes.append(msg)

    @property
    def bend_count(self) -> int:
        return len(self.bends)

    def distinct_radii(self, tol: float = 1e-3) -> list[float]:
        """Unique inside radii -- i.e. the punch nose radii you need to own."""
        out: list[float] = []
        for b in sorted(self.bends.values(), key=lambda x: x.r_inner):
            if not any(abs(b.r_inner - r) < tol for r in out):
                out.append(b.r_inner)
        return out

    def total_bend_length(self) -> float:
        return sum(b.length_material for b in self.bends.values())

    def features_with_role(self, *roles) -> list:
        wanted = set(roles)
        return [f for f in self.features.values() if f.role in wanted]

    def pierce_count(self) -> int:
        """Holes and slots -- what a punch or a laser has to start separately.

        Counted from the classified features rather than from the hole records,
        so it is right whether or not simple mode skipped the hole analysis.
        """
        from .features import PIERCING

        return len(self.features_with_role(*PIERCING))

    def secondary_operations(self) -> list:
        """Countersinks, counterbores and chamfers.

        These do not change the blank at all -- they are cut or machined into a
        sheet that flattens the same either way -- but they decide whether the
        part needs a second setup after forming, which is worth knowing before
        you quote it.
        """
        from .features import SECONDARY

        return self.features_with_role(*SECONDARY)
