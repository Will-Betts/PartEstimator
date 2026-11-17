"""What every face on the part actually IS -- kernel-free, so it is testable.

WHY THIS MODULE EXISTS
----------------------
Up to v5 the detector asked one question -- "is this a bend?" -- and treated
everything else as noise, with a blanket warning for any surface it did not
recognise. That is fine on a clean four-bend bracket and useless on a real
production part, where the same sheet carries countersunk holes, chamfered
edges, slots, corner reliefs and hardware clearance cutouts. A countersink is a
cone; the old code saw a cone, warned that the flat pattern would be wrong in
that region, and undermined confidence in an answer that was in fact correct.

The distinction that matters is between features that CHANGE THE DEVELOPMENT and
features that are merely CUT OR MACHINED INTO IT:

    changes the development      a bend, a formed dome, a louver, an emboss
    cut or machined into it      a hole, a slot, a countersink, a chamfer,
                                 a corner radius, the perimeter cut edge

Everything in the second column flattens with the sheet and needs no special
handling at all -- it simply has to be RECOGNISED, so that it is neither
mistaken for a bend nor reported as something that breaks the answer.

THE LOCAL SHEET NORMAL
----------------------
Every test here is made against the normal of the sheet *where the feature is*,
never against a global up. A part with flanges at four different angles has no
global up, and a countersink in a flange is at 90 degrees to a countersink in
the base. The local normal comes from the largest adjoining sheet face.

CONVEXITY IS FREE AND WORTH HAVING
----------------------------------
For a cylinder or cone the kernel's surface normal points radially outward. If
the face is REVERSED, the solid's outward normal points radially inward, which
means the material is OUTSIDE the surface -- a hole. If it is not reversed, the
material is inside -- a boss, or the outer surface of a bend. One flag, captured
at description time, separates a slot end from a corner radius and gives a free
consistency check on which cylinder of a bend pair is the inner one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from . import geom


class Role(str, Enum):
    """What a face is. Ordered roughly from 'sheet' outwards to 'unknown'."""

    SHEET = "sheet"                  # a face of the sheet's top or bottom skin
    BEND_INNER = "bend_inner"
    BEND_OUTER = "bend_outer"
    CUT_EDGE = "cut_edge"            # perimeter or cutout wall, one t across
    HOLE = "hole"                    # full round through-feature wall
    SLOT_END = "slot_end"            # partial cylinder, concave, axis through t
    CORNER_RADIUS = "corner_radius"  # partial cylinder, convex, axis through t
    COUNTERSINK = "countersink"      # cone, axis through the sheet, concave
    COUNTERBORE = "counterbore"      # flat-bottomed relief around a hole
    CHAMFER = "chamfer"              # machined bevel on an edge
    FILLET = "fillet"                # small blend radius on an edge
    FORMED = "formed"                # genuinely stretch-formed: dome, louver
    UNKNOWN = "unknown"

    @property
    def is_bend(self) -> bool:
        return self in (Role.BEND_INNER, Role.BEND_OUTER)

    @property
    def develops_cleanly(self) -> bool:
        """True if this feature does not invalidate the flat pattern.

        Everything cut or machined into a sheet flattens with it. Only genuinely
        formed surfaces -- which have non-zero Gaussian curvature and therefore
        no development at all -- make the answer wrong.
        """
        return self is not Role.FORMED


#: Roles that pierce the material, for pierce counts and cut length.
PIERCING = (Role.HOLE, Role.SLOT_END)

#: Roles a machinist wants on a setup sheet even though they do not change the
#: blank -- they decide whether a second operation is needed.
SECONDARY = (Role.COUNTERSINK, Role.COUNTERBORE, Role.CHAMFER, Role.FILLET)


@dataclass
class FaceDesc:
    """The geometry of one face, with no kernel types in sight.

    `smt.occ.topo.FaceInfo` converts into this. Keeping the classifier on plain
    numpy is what lets every rule below be tested on a machine with no CAD
    kernel installed -- which, given how much of this tool's history is bugs
    that only appeared once OCP was present, is worth the small duplication.
    """

    index: int
    kind: str                                # plane|cylinder|cone|sphere|torus
    area: float = 0.0
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3))

    normal: np.ndarray | None = None         # planes: outward, orientation-fixed
    point: np.ndarray | None = None

    axis_point: np.ndarray | None = None     # cylinders and cones
    axis_dir: np.ndarray | None = None
    radius: float = 0.0
    half_angle: float = 0.0                  # cones, radians
    sweep: float = 0.0                       # radians about the axis
    axial_length: float = 0.0                # extent along the axis

    #: True when the solid lies INSIDE the surface (a boss, a bend's outer
    #: face); False when it lies outside (a hole, a bend's inner face).
    convex: bool | None = None

    def __post_init__(self) -> None:
        self.centroid = np.asarray(self.centroid, dtype=float)
        for name in ("normal", "point", "axis_point", "axis_dir"):
            v = getattr(self, name)
            if v is not None:
                setattr(self, name, np.asarray(v, dtype=float))

    @property
    def is_curved(self) -> bool:
        return self.kind != "plane"

    @property
    def developable(self) -> bool:
        """Zero Gaussian curvature: flattens exactly, by Theorema Egregium."""
        return self.kind in ("plane", "cylinder", "cone")

    def extent(self) -> float:
        """A characteristic size, used to tell a bevel from a wall.

        sqrt(area) is crude but it is scale-correct and needs no boundary
        interrogation, which is what keeps this module kernel-free.
        """
        return math.sqrt(max(self.area, 0.0))


@dataclass
class Feature:
    """One classified face, plus whatever a machinist would want to know."""

    index: int
    role: Role
    reason: str = ""
    diameter: float | None = None
    included_angle: float | None = None      # countersinks and chamfers, degrees
    depth: float | None = None
    width: float | None = None               # chamfers
    #: Index of the sheet face this feature sits on, when one could be found.
    host: int | None = None

    def label(self, decimals: int = 3, per_mm: float = 1.0, suffix: str = "mm") -> str:
        """A short human description, in whatever unit the caller is using."""

        def n(x: float) -> str:
            return f"{x * per_mm:.{decimals}f}"

        if self.role is Role.HOLE and self.diameter:
            return f"hole {n(self.diameter)} {suffix} dia"
        if self.role is Role.COUNTERSINK:
            ang = f" {self.included_angle:.0f} deg" if self.included_angle else ""
            dia = f" {n(self.diameter)} {suffix} dia" if self.diameter else ""
            return f"countersink{ang}{dia}"
        if self.role is Role.COUNTERBORE and self.diameter:
            return f"counterbore {n(self.diameter)} {suffix} dia"
        if self.role is Role.SLOT_END and self.diameter:
            return f"slot end R{n(self.diameter / 2)} {suffix}"
        if self.role is Role.CORNER_RADIUS and self.diameter:
            return f"corner R{n(self.diameter / 2)} {suffix}"
        if self.role is Role.CHAMFER and self.width:
            ang = f" at {self.included_angle:.0f} deg" if self.included_angle else ""
            return f"chamfer {n(self.width)} {suffix}{ang}"
        if self.role is Role.FILLET and self.diameter:
            return f"fillet R{n(self.diameter / 2)} {suffix}"
        return self.role.value.replace("_", " ")


# ---------------------------------------------------------------------------
# step 1: which faces are the sheet
# ---------------------------------------------------------------------------


def sheet_plane_pairs(
    planes: list[FaceDesc], thickness: float, tol, overlap_frac: float = 0.75,
) -> list[tuple[int, int]]:
    """Planar faces that are the two skins of the sheet, paired up.

    THREE CONDITIONS, AND THE THIRD IS THE ONE THAT WAS MISSING.

    1. Antiparallel normals.
    2. Separated by exactly one thickness.
    3. FACING EACH OTHER, and overlapping when projected.

    Condition 3 earns its keep. A tray base and a flange folded back over it are
    antiparallel and can easily sit one thickness apart without being two sides
    of the same sheet; matching on distance alone paired them and produced a
    flange that did not exist. Two skins of one sheet have their material
    BETWEEN them, so each one's partner lies on its inward side -- and their
    centroids are nearly coincident once the thickness direction is projected
    out.

    Returns index pairs, larger face first.
    """
    out: list[tuple[int, int]] = []
    used: set[int] = set()

    # Largest first: the real sheet faces are the big ones, and taking them
    # first stops a sliver claiming a partner that a proper face wanted.
    order = sorted(planes, key=lambda f: -f.area)

    for i, a in enumerate(order):
        if a.index in used or a.normal is None:
            continue
        best: FaceDesc | None = None
        best_off = math.inf

        for b in order[i + 1:]:
            if b.index in used or b.normal is None:
                continue
            if not geom.antiparallel(a.normal, b.normal, tol.tangent_angular):
                continue

            # Signed, not absolute: the partner must lie on the INWARD side.
            d = float(geom.signed_plane_distance(b.point, a.point, a.normal))
            if d > 0:
                continue                      # b is outside a, not behind it
            if not tol.radius_close(abs(d), thickness):
                continue

            delta = b.centroid - a.centroid
            lateral = float(
                np.linalg.norm(delta - np.dot(delta, a.normal) * a.normal)
            )
            scale = max(a.extent(), b.extent(), 1e-9)
            if lateral > overlap_frac * scale:
                continue

            if lateral < best_off:
                best, best_off = b, lateral

        if best is not None:
            out.append((a.index, best.index))
            used.add(a.index)
            used.add(best.index)

    return out


def local_normal(
    idx: int, descs: dict[int, FaceDesc], adj: dict[int, set[int]],
    sheet: set[int], depth: int = 2,
) -> np.ndarray | None:
    """The sheet normal in the neighbourhood of face `idx`.

    Breadth-first out to `depth` rings, taking the largest sheet face found.
    Two rings is enough to cross a chamfer and its cut edge and still land on
    the skin; going further starts finding the sheet on the far side of a bend,
    where the normal means something entirely different.
    """
    seen = {idx}
    ring = {idx}
    for _ in range(depth + 1):
        found = [
            descs[i] for i in ring
            if i in sheet and descs.get(i) is not None and descs[i].normal is not None
        ]
        if found:
            return max(found, key=lambda f: f.area).normal
        nxt: set[int] = set()
        for i in ring:
            nxt |= adj.get(i, set()) - seen
        if not nxt:
            break
        seen |= nxt
        ring = nxt
    return None


# ---------------------------------------------------------------------------
# step 2: classify everything else against it
# ---------------------------------------------------------------------------


def classify(
    descs: list[FaceDesc],
    adj: dict[int, set[int]],
    thickness: float,
    tol,
    bend_faces: dict[int, Role] | None = None,
) -> dict[int, Feature]:
    """Assign a role to every face.

    `bend_faces` is the detector's own verdict on the cylinders it accepted as
    bends. The bend test is stricter than anything here -- it needs a concentric
    mate one thickness away AND tangency to two distinct planes -- so its answer
    wins wherever it has one.

    The ordering matters. Sheet faces are established first because every other
    rule is expressed relative to the local sheet normal; without them a
    countersink cannot be told from a conical formed feature.
    """
    by_index = {d.index: d for d in descs}
    bend_faces = bend_faces or {}
    out: dict[int, Feature] = {}

    planes = [d for d in descs if d.kind == "plane" and d.area > tol.min_face_area]
    pairs = sheet_plane_pairs(planes, thickness, tol)
    sheet: set[int] = {i for pair in pairs for i in pair}

    for i in sheet:
        out[i] = Feature(
            index=i, role=Role.SHEET, reason="paired across the thickness"
        )

    for idx, role in bend_faces.items():
        out[idx] = Feature(
            index=idx, role=role,
            reason="concentric pair, tangent to two planes",
        )

    for d in descs:
        if d.index in out:
            continue
        n = local_normal(d.index, by_index, adj, sheet)
        out[d.index] = _classify_one(d, n, thickness, tol, adj, sheet)

    _promote_counterbores(out, adj)
    return out


def _classify_one(
    d: FaceDesc, n: np.ndarray | None, thickness: float, tol,
    adj: dict[int, set[int]], sheet: set[int],
) -> Feature:
    """One face, given the sheet normal where it sits."""
    f = Feature(index=d.index, role=Role.UNKNOWN)
    f.host = next((i for i in sorted(adj.get(d.index, set())) if i in sheet), None)

    if d.area <= tol.min_face_area:
        f.role, f.reason = Role.UNKNOWN, "sliver"
        return f

    if d.kind == "plane":
        return _classify_plane(d, n, thickness, f)
    if d.kind == "cylinder":
        return _classify_cylinder(d, n, thickness, f)
    if d.kind == "cone":
        return _classify_cone(d, n, f)

    # Sphere, torus, B-spline: non-zero Gaussian curvature. These are the only
    # surfaces that genuinely have no development.
    f.role = Role.FORMED
    f.reason = f"{d.kind} surface -- not developable"
    return f


def _classify_plane(d: FaceDesc, n, thickness: float, f: Feature) -> Feature:
    """A planar face that is not part of the sheet skin.

    Three possibilities: it stands through the thickness (a cut edge), it lies
    at a bevel (a chamfer), or it is parallel to the skin without a partner (a
    counterbore floor, or a pocket).
    """
    if n is None or d.normal is None:
        f.role, f.reason = Role.UNKNOWN, "no sheet face nearby to reference"
        return f

    cos = float(np.dot(d.normal, n))
    ang = math.degrees(math.acos(max(-1.0, min(1.0, abs(cos)))))

    if ang > 80.0:
        f.role = Role.CUT_EDGE
        f.reason = "planar, standing through the thickness"
        f.depth = thickness
        return f

    if ang < 10.0:
        f.role = Role.COUNTERBORE
        f.reason = "planar, parallel to the sheet, no partner one thickness away"
        return f

    f.role = Role.CHAMFER
    f.reason = f"planar bevel at {ang:.0f} deg to the sheet"
    f.included_angle = ang
    # A chamfer is a narrow ribbon, so sqrt(area) overstates its width. The
    # geometric ceiling -- a bevel at this angle cannot take off more than the
    # thickness allows -- is the safer of the two readings.
    f.width = min(d.extent(), thickness / max(math.sin(math.radians(ang)), 1e-3))
    return f


def _classify_cylinder(d: FaceDesc, n, thickness: float, f: Feature) -> Feature:
    """A cylindrical face the bend detector did not claim.

    The bend detector has already taken every cylinder with a concentric mate
    one thickness away that is tangent to two distinct planes. What is left is
    round: a hole, a slot end, a corner radius or an edge fillet -- told apart
    by how the axis lies, how far it sweeps, and which side the material is on.
    """
    f.diameter = 2.0 * d.radius

    if n is None or d.axis_dir is None:
        f.role, f.reason = Role.UNKNOWN, "cylinder with no sheet face nearby"
        return f

    along = abs(float(np.dot(d.axis_dir, n)))
    through = d.axial_length >= 0.5 * thickness

    if along > 0.9:
        # Axis stands through the sheet: a round feature in plan view.
        if d.sweep > math.radians(350.0):
            f.role = Role.HOLE
            f.reason = "full cylinder, axis through the sheet"
            f.depth = d.axial_length
            return f
        if d.convex is False:
            f.role = Role.SLOT_END
            f.reason = f"{math.degrees(d.sweep):.0f} deg concave cylinder in the sheet"
            f.depth = d.axial_length
            return f
        f.role = Role.CORNER_RADIUS
        f.reason = f"{math.degrees(d.sweep):.0f} deg convex cylinder on the outside"
        return f

    if along < 0.2:
        # Axis lies in the plane of the sheet but the bend test did not take it.
        # A small radius here is an edge break; a large one with no concentric
        # mate is a bend that failed a gate, and that is worth saying out loud
        # rather than quietly dropping.
        if d.radius < 0.6 * thickness and not through:
            f.role = Role.FILLET
            f.reason = "small radius along an edge"
            return f
        f.role = Role.UNKNOWN
        f.reason = (
            f"R{d.radius:.3f} cylinder lying in the sheet plane with no "
            f"concentric mate -- possibly an unrecognised bend"
        )
        return f

    f.role = Role.FILLET
    f.reason = "cylindrical blend"
    return f


def _classify_cone(d: FaceDesc, n, f: Feature) -> Feature:
    """A conical face.

    This is the one that mattered. A countersink is a cone whose axis runs
    through the sheet and whose material is on the outside -- and it is
    perfectly developable, so it belongs in the flat pattern as a marking, not
    in a warning about formed features. v5 warned on every cone it saw, which
    meant a correct answer arrived looking untrustworthy.

    A cone whose axis lies IN the sheet plane is a tapered bend: a real thing,
    genuinely developable, but not something this version unfolds. That gets
    said plainly rather than silently mishandled.
    """
    half = math.degrees(abs(d.half_angle))
    f.diameter = 2.0 * d.radius

    if n is None or d.axis_dir is None:
        f.role, f.reason = Role.UNKNOWN, "cone with no sheet face nearby"
        return f

    along = abs(float(np.dot(d.axis_dir, n)))

    # Half angle is measured from the axis, so the included angle is twice it:
    # a 90 degree countersink has a 45 degree half angle. The standard included
    # angles are 82 (imperial), 90 (metric), 100 and 120.
    included = 2.0 * half

    if along > 0.9:
        if d.convex is False:
            f.role = Role.COUNTERSINK
            f.included_angle = included
            f.depth = d.axial_length
            f.reason = f"cone, axis through the sheet, {included:.0f} deg included"
            return f
        f.role = Role.CHAMFER
        f.included_angle = included
        f.reason = "conical chamfer around a boss or an outside corner"
        return f

    if along < 0.2:
        f.role = Role.UNKNOWN
        f.reason = (
            "conical surface lying in the sheet plane -- a tapered bend. It is "
            "developable, but this version unfolds cylindrical bends only, so "
            "the flat pattern will be wrong across it."
        )
        return f

    f.role = Role.CHAMFER
    f.included_angle = included
    f.reason = "conical bevel"
    return f


def _promote_counterbores(
    out: dict[int, Feature], adj: dict[int, set[int]],
) -> None:
    """A flat floor is only a counterbore if a hole runs out of it.

    Otherwise it is an unpaired planar face, which is worth flagging honestly
    rather than describing as a machining feature that is not there.
    """
    for f in out.values():
        if f.role is not Role.COUNTERBORE:
            continue
        neighbours = [out.get(i) for i in adj.get(f.index, set())]
        holes = [g for g in neighbours if g is not None and g.role is Role.HOLE]
        if holes:
            f.diameter = max((g.diameter or 0.0) for g in holes) or f.diameter
        else:
            f.role = Role.UNKNOWN
            f.reason = (
                "planar face parallel to the sheet with no partner one "
                "thickness away -- the part may not be uniform gauge here"
            )


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


@dataclass
class FeatureGroup:
    """Identical features collapsed into one line for the setup sheet."""

    role: Role
    count: int
    diameter: float | None = None
    included_angle: float | None = None

    def key(self) -> tuple:
        return (
            self.role.value,
            round(self.diameter, 4) if self.diameter else None,
            round(self.included_angle, 1) if self.included_angle else None,
        )


def group_features(features: dict[int, Feature], roles=None) -> list[FeatureGroup]:
    """Collapse a face-by-face classification into a countable summary.

    Twelve identical clearance holes should read as "12 x hole 6.5", not as
    twelve lines. Grouping is on role plus the dimensions that distinguish one
    tool from another, which is exactly what decides whether a second operation
    is needed.
    """
    wanted = set(roles) if roles else None
    buckets: dict[tuple, FeatureGroup] = {}

    for f in features.values():
        if wanted is not None and f.role not in wanted:
            continue
        g = FeatureGroup(
            role=f.role, count=0,
            diameter=f.diameter, included_angle=f.included_angle,
        )
        buckets.setdefault(g.key(), g).count += 1

    return sorted(
        buckets.values(), key=lambda g: (g.role.value, -(g.diameter or 0.0)),
    )


def counts_by_role(features: dict[int, Feature]) -> dict[Role, int]:
    out: dict[Role, int] = {}
    for f in features.values():
        out[f.role] = out.get(f.role, 0) + 1
    return out


def undevelopable(features: dict[int, Feature]) -> list[Feature]:
    """The features that genuinely make the flat pattern wrong.

    Deliberately short. A tool that warns about everything trains you to ignore
    it, and the whole point of this module is that a countersink is not a
    problem.
    """
    return [f for f in features.values() if not f.role.develops_cleanly]


# ---------------------------------------------------------------------------
# thickness, when there are no bends to measure it from
# ---------------------------------------------------------------------------


def thickness_plausible(value: float, tol) -> bool:
    """Could this be sheet metal at all?

    A bound, not a guess, and it is what stops the wrong pair of faces being
    chosen as the sheet. Sheet metal is thin BY DEFINITION: half an inch is
    already well outside anything a press brake air bends, and anything under a
    twentieth of a millimetre is noise rather than material.

    Without this, a part whose real skins are partly hidden -- behind a machined
    boss, a thick weld prep, a mounting pad -- can have some other pair of
    parallel faces measured as the material. That is worse than failing,
    because every downstream number then comes out confidently wrong instead of
    obviously wrong.
    """
    v = float(value)
    return tol.min_thickness <= v <= tol.max_thickness


def thickness_candidates(planes: list[FaceDesc], tol) -> list[tuple[float, float]]:
    """Plausible sheet thicknesses from planar evidence alone, with support.

    Needed because a FLAT BLANK has no bends, and every bend is an independent
    measurement of the thickness. With none of them, the thickness has to come
    from the two skins -- and the obvious way to do that is wrong.

    The obvious way is to take the smallest distance between any two
    antiparallel planar faces. On a blank with a 3 mm slot in 5 mm material the
    two walls of that slot are antiparallel and 3 mm apart, so the obvious way
    reports 3 mm and everything downstream is confidently wrong.

    What separates the real answer from the slot is WHICH SIDE THE MATERIAL IS
    ON. Two skins of a sheet have their material between them, so each one's
    partner lies on its inward side. The two walls of a slot have the material
    outside them, so each one's partner lies on its outward side and the
    candidate never arises. The same signed test as `sheet_plane_pairs`, run
    before the thickness is known.

    Returns (distance, supporting area) sorted by support, best first.
    """
    votes: dict[int, list] = {}

    for i, a in enumerate(planes):
        if a.normal is None:
            continue
        for b in planes[i + 1:]:
            if b.normal is None:
                continue
            if not geom.antiparallel(a.normal, b.normal, tol.tangent_angular):
                continue

            d = float(geom.signed_plane_distance(b.point, a.point, a.normal))
            if d >= -1e-9:
                continue                      # b is outside a, not behind it
            dist = abs(d)

            delta = b.centroid - a.centroid
            lateral = float(
                np.linalg.norm(delta - np.dot(delta, a.normal) * a.normal)
            )
            if lateral > 0.75 * max(a.extent(), b.extent(), 1e-9):
                continue

            # Bucket on the tolerance, so translation noise does not split one
            # real thickness into several near-identical candidates.
            key = None
            for k in votes:
                if tol.radius_close(votes[k][0], dist):
                    key = k
                    break
            if key is None:
                key = len(votes)
                votes[key] = [dist, 0.0, 0]
            votes[key][1] += min(a.area, b.area)
            votes[key][2] += 1

    out = [(v[0], v[1]) for v in votes.values()]
    out.sort(key=lambda x: -x[1])
    return out


def thickness_from_face(index: int, planes: list[FaceDesc], tol,
                        cap: bool = True) -> tuple[float, int] | None:
    """Thickness measured from ONE nominated face to its opposite number.

    The manual override. The user has pointed at a face and said "this is the
    sheet"; all that remains is to find what lies one material behind it.

    Same three conditions as the automatic pairing -- antiparallel, on the
    INWARD side, overlapping in projection -- but with the search anchored to a
    face that has been asserted rather than inferred, so the nearest qualifying
    partner wins outright. No cluster, no vote, no ranking by supporting area.

    `cap` applies the sheet-metal plausibility bound. It can be turned off,
    because a person pointing at two specific faces has better information than
    a rule of thumb, and refusing them would make the override useless in the
    one situation it exists for.

    Returns (thickness, partner index), or None if nothing qualifies.
    """
    by_index = {f.index: f for f in planes}
    face = by_index.get(index)
    if face is None or face.normal is None:
        return None

    best = None
    best_gap = math.inf

    for other in planes:
        if other.index == index or other.normal is None:
            continue
        if not geom.antiparallel(face.normal, other.normal, tol.tangent_angular):
            continue

        d = float(geom.signed_plane_distance(other.point, face.point, face.normal))
        if d >= -1e-9:
            continue                       # outside the face, not behind it
        gap = abs(d)
        if cap and not thickness_plausible(gap, tol):
            continue

        delta = other.centroid - face.centroid
        lateral = float(
            np.linalg.norm(delta - np.dot(delta, face.normal) * face.normal)
        )
        if lateral > 0.75 * max(face.extent(), other.extent(), 1e-9):
            continue

        if gap < best_gap:
            best, best_gap = other.index, gap

    if best is None:
        return None
    return best_gap, best
