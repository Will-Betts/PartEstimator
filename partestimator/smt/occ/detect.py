"""Feature recognition: find the air bends and the flanges they connect.

THE BEND SIGNATURE
------------------
A brake bend is a PAIR of cylindrical faces where:

  1. the axes are coaxial;
  2. the radii differ by exactly the material thickness;
  3. the sweeps and axial extents match;
  4. both are TANGENT to the planar faces they adjoin;
  5. the axis is PERPENDICULAR to those planes' normals.

Conditions 4 and 5 are the discriminator, and they are evaluated locally --
against each cylinder's own neighbours rather than against a global "sheet
normal", which is meaningless on a part whose flanges are not parallel.

                     tangent to neighbours?   axis . adjoining normal   sweep
    bend                     yes                      ~ 0            < 180 deg
    corner fillet            yes                      ~ +/-1         <= 90 deg
    hole                     no                       ~ +/-1         360 deg

The corner fillet is the subtle one: it IS tangent to the perimeter wall faces
it meets, and those walls have normals perpendicular to the fillet axis, so it
can pass tests 4 and 5. What excludes it is condition 1 and 2 -- a rounded
corner is a single cylinder with no concentric mate one thickness away.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .. import geom
from ..config import Tolerance, UnfoldOptions
from ..features import (
    Feature,
    Role,
    classify,
    sheet_plane_pairs,
    thickness_candidates,
    thickness_from_face,
    thickness_plausible,
)
from ..model import Bend, BendKind, Flange, Hole, Part
from ..materials import snap_to_gauge
from ..diag import LOGGER, Stage, null_stage
from .topo import (
    FaceInfo,
    MergedCylinder,
    adjacent_faces,
    describe_all,
    face_loops,
    face_loops_safe,
    merge_coaxial,
    solid_volume,
)


class DetectionError(Exception):
    pass


@dataclass
class _Candidate:
    inner: MergedCylinder
    outer: MergedCylinder
    delta_r: float


# ---------------------------------------------------------------------------
# step 1: pair coaxial cylinders
# ---------------------------------------------------------------------------


def coaxial_pairs(
    merged: list[MergedCylinder], tol: Tolerance
) -> list[_Candidate]:
    """Every pair of merged cylinders sharing an axis, smaller radius first.

    Grouped by axis line first: two cylinders on different axes can never pair,
    and comparing all of them against each other is wasted work on a part with
    many holes. The grouping uses exact tests, so nothing is lost -- see
    `geom.group_by_line`.
    """
    out: list[_Candidate] = []

    for line in geom.group_by_line(
        merged, lambda c: (c.axis_point, c.axis_dir),
        ang_tol=tol.angular, lin_tol=tol.coaxial,
    ):
        for i, a in enumerate(line):
            for b in line[i + 1:]:
                lo, hi = (a, b) if a.radius < b.radius else (b, a)
                dr = hi.radius - lo.radius
                if dr <= 1e-9:
                    continue
                out.append(_Candidate(inner=lo, outer=hi, delta_r=dr))
    return out


# ---------------------------------------------------------------------------
# step 2: thickness
# ---------------------------------------------------------------------------


def _cluster(values: list[float], tol: Tolerance) -> list[list[float]]:
    clusters: list[list[float]] = []
    for v in sorted(values):
        for c in clusters:
            if tol.radius_close(c[0], v):
                c.append(v)
                break
        else:
            clusters.append([v])
    return clusters


def planar_pair_distances(planes: list[FaceInfo], tol: Tolerance) -> list[float]:
    """Distances between antiparallel planar faces -- thickness candidates."""
    out = []
    for i, a in enumerate(planes):
        for b in planes[i + 1:]:
            if not geom.antiparallel(a.normal, b.normal, tol.angular):
                continue
            d = abs(geom.signed_plane_distance(b.point, a.point, a.normal))
            if d > 1e-9:
                out.append(d)
    return out


def infer_thickness(
    candidates: list[_Candidate],
    planes: list[FaceInfo],
    tol: Tolerance,
    part: Part | None = None,
) -> float:
    """Infer sheet thickness, preferring the evidence from the bends.

    Every coaxial pair contributes an independent measurement of the thickness,
    so on a part with several bends this is heavily over-determined. If the
    bends disagree with each other the part is not uniform-gauge sheet metal and
    we fail loudly rather than produce a confidently wrong flat.
    """
    if candidates:
        clusters = _cluster([c.delta_r for c in candidates], tol)

        # Throw out anything that is not a plausible sheet BEFORE ranking, not
        # after. A cluster of implausible values can easily be the biggest one
        # -- a part with many identical bosses gives many identical parallel
        # pairs -- so filtering afterwards would already have lost.
        plausible = [c for c in clusters if thickness_plausible(np.mean(c), tol)]
        rejected = [c for c in clusters if c not in plausible]
        if rejected and part is not None:
            values = ", ".join(f"{np.mean(c):.3f}" for c in rejected[:6])
            LOGGER.info("thickness candidates rejected as implausible: %s", values)

        if not plausible:
            found = ", ".join(f"{np.mean(c):.3f}" for c in clusters[:6])
            raise DetectionError(
                f"no plausible sheet thickness. Concentric face pairs differ by "
                f"{found} mm, and sheet metal must be between "
                f"{tol.min_thickness:g} and {tol.max_thickness:g} mm "
                f"({tol.max_thickness / 25.4:.2f} in). Either this is not a "
                f"brake-formed part, or the model units are wrong -- or pick "
                f"the sheet face by hand in the 3D view."
            )

        clusters = plausible
        clusters.sort(key=len, reverse=True)
        best = clusters[0]
        t = float(np.mean(best))

        if len(clusters) > 1 and len(clusters[1]) == len(best):
            raise DetectionError(
                "ambiguous thickness: coaxial face pairs suggest both "
                f"{np.mean(best):.4f} and {np.mean(clusters[1]):.4f} mm with equal "
                "support. Pass an explicit thickness."
            )

        # Cross-check against the planar evidence.
        pdists = planar_pair_distances(planes, tol)
        if pdists and part is not None:
            if not any(tol.radius_close(d, t) for d in pdists):
                part.warn(
                    f"thickness {t:.4f} mm inferred from bend radii is not confirmed "
                    f"by any pair of parallel faces. Check that the part is uniform gauge."
                )
        return t

    # No bends at all -- a FLAT BLANK, which is a perfectly ordinary thing to
    # want to open. The thickness has to come from the two skins instead.
    #
    # The tempting one-liner here is min(planar_pair_distances), and it is
    # wrong: on a blank with a 3 mm slot in 5 mm material the two walls of the
    # slot are antiparallel and 3 mm apart, so it reports 3 mm. What rules the
    # slot out is that its material lies OUTSIDE the pair, not between them.
    # See `features.thickness_candidates`.
    cands = [
        (d, w) for d, w in thickness_candidates([f.to_desc() for f in planes], tol)
        if thickness_plausible(d, tol)
    ]
    if cands:
        t = float(cands[0][0])
        if len(cands) > 1 and cands[1][1] > 0.6 * cands[0][1]:
            LOGGER.warning(
                "thickness is ambiguous: %.4f mm and %.4f mm have comparable "
                "support", cands[0][0], cands[1][0],
            )
            if part is not None:
                part.warn(
                    f"thickness could be {cands[0][0]:.4f} mm or "
                    f"{cands[1][0]:.4f} mm -- both are supported by similar face "
                    f"area. Set it explicitly if the value shown looks wrong."
                )
        return t

    pdists = planar_pair_distances(planes, tol)
    if not pdists:
        raise DetectionError(
            "cannot infer thickness: the part has no coaxial cylinder pairs and no "
            "parallel planar faces. Is this a sheet metal part?"
        )
    return float(min(pdists))


# ---------------------------------------------------------------------------
# step 3: classify pairs as bends
# ---------------------------------------------------------------------------


@dataclass
class TangentPlane:
    """One DISTINCT plane a cylinder runs tangentially into.

    Distinct plane, not distinct face. A flange whose face is split by a cutout
    presents several coplanar faces to the same bend; treating those as separate
    sides made a bend look like it joined a flange to itself, and it was thrown
    away. Grouping coplanar faces first is what fixes that.
    """

    normal: np.ndarray
    point: np.ndarray
    faces: list[FaceInfo]

    @property
    def area(self) -> float:
        return sum(f.area for f in self.faces)

    @property
    def primary(self) -> FaceInfo:
        """The largest face on this plane -- the one to reference the flange by."""
        return max(self.faces, key=lambda f: f.area)


def tangent_planes(
    cyl: MergedCylinder, planes: dict[int, FaceInfo], adj: dict[int, set[int]],
    tol: Tolerance,
) -> list[TangentPlane]:
    """Distinct planes this cylinder runs tangentially into, largest first."""
    neighbours: set[int] = set()
    for f in cyl.faces:
        neighbours |= adj.get(f.index, set())

    # Tangency is a distance comparison against the radius, so the window has to
    # scale with the radius. A fixed 1e-4 mm is 0.1 micron -- fine for natively
    # exported geometry, too tight for STEP that has been through a translator.
    lin_tol = max(tol.tangent, tol.tangent_rel * cyl.radius)

    found: list[TangentPlane] = []
    for idx in sorted(neighbours):
        p = planes.get(idx)
        if p is None:
            continue
        if not geom.cylinder_tangent_to_plane(
            cyl.axis_point, cyl.axis_dir, cyl.radius, p.point, p.normal,
            ang_tol=tol.tangent_angular, lin_tol=lin_tol,
        ):
            continue

        for tp in found:
            same_dir = geom.codirectional(tp.normal, p.normal, tol.tangent_angular)
            coplanar = abs(
                geom.signed_plane_distance(p.point, tp.point, tp.normal)
            ) < lin_tol
            if same_dir and coplanar:
                tp.faces.append(p)
                break
        else:
            found.append(TangentPlane(normal=p.normal, point=p.point, faces=[p]))

    found.sort(key=lambda t: -t.area)
    return found


@dataclass
class Funnel:
    """Why each candidate was or was not accepted as a bend.

    Detection is a series of gates, and when it finds nothing the only question
    that matters is WHICH gate ate the candidates. Guessing at that from the
    outside is miserable, so every rejection is recorded here with the actual
    numbers and rendered on demand.
    """

    faces: int = 0
    planes: int = 0
    cylinders: int = 0
    merged: int = 0
    pairs: int = 0

    thickness_used: float = 0.0
    rejected_thickness: list[float] = field(default_factory=list)
    rejected_sweep: list[float] = field(default_factory=list)
    rejected_tangency: list[tuple[float, int, int]] = field(default_factory=list)
    accepted: int = 0

    def report(self) -> str:
        lines = [
            "detection funnel",
            f"  faces                     {self.faces}",
            f"    planar                  {self.planes}",
            f"    cylindrical             {self.cylinders}",
            f"  merged into bands         {self.merged}",
            f"  concentric pairs          {self.pairs}",
            f"  thickness used            {self.thickness_used:.4f} mm",
        ]
        if self.rejected_thickness:
            vals = sorted(set(round(v, 4) for v in self.rejected_thickness))
            shown = ", ".join(f"{v:g}" for v in vals[:8])
            lines.append(
                f"  REJECTED, dr != thickness {len(self.rejected_thickness)}"
                f"   dr values: {shown}"
            )
        if self.rejected_sweep:
            lines.append(
                f"  REJECTED, sweep too small {len(self.rejected_sweep)}"
            )
        if self.rejected_tangency:
            lines.append(
                f"  REJECTED, not tangent to  {len(self.rejected_tangency)}"
                "   two planes on both surfaces"
            )
            for r, ni, no in self.rejected_tangency[:6]:
                lines.append(
                    f"      R{r:.3f}: inner touched {ni} plane(s), "
                    f"outer touched {no}"
                )
        lines.append(f"  ACCEPTED as bends         {self.accepted}")
        return "\n".join(lines)

    def diagnosis(self) -> str | None:
        """Plain-language guess at the cause when nothing was accepted."""
        if self.accepted:
            return None
        if self.cylinders == 0:
            return (
                "No cylindrical faces at all. Either this is not a bent sheet "
                "metal part, or the bends were modelled as something other than "
                "cylinders."
            )
        if self.pairs == 0:
            return (
                "Cylindrical faces exist but none are CONCENTRIC PAIRS. A brake "
                "bend has an inner and an outer cylinder sharing an axis. If the "
                "model has only one surface per bend it may be a surface/midplane "
                "model rather than a solid, or the bends may be single-sided."
            )
        if self.rejected_thickness and not self.rejected_tangency:
            vals = sorted(set(round(v, 4) for v in self.rejected_thickness))
            return (
                f"Every concentric pair was rejected because its radius "
                f"difference did not match the thickness of "
                f"{self.thickness_used:.4f} mm. The differences actually found "
                f"were {', '.join(f'{v:g}' for v in vals[:8])} mm. If one of "
                f"those is your real material thickness, set it explicitly "
                f"(--thickness, or untick 'Infer from geometry') -- and check "
                f"whether the model is in inches."
            )
        if self.rejected_tangency:
            return (
                "Concentric pairs of the right thickness exist, but they are not "
                "tangent to two distinct planes on both surfaces. That usually "
                "means the flanges are not planar where they meet the bend, or "
                "the geometry is loose enough to fail the tangency test."
            )
        return "No candidates survived; see the funnel above."


def classify_bends(
    candidates: list[_Candidate],
    thickness: float,
    planes: dict[int, FaceInfo],
    adj: dict[int, set[int]],
    tol: Tolerance,
    part: Part,
    funnel: Funnel | None = None,
) -> list[tuple[_Candidate, list[TangentPlane], list[TangentPlane]]]:
    """Keep only the coaxial pairs that are genuinely brake bends."""
    kept = []
    funnel = funnel or Funnel()
    funnel.thickness_used = thickness
    min_sweep = math.radians(tol.min_bend_angle_deg)
    max_sweep = math.radians(tol.max_air_bend_deg)

    for c in candidates:
        if not tol.radius_close(c.delta_r, thickness):
            funnel.rejected_thickness.append(c.delta_r)
            continue
        if c.inner.sweep < min_sweep:
            funnel.rejected_sweep.append(c.inner.sweep)
            continue

        inner_tp = tangent_planes(c.inner, planes, adj, tol)
        outer_tp = tangent_planes(c.outer, planes, adj, tol)

        # A bend runs into one flange on each side, on both surfaces. Counting
        # distinct PLANES rather than faces is what makes this survive a flange
        # whose face has been split by a cutout.
        if len(inner_tp) < 2 or len(outer_tp) < 2:
            funnel.rejected_tangency.append(
                (c.inner.radius, len(inner_tp), len(outer_tp))
            )
            continue

        if len(inner_tp) > 2:
            part.warn(
                f"bend at R{c.inner.radius:.3f} touches {len(inner_tp)} distinct "
                f"planes; using the two largest."
            )

        if abs(c.inner.sweep - c.outer.sweep) > math.radians(1.0):
            part.warn(
                f"bend at radius {c.inner.radius:.3f}: inner and outer sweeps differ "
                f"({math.degrees(c.inner.sweep):.2f} vs "
                f"{math.degrees(c.outer.sweep):.2f} deg). Using the inner face."
            )

        if c.inner.sweep > max_sweep:
            part.warn(
                f"a {math.degrees(c.inner.sweep):.1f} deg bend at radius "
                f"{c.inner.radius:.3f} mm exceeds the air-bend limit. It is a hem or "
                f"curl; this version unfolds it with the standard K-factor formula, "
                f"which does NOT hold for crushed material. Verify against a sample."
            )

        funnel.accepted += 1
        kept.append((c, inner_tp[:2], outer_tp[:2]))
    return kept


# ---------------------------------------------------------------------------
# step 4: flange pairing
# ---------------------------------------------------------------------------


class _UnionFind:
    def __init__(self):
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def pair_flange_faces(
    bends_info, thickness: float, tol: Tolerance
) -> _UnionFind:
    """Match each flange's two faces using the bends themselves.

    A generic search over all parallel face pairs is fragile on L-shaped and
    cut-out flanges. The bends give it to us almost for free: for one bend, the
    inner cylinder touches two planes and the outer touches two more, and each
    inner plane has exactly one antiparallel partner one thickness away among
    the outer pair. Two candidates, so the match is essentially unambiguous.
    """
    uf = _UnionFind()
    for _c, inner_tp, outer_tp in bends_info:
        for inner in inner_tp:
            ip = inner.primary

            # EVERY coplanar face on this plane, not just the largest.
            #
            # This is the fix for two tabs sharing one bend line. A plane can
            # carry several DISCONNECTED regions of sheet -- two tabs bent up
            # from the same bend, a flange split by a relief cut -- and they are
            # all one flange for unfolding purposes, because being coplanar and
            # on the same side of the same bend means they all take the same
            # rigid transform.
            #
            # Taking only `primary` here meant the smaller tab never entered the
            # union-find at all, so it was not part of any flange and simply
            # never appeared in the flat pattern. The bend was found, the big
            # tab was found, and the second one vanished silently.
            for f in inner.faces:
                uf.find(f.index)
                if f.index != ip.index:
                    uf.union(ip.index, f.index)

            for outer in outer_tp:
                op = outer.primary
                if not geom.antiparallel(ip.normal, op.normal, tol.angular):
                    continue
                d = abs(geom.signed_plane_distance(op.point, ip.point, ip.normal))
                if not tol.radius_close(d, thickness):
                    continue
                # Join the whole opposite plane too: same argument, other side
                # of the material.
                for f in outer.faces:
                    uf.union(ip.index, f.index)
                break
    return uf


def pair_leftovers(
    uf: _UnionFind, planes: dict[int, FaceInfo], thickness: float, tol: Tolerance
) -> None:
    """Generic fallback for flange faces no bend told us about."""
    groups: dict[int, list[int]] = {}
    for idx in list(uf.parent):
        groups.setdefault(uf.find(idx), []).append(idx)
    singles = [g[0] for g in groups.values() if len(g) == 1]

    for idx in singles:
        a = planes[idx]
        best, best_d = None, math.inf
        for jdx, b in planes.items():
            if jdx == idx or uf.find(jdx) == uf.find(idx):
                continue
            if not geom.antiparallel(a.normal, b.normal, tol.angular):
                continue
            d = abs(geom.signed_plane_distance(b.point, a.point, a.normal))
            if tol.radius_close(d, thickness) and abs(d - thickness) < best_d:
                best, best_d = jdx, abs(d - thickness)
        if best is not None:
            uf.union(idx, best)


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------


def detect(
    shape, opts: UnfoldOptions, stage: Stage | None = None,
    file_units: str | None = None, loop_cache: dict | None = None,
) -> Part:
    """Recognise the sheet metal features of a solid.

    `stage` reports progress. Every step that can take real time on a real part
    announces itself with a face count and a timing, so a stall names its own
    cause instead of looking like a hang.
    """
    stage = stage or null_stage()
    tol = opts.tol

    funnel = Funnel()

    stage("Reading faces...")
    infos = describe_all(shape)
    kinds: dict[str, int] = {}
    for f in infos:
        kinds[f.kind] = kinds.get(f.kind, 0) + 1
    stage.detail(
        f"{len(infos)} faces: "
        + ", ".join(f"{n} {k}" for k, n in sorted(kinds.items()))
    )
    stage.done()

    stage.check_cancelled()
    stage("Mapping face adjacency...")
    adj = adjacent_faces(shape, infos)
    stage.done()

    planes = {
        f.index: f for f in infos
        if f.kind == "plane" and f.area > tol.min_face_area
    }
    cylinders = [
        f for f in infos
        if f.kind == "cylinder" and f.area > tol.min_face_area
    ]
    stage.detail(f"{len(planes)} planar faces, {len(cylinders)} cylindrical")
    funnel.faces = len(infos)
    funnel.planes = len(planes)
    funnel.cylinders = len(cylinders)

    part = Part(thickness=0.0, material=opts.material)
    part.file_units = file_units
    part.funnel = funnel
    part.solid_bent = shape
    part.volume = solid_volume(shape)
    # Kept so the 3D view does not have to interrogate every face a second
    # time; describe_all is the single most expensive read in the pipeline.
    part.face_infos = infos
    # Shared with the 3D view so face outlines are traced once, not twice.
    part.loop_cache = loop_cache if loop_cache is not None else {}

    # NOTE: v5 warned here on every cone, torus and spline it saw, which put a
    # scary paragraph about undevelopable geometry on the screen for a part
    # whose only sin was a countersunk hole. Cones are now classified properly
    # further down -- see the feature pass -- and only genuinely non-developable
    # surfaces get a warning.

    stage.check_cancelled()
    stage("Merging split bend faces...")
    merged = merge_coaxial(cylinders, tol)
    funnel.merged = len(merged)
    stage.done(f"{len(cylinders)} -> {len(merged)} logical bands")
    if len(merged) < len(cylinders):
        part.note(
            f"merged {len(cylinders)} cylindrical faces into {len(merged)} logical "
            f"bands (holes or cutouts crossing a bend split its face)."
        )

    stage.check_cancelled()
    stage("Pairing concentric faces...")
    candidates = coaxial_pairs(merged, tol)
    funnel.pairs = len(candidates)
    stage.done(f"{len(candidates)} coaxial pair(s)")

    stage.check_cancelled()
    stage("Inferring thickness...")
    # MEASURED thickness. Classification must use this, never a snapped value.
    #
    # An earlier version snapped to the nearest standard gauge here, then tested
    # candidates against the snapped number. Snapping was allowed to move the
    # value by up to 2% while the match tolerance was 1%, so any part landing in
    # that window had EVERY bend rejected. An inch-gauge sheet (1.905 mm snapping
    # to 2.0, a 5% move) failed completely and silently. The nominal gauge is
    # cosmetic; it is recorded for reporting and kept well away from the geometry.
    thickness = None

    # A face the user pointed at wins over anything inferred. They can see the
    # model; the inference cannot, and on a part with a boss or a weld prep
    # there is genuinely more than one defensible answer.
    if opts.sheet_face is not None:
        picked = thickness_from_face(
            opts.sheet_face, [f.to_desc() for f in planes.values()], tol,
            cap=False,
        )
        if picked is not None:
            thickness, partner = picked
            part.sheet_face = opts.sheet_face
            part.note(
                f"thickness {thickness:.4f} mm measured from the face you "
                f"selected (face {opts.sheet_face}) to face {partner}. The "
                f"automatic inference was overridden."
            )
            if not thickness_plausible(thickness, tol):
                part.warn(
                    f"the selected face gives a thickness of {thickness:.3f} mm "
                    f"({thickness / 25.4:.3f} in), which is outside the range "
                    f"this tool treats as sheet metal. It has been used as "
                    f"asked, but check it is the face you meant."
                )
        else:
            part.warn(
                f"face {opts.sheet_face} was selected as the sheet, but nothing "
                f"lies parallel behind it to measure a thickness against. "
                f"Falling back to automatic detection -- try the face on the "
                f"opposite side, or a larger one."
            )

    if thickness is None:
        thickness = opts.thickness or infer_thickness(
            candidates, list(planes.values()), tol, part
        )
    part.thickness = thickness
    part.measured_thickness = thickness

    if opts.thickness is None:
        # Match against stock from the right tradition. A 0.075 inch sheet is
        # 1.905 mm; measured against a metric stock list the nearest entry is
        # 2.0 mm and it looks like a 5% error, when in fact it is exactly a
        # standard imperial gauge.
        from ..units import (
            choose_display_unit, nearest_gauge_number_in, standard_gauges_mm_for,
        )

        unit = choose_display_unit(opts.display_units, part.file_units)
        stock = standard_gauges_mm_for(unit)
        snapped = min(stock, key=lambda g: abs(g - thickness))
        shift = abs(snapped - thickness) / thickness if thickness else 0.0
        part.nominal_thickness = snapped

        if unit.name == "in":
            gauge_no = nearest_gauge_number_in(thickness)
            label = (f'{snapped / 25.4:.4f}"'
                     + (f" (~{gauge_no} ga)" if gauge_no else ""))
        else:
            label = f"{snapped:.3f} mm"

        if shift > opts.gauge_snap_warn:
            part.warn(
                f"measured thickness {thickness:.4f} mm is {shift * 100:.1f}% from the "
                f"nearest standard gauge ({label}). Verify the model units and that "
                f"the part is uniform gauge."
            )
        elif abs(snapped - thickness) > 1e-9:
            part.note(
                f"measured thickness {thickness:.4f} mm; nearest stock {label}. "
                f"Detection and allowances use the MEASURED value."
            )
    stage.done(f"t = {thickness:.4f} mm (measured)")

    stage.check_cancelled()
    stage("Classifying bends...")
    bends_info = classify_bends(
        candidates, thickness, planes, adj, tol, part, funnel
    )
    stage.done(f"{len(bends_info)} bend(s)")

    # --- classify every other face -------------------------------------
    #
    # This is the pass that lets a real production part through. Up to v5 the
    # detector answered one question -- "is this a bend?" -- and shrugged at
    # everything else, with a blanket warning for any surface it did not
    # recognise. A countersunk hole is a cone, so a perfectly correct flat
    # pattern arrived with a paragraph attached saying it was probably wrong.
    #
    # Now every face gets a role, and only the ones that genuinely have no
    # development are worth warning about. See `smt.features`.
    stage.check_cancelled()
    stage("Classifying features...")

    bend_roles: dict[int, Role] = {}
    for c, _i, _o in bends_info:
        for f in c.inner.faces:
            bend_roles[f.index] = Role.BEND_INNER
        for f in c.outer.faces:
            bend_roles[f.index] = Role.BEND_OUTER

    descs = [f.to_desc() for f in infos]
    part.features = classify(descs, adj, thickness, tol, bend_roles)

    from ..features import counts_by_role, group_features, undevelopable

    counts = counts_by_role(part.features)
    part.feature_groups = group_features(
        part.features,
        roles=(Role.HOLE, Role.SLOT_END, Role.COUNTERSINK, Role.COUNTERBORE,
               Role.CHAMFER, Role.CORNER_RADIUS, Role.FILLET),
    )
    stage.detail(
        ", ".join(
            f"{n} {r.value.replace('_', ' ')}"
            for r, n in sorted(counts.items(), key=lambda kv: -kv[1])
        )
    )

    for f in undevelopable(part.features):
        part.warn(
            f"face {f.index}: {f.reason}. Formed features -- domes, louvers, "
            f"embosses -- are stretched, not bent, and have no exact flat "
            f"pattern. The blank will be wrong across that region."
        )

    # An unrecognised cylinder lying in the sheet plane is the shape of a bend
    # that failed a gate. That is worth surfacing loudly, because it is the one
    # misclassification that silently changes the blank length.
    for f in part.features.values():
        if f.role is Role.UNKNOWN and "unrecognised bend" in f.reason:
            part.warn(
                f"face {f.index}: {f.reason}. If that is a bend, the flat "
                f"pattern is short by its allowance -- check the detection "
                f"funnel in the Activity log."
            )
    stage.done()

    # --- sheet faces, and only sheet faces, may become flanges ----------
    #
    # `planes` holds EVERY planar face: chamfer bevels, cut edges, counterbore
    # floors. Letting those into flange pairing is how a chamfer two thicknesses
    # from another chamfer becomes a flange that does not exist. Only faces the
    # feature pass identified as sheet skin are eligible.
    sheet_planes = {
        idx: f for idx, f in planes.items()
        if part.features.get(idx) is not None
        and part.features[idx].role is Role.SHEET
    }
    if not sheet_planes:
        sheet_planes = planes
        LOGGER.warning(
            "no face was identified as sheet skin; falling back to all %d "
            "planar faces for flange pairing", len(planes),
        )

    if not bends_info:
        # Say WHY, with numbers, rather than shrugging.
        for line in funnel.report().splitlines():
            stage.detail(line)
        LOGGER.info("no bends detected\n%s", funnel.report())

        if sheet_planes and len(sheet_planes) >= 2:
            # A FLAT BLANK. This is an ordinary thing to open -- you still want
            # the size, the thickness, the hole count and a DXF -- so it is a
            # note, not a warning. v5 treated it as a failure.
            part.note(
                "no bends: this is a flat blank. The flat pattern is the part "
                "itself, and every dimension below is measured directly."
            )
            part.is_blank = True
        else:
            why = funnel.diagnosis()
            part.warn("No bends detected. " + (why or ""))

    stage.check_cancelled()
    stage("Pairing flange faces...")
    uf = pair_flange_faces(bends_info, thickness, tol)
    pair_leftovers(uf, sheet_planes, thickness, tol)

    # --- build flange nodes --------------------------------------------
    flange_of_face: dict[int, int] = {}
    groups: dict[int, list[int]] = {}
    for idx in list(uf.parent):
        groups.setdefault(uf.find(idx), []).append(idx)

    if not groups:
        # Nothing was paired, because nothing was paired BY A BEND. On a flat
        # blank that is the expected outcome, and the sheet pairs found during
        # feature classification already say exactly which faces are the two
        # skins -- so use them directly rather than reporting no geometry.
        pairs = sheet_plane_pairs(
            [f.to_desc() for f in sheet_planes.values()], thickness, tol
        )
        for a_idx, b_idx in pairs:
            groups[a_idx] = [a_idx, b_idx]
        if groups:
            stage.detail(
                f"flat blank: {len(groups)} sheet region(s) from the skin pairs"
            )
        else:
            raise DetectionError(
                "no flange could be identified. The part has no bends and no "
                "pair of planar faces one thickness apart facing each other, "
                "so there is nothing recognisable as sheet."
            )

    stage.done(f"{len(groups)} flange(s)")

    stage.check_cancelled()
    stage("Extracting flange outlines...")
    for fid, (_root, members) in enumerate(sorted(groups.items())):
        members = sorted(members, key=lambda i: -planes[i].area)
        a = planes[members[0]]

        # Split the group by which SIDE of the material each face is on, and
        # keep them all. A flange is not necessarily one face per side: two
        # tabs on the same bend line, or a flange cut into separate regions,
        # give several coplanar faces that all move together.
        side_a = [
            i for i in members
            if geom.codirectional(planes[i].normal, a.normal, tol.tangent_angular)
        ]
        side_b = [
            i for i in members
            if geom.antiparallel(planes[i].normal, a.normal, tol.tangent_angular)
        ]
        b = planes[side_b[0]] if side_b else None

        # Loops are cached as a list of REGIONS, one per face, each of which is
        # [outer wire, *holes]. Flattening them into a single list would make
        # the second region's outer wire look like a hole in the first.
        regions_a = [
            face_loops_safe(planes[i].face, i, cache=part.loop_cache)
            for i in side_a
        ]
        regions_b = [
            face_loops_safe(planes[i].face, i, cache=part.loop_cache)
            for i in side_b
        ]

        total_area = sum(planes[i].area for i in side_a) or a.area
        centroid = (
            sum(planes[i].centroid * planes[i].area for i in side_a) / total_area
            if total_area > 0 else a.centroid
        )

        part.flanges[fid] = Flange(
            id=fid,
            normal=a.normal,
            origin=a.point,
            origin_b=b.point if b is not None else None,
            centroid=centroid,
            area=total_area,
            face=a.face,
            partner_face=b.face if b is not None else None,
            faces_a=[planes[i].face for i in side_a],
            faces_b=[planes[i].face for i in side_b],
            # Cached now, while the kernel is already loaded. This is what lets
            # the GUI redraw at any K without touching OCC.
            loops_a=[r for r in regions_a if r],
            loops_b=[r for r in regions_b if r] or None,
        )
        for m in members:
            flange_of_face[m] = fid

        if len(side_a) > 1:
            part.note(
                f"flange {fid} is {len(side_a)} separate regions of sheet on "
                f"the same plane -- tabs sharing a bend line, or a flange split "
                f"by a relief. All of them are in the flat pattern."
            )
        if b is None:
            part.warn(
                f"flange {fid} has no matching opposite face; the part may not be "
                f"uniform gauge in that region."
            )

    stage.done()

    # --- build bends ----------------------------------------------------
    stage.check_cancelled()
    stage("Building bend records...")
    for bid, (c, inner_tp, _outer_tp) in enumerate(bends_info):
        fa = flange_of_face.get(inner_tp[0].primary.index)
        fb = flange_of_face.get(inner_tp[1].primary.index)
        if fa is None or fb is None or fa == fb:
            part.warn(
                f"skipping a bend at radius {c.inner.radius:.3f} mm: could not resolve "
                f"the two distinct flanges it connects."
            )
            continue
        part.bends[bid] = Bend(
            id=bid,
            axis_point=c.inner.axis_point,
            axis_dir=c.inner.axis_dir,
            r_inner=c.inner.radius,
            thickness=thickness,
            angle=c.inner.sweep,
            length_envelope=c.inner.length_envelope,
            length_material=c.inner.length_material,
            flange_a=fa,
            flange_b=fb,
            v_min=c.inner.v_min,
            v_max=c.inner.v_max,
            kind=BendKind.AIR,
            inner_face=c.inner,
            outer_face=c.outer,
            axis_x_dir=c.inner.faces[0].x_dir,
            u_min=c.inner.u_min,
            u_max=c.inner.u_max,
            # The real trimmed boundary of every piece of the bend face. A
            # cutout crossing the bend splits it, so take them all.
            inner_loops=[
                loop
                for f in c.inner.faces
                for loop in face_loops_safe(f.face, f.index, cache=part.loop_cache)
            ],
        )

    stage.done()

    # --- holes ----------------------------------------------------------
    # Skipped in simple mode. The flange outlines already carry their hole
    # wires, so the preview and the DXF still show every cutout -- this only
    # skips building the separate Hole records used for pierce counts and the
    # hole-near-bend warning.
    if opts.simple_mode:
        stage.detail("simple mode: skipping hole analysis")
        return part

    stage("Finding holes...")
    # Driven by the feature roles rather than by a second, independent guess.
    # v5 re-derived "is this a hole?" here from the sweep angle alone, which
    # meant a slot end and a corner radius were both silently dropped and the
    # pierce count was wrong on any part with a slot in it.
    bend_cyls = {id(c.inner) for c, _, _ in bends_info} | {
        id(c.outer) for c, _, _ in bends_info
    }
    for hid, m in enumerate(merged):
        if id(m) in bend_cyls:
            continue
        role = None
        for f in m.faces:
            feat = part.features.get(f.index)
            if feat is not None:
                role = feat.role
                break
        if role not in (Role.HOLE, Role.SLOT_END):
            continue
        part.holes.append(
            Hole(
                id=hid,
                centre=m.centroid,
                diameter=2.0 * m.radius if role is Role.HOLE else None,
                perimeter=m.radius * m.sweep,
            )
        )
    stage.done(f"{len(part.holes)} pierced feature(s)")

    return part


def classify_shape(shape, thickness: float, tol, infos=None):
    """Roles for every face of a shape, with no bend detection at all.

    Used on the UNFOLDED solid. Once a part is flat its bends are gone, so
    running the full detector over it would find nothing and say so loudly; but
    the countersinks, chamfers and slots are all still there and still want
    colouring in the 3D view. This is the classifier on its own.

    Returns (roles, infos).
    """
    from ..features import classify

    if infos is None:
        infos = describe_all(shape)
    adj = adjacent_faces(shape, infos)
    features = classify([f.to_desc() for f in infos], adj, thickness, tol)
    return {i: f.role for i, f in features.items()}, infos
