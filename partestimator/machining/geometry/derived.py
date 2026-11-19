"""Derived features for machine-learning cycle-time models.

This module turns the raw geometry results into a flat feature row. The
design rules behind what's here (and what deliberately isn't):

  * NOTHING POSITIONAL. Absolute centroid, bounding-box axes and any world
    coordinate are artifacts of where the modeller put the origin and how
    the part happened to be oriented when saved. Two identical parts can
    differ in all of them, so they are pure noise to a similarity model.

  * NO RAW TOPOLOGY COUNTS as complexity measures. Face and edge counts are
    partly an exporter artifact -- a single bore is one cylindrical face in
    some files and two half-cylinders in others (this codebase had to be
    fixed for exactly that). They masquerade as difficulty measures while
    encoding authoring noise.

  * ABSOLUTE QUANTITIES WHERE THEY MULTIPLY. Areas are emitted in square
    inches rather than as fractions, because finishing time is
    area / (stepover x feed) -- the absolute value is what the rate
    multiplies. Handing the model the product directly saves it having to
    learn a multiplication, which matters at realistic sample sizes.
    A few dimensionless ratios are kept where they describe the part's
    character independently of size (solidity, removal ratio, aspect
    ratios, A/V^(2/3)).

  * NO EXACT DUPLICATES. Anything derivable by a single arithmetic step
    from other exported columns has been removed: weights (volume x
    density), hull volume (implied by solidity), area fractions (implied
    by absolute areas and the total), and every "x / bbox diagonal" or
    "count / total" normalisation.

  * DIMENSIONS SORTED. Bounding-box extents are used largest-to-smallest
    internally so aspect ratios don't depend on which axis the modeller
    called X. The raw extents are NOT exported -- stock dimensions carry
    the same envelope information in the form that is actually bought.

UNITS: every dimensional feature is emitted in IMPERIAL -- inches, in^2,
in^3 -- with the unit in the key name (`_in`, `_in2`, `_in3`). Internally
the geometry pipeline works in millimetres; conversion happens here, at
the boundary. Dimensionless features (ratios, counts, angles in degrees)
carry no length suffix.

QUALITY FLAGS (`hull_is_approximate`, `rejected_hole_count`) are NOT
features -- they describe how much to trust the row. They are emitted by
`export.py` as metadata columns so suspect rows can be filtered during
training rather than silently skewing a fit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import axis_stats, features, materials
from . import stock as stock_mod

MM_PER_IN = 25.4
MM2_PER_IN2 = MM_PER_IN ** 2
MM3_PER_IN3 = MM_PER_IN ** 3


def _in(mm: float) -> float:
    return float(mm) / MM_PER_IN


def _in2(mm2: float) -> float:
    return float(mm2) / MM2_PER_IN2


def _in3(mm3: float) -> float:
    return float(mm3) / MM3_PER_IN3

# Hole-axis orientation counting lives in geometry.axis_stats, which
# reports counts at several tolerances at once. A single fixed tolerance
# was removed from here on purpose: on contoured work the answer depends
# entirely on the tolerance, so publishing one number would be misleading.
#
# Planar faces group into one "level" within these tolerances. The angular
# tolerance is deliberately fine: on contoured work, faces a fraction of a
# degree apart are genuinely different tool orientations. The OFFSET
# tolerance is what usually decides whether two faces are the same level.
PLANE_NORMAL_TOLERANCE_RAD = np.radians(0.1)
PLANE_OFFSET_TOLERANCE_MM = 0.05
# Co-axial features must share an axis line to this accuracy.
COAXIAL_ANGLE_TOLERANCE_RAD = np.radians(1.0)
COAXIAL_OFFSET_TOLERANCE_MM = 0.05
# A cone whose small end tapers essentially to a point is a drill tip.
DRILL_POINT_RADIUS_FRACTION = 0.25
# Radius thresholds for "how much surface needs a small cutter", in inches
# and named in fractional form since that's how end mills are bought.
SMALL_RADIUS_THRESHOLDS_IN = ((1.0 / 32.0, "1_32"), (1.0 / 16.0, "1_16"),
                               (1.0 / 8.0, "1_8"), (1.0 / 4.0, "1_4"))


@dataclass
class DerivedFeatures:
    values: dict = field(default_factory=dict)

    def __getitem__(self, key):
        return self.values[key]

    def get(self, key, default=None):
        return self.values.get(key, default)


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _same_axis_line(p1, d1, p2, d2,
                     angle_tol=COAXIAL_ANGLE_TOLERANCE_RAD,
                     offset_tol=COAXIAL_OFFSET_TOLERANCE_MM) -> bool:
    """Do two (point, direction) pairs describe the same infinite line?
    Direction-insensitive, so opposite-pointing axes still match."""
    d1 = _unit(d1)
    d2 = _unit(d2)
    if float(np.linalg.norm(np.cross(d1, d2))) > np.sin(angle_tol):
        return False
    delta = np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float)
    perpendicular = delta - np.dot(delta, d1) * d1
    return float(np.linalg.norm(perpendicular)) <= offset_tol


def count_planar_levels(planes, normal_tol=PLANE_NORMAL_TOLERANCE_RAD,
                         offset_tol=PLANE_OFFSET_TOLERANCE_MM):
    """Groups planar faces into distinct (normal, offset) levels.

    The number of levels sharing the dominant normal is a cheap proxy for
    how many Z-steps a roughing strategy needs -- something raw volume
    can't reveal. Returns (total_levels, levels_on_dominant_normal).
    """
    if not planes:
        return 0, 0

    groups: list[dict] = []
    cos_tol = np.cos(normal_tol)
    for p in planes:
        n = _unit(p.normal)
        offset = float(p.offset_mm)
        placed = False
        for g in groups:
            dot = float(np.dot(g["normal"], n))
            if abs(dot) >= cos_tol:
                # Flip the offset too if the normal is anti-parallel.
                eff = offset if dot > 0 else -offset
                if abs(eff - g["offset"]) <= offset_tol:
                    g["area"] += p.area_mm2
                    placed = True
                    break
        if not placed:
            groups.append({"normal": n, "offset": offset, "area": p.area_mm2})

    # Which normal direction carries the most area?
    dir_area: list[list] = []
    for g in groups:
        placed = False
        for d in dir_area:
            if abs(float(np.dot(d[0], g["normal"]))) >= cos_tol:
                d[1] += g["area"]
                d[2] += 1
                placed = True
                break
        if not placed:
            dir_area.append([g["normal"], g["area"], 1])
    dominant = max(dir_area, key=lambda d: d[1]) if dir_area else None
    return len(groups), (dominant[2] if dominant else 0)


def classify_holes(holes, cones):
    """Labels each hole using co-axial neighbours.

    * blind      -- a co-axial cone tapering to a point sits at one end
                    (the drill tip a twist drill leaves behind)
    * countersunk-- a co-axial cone flaring wider than the bore
    * counterbored -- a co-axial bore of larger radius, axially adjacent

    Anything with no capping cone is reported as through. That is a
    heuristic: a flat-bottomed blind hole (end-milled, or a flat-bottom
    drill) leaves a planar cap rather than a cone and will be missed here.
    """
    n = len(holes)
    blind = [False] * n
    countersunk = [False] * n
    counterbored = [False] * n

    for i, h in enumerate(holes):
        if h.axis_direction is None:
            continue
        for c in cones:
            if not _same_axis_line(h.axis_point, h.axis_direction, c.axis_point, c.axis_direction):
                continue
            if c.radius_min_mm <= h.radius_mm * DRILL_POINT_RADIUS_FRACTION:
                blind[i] = True          # tapers to a point -> drill tip
            elif c.radius_max_mm > h.radius_mm * 1.1:
                countersunk[i] = True    # flares wider than the bore

        for j, other in enumerate(holes):
            if i == j or other.axis_direction is None:
                continue
            if not _same_axis_line(h.axis_point, h.axis_direction,
                                    other.axis_point, other.axis_direction):
                continue
            if other.radius_mm > h.radius_mm * 1.1:
                counterbored[i] = True   # a wider bore shares this axis

    return {
        "blind": blind,
        "countersunk": countersunk,
        "counterbored": counterbored,
    }


def compute_derived_features(results: dict, material: str = "") -> dict:
    """Builds the flat feature dictionary from a pipeline result dict.

    All dimensional outputs are imperial; see the module docstring.
    `material` (optional) adds a weight estimate.
    """
    out: dict = {}

    props = results.get("shape_props")
    rect = results.get("rect_box")
    cyl = results.get("cylinder")
    holes = results.get("holes") or []
    fillets = results.get("fillets") or []
    cones = results.get("cones") or []
    planes = results.get("planes") or []
    split = results.get("surface_split") or {}
    stock = results.get("stock")

    volume = float(props.volume_mm3) if props else 0.0
    area = float(props.surface_area_mm2) if props else 0.0

    # ---------------------------------------------------------------- size
    out["volume_in3"] = _in3(volume)
    out["surface_area_in2"] = _in2(area)

    # Material enters the formula as a number, not just a label. Density is
    # the one property we hold; a machinability rating would be a better
    # predictor and is a natural place to extend this table.
    density = materials.density_lb_per_in3(material) if material else None
    if density is not None:
        out["material_density_lb_in3"] = float(density)

    # Bounding-box EXTENTS are not exported -- stock dimensions carry the
    # same envelope in the form actually purchased. What the box uniquely
    # provides is the part's true (unquantised) proportions, which stock
    # cannot give because its thickness is rounded to the next quarter inch.
    diagonal_in = 0.0
    if rect is not None:
        dims = np.sort(np.asarray(rect.extents_mm, dtype=float))[::-1]
        diagonal_in = _in(float(np.linalg.norm(dims)))
        out["aspect_max_min"] = float(dims[0] / dims[2]) if dims[2] > 0 else 0.0
        out["aspect_max_mid"] = float(dims[0] / dims[1]) if dims[1] > 0 else 0.0
        out["removal_ratio_box"] = (float((rect.volume_mm3 - volume) / rect.volume_mm3)
                                     if rect.volume_mm3 > 0 else 0.0)

    if cyl is not None:
        out["removal_ratio_cyl"] = (float((cyl.volume_mm3 - volume) / cyl.volume_mm3)
                                     if cyl.volume_mm3 > 0 else 0.0)
    # The stock actually chosen decides this, not just the best-fitting box:
    # the programmer may put a prismatic part on bar, or a round one on plate.
    if stock is not None:
        out["stock_is_round"] = 1 if stock.is_round else 0
    else:
        out["stock_is_round"] = 1 if results.get("primary") == "cylindrical" else 0
    out["stock_is_extrusion"] = 1 if (stock is not None and stock.is_extrusion) else 0

    # ------------------------------------------------------------- turning
    # How much of the part a lathe can generate. turned_area_fraction is
    # what decides mill vs turn by default; the envelope and areas are what
    # the turning estimate is built from. All zero when there is no
    # enclosing cylinder to turn about.
    tg = results.get("turning")
    if tg is not None and tg.total_area_mm2 > 0:
        out["turned_area_fraction"] = float(tg.turned_fraction)
        out["turn_od_area_in2"] = _in2(tg.turned_od_area_mm2)
        out["turn_id_area_in2"] = _in2(tg.turned_id_area_mm2)
        out["turn_face_area_in2"] = _in2(tg.turned_face_area_mm2)
        out["turn_envelope_volume_in3"] = _in3(tg.envelope_volume_mm3)
        out["turn_max_diameter_in"] = _in(tg.max_diameter_mm)
        out["turn_length_in"] = _in(tg.length_mm)
        out["turn_axial_hole_count"] = len(tg.axial_holes)
        out["turn_off_axis_hole_count"] = int(tg.off_axis_hole_count)
    else:
        for k in ("turned_area_fraction", "turn_od_area_in2", "turn_id_area_in2",
                  "turn_face_area_in2", "turn_envelope_volume_in3",
                  "turn_max_diameter_in", "turn_length_in"):
            out[k] = 0.0
        out["turn_axial_hole_count"] = 0
        out["turn_off_axis_hole_count"] = 0

    # --------------------------------------------------------------- stock
    # What actually gets bought and cut -- a better roughing driver than the
    # bounding box, since it includes the real allowances.
    if stock is not None:
        out.update(stock_mod.stock_features(stock, _in3(volume)))

    # ------------------------------------------------------------ shape
    hull_v = results.get("hull_volume_mm3") or 0.0
    if hull_v > 0:
        # Solidity: 1.0 is a convex blob, low values mean deep pockets,
        # webs and open structure -- i.e. a lot of material to clear.
        out["solidity"] = float(volume / hull_v)

    if volume > 0:
        # Dimensionless surface complexity: a sphere gives ~4.836, and the
        # value climbs with intricacy independently of size.
        out["area_over_volume_23"] = float(area / (volume ** (2.0 / 3.0)))

    # --------------------------------------------------------- surfaces
    # ABSOLUTE areas, not fractions: finishing time is
    # area / (stepover x feed), so the absolute value is what the rate
    # multiplies. Fractions remain recoverable by dividing by
    # surface_area_in2 if a scale-free view is wanted.
    #
    # The three-way split is exhaustive (analytic + freeform + other =
    # total), and the axis it splits on is the one that matters for
    # strategy: analytic faces are cut with prismatic toolpaths at full
    # width, freeform faces need a ball mill stepping over at a fraction
    # of the tool diameter, which is an order of magnitude slower per
    # square inch.
    out["area_analytic_in2"] = _in2(split.get("analytic_area_mm2", 0.0))
    out["area_freeform_in2"] = _in2(split.get("freeform_area_mm2", 0.0))
    out["area_other_surface_in2"] = _in2(split.get("other_area_mm2", 0.0))
    by_type = split.get("by_type_mm2") or {}
    for label, key in (("Plane", "area_planar_in2"), ("Cylinder", "area_cylindrical_in2"),
                        ("Cone", "area_conical_in2"), ("Sphere", "area_spherical_in2"),
                        ("Torus", "area_toroidal_in2")):
        out[key] = _in2(by_type.get(label, 0.0))

    # ------------------------------------------------------------ holes
    out["hole_count"] = len(holes)
    out["hole_total_depth_in"] = _in(sum(h.depth_mm for h in holes))
    if holes:
        diameters = np.array([h.diameter_mm for h in holes], dtype=float)
        depths = np.array([h.depth_mm for h in holes], dtype=float)
        ratios = np.array([h.depth_to_diameter_ratio for h in holes], dtype=float)
        ratios = ratios[np.isfinite(ratios)]
        out["hole_dia_min_in"] = _in(diameters.min())
        out["hole_dia_max_in"] = _in(diameters.max())
        out["hole_dia_mean_in"] = _in(diameters.mean())
        out["hole_depth_max_in"] = _in(depths.max())
        out["hole_depth_avg_in"] = _in(depths.mean())
        out["hole_max_depth_ratio"] = float(ratios.max()) if len(ratios) else 0.0
        # Mean of the PER-HOLE ratios, not mean depth over mean diameter --
        # those differ whenever diameters vary, and only the former is the
        # average pecking penalty.
        out["hole_avg_depth_ratio"] = float(ratios.mean()) if len(ratios) else 0.0
        out["hole_distinct_diameters"] = len(results.get("hole_groups") or [])
        out["deep_hole_count"] = len(results.get("deep_holes") or [])

        # --- drilling orientations -------------------------------------
        # On contoured parts each hole is normal to a curved surface, so
        # the axes form a continuum with no natural gap to cluster on. A
        # count at a single tolerance is therefore fragile; report counts
        # at several, plus spread statistics that need no clustering rule.
        stats = axis_stats.compute_axis_stats(
            [h.axis_direction for h in holes if h.axis_direction is not None])
        for tol, count in stats.counts_by_tolerance.items():
            label = str(tol).replace(".", "p")
            out[f"hole_axis_orientations_{label}deg"] = int(count)
        out["hole_frac_on_main_axis"] = float(stats.frac_on_dominant)
        out["hole_axis_max_deviation_deg"] = float(stats.max_deviation_deg)
        out["hole_axis_mean_deviation_deg"] = float(stats.mean_deviation_deg)
        out["hole_axis_p90_deviation_deg"] = float(stats.p90_deviation_deg)
        out["hole_frac_axis_within_1deg"] = float(stats.frac_within_1deg)
        out["hole_frac_axis_within_5deg"] = float(stats.frac_within_5deg)

        # Entry and termination types come from geometry.holes, which
        # assembles co-axial cylinders, cones and capping planes into whole
        # holes rather than guessing from isolated faces.
        terminations = [getattr(h, "termination", "through") for h in holes]
        entries = [getattr(h, "entry_type", "plain") for h in holes]
        out["hole_through_count"] = terminations.count("through")
        out["hole_blind_count"] = sum(1 for t in terminations if t.startswith("blind"))
        # Flat bottoms are kept separate from the general blind count
        # because they are a DIFFERENT OPERATION: a flat-bottomed hole
        # cannot be twist-drilled and needs an end mill or flat-bottom
        # drill. The tapered count is just blind minus flat.
        out["hole_blind_flat_count"] = terminations.count("blind_flat")
        out["hole_plain_count"] = entries.count("plain")
        out["hole_counterbored_count"] = sum(1 for e in entries if "counterbore" in e)
        out["hole_countersunk_count"] = sum(1 for e in entries if "countersink" in e)
        out["hole_stepped_count"] = sum(1 for h in holes if len(getattr(h, "steps", []) or []) > 1)
        majors = np.array([getattr(h, "major_diameter_mm", h.diameter_mm) for h in holes], dtype=float)
        out["hole_major_dia_max_in"] = _in(majors.max())
        out["internal_groove_count"] = sum(getattr(h, "internal_groove_count", 0) for h in holes)

        # Tool access along the hole axis, from the ray-cast probe.
        # "Captured" = every open mouth blocked, so no straight approach.
        captured = sum(1 for h in holes if getattr(h, "is_captured", False))
        obstructed = sum(1 for h in holes if getattr(h, "is_obstructed", False))
        out["hole_captured_count"] = captured
        out["hole_obstructed_count"] = obstructed

        clearances = [getattr(h, "entry_clearance_mm", None) for h in holes]
        clearances = [c for c in clearances if c is not None]
        out["hole_min_entry_clearance_in"] = _in(min(clearances)) if clearances else 0.0

        walls = [getattr(h, "wall_thickness_mm", None) for h in holes]
        walls = [w for w in walls if w is not None]
        out["hole_min_wall_thickness_in"] = _in(min(walls)) if walls else 0.0
    else:
        zeros = ["hole_dia_min_in", "hole_dia_max_in", "hole_dia_mean_in",
                 "hole_depth_max_in", "hole_depth_avg_in", "hole_max_depth_ratio",
                 "hole_avg_depth_ratio", "hole_distinct_diameters", "deep_hole_count",
                 "hole_frac_on_main_axis", "hole_axis_max_deviation_deg",
                 "hole_axis_mean_deviation_deg", "hole_axis_p90_deviation_deg",
                 "hole_frac_axis_within_1deg", "hole_frac_axis_within_5deg",
                 "hole_blind_count", "hole_through_count", "hole_blind_flat_count",
                 "hole_plain_count", "hole_countersunk_count", "hole_counterbored_count",
                 "hole_stepped_count", "hole_major_dia_max_in", "internal_groove_count",
                 "hole_captured_count", "hole_obstructed_count",
                 "hole_min_entry_clearance_in", "hole_min_wall_thickness_in"]
        zeros += [f"hole_axis_orientations_{str(t).replace('.', 'p')}deg"
                  for t in axis_stats.AXIS_TOLERANCES_DEG]
        for key in zeros:
            out[key] = 0

    # ------------------------------------------------- corners / tooling
    radii = [f.radius_mm for f in fillets]
    out["min_internal_radius_in"] = _in(min(radii)) if radii else 0.0
    out["internal_fillet_count"] = len(fillets)
    out["internal_fillet_area_in2"] = _in2(sum(f.area_mm2 for f in fillets))
    # How much surface needs a genuinely small cutter -- one tiny fillet is
    # very different from hundreds of them. Thresholds are fractional-inch
    # end mill sizes.
    for threshold_in, label in SMALL_RADIUS_THRESHOLDS_IN:
        threshold_mm = threshold_in * MM_PER_IN
        out[f"area_radius_under_{label}in_in2"] = _in2(
            sum(f.area_mm2 for f in fillets if f.radius_mm < threshold_mm)
        )

    # ------------------------------------------------------------ levels
    total_levels, dominant_levels = count_planar_levels(planes)
    out["planar_level_count"] = total_levels
    out["planar_levels_dominant_normal"] = dominant_levels

    # ------------------------------------------------------------- edges
    ec = results.get("edge_convexity")
    if ec is not None:
        out["concave_edge_length_in"] = _in(ec.concave_length_mm)
        out["convex_edge_length_in"] = _in(ec.convex_length_mm)
        out["smooth_edge_length_in"] = _in(ec.smooth_length_mm)
        out["concave_edge_count"] = int(ec.concave_edge_count)

    # -------------------------------------------------------- thin walls
    tw = results.get("thin_wall")
    if tw is not None and getattr(tw, "min_thickness_mm", None) is not None:
        # The MINIMUM of a sampled thickness scan is the least trustworthy
        # number this program can produce. It is an extreme-order statistic
        # over thousands of rays, so a single bad sample sets it, and the
        # bad samples have a systematic cause: a ray fired near a fillet or
        # a concave corner hits the neighbouring face of the same wall
        # almost immediately. That distance is geometrically real and
        # physically meaningless, and it reports a thin wall that exists
        # nowhere on the part.
        #
        # Two changes fix it. Corner and grazing hits are now rejected at
        # source (see features._is_opposing_hit), and the exported figures
        # are area-weighted PERCENTILES rather than the extreme. p10 is the
        # headline number: the thickness that 10% of the measured surface
        # area falls below. It answers "is there a thin REGION here",
        # which is the thing that changes feeds and depth of cut, and it
        # cannot be set by one unlucky ray.
        out["wall_thickness_p10_in"] = _in(tw.p10_thickness_mm)
        out["wall_thickness_p25_in"] = _in(tw.p25_thickness_mm)
        out["wall_thickness_median_in"] = _in(tw.median_thickness_mm)
        # Retained for reference and for spotting genuine breakthrough
        # risk, but no longer the primary signal. Treat a min far below p10
        # as a hint to go and look at the model, not as a measurement.
        out["wall_thickness_min_in"] = _in(tw.min_thickness_mm)
        # Area-weighted, so hundreds of small blend faces cannot outvote
        # the walls the way a raw sample count did.
        out["thin_area_fraction"] = float(getattr(tw, "thin_area_fraction", 0.0))
        out["thin_wall_threshold_in"] = _in(getattr(tw, "threshold_mm",
                                                     features.THIN_WALL_DEFAULT_THRESHOLD_MM))
        # Flagged off AREA, not off a percentile and not off the minimum.
        # The minimum fires on a single artifact; a percentile misses a
        # genuine thin web that covers less area than the percentile it is
        # measured at (a 6% web leaves p10 untouched by definition). Area
        # fraction is the measure that answers "is there a thin region
        # here" without being hostage to either failure.
        out["has_thin_walls"] = (
            1 if out["thin_area_fraction"] >= features.MIN_THIN_AREA_FRACTION else 0)

    # The undercut heuristic is not exported: the check is off by default,
    # and it only inspects face normals rather than real tool collisions,
    # so a column that is empty on most rows and coarse on the rest would
    # be more misleading than useful.

    return out


def quality_flags(results: dict) -> dict:
    """Row-level trust indicators, kept OUT of the feature vector.

    These say how much to believe a row rather than anything about the
    part: `hull_is_approximate` marks a row whose solidity came from a
    reduced hull, and a high `rejected_hole_count` means several
    cylindrical features failed the enclosure test, so the hole columns
    deserve a second look. Exported by `export.py` as metadata so suspect
    rows can be filtered during fitting instead of quietly skewing it.
    """
    tw = results.get("thin_wall")
    return {
        "hull_is_approximate": 1 if results.get("hull_is_approximate") else 0,
        "rejected_hole_count": int(results.get("rejected_hole_count") or 0),
        # A part with many holes can exhaust the probe time budget. The
        # holes are still counted, but their accessibility fields are
        # defaults rather than measurements, so any row with a non-zero
        # count here should be excluded when fitting the captured- and
        # obstructed-hole terms.
        "holes_unprobed": int(results.get("holes_unprobed") or 0),
        "probe_timed_out": 1 if results.get("probe_timed_out") else 0,
        # How much of the thickness scan survived artifact rejection. A
        # high proportion of rejects means the part is mostly blends and
        # corners, so the thickness percentiles rest on few samples.
        "wall_samples_used": int(getattr(tw, "samples_checked", 0) or 0),
        "wall_samples_rejected": int(getattr(tw, "samples_rejected", 0) or 0),
        "face_count": int(results.get("face_count") or 0),
        "tessellation_deflection_mm": float(results.get("deflection_mm") or 0.0),
    }
