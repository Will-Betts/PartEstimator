"""Approximate door-to-door cycle time from the extracted geometry.

This is a BOTTOM-UP estimate built from published cutting data. It is not
a fitted model and it does not know anything about your shop. Treat it as
a physically-grounded starting point and a sanity check on the regression
you will eventually fit against real job records -- which is why the
export carries both `estimated_cycle_time_min` (this) and the blank
`cycle_time_min` (measured). The gap between them is the interesting
quantity.

TIME BASIS: door-to-door per part.

    machine cycle = cut time + rapids + tool changes
    door-to-door  = machine cycle + load/unload

THE FIVE CUTTING TERMS

  1. Outside roughing   (stock_volume - hull_volume) / MRR_open
     Clearing down to the part's convex envelope. Big tool, full width,
     open air, so this runs at the full rate.

  2. Pocket roughing    (hull_volume - part_volume) / MRR_pocket
     Everything enclosed by the envelope: pockets, bores, slots. The
     tool is radially buried and chip evacuation is poor, so the rate is
     derated hard against open milling.

  3. Analytic finishing  area_analytic / (stepover x feed)
     Planes, bores, chamfers. Finished with a flat tool at a wide
     stepover, so it is fast per square inch.

  4. Freeform finishing  area_freeform / (stepover x feed)
     Contoured surface, ball nose, stepover set by the SCALLOP HEIGHT
     the Ra target allows. This is the term that explodes on a tight
     finish: halving the allowed scallop doubles the pass count.

  5. Hole time           sum over holes of depth / (ipr x rpm)
     Plus a peck penalty past 4x diameter, where the drill has to
     retract to clear chips.

MRR IS BOUNDED TWO WAYS and the estimator takes the lower:

    power bound = (spindle_hp x drive_efficiency) / unit_power
    tool bound  = ae x ap x feed_rate

On a 25 hp 40-taper in aluminium the power bound lands near 70 in^3/min
and the tool bound near 20, so the tool is the constraint and quoting
horsepower would overstate the rate three-fold. In titanium the two
converge and power starts to matter. Reporting which bound won, per
term, is part of the output.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import shop

MM_PER_IN = 25.4

# --------------------------------------------------------------- machine
# A 40-taper VMC, the configuration the estimates were scoped to.
DEFAULT_SPINDLE_HP = 22.0
# Belt/gear losses between motor plate rating and the cut. 0.85 is the
# usual assumption for a direct-drive or belted 40-taper.
DRIVE_EFFICIENCY = 0.85

# Turn-mill: the main spindle turns, live tooling does the milled
# features at a fraction of the main spindle's power.
DEFAULT_LATHE_HP = 25.0
LIVE_TOOL_HP = 8.0

# ------------------------------------------------------------- tooling
# The assumed tool set. These are what the published sfm/ipt figures in
# machining_rates.csv were quoted against, so they belong together.
ROUGH_TOOL_DIA_IN = 0.625
ROUGH_TOOL_FLUTES = 4
# Radial and axial engagement as fractions of tool diameter. A 30%
# stepover at full flute depth is a conventional roughing cut; it is not
# high-efficiency/trochoidal, which would take a much smaller ae at a
# much higher feed.
ROUGH_RADIAL_FRACTION = 0.30
ROUGH_AXIAL_FRACTION = 1.00

# Pocketing derate: buried tool, worse chip evacuation, more corner
# slowdown. Sits in the middle of the 40-60% band quoted for enclosed
# milling against open-face milling.
POCKET_MRR_FRACTION = 0.50

FINISH_TOOL_DIA_IN = 0.500
FINISH_TOOL_FLUTES = 4
# Flat-tool finishing stepover on analytic faces.
FINISH_ANALYTIC_STEPOVER_FRACTION = 0.65
BALL_TOOL_RADIUS_IN = 0.250

# Finishing runs lighter than roughing, so the per-tooth load drops.
FINISH_IPT_FRACTION = 0.60
# ... and the surface speed rises, since there is almost no radial load.
FINISH_SFM_FRACTION = 1.25

# --------------------------------------------------------------- holes
# Past this depth-to-diameter ratio a drill must peck, which costs
# retract and re-approach time on every peck.
PECK_DEPTH_RATIO = 4.0
PECK_TIME_PENALTY = 0.35      # fraction added to drill time once pecking

# ------------------------------------------------------- non-cutting
TOOL_CHANGE_S = 6.0           # chip-to-chip on a typical 40-taper carousel
RAPID_FRACTION = 0.15         # rapids/approach/retract as a share of cut time
MIN_LOAD_UNLOAD_MIN = 0.50    # a small part in a vise
LOAD_UNLOAD_PER_LB_MIN = 0.08 # heavier parts take longer to handle

# Surface finish tiers, micro-inches Ra. The scallop height a finish
# allows is what sets the ball-nose stepover.
FINISH_TIERS_UIN = (250.0, 125.0, 63.0, 32.0, 16.0)


def rpm_from_sfm(sfm: float, diameter_in: float) -> float:
    """RPM = 3.82 x sfm / diameter."""
    if diameter_in <= 0:
        return 0.0
    return 3.82 * sfm / diameter_in


def scallop_stepover_in(ra_uin: float, ball_radius_in: float = BALL_TOOL_RADIUS_IN) -> float:
    """Ball-nose stepover that leaves a scallop consistent with an Ra target.

    Geometry: for stepover ae and ball radius R, the cusp left between
    passes is h = R - sqrt(R^2 - (ae/2)^2), which inverts to

        ae = 2 x sqrt(2 R h - h^2)

    The link from Ra to peak cusp height h is the approximation
    h ~= 4 x Ra -- a cusp is a near-triangular profile, whose average
    roughness is roughly a quarter of its peak height. That is a rule of
    thumb, not a standard, and it is the single loosest assumption in
    this module: it is the reason a tight-finish estimate should be
    trusted less than a roughing estimate.
    """
    ra_in = max(float(ra_uin), 1.0) * 1e-6
    h = 4.0 * ra_in
    R = max(float(ball_radius_in), 1e-6)
    h = min(h, R * 0.5)                      # cusp cannot approach the radius
    inner = 2.0 * R * h - h * h
    if inner <= 0:
        return 1e-4
    return max(2.0 * math.sqrt(inner), 1e-4)


@dataclass
class TimeTerm:
    """One line of the estimate, with the arithmetic kept visible."""
    name: str
    minutes: float = 0.0
    quantity: float = 0.0        # in^3, in^2 or in, depending on the term
    quantity_units: str = ""
    rate: float = 0.0            # in^3/min or in^2/min or in/min
    rate_units: str = ""
    limited_by: str = ""         # "power", "tool", or "" where not applicable
    detail: str = ""


@dataclass
class CycleEstimate:
    terms: list[TimeTerm] = field(default_factory=list)
    cut_minutes: float = 0.0
    rapid_minutes: float = 0.0
    tool_change_minutes: float = 0.0
    load_unload_minutes: float = 0.0
    machine_cycle_minutes: float = 0.0
    door_to_door_minutes: float = 0.0
    material: str = ""
    removal_type: str = ""
    finish_uin: float = 0.0
    operations: int = 1
    calibration: float = 1.0
    index_minutes: float = 0.0
    deburr_minutes: float = 0.0
    tool_count: int = 0
    rapid_fraction: float = RAPID_FRACTION
    warnings: list[str] = field(default_factory=list)

    def as_rows(self) -> list[tuple[str, str, str]]:
        """(label, value, note) triples for display."""
        rows = [(t.name, f"{t.minutes:,.2f} min", t.detail)
                for t in self.terms if t.name != "Deburring"]
        rows.append(("Cut subtotal", f"{self.cut_minutes:,.2f} min", ""))
        rows.append(("Rapids / approach", f"{self.rapid_minutes:,.2f} min",
                     f"{self.rapid_fraction:.0%} of cut time"))
        rows.append(("Tool changes", f"{self.tool_change_minutes:,.2f} min",
                     f"{self.tool_count} tool(s)"
                     + (f" x {self.operations} ops" if self.operations > 1 else "")))
        rows.append(("Machine cycle", f"{self.machine_cycle_minutes:,.2f} min",
                     f"{self.operations} operation(s)" if self.operations > 1 else ""))
        rows.append(("Load / unload", f"{self.load_unload_minutes:,.2f} min",
                     f"x{self.operations} setups" if self.operations > 1 else ""))
        if self.deburr_minutes > 0:
            rows.append(("Deburring", f"{self.deburr_minutes:,.2f} min",
                         "hand work at the open door — outside the machine cycle"))
        if self.calibration != 1.0:
            rows.append(("Shop calibration", f"x{self.calibration:,.2f}",
                         "calibrate_overall in shop_profile.csv"))
        rows.append(("DOOR TO DOOR", f"{self.door_to_door_minutes:,.2f} min", ""))
        return rows


def clamp_rpm(rpm: float, profile=None) -> tuple[float, bool]:
    """Spindle speed the machine can actually deliver.

    Without this, a small tool at a high published surface speed is
    modelled spinning faster than the spindle turns -- a 0.190 in drill
    at 680 sfm asks for 13,672 rpm on a 12,000 rpm machine, and the
    resulting feed is one the machine will never command.
    """
    ceiling = profile.get("spindle_max_rpm", 0.0) if profile else 0.0
    if ceiling > 0 and rpm > ceiling:
        return ceiling, True
    return rpm, False


def clamp_feed(feed_ipm: float, profile=None) -> float:
    cap = profile.get("max_cutting_feed_ipm", 0.0) if profile else 0.0
    return min(feed_ipm, cap) if cap > 0 else feed_ipm


def _mrr_open(table, material: str, removal_type: str, spindle_hp: float,
               profile=None):
    """Open-face roughing MRR in in^3/min, and which bound set it.

    Returns the AVERAGE rate over a real toolpath, not the peak. The
    difference is the duty factor, and it is the largest correction in
    this module -- see geometry/shop.py for why a published rate is a
    peak rate.
    """
    unit_power = table.value(material, removal_type, "unit_power")
    sfm = table.value(material, removal_type, "rough_open_sfm")
    ipt = table.value(material, removal_type, "rough_open_ipt")
    if not (unit_power and sfm and ipt):
        return None, "", ""

    efficiency = profile.get("drive_efficiency", DRIVE_EFFICIENCY) if profile else DRIVE_EFFICIENCY
    power_bound = (spindle_hp * efficiency) / unit_power

    rpm, clamped = clamp_rpm(rpm_from_sfm(sfm, ROUGH_TOOL_DIA_IN), profile)
    feed_ipm = clamp_feed(rpm * ROUGH_TOOL_FLUTES * ipt, profile)
    ae = ROUGH_TOOL_DIA_IN * ROUGH_RADIAL_FRACTION
    ap = ROUGH_TOOL_DIA_IN * ROUGH_AXIAL_FRACTION
    tool_bound = ae * ap * feed_ipm

    if tool_bound <= power_bound:
        peak, bound = tool_bound, "tool"
        detail = (f"{ROUGH_TOOL_DIA_IN:.3f}\" {ROUGH_TOOL_FLUTES}FL at {sfm:,.0f} sfm"
                  f"{' (rpm clamped)' if clamped else ''}, "
                  f"{ipt:.4f} ipt -> {feed_ipm:,.0f} ipm, "
                  f"ae {ae:.3f}\" x ap {ap:.3f}\"")
    else:
        peak, bound = power_bound, "power"
        detail = (f"{spindle_hp:.0f} hp x {efficiency:.2f} / {unit_power:.2f} "
                  f"hp per in3/min")

    duty = profile.get("rough_open_duty", 1.0) if profile else 1.0
    return peak * duty, bound, detail + f", duty {duty:.0%}"


def finish_ra_factor(finish_uin: float, profile=None) -> float:
    """How much more a tight finish costs on an ANALYTIC surface.

    This one is empirical and says so. The geometry says analytic
    finishing should be insensitive to the Ra target: a flat tool leaves
    a flat floor, so the stepover creates no cusp at all, and the
    feed-direction marks work out near 1 uin Ra -- far below any target
    anyone asks for. By cusp theory alone, a 16 uin wall and a 125 uin
    wall cost the same.

    They do not, in a real shop. A tight finish means a lighter radial
    depth to keep the tool from deflecting, a slower feed to stay out of
    chatter, sharper tooling changed more often, and often a spring pass
    (charged separately). None of that is geometry, so it is a lookup
    rather than a formula, interpolated between the tiers in
    shop_profile.csv and meant to be tuned against measured parts.

    Contrast this with the FREEFORM term, where the Ra target drives a
    genuine geometric scallop and the stepover really does change.
    """
    if profile is None:
        return 1.0
    tiers = []
    for ra in FINISH_TIERS_UIN:                     # 250, 125, 63, 32, 16
        label = f"{ra:.0f}".rstrip("0").rstrip(".") if ra >= 1 else str(ra)
        key = f"finish_ra_factor_{int(ra)}uin"
        if key in profile.values:
            tiers.append((float(ra), profile.get(key)))
    if not tiers:
        return 1.0
    tiers.sort(key=lambda kv: kv[0])                # ascending Ra
    ra = max(float(finish_uin), 0.1)
    if ra <= tiers[0][0]:
        return tiers[0][1]
    if ra >= tiers[-1][0]:
        return tiers[-1][1]
    for (ra_lo, f_lo), (ra_hi, f_hi) in zip(tiers, tiers[1:]):
        if ra_lo <= ra <= ra_hi:
            # Interpolate in log(Ra); the tiers are roughly a geometric
            # series, so linear interpolation would bias toward the
            # coarser end of each interval.
            t = (math.log(ra) - math.log(ra_lo)) / (math.log(ra_hi) - math.log(ra_lo))
            return f_lo + t * (f_hi - f_lo)
    return 1.0


def _finish_rate_analytic(table, material: str, removal_type: str,
                           profile=None, finish_uin: float = 125.0,
                           tool_dia_in: float | None = None) -> tuple[float, str]:
    """in^2/min for flat-tool finishing of analytic faces.

    `tool_dia_in` defaults to the standard finisher. Passing a smaller
    one is how the small-cutter term is priced -- and the rpm clamp
    inside is what makes that genuinely slow rather than merely
    proportionally slower.
    """
    dia = tool_dia_in or FINISH_TOOL_DIA_IN
    sfm = table.value(material, removal_type, "rough_open_sfm")
    ipt = table.value(material, removal_type, "rough_open_ipt")
    if not (sfm and ipt):
        return 0.0, ""
    sfm *= FINISH_SFM_FRACTION
    ipt *= FINISH_IPT_FRACTION
    rpm, clamped = clamp_rpm(rpm_from_sfm(sfm, dia), profile)
    feed_ipm = clamp_feed(rpm * FINISH_TOOL_FLUTES * ipt, profile)
    stepover = dia * FINISH_ANALYTIC_STEPOVER_FRACTION
    duty = profile.get("finish_duty", 1.0) if profile else 1.0
    ra_factor = finish_ra_factor(finish_uin, profile)
    rate = stepover * feed_ipm * duty / (ra_factor if ra_factor > 0 else 1.0)
    return rate, (
        f"{dia:.3f}\" flat at {sfm:,.0f} sfm"
        f"{' (rpm clamped)' if clamped else ''}, "
        f"stepover {stepover:.3f}\" -> {feed_ipm:,.0f} ipm, duty {duty:.0%}, "
        f"{finish_uin:.0f} µin Ra factor ×{ra_factor:.2f}")


def _finish_rate_freeform(table, material: str, removal_type: str,
                           finish_uin: float, profile=None) -> tuple[float, str]:
    """in^2/min for ball-nose finishing, stepover driven by the Ra target."""
    sfm = table.value(material, removal_type, "rough_open_sfm")
    ipt = table.value(material, removal_type, "rough_open_ipt")
    if not (sfm and ipt):
        return 0.0, ""
    sfm *= FINISH_SFM_FRACTION
    ipt *= FINISH_IPT_FRACTION
    rpm, _clamped = clamp_rpm(rpm_from_sfm(sfm, BALL_TOOL_RADIUS_IN * 2.0), profile)
    feed_ipm = clamp_feed(rpm * FINISH_TOOL_FLUTES * ipt, profile)
    duty = profile.get("finish_duty", 1.0) if profile else 1.0
    feed_ipm *= duty
    stepover = scallop_stepover_in(finish_uin)
    detail = (f"{BALL_TOOL_RADIUS_IN * 2:.3f}\" ball at {sfm:,.0f} sfm, "
              f"{finish_uin:.0f} µin Ra -> stepover {stepover:.4f}\", "
              f"{feed_ipm:,.0f} ipm")
    return stepover * feed_ipm, detail


def estimate_cycle_time(features: dict, results: dict, table,
                         material: str, finish_uin: float,
                         removal_type: str = "mill",
                         spindle_hp: float | None = None,
                         load_unload_min: float | None = None,
                         profile=None,
                         operations: int | None = None,
                         tapped_override: int | None = None) -> CycleEstimate:
    """Builds the term-by-term estimate.

    `features` is the derived feature vector, `results` the raw analysis
    dict, `table` a loaded RateTable, `profile` a loaded ShopProfile.
    """
    if removal_type == "lathe":
        # Turning is priced by its own model -- see geometry/turn_time.py.
        from . import turn_time

        return turn_time.estimate_turning(
            features, results, table, material, finish_uin, profile,
            operations=operations, tapped_override=tapped_override,
            load_unload_min=load_unload_min)

    est = CycleEstimate(material=material, removal_type=removal_type,
                        finish_uin=float(finish_uin))

    def p(key, default=0.0):
        return profile.get(key, default) if profile else default

    if spindle_hp is None:
        if profile:
            spindle_hp = p("lathe_hp", DEFAULT_LATHE_HP) if removal_type == "lathe" \
                else p("spindle_hp", DEFAULT_SPINDLE_HP)
        else:
            spindle_hp = DEFAULT_LATHE_HP if removal_type == "lathe" else DEFAULT_SPINDLE_HP

    if table.get(material, removal_type, "unit_power") is None:
        est.warnings.append(
            f"No rate data for {material!r}. Add a row to machining_rates.csv.")
        return est

    cal_rough = p("calibrate_roughing", 1.0) or 1.0
    cal_finish = p("calibrate_finishing", 1.0) or 1.0
    cal_holes = p("calibrate_holes", 1.0) or 1.0

    part_vol = float(features.get("volume_in3") or 0.0)
    stock_vol = float(features.get("stock_volume_in3") or 0.0)
    solidity = float(features.get("solidity") or 0.0)
    hull_vol = (part_vol / solidity) if solidity > 0 else part_vol

    mrr_open, bound, detail = _mrr_open(table, material, removal_type,
                                         spindle_hp, profile)
    if not mrr_open:
        est.warnings.append("Incomplete roughing data; roughing not estimated.")
        mrr_open = 0.0
    mrr_open /= cal_rough if cal_rough else 1.0

    # --- 1. outside roughing -------------------------------------------
    outside_vol = max(stock_vol - hull_vol, 0.0)
    if mrr_open > 0 and outside_vol > 0:
        est.terms.append(TimeTerm(
            name="Outside roughing", minutes=outside_vol / mrr_open,
            quantity=outside_vol, quantity_units="in³",
            rate=mrr_open, rate_units="in³/min",
            limited_by=bound,
            detail=f"stock − hull = {outside_vol:,.2f} in³ at "
                   f"{mrr_open:,.2f} in³/min ({bound}-limited: {detail})"))

    # --- 2. pocket roughing --------------------------------------------
    # Two separate reductions, not one. `pocket_engagement` is a RATE
    # reduction (a buried tool cuts slower even when fully engaged, from
    # poor chip evacuation and higher radial load). The duty ratio is a
    # TIME reduction (helical entry at every level, corner slowdowns,
    # linking moves, retracts between stepdowns). Conflating them hides
    # which one is wrong when the estimate misses.
    pocket_vol = max(hull_vol - part_vol, 0.0)
    engagement = p("pocket_engagement", POCKET_MRR_FRACTION) or POCKET_MRR_FRACTION
    duty_ratio = 1.0
    if profile:
        open_duty = p("rough_open_duty", 1.0) or 1.0
        duty_ratio = (p("rough_pocket_duty", open_duty) or open_duty) / open_duty
    mrr_pocket = mrr_open * engagement * duty_ratio
    if mrr_pocket > 0 and pocket_vol > 0:
        est.terms.append(TimeTerm(
            name="Pocket roughing", minutes=pocket_vol / mrr_pocket,
            quantity=pocket_vol, quantity_units="in³",
            rate=mrr_pocket, rate_units="in³/min",
            limited_by=bound,
            detail=f"hull − part = {pocket_vol:,.2f} in³ at "
                   f"{mrr_pocket:,.2f} in³/min "
                   f"(engagement {engagement:.0%}, duty {p('rough_pocket_duty', 1.0):.0%})"))

    # --- 2b. rest roughing ----------------------------------------------
    # The main rougher cannot reach a corner tighter than its own radius,
    # so whatever it leaves behind needs a second, smaller, slower tool.
    # This was missing entirely, and on a part with small internal radii
    # it is a substantial share of roughing.
    rest_fraction = p("rest_rough_fraction", 0.0)
    rest_rate_fraction = p("rest_rough_rate_fraction", 0.35) or 0.35
    if rest_fraction > 0 and pocket_vol > 0 and mrr_pocket > 0:
        rest_vol = pocket_vol * rest_fraction
        rest_rate = mrr_pocket * rest_rate_fraction
        min_rad = float(features.get("min_internal_radius_in") or 0.0)
        reach_note = ""
        if 0 < min_rad < ROUGH_TOOL_DIA_IN / 2.0:
            reach_note = (f"; smallest internal radius {min_rad:.3f}\" is below the "
                          f"{ROUGH_TOOL_DIA_IN / 2:.3f}\" rougher radius")
        est.terms.append(TimeTerm(
            name="Rest roughing", minutes=rest_vol / rest_rate,
            quantity=rest_vol, quantity_units="in³",
            rate=rest_rate, rate_units="in³/min",
            detail=f"{rest_fraction:.0%} of pocket volume left by the rougher, "
                   f"cleaned at {rest_rate:,.2f} in³/min{reach_note}"))

    # --- 2c. semi-finishing ---------------------------------------------
    # Roughing leaves a scalloped surface. Going straight to the finisher
    # would overload it and leave witness marks, so there is almost always
    # an intermediate pass -- and it was not being charged.
    total_area = (float(features.get("area_analytic_in2") or 0.0)
                  + float(features.get("area_freeform_in2") or 0.0))
    if profile and profile.flag("semi_finish_enabled") and total_area > 0:
        sfm_v = table.value(material, removal_type, "rough_open_sfm")
        ipt_v = table.value(material, removal_type, "rough_open_ipt")
        if sfm_v and ipt_v:
            rpm_sf, _c = clamp_rpm(rpm_from_sfm(sfm_v, FINISH_TOOL_DIA_IN), profile)
            feed_sf = clamp_feed(rpm_sf * FINISH_TOOL_FLUTES * ipt_v, profile)
            step_sf = FINISH_TOOL_DIA_IN * p("semi_finish_stepover_fraction", 0.40)
            rate_sf = step_sf * feed_sf * p("finish_duty", 1.0)
            if rate_sf > 0:
                est.terms.append(TimeTerm(
                    name="Semi-finishing", minutes=total_area / rate_sf,
                    quantity=total_area, quantity_units="in²",
                    rate=rate_sf, rate_units="in²/min",
                    detail=f"{total_area:,.2f} in² at {rate_sf:,.1f} in²/min "
                           f"({FINISH_TOOL_DIA_IN:.3f}\" at {step_sf:.3f}\" stepover)"))

    # --- thin-wall feed derate -------------------------------------------
    # Applied to FINISHING ONLY, on purpose. While roughing there is still
    # material on both sides, so the wall is not thin yet -- it only
    # becomes thin as the last passes go on, which is exactly when the
    # feed has to come down.
    thin_multiplier = 1.0
    thin_note = ""
    if profile and profile.flag("thin_wall_derate_enabled"):
        thin_frac = float(features.get("thin_area_fraction") or 0.0)
        derate = p("thin_wall_feed_derate", 1.0) or 1.0
        if thin_frac > 0 and derate < 1.0:
            thin_multiplier = (1.0 - thin_frac) + thin_frac / derate
            thin_note = (f", thin-wall derate x{thin_multiplier:.2f} "
                         f"({thin_frac:.0%} of area at {derate:.0%} feed)")

    # --- 3. analytic finishing -----------------------------------------
    area_analytic = float(features.get("area_analytic_in2") or 0.0)
    rate_a, detail_a = _finish_rate_analytic(table, material, removal_type,
                                              profile, finish_uin)
    rate_a /= cal_finish if cal_finish else 1.0
    rate_a /= thin_multiplier
    detail_a += thin_note

    # --- small-cutter area ------------------------------------------------
    # Surface whose radius is below the standard finisher's radius cannot
    # be cut with it. The replacement tool is slower twice over: a smaller
    # diameter sweeps less per pass, AND it wants an rpm the spindle
    # cannot reach, so its surface speed collapses as well. That second
    # effect is the one that surprises -- a 1/8 cutter is ~4x slower per
    # square inch here, not the ~2x its diameter alone suggests.
    small_area = 0.0
    if profile and profile.flag("small_cutter_derate_enabled"):
        small_area = float(features.get("area_radius_under_1_4in_in2") or 0.0)
        small_area = min(small_area, area_analytic)
    if small_area > 0:
        min_rad = float(features.get("min_internal_radius_in") or 0.0)
        small_dia = max(2.0 * min_rad, p("small_cutter_min_dia_in", 0.0625))
        small_dia = min(small_dia, FINISH_TOOL_DIA_IN)
        rate_small, detail_small = _finish_rate_analytic(
            table, material, removal_type, profile, finish_uin,
            tool_dia_in=small_dia)
        rate_small /= (cal_finish if cal_finish else 1.0) * thin_multiplier
        area_analytic -= small_area
        if rate_small > 0:
            est.terms.append(TimeTerm(
                name="Finishing, small cutter", minutes=small_area / rate_small,
                quantity=small_area, quantity_units="in²",
                rate=rate_small, rate_units="in²/min",
                detail=f"{small_area:,.2f} in² needs a {small_dia:.3f}\" cutter "
                       f"(smallest internal radius {min_rad:.3f}\"): "
                       f"{rate_small:,.1f} in²/min vs {rate_a:,.1f} for the "
                       f"{FINISH_TOOL_DIA_IN:.3f}\" finisher"))

    if rate_a > 0 and area_analytic > 0:
        est.terms.append(TimeTerm(
            name="Finishing, analytic", minutes=area_analytic / rate_a,
            quantity=area_analytic, quantity_units="in²",
            rate=rate_a, rate_units="in²/min",
            detail=f"{area_analytic:,.2f} in² at {rate_a:,.1f} in²/min ({detail_a})"))

    # --- 4. freeform finishing -----------------------------------------
    area_free = float(features.get("area_freeform_in2") or 0.0)
    rate_f, detail_f = _finish_rate_freeform(table, material, removal_type,
                                              finish_uin, profile)
    rate_f /= cal_finish if cal_finish else 1.0
    rate_f /= thin_multiplier
    detail_f += thin_note
    if rate_f > 0 and area_free > 0:
        est.terms.append(TimeTerm(
            name="Finishing, freeform", minutes=area_free / rate_f,
            quantity=area_free, quantity_units="in²",
            rate=rate_f, rate_units="in²/min",
            detail=f"{area_free:,.2f} in² at {rate_f:,.1f} in²/min ({detail_f})"))

    # --- 4b. spring pass -------------------------------------------------
    # Below a threshold Ra, a wall gets re-cut at nominal depth to take
    # out whatever the tool deflected away from.
    #
    # ANALYTIC AREA ONLY, deliberately. A spring pass is a wall and floor
    # technique: leave a few thou, take it, then repeat at zero depth. On
    # a contoured surface you do not re-run every stepover pass -- you
    # take a finer stepover, and the scallop model upstream has already
    # charged for that. Charging a freeform repeat as well would count
    # the same tightening twice, and at the ball-nose rate it produced a
    # 2.2x cliff at the threshold that no real part shows.
    spring_threshold = p("spring_pass_below_uin", 0.0)
    spring_fraction = p("spring_pass_fraction", 1.0)
    if (spring_threshold > 0 and finish_uin <= spring_threshold
            and spring_fraction > 0 and rate_a > 0 and area_analytic > 0):
        spring_min = (area_analytic / rate_a) * spring_fraction
        est.terms.append(TimeTerm(
            name="Spring pass", minutes=spring_min,
            quantity=area_analytic, quantity_units="in²",
            detail=f"{finish_uin:.0f} µin Ra is at or below the "
                   f"{spring_threshold:.0f} µin threshold; walls re-cut at "
                   f"{spring_fraction:.0%} of a finish pass "
                   f"(freeform excluded — the scallop stepover already "
                   f"covers tightening a contour)"))

    # --- 5. holes and 6. tapping ---------------------------------------
    hole_minutes, tap_minutes, hole_detail, tap_detail = _hole_time(
        table, material, removal_type, results, features, profile, tapped_override)
    hole_minutes *= cal_holes
    tap_minutes *= cal_holes
    if hole_minutes > 0:
        est.terms.append(TimeTerm(
            name="Holes", minutes=hole_minutes,
            quantity=float(features.get("hole_total_depth_in") or 0.0),
            quantity_units="in", detail=hole_detail))
    if tap_minutes > 0:
        est.terms.append(TimeTerm(
            name="Tapping", minutes=tap_minutes,
            quantity=0.0, quantity_units="", detail=tap_detail))

    # --- 7. deburring -----------------------------------------------------
    # Door-to-door includes breaking edges at the machine, and nothing was
    # charged for it at all. Convex edges carry most of the burr: a burr
    # forms where the tool exits the material, so outside corners are the
    # target. An internal corner usually carries the cutter radius already
    # and throws little, hence the much lower concave share. Tangent
    # (smooth) edges are excluded entirely -- there is no edge to break.
    if profile and profile.flag("deburr_enabled"):
        convex = float(features.get("convex_edge_length_in") or 0.0)
        concave = float(features.get("concave_edge_length_in") or 0.0)
        rate_ipm = p("deburr_rate_ipm", 0.0)
        length = (convex * p("deburr_convex_fraction", 1.0)
                  + concave * p("deburr_concave_fraction", 0.0))
        deburr_min = (length / rate_ipm) if rate_ipm > 0 else 0.0
        deburr_min += int(features.get("hole_count") or 0) * p("deburr_hole_s", 0.0) / 60.0
        if deburr_min > 0:
            est.terms.append(TimeTerm(
                name="Deburring", minutes=deburr_min,
                quantity=length, quantity_units="in",
                rate=rate_ipm, rate_units="in/min",
                detail=f"{length:,.1f} in of edge at {rate_ipm:,.0f} in/min "
                       f"({convex:,.0f} in convex x{p('deburr_convex_fraction', 1.0):.2f}"
                       f" + {concave:,.0f} in concave x{p('deburr_concave_fraction', 0.0):.2f});"
                       f" tangent edges excluded"))

    # --- non-cutting ----------------------------------------------------
    # Deburring is hand work at the open door. It belongs in door-to-door
    # but NOT in the machine cycle, and it must not attract the rapid
    # allowance -- there is no spindle involved.
    est.deburr_minutes = sum(t.minutes for t in est.terms if t.name == "Deburring")
    est.cut_minutes = sum(t.minutes for t in est.terms
                          if t.name != "Deburring")
    # Hole and tap terms already carry their own positioning and retract
    # time, so the blanket rapid allowance must not be charged on them
    # again.
    per_hole_counted = sum(t.minutes for t in est.terms
                           if t.name in ("Holes", "Tapping"))
    rapid_base = max(est.cut_minutes - per_hole_counted, 0.0)
    est.rapid_fraction = p("rapid_fraction", RAPID_FRACTION)
    est.rapid_minutes = rapid_base * est.rapid_fraction

    tools = _tool_count(features, tapped=tap_minutes > 0,
                        semi_finish=bool(profile and profile.flag("semi_finish_enabled")),
                        rest_rough=rest_fraction > 0)
    est.tool_count = tools
    est.tool_change_minutes = tools * p("tool_change_s", TOOL_CHANGE_S) / 60.0

    # --- 5-axis indexing --------------------------------------------------
    # Every distinct tool orientation is a rotary index the machine has to
    # make, and the extractor already counts them. The model charged
    # nothing for any of it, which matters most on exactly the contoured
    # work this tool was built for.
    index_s = p("index_time_s", 0.0)
    orientations = int(features.get("hole_axis_orientations_5p0deg") or 0)
    est.index_minutes = 0.0
    if index_s > 0 and orientations > 1:
        # One index per orientation beyond the first, per cutting tool that
        # has to visit it.
        est.index_minutes = (orientations - 1) * index_s / 60.0
        est.terms.append(TimeTerm(
            name="5-axis indexing", minutes=est.index_minutes,
            quantity=float(orientations), quantity_units="orientations",
            detail=f"{orientations} distinct tool orientations at 5°, "
                   f"{index_s:.0f} s per index"))
        est.cut_minutes = sum(t.minutes for t in est.terms)

    est.machine_cycle_minutes = (est.cut_minutes + est.rapid_minutes
                                  + est.tool_change_minutes)

    if load_unload_min is None:
        weight_lb = (float(features.get("stock_volume_in3") or 0.0)
                     * float(features.get("material_density_lb_in3") or 0.0))
        load_unload_min = (p("load_unload_base_min", MIN_LOAD_UNLOAD_MIN)
                           + weight_lb * p("load_unload_per_lb_min",
                                           LOAD_UNLOAD_PER_LB_MIN))
    est.load_unload_minutes = load_unload_min

    # --- operations ------------------------------------------------------
    # Each setup is its own machine cycle and its own load/unload. The
    # cutting work is split across them rather than duplicated -- the part
    # only has so much material to remove -- but the fixed overheads
    # repeat, which is why a two-op part is more than twice as expensive
    # as half a one-op part.
    if operations is None:
        operations = int(p("operations", 1.0)) or 1
    operations = max(int(operations), 1)
    est.operations = operations
    if operations > 1:
        extra_cycles = operations - 1
        est.tool_change_minutes *= operations
        est.load_unload_minutes *= operations
        est.machine_cycle_minutes = (est.cut_minutes + est.rapid_minutes
                                      + est.tool_change_minutes)
        est.warnings.append(
            f"{operations} operations charged: tool changes and load/unload "
            f"repeat per setup ({extra_cycles} extra).")

    est.door_to_door_minutes = (est.machine_cycle_minutes
                                 + est.load_unload_minutes
                                 + est.deburr_minutes)

    overall = p("calibrate_overall", 1.0) or 1.0
    if overall != 1.0:
        est.door_to_door_minutes *= overall
        est.calibration = overall

    _add_warnings(est, features, removal_type)
    return est


def _tool_count(features: dict, tapped: bool = False,
                 semi_finish: bool = False, rest_rough: bool = False) -> int:
    """Distinct tools the part implies: rougher, finisher, ball, plus one
    drill per distinct hole diameter, a spot drill, a chamfer tool and a
    tap where those apply."""
    tools = 1                                        # rougher
    if float(features.get("area_analytic_in2") or 0.0) > 0:
        tools += 1
    if float(features.get("area_freeform_in2") or 0.0) > 0:
        tools += 1
    tools += int(features.get("hole_distinct_diameters") or 0)
    # One extra tool for counterboring, however many counterbores there
    # are -- the same endmill does them all. (Parenthesised deliberately:
    # `x or 0 > 0` binds as `x or (0 > 0)` and would add the whole count.)
    if int(features.get("hole_counterbored_count") or 0) > 0:
        tools += 1
    if int(features.get("hole_count") or 0) > 0:
        tools += 2                                   # spot drill + chamfer
    if tapped:
        tools += 1
    if semi_finish:
        tools += 1
    if rest_rough:
        tools += 1
    return tools


def _hole_time(table, material: str, removal_type: str,
                results: dict, features: dict, profile=None,
                tapped_override: int | None = None) -> tuple[float, float, str, str]:
    """Full hole CYCLE time, not just the seconds the drill is cutting.

    Returns (drill_minutes, tap_minutes, drill_note, tap_note).

    The earlier version charged only `depth / feed`, which for a
    0.250 in hole half an inch deep is 0.49 seconds. The real cycle is
    nearer four seconds once you count the rapid to position, the move
    down to the clearance plane and the retract -- and that is before a
    spot drill or a chamfer. On a part with sixty holes the difference
    is minutes, not seconds, which is why hole time was the second
    biggest source of optimism in this model.
    """
    sfm = table.value(material, removal_type, "drill_sfm")
    ipr_per_in = table.value(material, removal_type, "drill_ipr_per_inch_dia")
    if not (sfm and ipr_per_in):
        return 0.0, 0.0, "", ""

    def p(key, default=0.0):
        return profile.get(key, default) if profile else default

    per_hole_overhead_s = (p("hole_position_s") + p("hole_approach_s")
                           + p("hole_retract_s"))
    spot_s = p("spot_drill_s") if (profile and profile.flag("spot_drill_enabled")) else 0.0
    chamfer_s = p("chamfer_s") * p("chamfer_fraction", 1.0)

    holes = results.get("holes") or []
    tapped = shop.identify_tapped_holes(holes) if holes else []
    if tapped_override is not None:
        # Honour the user's count: keep the largest-diameter candidates,
        # or if they named more than we detected, scale the known ones.
        tapped = tapped[:max(int(tapped_override), 0)]
    tapped_ids = {id(h) for h, _l, _p in tapped}

    cut_min = 0.0
    overhead_min = 0.0
    pecked = 0
    counted = 0

    for h in holes:
        dia_in = float(getattr(h, "nominal_diameter_mm", 0.0)) / MM_PER_IN
        depth_in = float(getattr(h, "depth_mm", 0.0)) / MM_PER_IN
        if dia_in <= 0 or depth_in <= 0:
            continue
        counted += 1
        rpm, _c = clamp_rpm(rpm_from_sfm(sfm, dia_in), profile)
        feed_ipm = clamp_feed(rpm * ipr_per_in * dia_in, profile)
        if feed_ipm > 0:
            t = depth_in / feed_ipm
            if depth_in / dia_in > PECK_DEPTH_RATIO:
                t *= (1.0 + PECK_TIME_PENALTY)
                pecked += 1
            cut_min += t
        overhead_min += (per_hole_overhead_s + spot_s + chamfer_s) / 60.0

    if counted == 0:
        # Aggregate fallback when the hole list is not in hand.
        depth_total = float(features.get("hole_total_depth_in") or 0.0)
        dia_mean = float(features.get("hole_dia_mean_in") or 0.0)
        n = int(features.get("hole_count") or 0)
        if depth_total <= 0 or dia_mean <= 0:
            return 0.0, 0.0, "", ""
        rpm, _c = clamp_rpm(rpm_from_sfm(sfm, dia_mean), profile)
        feed_ipm = clamp_feed(rpm * ipr_per_in * dia_mean, profile)
        cut_min = depth_total / feed_ipm if feed_ipm > 0 else 0.0
        overhead_min = n * (per_hole_overhead_s + spot_s + chamfer_s) / 60.0
        total = cut_min + overhead_min
        return total, 0.0, (
            f"{n} hole(s), {depth_total:,.2f} in total depth at mean ⌀"
            f"{dia_mean:.3f}\" (aggregate); "
            f"{overhead_min:,.2f} min of cycle overhead"), ""

    drill_total = cut_min + overhead_min
    note = (f"{counted} hole(s) at {sfm:,.0f} sfm — "
            f"{cut_min:,.2f} min cutting + {overhead_min:,.2f} min cycle overhead "
            f"({per_hole_overhead_s + spot_s + chamfer_s:.1f} s each)")
    if pecked:
        note += f"; {pecked} pecked (+{PECK_TIME_PENALTY:.0%})"

    # --- tapping -------------------------------------------------------
    tap_min = 0.0
    tap_note = ""
    if tapped:
        tap_rpm = p("tap_rpm", 600.0)
        retract_ratio = p("tap_retract_ratio", 1.0) or 1.0
        tap_overhead_s = p("tap_overhead_s")
        depth_fraction = p("tap_depth_fraction", 0.75)
        for h, label, pitch_in in tapped:
            depth_in = float(getattr(h, "depth_mm", 0.0)) / MM_PER_IN * depth_fraction
            if depth_in <= 0 or pitch_in <= 0:
                continue
            feed_ipm = tap_rpm * pitch_in          # pitch-synchronised
            if feed_ipm <= 0:
                continue
            tap_min += depth_in / feed_ipm                    # in
            tap_min += depth_in / (feed_ipm * retract_ratio)  # back out
            tap_min += tap_overhead_s / 60.0
        labels = sorted({l for _h, l, _p in tapped})
        tap_note = (f"{len(tapped)} tapped hole(s) at {tap_rpm:,.0f} rpm "
                    f"({', '.join(labels[:4])}{'…' if len(labels) > 4 else ''}); "
                    f"detected from tap-drill diameters")
    return drill_total, tap_min, note, tap_note


def _add_warnings(est: CycleEstimate, features: dict, removal_type: str):
    if features.get("has_thin_walls"):
        est.warnings.append(
            "Thin walls present — real feeds will be lower than these "
            "published rates allow for.")
    if float(features.get("hole_captured_count") or 0) > 0 and removal_type == "mill":
        est.warnings.append(
            f"{int(features['hole_captured_count'])} captured hole(s) — "
            "implies an extra setup or special tooling not costed here.")
    min_rad = float(features.get("min_internal_radius_in") or 0.0)
    if 0 < min_rad < FINISH_TOOL_DIA_IN / 2.0:
        est.warnings.append(
            f"Smallest internal radius {min_rad:.3f}\" needs a cutter below the "
            f"{FINISH_TOOL_DIA_IN:.3f}\" assumed here; corner work will run slower.")
    if est.door_to_door_minutes > 0 and est.cut_minutes / max(est.door_to_door_minutes, 1e-9) < 0.3:
        est.warnings.append(
            "Non-cutting time dominates — the estimate is sensitive to the "
            "load/unload and tool-change assumptions rather than to geometry.")
