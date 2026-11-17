"""Cycle time on a turn-mill: turning is not milling.

Up to v2.5, choosing "Lathe" swapped the spindle horsepower and nothing else:
the part was still priced as an end mill clearing volume at a milling rate.
A lathe removes material completely differently, so this module prices it as
a lathe does it:

  OD roughing   Stock minus the turned envelope, station by station along the
                axis. Each station takes ceil(radial stock / DOC) passes, and
                every pass runs at its OWN diameter: constant surface speed
                means rpm rises as the diameter falls, until the spindle
                ceiling (lathe_max_rpm) clamps it and surface speed collapses.
                Time per pass = length / (ipr x rpm). Power bounds rpm too:
                MRR = 12 x sfm x ipr x DOC must stay under hp / unit power.
  Facing        One pass across each end: integral of dr / (ipr x rpm(r)),
                with the same CSS-then-clamp behaviour toward the centre.
  ID boring     Axial holes larger than turn_bore_above_in are drilled to that
                size, then bored out pass by pass like OD roughing, inside out.
  Finishing     Turned area (OD, ID, faces) at the feed the Ra target allows:
                Ra = f^2 / (32 r_nose), divided by a practical factor, capped
                at the catalogue finishing feed. A turned surface is generated
                at 12 x sfm x f in^2/min.
  Live tooling  What the envelope holds that the part does not -- flats,
                slots, cross features -- milled by the driven tool at its own
                horsepower and rpm ceiling, plus finishing the non-turned area.
  Holes         Axial holes by the turret with the PART spinning (lathe rpm
                ceiling); off-axis holes by the live tool (live rpm ceiling).
  Grooves, part-off, C-axis indexing, turret indexing, rapids, load/unload.

Every rate comes from the `lathe` rows of machining_rates.csv; every machine
limit and duty factor from shop_profile.csv.
"""
from __future__ import annotations

import math

from . import cycle_time as ct
from . import shop
from . import stock as stock_mod

MM_PER_IN = 25.4


class _LiveToolProfile:
    """The shop profile as the live tool sees it: its own rpm ceiling."""

    def __init__(self, profile):
        self._p = profile
        self.values = dict(profile.values) if profile else {}
        if profile:
            self.values["spindle_max_rpm"] = profile.get("live_tool_max_rpm", 6000.0)

    def get(self, key, default=0.0):
        return float(self.values.get(key, default))

    def flag(self, key, default=False):
        return bool(round(self.get(key, 1.0 if default else 0.0)))


def _rpm(sfm: float, dia_in: float, ceiling: float) -> float:
    if dia_in <= 1e-6:
        return ceiling
    return min(3.82 * sfm / dia_in, ceiling) if ceiling > 0 else 3.82 * sfm / dia_in


def _pass_minutes(length_in, dia_in, sfm, ipr, doc, ceiling, hp_eff, unit_power):
    """One turning pass: rpm from CSS, clamped by the spindle and by power."""
    rpm = _rpm(sfm, dia_in, ceiling)
    if hp_eff > 0 and unit_power > 0 and dia_in > 0:
        # MRR = pi * D * doc * ipr * rpm  must not exceed hp_eff / unit_power.
        rpm_power = (hp_eff / unit_power) / (math.pi * dia_in * doc * ipr)
        rpm = min(rpm, rpm_power)
    if rpm <= 0 or ipr <= 0:
        return 0.0
    return length_in / (ipr * rpm)


def _facing_minutes(radius_in, sfm, ipr, ceiling):
    """Integral of dr / (ipr * rpm(r)) from the OD to the centre."""
    if radius_in <= 0 or sfm <= 0 or ipr <= 0:
        return 0.0
    # Below r_c the spindle is at its ceiling.
    r_c = 3.82 * sfm / (2.0 * ceiling) if ceiling > 0 else 0.0
    r_c = min(r_c, radius_in)
    t_css = (radius_in ** 2 - r_c ** 2) / (3.82 * sfm * ipr)
    t_clamped = (r_c / (ipr * ceiling)) if ceiling > 0 else 0.0
    return t_css + t_clamped


def estimate_turning(features: dict, results: dict, table, material: str,
                     finish_uin: float, profile, operations=None,
                     tapped_override=None, load_unload_min=None) -> ct.CycleEstimate:
    est = ct.CycleEstimate(material=material, removal_type="lathe",
                           finish_uin=float(finish_uin))

    def p(key, default=0.0):
        return profile.get(key, default) if profile else default

    tg = results.get("turning")
    if tg is None:
        est.warnings.append("No turning axis could be found for this part; "
                            "estimated as milled instead.")
        return ct.estimate_cycle_time(features, results, table, material, finish_uin,
                                      removal_type="mill", profile=profile,
                                      operations=operations, tapped_override=tapped_override,
                                      load_unload_min=load_unload_min)

    v = lambda op: table.value(material, "lathe", op)  # noqa: E731
    sfm, ipr, doc = v("turn_rough_sfm"), v("turn_rough_ipr"), v("turn_rough_doc_in")
    sfm_f, ipr_f_max = v("turn_finish_sfm"), v("turn_finish_ipr_max")
    unit_power = v("unit_power")
    if not all((sfm, ipr, doc, sfm_f, ipr_f_max, unit_power)):
        est.warnings.append(f"No turning data for {material!r}. Add 'lathe' rows to "
                            f"machining_rates.csv.")
        return est

    ceiling = p("lathe_max_rpm", 4000.0)
    hp_eff = p("lathe_hp", ct.DEFAULT_LATHE_HP) * p("drive_efficiency", ct.DRIVE_EFFICIENCY)
    duty = p("turn_rough_duty", 0.8) or 0.8
    cal_rough = p("calibrate_roughing", 1.0) or 1.0
    cal_finish = p("calibrate_finishing", 1.0) or 1.0
    cal_holes = p("calibrate_holes", 1.0) or 1.0
    allow = p("turn_finish_allowance_in", 0.015)

    # --- stock: round bar sized by the stock rules ------------------------
    bar = stock_mod.cylindrical_stock(tg.max_diameter_mm, tg.length_mm)
    r_stock = bar.thickness_or_diameter_in / 2.0
    z = tg.profile_z_mm
    r = tg.profile_r_mm / MM_PER_IN
    dz = (tg.length_mm / len(r)) / MM_PER_IN if len(r) else 0.0

    # --- 1. OD roughing ----------------------------------------------------
    t_od = 0.0
    passes_total = 0
    vol_od = 0.0
    for ri in r:
        radial = r_stock - ri - allow
        if radial <= 1e-6:
            continue
        vol_od += math.pi * ((r_stock ** 2) - (ri + allow) ** 2) * dz
        n = int(math.ceil(radial / doc))
        step = radial / n
        for k in range(n):
            d_pass = 2.0 * (r_stock - (k + 0.5) * step)
            t_od += _pass_minutes(dz, d_pass, sfm, ipr, step, ceiling, hp_eff, unit_power)
        passes_total += n
    t_od = t_od / duty * cal_rough
    if t_od > 0:
        est.terms.append(ct.TimeTerm(
            name="OD roughing (turn)", minutes=t_od, quantity=vol_od,
            quantity_units="in³",
            rate=(vol_od / t_od) if t_od else 0.0, rate_units="in³/min",
            limited_by="",
            detail=(f"⌀{bar.thickness_or_diameter_in:.3f}\" bar to the turned profile: "
                    f"{vol_od:,.2f} in³, up to {max(1, int(math.ceil((r_stock - r.min()) / doc)))} "
                    f"passes at {doc:.3f}\" DOC, {ipr:.4f} ipr, {sfm:,.0f} sfm CSS "
                    f"(rpm ceiling {ceiling:,.0f}), duty {duty:.0%}")))

    # --- 2. facing ------------------------------------------------------------
    face_ipr = ipr
    t_face = 2.0 * _facing_minutes(r_stock, sfm, face_ipr, ceiling) / duty * cal_rough
    if t_face > 0:
        est.terms.append(ct.TimeTerm(
            name="Facing", minutes=t_face, quantity=2, quantity_units="faces",
            detail=f"both ends, one pass each to the centre at {face_ipr:.4f} ipr; "
                   f"spindle reaches its {ceiling:,.0f} rpm ceiling inside "
                   f"r = {min(3.82 * sfm / (2 * ceiling), r_stock):.3f}\""))

    # --- 3. ID boring ----------------------------------------------------------
    bore_above = p("turn_bore_above_in", 0.75)
    t_bore = 0.0
    bored = 0
    for _nom, major_mm, depth_mm in tg.axial_holes:
        d_in, L_in = major_mm / MM_PER_IN, depth_mm / MM_PER_IN
        if d_in <= bore_above or L_in <= 0:
            continue
        bored += 1
        radial = (d_in - bore_above) / 2.0 - allow
        if radial <= 0:
            continue
        n = int(math.ceil(radial / doc))
        step = radial / n
        for k in range(n):
            d_pass = bore_above + 2.0 * (k + 0.5) * step
            t_bore += _pass_minutes(L_in, d_pass, sfm, ipr, step, ceiling, hp_eff, unit_power)
    t_bore = t_bore / duty * cal_rough
    if t_bore > 0:
        est.terms.append(ct.TimeTerm(
            name="ID boring", minutes=t_bore, quantity=bored, quantity_units="bores",
            detail=f"{bored} axial bore(s) over ⌀{bore_above:.3f}\": drilled to "
                   f"⌀{bore_above:.3f}\" then bored at {doc:.3f}\" DOC"))

    # --- 4. finish turning --------------------------------------------------
    area_turned = (tg.turned_od_area_mm2 + tg.turned_id_area_mm2
                   + tg.turned_face_area_mm2) / MM_PER_IN ** 2
    nose = p("turn_nose_radius_in", 1.0 / 32.0)
    practical = p("turn_ra_practical_factor", 1.5) or 1.5
    ra_theory_in = max(float(finish_uin), 1.0) / practical * 1e-6
    f_ra = math.sqrt(32.0 * nose * ra_theory_in)
    f_fin = min(f_ra, ipr_f_max)
    d_mean = float(2.0 * r.mean()) if len(r) else tg.max_diameter_mm / MM_PER_IN
    sfm_eff = min(sfm_f, ceiling * math.pi * d_mean / 12.0) if ceiling > 0 else sfm_f
    fin_rate = 12.0 * sfm_eff * f_fin * (p("turn_finish_duty", 0.9) or 0.9)
    t_fin = (area_turned / fin_rate * cal_finish) if fin_rate > 0 else 0.0
    if t_fin > 0:
        limit = "Ra target" if f_ra < ipr_f_max else "catalogue finishing feed"
        est.terms.append(ct.TimeTerm(
            name="Finish turning", minutes=t_fin, quantity=area_turned,
            quantity_units="in²", rate=fin_rate, rate_units="in²/min",
            detail=(f"{area_turned:,.2f} in² turned (OD, bore, faces) at {f_fin:.4f} ipr "
                    f"(set by the {limit}: {finish_uin:.0f} µin Ra, r{nose:.4f}\" nose, "
                    f"x{practical:.1f} practical) and {sfm_eff:,.0f} sfm")))
        spring_threshold = p("spring_pass_below_uin", 0.0)
        if spring_threshold > 0 and finish_uin <= spring_threshold:
            sp = t_fin * p("spring_pass_fraction", 1.0)
            est.terms.append(ct.TimeTerm(
                name="Spring pass", minutes=sp, quantity=area_turned, quantity_units="in²",
                detail=f"{finish_uin:.0f} µin Ra is at or below {spring_threshold:.0f} µin: "
                       f"one repeat of the finish profile"))

    # --- 5. live tooling ---------------------------------------------------
    part_in3 = float(features.get("volume_in3") or 0.0)
    env_in3 = tg.envelope_volume_mm3 / MM_PER_IN ** 3
    axial_in3 = tg.axial_hole_volume_mm3 / MM_PER_IN ** 3
    live_vol = max(env_in3 - part_in3 - axial_in3, 0.0)
    if live_vol < 1e-3 * max(env_in3, 1e-9):
        live_vol = 0.0
    live_profile = _LiveToolProfile(profile)
    live_hp = p("live_tool_hp", ct.LIVE_TOOL_HP)
    if live_vol > 0:
        mrr, bound, detail = ct._mrr_open(table, material, "mill", live_hp, live_profile)
        if mrr:
            mrr *= p("pocket_engagement", ct.POCKET_MRR_FRACTION) or ct.POCKET_MRR_FRACTION
            t_live = live_vol / mrr * cal_rough
            est.terms.append(ct.TimeTerm(
                name="Live-tool milling", minutes=t_live, quantity=live_vol,
                quantity_units="in³", rate=mrr, rate_units="in³/min", limited_by=bound,
                detail=(f"{live_vol:,.2f} in³ inside the turned envelope that turning "
                        f"cannot reach (flats, slots, cross features), {live_hp:.0f} hp "
                        f"live tool, {bound}-limited: {detail}")))
    other_in2 = tg.other_area_mm2 / MM_PER_IN ** 2
    if other_in2 > 0:
        rate_l, det_l = ct._finish_rate_analytic(table, material, "mill", live_profile,
                                                 finish_uin)
        if rate_l > 0:
            est.terms.append(ct.TimeTerm(
                name="Live-tool finishing", minutes=other_in2 / rate_l * cal_finish,
                quantity=other_in2, quantity_units="in²", rate=rate_l,
                rate_units="in²/min",
                detail=f"{other_in2:,.2f} in² not generated by turning ({det_l})"))

    # --- 6. holes -----------------------------------------------------------
    hole_min, tap_min, hole_note, tap_note = _lathe_holes(
        table, material, results, features, profile, live_profile, tapped_override, tg)
    if hole_min > 0:
        est.terms.append(ct.TimeTerm(name="Holes", minutes=hole_min * cal_holes,
                                     quantity=float(features.get("hole_total_depth_in") or 0),
                                     quantity_units="in", detail=hole_note))
    if tap_min > 0:
        est.terms.append(ct.TimeTerm(name="Tapping", minutes=tap_min * cal_holes,
                                     detail=tap_note))

    # --- 7. grooves and part-off -------------------------------------------
    grooves = sum(int(getattr(h, "internal_groove_count", 0) or 0)
                  for h in (results.get("holes") or []))
    if grooves:
        est.terms.append(ct.TimeTerm(
            name="Grooving", minutes=grooves * p("turn_groove_s", 20.0) / 60.0,
            quantity=grooves, quantity_units="grooves",
            detail=f"{grooves} internal groove(s) at {p('turn_groove_s', 20.0):.0f} s each"))
    if profile and profile.flag("turn_part_off_enabled"):
        part_sfm = sfm * p("turn_part_sfm_fraction", 0.7)
        part_ipr = p("turn_part_ipr", 0.003)
        t_part = _facing_minutes(r_stock, part_sfm, part_ipr, ceiling)
        est.terms.append(ct.TimeTerm(
            name="Part-off", minutes=t_part, quantity=1, quantity_units="cut",
            detail=f"⌀{2 * r_stock:.3f}\" at {part_ipr:.4f} ipr, {part_sfm:,.0f} sfm"))

    # --- 8. deburring (most turned edges are chamfered in the program) -----
    if profile and profile.flag("deburr_enabled"):
        convex = float(features.get("convex_edge_length_in") or 0.0)
        frac = p("turn_deburr_fraction", 0.3)
        rate_ipm = p("deburr_rate_ipm", 0.0)
        length = convex * frac
        if rate_ipm > 0 and length > 0:
            est.terms.append(ct.TimeTerm(
                name="Deburring", minutes=length / rate_ipm, quantity=length,
                quantity_units="in", rate=rate_ipm, rate_units="in/min",
                detail=f"{frac:.0%} of {convex:,.1f} in convex edge; the rest is "
                       f"chamfered in the turning program"))

    # --- non-cutting ----------------------------------------------------------
    est.deburr_minutes = sum(t.minutes for t in est.terms if t.name == "Deburring")
    est.cut_minutes = sum(t.minutes for t in est.terms if t.name != "Deburring")
    per_hole = sum(t.minutes for t in est.terms if t.name in ("Holes", "Tapping"))
    est.rapid_fraction = p("turn_rapid_fraction", 0.10)
    est.rapid_minutes = max(est.cut_minutes - per_hole, 0.0) * est.rapid_fraction

    tools = 3                                                   # rough, finish, part-off
    tools += int(features.get("hole_distinct_diameters") or 0)
    tools += 1 if bored else 0
    tools += 1 if grooves else 0
    tools += 1 if live_vol > 0 or other_in2 > 0 else 0
    tools += 1 if tap_min > 0 else 0
    est.tool_count = tools
    est.tool_change_minutes = tools * p("turret_index_s", 1.5) / 60.0

    orient = int(features.get("hole_axis_orientations_5p0deg") or 0)
    c_idx = p("c_axis_index_s", 0.0)
    if c_idx > 0 and orient > 1:
        est.index_minutes = (orient - 1) * c_idx / 60.0
        est.terms.append(ct.TimeTerm(
            name="C-axis indexing", minutes=est.index_minutes, quantity=orient,
            quantity_units="orientations",
            detail=f"{orient} hole orientations, {c_idx:.0f} s per index"))
        est.cut_minutes += est.index_minutes

    est.machine_cycle_minutes = est.cut_minutes + est.rapid_minutes + est.tool_change_minutes
    est.load_unload_minutes = (load_unload_min if load_unload_min is not None
                               else p("lathe_load_unload_min", 0.35))
    ops = max(int(operations if operations is not None else p("operations", 1.0)) or 1, 1)
    est.operations = ops
    if ops > 1:
        est.tool_change_minutes *= ops
        est.load_unload_minutes *= ops
        est.machine_cycle_minutes = est.cut_minutes + est.rapid_minutes + est.tool_change_minutes
    est.door_to_door_minutes = (est.machine_cycle_minutes + est.load_unload_minutes
                                + est.deburr_minutes)
    overall = p("calibrate_overall", 1.0) or 1.0
    if overall != 1.0:
        est.door_to_door_minutes *= overall
        est.calibration = overall

    frac_turned = tg.turned_fraction
    if frac_turned < p("turn_auto_fraction", 0.5):
        est.warnings.append(
            f"Only {frac_turned:.0%} of this part's surface is turned — most of it would be "
            f"live-tool work. A machining center is probably the better choice.")
    if 2 * r_stock < 0.25:
        est.warnings.append("Very small diameter: rpm is clamped for most of the cut. "
                            "A Swiss-type machine would run this faster.")
    return est


def _lathe_holes(table, material, results, features, profile, live_profile,
                 tapped_override, tg):
    """Axial holes at the LATHE rpm ceiling; off-axis holes on the live tool."""
    from .turning import _coaxial

    holes = results.get("holes") or []
    if not holes:
        return 0.0, 0.0, "", ""
    axis, origin = tg.axis_direction, tg.axis_point_mm
    tol = 0.05 + 1e-3 * tg.max_diameter_mm / 2.0
    axial = [h for h in holes if _coaxial(h.axis_direction, h.axis_point, origin, axis, tol)]
    off = [h for h in holes if h not in axial]

    class _LatheSpindle(_LiveToolProfile):
        def __init__(self, profile):
            super().__init__(profile)
            if profile:
                self.values["spindle_max_rpm"] = profile.get("lathe_max_rpm", 4000.0)

    total_h = total_t = 0.0
    notes = []
    tap_notes = []
    for group, prof, label in ((axial, _LatheSpindle(profile), "axial (turret)"),
                               (off, live_profile, "off-axis (live tool)")):
        if not group:
            continue
        h_min, t_min, h_note, t_note = ct._hole_time(
            table, material, "mill", {"holes": group}, features, prof, tapped_override)
        total_h += h_min
        total_t += t_min
        if h_note:
            notes.append(f"{label}: {h_note}")
        if t_note:
            tap_notes.append(f"{label}: {t_note}")
    return total_h, total_t, "; ".join(notes), "; ".join(tap_notes)
