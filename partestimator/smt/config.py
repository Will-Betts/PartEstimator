"""Tolerances and run options.

Tolerance philosophy: be STRICT on angular / coaxiality tests (these are exact
constructions in any CAD kernel and survive translation cleanly) but GENEROUS on
radius and thickness matching. Imported and translated STEP routinely carries
dimensional noise at the 1e-4 level that would fail a tight comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Tolerance:
    # Angular tests: coaxiality, parallelism, tangency direction.
    angular: float = 1e-6
    # Coaxiality: perpendicular offset between two axis lines (mm).
    coaxial: float = 1e-5
    # Tangency standoff: |axis-to-plane distance - radius| (mm).
    tangent: float = 1e-4
    # ...plus this fraction of the radius. A fixed 0.1 micron window is fine for
    # natively exported geometry but too tight for STEP that has been through a
    # translator, where a large-radius bend accumulates more absolute error.
    tangent_rel: float = 2e-4
    # Angular tolerance for the tangency tests specifically. Looser than the
    # coaxiality tolerance because it compares surface normals, which carry more
    # noise through a translation than an axis direction does.
    tangent_angular: float = 1e-4
    # Relative tolerance when matching radii and thicknesses.
    radius_rel: float = 0.01
    # Absolute floor for the above, so thin material still gets a usable window.
    radius_abs: float = 1e-3
    # Below this a face is treated as a sliver and ignored (mm^2).
    min_face_area: float = 1e-6
    # Sheet metal is THIN, and that is a usable constraint rather than a
    # platitude. Half an inch is already far outside press brake work -- 12 mm
    # plate is not air bent on a brake -- so any thickness candidate above this
    # is not the sheet, it is two unrelated faces that happen to be parallel.
    #
    # This is what stops the wrong pair of faces being chosen. Without it, a
    # part whose real skins are hidden behind a machined boss can have the boss
    # measured as the material, and every downstream number is then confidently
    # wrong rather than obviously wrong.
    max_thickness: float = 12.7          # 0.5 inch
    # Below this it is foil, not sheet, and almost certainly noise.
    min_thickness: float = 0.05
    # Bend sweeps below this are noise, not bends.
    min_bend_angle_deg: float = 0.5
    # Above this a cylinder is a hem or curl, not an air bend.
    max_air_bend_deg: float = 150.0

    def radius_close(self, a: float, b: float) -> bool:
        return abs(a - b) <= max(self.radius_abs, self.radius_rel * max(abs(a), abs(b)))


@dataclass
class UnfoldOptions:
    """Everything the caller can dial."""

    # K-factor. If None, looked up from the material table as K = f(r/t).
    k_factor: float | None = None
    # Forming method -- affects the K lookup.
    method: str = "air"  # air | bottoming | coining
    material: str = "CRS"
    # Override the inferred thickness (mm). None = infer from geometry.
    thickness: float | None = None
    # Report the nearest standard gauge alongside the measured thickness.
    # NOTE: this never changes the value used for detection or allowances --
    # see the comment in occ.detect where the thickness is resolved.
    snap_to_gauge: bool = True
    # Display and export units: "auto" follows the STEP file, else "mm" / "in".
    display_units: str = "auto"
    # Warn if gauge snapping moves the value by more than this fraction.
    gauge_snap_warn: float = 0.02
    # Face index to hold fixed during unfolding. None = largest planar face.
    base_face: int | None = None

    # B-rep face index the user has declared to BE the sheet surface.
    #
    # When set, the thickness is measured from this face to whatever lies one
    # material behind it, and that measurement overrides everything the
    # automatic inference would have concluded. Recognition is then re-derived
    # from it. This is the manual override for a part where the geometry
    # genuinely is ambiguous -- a boss, a weld prep or a mounting pad can leave
    # more than one defensible answer, and the person looking at the model can
    # settle it in one click where an algorithm cannot.
    sheet_face: int | None = None
    # Grain direction constraint for nesting: None | 0 | 90 (degrees).
    grain_deg: int | None = None
    # Scrap allowance added to the nest rectangle, as a fraction.
    scrap_factor: float = 0.0
    # Explicit V-die width (mm). None = selected from the die table.
    v_die: float | None = None
    # Press capacity for the tonnage warning (metric tonnes). None = no check.
    press_capacity_t: float | None = None
    # Material unit cost for the cost estimate (currency per kg).
    cost_per_kg: float | None = None

    # Simple mode: do the minimum needed to produce a flat pattern.
    #
    # Skips building the separate Hole records used for pierce counts. It does
    # NOT skip the hole geometry itself -- the flange outlines carry their own
    # hole wires, so cutouts still appear in the preview and the DXF -- and it
    # no longer skips feature classification, which is cheap and is what keeps
    # a countersunk part from looking like a failure.
    simple_mode: bool = True

    # How the finished blank is turned before it is measured and exported.
    #
    #   "min_area"     rotate until the tightest rectangle is square to the
    #                  axes, long side along X. This is the default because the
    #                  bounding box is what a material cost estimate is built
    #                  on, and a blank left at whatever angle the model happened
    #                  to be drawn at can report three times the area it needs.
    #   "as_modelled"  leave it where the unfold put it.
    #   "grain"        square to the rolling direction. Implied whenever
    #                  `grain_deg` is set, because that constraint comes from
    #                  the material and outranks any envelope saving.
    #
    # See `smt.orient`.
    orient: str = "min_area"

    # Move the finished blank into the positive quadrant, corner at the origin.
    orient_to_origin: bool = True

    tol: Tolerance = field(default_factory=Tolerance)
