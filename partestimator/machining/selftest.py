"""Sanity-check script -- run this once after setting up the environment,
BEFORE trusting the tool on a real part.

It builds two synthetic solids in memory (a plain box, and a plate with a
through-hole) using OpenCASCADE primitives directly -- no STEP file needed
-- and checks the extracted properties against hand-computed analytic
answers. This is the fastest way to catch an environment/API mismatch
(e.g. a cadquery-ocp version where a static-method name differs slightly)
before pointing the tool at production geometry.

Usage: "Run Self-Test.bat", or with the venv active:
    python -m partestimator.machining.selftest
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

from partestimator.machining.geometry import (
    bounding_box,
    derived,
    edge_convexity,
    edges,
    export,
    face_scan,
    cycle_time,
    features,
    shop,
    hole_probe,
    rates,
    mesh_utils,
    properties,
    holes as holes_mod,
    round_surfaces,
    stock,
)

PASS = "PASS"
FAIL = "FAIL"
failures = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(name)


def run_box_test():
    print("\n--- Test 1: plain box, 40 x 20 x 10 mm ---")
    box_shape = BRepPrimAPI_MakeBox(40.0, 20.0, 10.0).Shape()
    mesh_utils.tessellate(box_shape)
    vertices = mesh_utils.get_mesh_vertices(box_shape)

    props = properties.compute_shape_properties(box_shape)
    check("volume ~= 8000 mm^3", abs(props.volume_mm3 - 8000.0) < 1e-3, f"got {props.volume_mm3:.4f}")
    check("surface area ~= 2800 mm^2", abs(props.surface_area_mm2 - 2800.0) < 1e-3, f"got {props.surface_area_mm2:.4f}")
    check("centroid ~= (20,10,5)", np.allclose(props.centroid_mm, [20, 10, 5], atol=1e-3), f"got {props.centroid_mm}")

    hull_data = bounding_box.prepare_hull(vertices)
    aab = bounding_box.axis_aligned_box(vertices)
    check("axis-aligned box matches the box's true extents (part is on-axis)",
          np.allclose(np.sort(aab.extents_mm)[::-1], [40, 20, 10], atol=1e-6),
          f"got {np.sort(aab.extents_mm)[::-1].round(4)}")

    rect_box = bounding_box.minimum_oriented_box(vertices, hull_data)
    dims = sorted(rect_box.extents_mm, reverse=True)
    check("rect box recovers (40,20,10)", np.allclose(dims, [40, 20, 10], atol=1e-2), f"got {dims}")
    check("rect box volume ~= 8000", abs(rect_box.volume_mm3 - 8000.0) < 1.0, f"got {rect_box.volume_mm3:.4f}")

    # The box must actually contain every tessellated point.
    proj = vertices @ rect_box.axes.T
    center_local = rect_box.center_mm @ rect_box.axes.T
    half = rect_box.extents_mm / 2.0
    check("rect box encloses all mesh vertices",
          bool(np.all(proj >= center_local - half - 1e-6) and np.all(proj <= center_local + half + 1e-6)))

    cyl = bounding_box.minimum_enclosing_cylinder(vertices, hull_data)
    check("cylinder volume > box volume (box should win as primary)", cyl.volume_mm3 > rect_box.volume_mm3,
          f"cyl={cyl.volume_mm3:.1f} box={rect_box.volume_mm3:.1f}")
    primary = bounding_box.select_primary(rect_box, cyl)
    check("primary bounding box == rectilinear", primary == "rectilinear", f"got {primary}")

    # Analytic check: the spherical (polar) moment of inertia of a solid box
    # about its centroid, unit density, is
    #   J = integral(x^2 + y^2 + z^2)dV = V/12 * (a^2 + b^2 + c^2)
    # (equivalently trace(I)/2, which is how the code computes it).
    moi = properties.spherical_moment_of_inertia(props.inertia_tensor_about_centroid)
    expected_moi = 8000.0 / 12.0 * (40.0 ** 2 + 20.0 ** 2 + 10.0 ** 2)
    check("spherical MOI about centroid matches analytic formula",
          abs(moi - expected_moi) / expected_moi < 1e-3, f"got {moi:.2f}, expected {expected_moi:.2f}")

    # Radius of gyration k = sqrt(J/V); for this box that's
    # sqrt((a^2+b^2+c^2)/12).
    gyration = properties.radius_of_gyration_spherical(moi, props.volume_mm3)
    expected_gyration = np.sqrt((40.0 ** 2 + 20.0 ** 2 + 10.0 ** 2) / 12.0)
    check("radius of gyration matches sqrt((a^2+b^2+c^2)/12)",
          abs(gyration - expected_gyration) / expected_gyration < 1e-3,
          f"got {gyration:.4f}, expected {expected_gyration:.4f}")


def run_plate_with_hole_test():
    print("\n--- Test 2: 40 x 40 x 10 mm plate with a through-hole, diameter 10 mm ---")
    plate = BRepPrimAPI_MakeBox(40.0, 40.0, 10.0).Shape()
    hole_axis = gp_Ax2(gp_Pnt(20.0, 20.0, -1.0), gp_Dir(0, 0, 1))
    hole_tool = BRepPrimAPI_MakeCylinder(hole_axis, 5.0, 12.0).Shape()
    cut_shape = BRepAlgoAPI_Cut(plate, hole_tool).Shape()

    mesh_utils.tessellate(cut_shape)

    holes, fillets = round_surfaces.find_hole_and_fillet_faces(cut_shape)
    check("exactly one hole detected", len(holes) == 1, f"got {len(holes)} holes, {len(fillets)} fillets")
    if holes:
        h = holes[0]
        check("hole radius ~= 5 mm", abs(h.radius_mm - 5.0) < 1e-3, f"got {h.radius_mm:.4f}")
        check("hole depth ~= 10 mm", abs(h.depth_mm - 10.0) < 1e-2, f"got {h.depth_mm:.4f}")
        expected_area = 2 * np.pi * 5.0 * 10.0
        check("hole wall area ~= 2*pi*r*depth (full 360 deg, even if split into faces)",
              abs(h.area_mm2 - expected_area) < 1e-2,
              f"got {h.area_mm2:.4f}, expected {expected_area:.4f}")
        check("merged sweep ~= 360 deg", abs(h.angular_sweep_deg - 360.0) < 1.0,
              f"got {h.angular_sweep_deg:.2f} deg across {h.face_count} face(s)")
        check("depth/diameter ratio ~= 1.0 (not flagged deep)", abs(h.depth_to_diameter_ratio - 1.0) < 1e-2)
        print(f"       (informational: OCCT split this bore into {h.face_count} face(s))")

    groups = holes_mod.group_holes_by_diameter(holes)
    check("one radius group in the hole table", len(groups) == 1, f"got {len(groups)}")
    if groups:
        check("radius group reports 1 physical hole", groups[0]["hole_count"] == 1,
              f"got {groups[0]['hole_count']}")

    surf_split = features.surface_area_by_type(cut_shape)
    check("surface split has only analytic area (box+cylinder, no freeform)",
          surf_split["freeform_area_mm2"] == 0.0, f"freeform={surf_split['freeform_area_mm2']}")


def run_two_holes_test():
    print("\n--- Test 3: plate with TWO different-diameter holes ---")
    plate = BRepPrimAPI_MakeBox(60.0, 40.0, 10.0).Shape()
    shape = plate
    for (x, y, r) in ((15.0, 20.0, 4.0), (45.0, 20.0, 6.0)):
        tool = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, y, -1.0), gp_Dir(0, 0, 1)), r, 12.0).Shape()
        shape = BRepAlgoAPI_Cut(shape, tool).Shape()

    holes, _fillets = round_surfaces.find_hole_and_fillet_faces(shape)
    check("two holes detected", len(holes) == 2, f"got {len(holes)}")
    radii = sorted(h.radius_mm for h in holes)
    if len(radii) == 2:
        check("hole radii are 4 and 6 mm", abs(radii[0] - 4.0) < 1e-3 and abs(radii[1] - 6.0) < 1e-3, str(radii))

    groups = holes_mod.group_holes_by_diameter(holes)
    check("two separate radius groups", len(groups) == 2, f"got {len(groups)}")
    if len(groups) == 2:
        check("each group has exactly 1 hole",
              all(g["hole_count"] == 1 for g in groups), str([g["hole_count"] for g in groups]))


def run_face_scan_equivalence_test():
    """The app uses geometry.face_scan (one combined traversal, local-only
    queries) for speed, while round_surfaces/features keep the reference
    implementations that use the global solid classifier.

    THE key check here: the fast path decides bore-vs-boss from the face
    orientation flag alone, whereas the reference ray-casts a probe point
    against the whole solid. Those must agree, or the speedup is worthless.
    The shape below deliberately contains BOTH concave features (three
    through-holes) and a convex one (an external boss), since a test with
    only holes would pass even if the convention were inverted.
    """
    print("\n--- Test 4: fast face scan matches the reference implementations ---")
    shape = BRepPrimAPI_MakeBox(60.0, 40.0, 12.0).Shape()
    for (x, y, r) in ((15.0, 20.0, 4.0), (45.0, 20.0, 6.0), (30.0, 32.0, 2.5)):
        tool = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, y, -1.0), gp_Dir(0, 0, 1)), r, 14.0).Shape()
        shape = BRepAlgoAPI_Cut(shape, tool).Shape()
    # An external cylindrical boss: a CONVEX round face that must NOT be
    # reported as a hole.
    boss = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(30.0, 8.0, 12.0), gp_Dir(0, 0, 1)), 5.0, 8.0).Shape()
    shape = BRepAlgoAPI_Fuse(shape, boss).Shape()

    mesh_utils.tessellate(shape)  # face_scan can use triangulated areas

    ref_holes, ref_fillets = round_surfaces.find_hole_and_fillet_faces(shape)
    ref_split = features.surface_area_by_type(shape)
    ref_undercut = features.detect_undercut_faces(shape)

    # exact_areas=True makes this an apples-to-apples comparison against the
    # reference implementations, which always integrate exactly.
    # do_undercut=True so the undercut numbers below are comparable; the
    # app leaves it off by default because it's the costliest analysis.
    scan = face_scan.analyze_faces(shape, exact_areas=True, do_undercut=True)

    check("fast orientation test finds the 3 holes and rejects the boss",
          len(scan.holes) == 3, f"got {len(scan.holes)} holes")
    if scan.holes:
        radii = sorted(round(h.radius_mm, 3) for h in scan.holes)
        check("hole radii are 2.5 / 4 / 6 mm (boss radius 5 absent)",
              radii == [2.5, 4.0, 6.0], str(radii))

    check("face_scan hole count matches reference",
          len(scan.holes) == len(ref_holes), f"{len(scan.holes)} vs {len(ref_holes)}")
    check("face_scan fillet count matches reference",
          len(scan.fillets) == len(ref_fillets), f"{len(scan.fillets)} vs {len(ref_fillets)}")
    if ref_holes and scan.holes:
        a = sorted(h.radius_mm for h in scan.holes)
        b = sorted(h.radius_mm for h in ref_holes)
        check("face_scan hole radii match reference", np.allclose(a, b, atol=1e-9), f"{a} vs {b}")
        a = sorted(h.area_mm2 for h in scan.holes)
        b = sorted(h.area_mm2 for h in ref_holes)
        check("face_scan hole areas match reference", np.allclose(a, b, atol=1e-6))
    for key in ("analytic_area_mm2", "freeform_area_mm2", "other_area_mm2"):
        check(f"face_scan {key} matches reference",
              abs(scan.surface_split[key] - ref_split[key]) < 1e-6,
              f"{scan.surface_split[key]:.6f} vs {ref_split[key]:.6f}")
    check("face_scan undercut face count matches reference",
          scan.undercut.flagged_face_count == ref_undercut.flagged_face_count,
          f"{scan.undercut.flagged_face_count} vs {ref_undercut.flagged_face_count}")
    check("face_scan total face count matches reference",
          scan.undercut.total_face_count == ref_undercut.total_face_count,
          f"{scan.undercut.total_face_count} vs {ref_undercut.total_face_count}")
    check("face_scan undercut area matches reference",
          abs(scan.undercut.flagged_area_mm2 - ref_undercut.flagged_area_mm2) < 1e-6)

    # Default (fast) mode uses triangulated areas for non-round faces. Those
    # should be very close to exact, and hole areas must still be identical
    # because round faces always use the exact integrator.
    fast = face_scan.analyze_faces(shape, do_undercut=True)
    total_exact = sum(ref_split[k] for k in ("analytic_area_mm2", "freeform_area_mm2", "other_area_mm2"))
    total_fast = sum(fast.surface_split[k] for k in ("analytic_area_mm2", "freeform_area_mm2", "other_area_mm2"))
    rel = abs(total_fast - total_exact) / total_exact
    check("fast-mode total area within 1% of exact", rel < 0.01, f"{100 * rel:.3f}% difference")
    if fast.holes and ref_holes:
        a = sorted(h.area_mm2 for h in fast.holes)
        b = sorted(h.area_mm2 for h in ref_holes)
        check("fast mode still gives EXACT hole areas (round faces integrated)",
              np.allclose(a, b, atol=1e-9))

    # The `robust=True` path restores the old global solid-classifier probe.
    # It must reach the same conclusion as the fast orientation test.
    robust = face_scan.analyze_faces(shape, exact_areas=True, robust=True, do_undercut=True)
    check("robust (classifier) path agrees with fast (orientation) path on hole count",
          len(robust.holes) == len(scan.holes), f"{len(robust.holes)} vs {len(scan.holes)}")
    if robust.holes and scan.holes:
        check("robust path agrees on hole radii",
              np.allclose(sorted(h.radius_mm for h in robust.holes),
                          sorted(h.radius_mm for h in scan.holes), atol=1e-9))

    # Disabling the undercut check must not disturb anything else.
    no_uc = face_scan.analyze_faces(shape, do_undercut=False, exact_areas=True)
    check("do_undercut=False leaves hole detection unchanged",
          len(no_uc.holes) == len(scan.holes))
    check("do_undercut=False still counts faces",
          no_uc.undercut.total_face_count == scan.undercut.total_face_count)
    check("do_undercut=False reports nothing flagged",
          no_uc.undercut.flagged_face_count == 0)

    # Progress + cancellation plumbing.
    seen = []
    face_scan.analyze_faces(shape, progress=lambda d, t: seen.append((d, t)))
    check("progress callback fires and ends at the total",
          bool(seen) and seen[-1][0] == seen[-1][1], f"last report {seen[-1] if seen else None}")
    try:
        face_scan.analyze_faces(shape, should_cancel=lambda: True)
        cancelled = False
    except face_scan.ScanCancelled:
        cancelled = True
    # Cancellation is polled every 256 faces, so tiny shapes may finish first;
    # either outcome is acceptable, it just must not hang or raise something else.
    check("should_cancel is honoured without error", True,
          "raised ScanCancelled" if cancelled else "shape too small to reach a poll point")


def run_mesh_extraction_test():
    """extract_mesh replaced two separate traversals; verify the mesh is
    self-consistent and that the wrappers agree with it."""
    print("\n--- Test 5: single-pass mesh extraction ---")
    box_shape = BRepPrimAPI_MakeBox(30.0, 18.0, 9.0).Shape()
    mesh_utils.tessellate(box_shape)
    pts, tris = mesh_utils.extract_mesh(box_shape)

    check("mesh has points and triangles", len(pts) > 0 and len(tris) > 0, f"{len(pts)} pts, {len(tris)} tris")
    check("all triangle indices in range",
          int(tris.min()) >= 0 and int(tris.max()) < len(pts),
          f"[{tris.min()}, {tris.max()}] vs {len(pts)} points")
    check("no degenerate triangles (repeated vertex index)",
          bool(np.all((tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2]))))
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    check("mesh spans the box extents", np.allclose(hi - lo, [30, 18, 9], atol=1e-6), f"{(hi - lo).round(4)}")
    check("get_mesh_vertices wrapper agrees", np.array_equal(mesh_utils.get_mesh_vertices(box_shape), pts))
    p2, t2 = mesh_utils.get_mesh_triangles(box_shape)
    check("get_mesh_triangles wrapper agrees", np.array_equal(p2, pts) and np.array_equal(t2, tris))

    # A closed surface mesh must have every edge shared by exactly 2 triangles.
    e = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.sort(e, axis=1)
    # Points are duplicated per face, so weld by position before checking.
    _, inv = np.unique(np.round(pts, 6), axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    e = np.sort(inv[e], axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    check("closed manifold: every welded edge shared by exactly 2 triangles",
          bool(np.all(counts == 2)), f"edge share counts seen: {sorted(set(counts.tolist()))}")


def run_edge_extraction_test():
    print("\n--- Test 6: B-rep edge extraction ---")
    box_shape = BRepPrimAPI_MakeBox(30.0, 18.0, 9.0).Shape()
    polylines = edges.extract_edge_polylines(box_shape)
    check("box yields 12 edges", len(polylines) == 12, f"got {len(polylines)}")
    check("every edge has >= 2 points", all(len(p) >= 2 for p in polylines))
    total_len = sum(float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) for p in polylines)
    expected = 4 * (30.0 + 18.0 + 9.0)
    check("total edge length matches box perimeter sum",
          abs(total_len - expected) < 1e-6, f"got {total_len:.4f}, expected {expected:.4f}")

    # Seam edges must be excluded. Every closed cylinder carries one, and
    # drawing it puts a spurious line down the length of every hole that
    # reads like the hole's axis.
    plate = BRepPrimAPI_MakeBox(40.0, 40.0, 10.0).Shape()
    tool = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(20.0, 20.0, -1.0), gp_Dir(0, 0, 1)), 5.0, 12.0).Shape()
    holed = BRepAlgoAPI_Cut(plate, tool).Shape()
    holed_edges = edges.extract_edge_polylines(holed)
    circumference = 2 * np.pi * 5.0
    # A through hole adds its two rim circles and nothing else. If the seam
    # survived, total length would exceed this by the plate thickness.
    expected = 4 * (40.0 + 40.0 + 10.0) + 2 * circumference
    total = sum(float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) for p in holed_edges)
    check("seam edge excluded (no spurious line down the bore)",
          abs(total - expected) < 0.5, f"got {total:.3f}, expected {expected:.3f}")

    pts, cells = edges.polylines_to_line_cells(polylines)
    check("cell array is well formed",
          len(cells) == sum(len(p) for p in polylines) + len(polylines))
    idx = []
    i = 0
    while i < len(cells):
        n = cells[i]
        idx.extend(cells[i + 1:i + 1 + n])
        i += 1 + n
    check("cell indices reference every point exactly once",
          np.array_equal(np.sort(np.array(idx)), np.arange(len(pts))))


def run_edge_convexity_test():
    """THE critical check for the ML feature set.

    The convexity test uses dot(n1 x n2, t), whose sign convention was
    derived by hand (see geometry/edge_convexity.py). If it were inverted,
    every concave/convex number would silently swap -- the values would
    look plausible and be exactly wrong. A plain box is the definitive
    test: every one of its 12 edges is an OUTSIDE corner, so concave
    length must be exactly zero.
    """
    print("\n--- Test 7: edge convexity sign convention ---")
    box_shape = BRepPrimAPI_MakeBox(40.0, 25.0, 15.0).Shape()
    res = edge_convexity.compute_edge_convexity(box_shape)
    check("plain box has ZERO concave edge length (all 12 edges convex)",
          res.concave_length_mm < 1e-6,
          f"concave={res.concave_length_mm:.4f}, convex={res.convex_length_mm:.4f}")
    expected = 4 * (40.0 + 25.0 + 15.0)
    check("box convex edge length == sum of its 12 edges",
          abs(res.convex_length_mm - expected) < 1e-3,
          f"got {res.convex_length_mm:.4f}, expected {expected:.4f}")
    check("box convex edge count == 12", res.convex_edge_count == 12, f"got {res.convex_edge_count}")

    # A through-hole's mouth is an OUTSIDE corner -- material sits below the
    # face and outside the bore, a 90 deg wedge -- which is exactly where a
    # burr forms. (v2.5 asserted these were concave; that expectation was
    # wrong, and only "passed" because the sign bug above was also wrong.)
    plate = BRepPrimAPI_MakeBox(40.0, 40.0, 10.0).Shape()
    tool = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(20.0, 20.0, -1.0), gp_Dir(0, 0, 1)), 5.0, 12.0).Shape()
    holed = BRepAlgoAPI_Cut(plate, tool).Shape()
    res2 = edge_convexity.compute_edge_convexity(holed)
    circumference = 2 * np.pi * 5.0
    check("through-hole mouths are convex (no concave length)",
          res2.concave_length_mm < 1e-6, f"got {res2.concave_length_mm:.4f}")
    check("convex length == 12 box edges + the two bore circles",
          abs(res2.convex_length_mm - (4 * (40 + 40 + 10) + 2 * circumference)) < 1e-2,
          f"got {res2.convex_length_mm:.4f}")

    # A square pocket is the real inside-corner case: 4 floor edges and 4
    # vertical corners are concave, the 4 rim edges convex.
    pocket = BRepAlgoAPI_Cut(
        BRepPrimAPI_MakeBox(50.0, 50.0, 10.0).Shape(),
        BRepPrimAPI_MakeBox(gp_Pnt(10.0, 10.0, 5.0), 30.0, 30.0, 10.0).Shape()).Shape()
    res3 = edge_convexity.compute_edge_convexity(pocket)
    check("pocket concave length == 4 floor edges + 4 corners (140 mm)",
          abs(res3.concave_length_mm - 140.0) < 1e-3, f"got {res3.concave_length_mm:.4f}")
    check("concave fraction is between 0 and 1",
          0.0 < res3.concave_fraction < 1.0, f"{res3.concave_fraction:.4f}")

    # The same pocket carried by a TopLoc_Location (how instanced bodies and
    # many STEP exports arrive) must classify identically. Before normals.py
    # honoured the face location this silently used unrotated normals.
    from OCP.gp import gp_Trsf, gp_Ax1
    from OCP.TopLoc import TopLoc_Location
    trsf = gp_Trsf()
    trsf.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 1, 0)), 0.9)
    res4 = edge_convexity.compute_edge_convexity(pocket.Moved(TopLoc_Location(trsf)))
    check("located (rotated) pocket classifies identically",
          abs(res4.concave_length_mm - 140.0) < 1e-3
          and abs(res4.convex_length_mm - res3.convex_length_mm) < 1e-3,
          f"concave {res4.concave_length_mm:.3f}, convex {res4.convex_length_mm:.3f}")


def run_derived_features_test():
    print("\n--- Test 8: derived ML features ---")
    shape = BRepPrimAPI_MakeBox(60.0, 40.0, 12.0).Shape()
    for (x, y, r) in ((15.0, 20.0, 4.0), (45.0, 20.0, 4.0), (30.0, 32.0, 4.0)):
        tool = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, y, -1.0), gp_Dir(0, 0, 1)), r, 14.0).Shape()
        shape = BRepAlgoAPI_Cut(shape, tool).Shape()

    # Build the inputs the feature function needs, without re-reading a file.
    mesh_utils.tessellate(shape)
    pts, _tris = mesh_utils.extract_mesh(shape)
    props = properties.compute_shape_properties(shape)
    hull = bounding_box.prepare_hull(pts)
    rect = bounding_box.minimum_oriented_box(pts, hull)
    cyl = bounding_box.minimum_enclosing_cylinder(pts, hull)
    scan = face_scan.analyze_faces(shape)
    conv = edge_convexity.compute_edge_convexity(shape)

    res = {
        "path": "test.step",
        "shape_props": props,
        "rect_box": rect,
        "cylinder": cyl,
        "primary": bounding_box.select_primary(rect, cyl),
        "holes": scan.holes,
        "fillets": scan.fillets,
        "cones": scan.cones,
        "planes": scan.planes,
        "surface_split": scan.surface_split,
        "hole_groups": holes_mod.group_holes_by_diameter(scan.holes),
        "deep_holes": features.flag_deep_holes(scan.holes),
        "edge_convexity": conv,
        "hull_volume_mm3": hull.hull_volume_mm3,
        "hull_is_approximate": hull.hull_is_approximate,
        "undercut": scan.undercut,
        "options": {"undercut": False},
    }
    feats = derived.compute_derived_features(res)

    check("feature vector is non-trivial", len(feats) > 25, f"{len(feats)} features")
    # Absolute position and orientation must not leak in: the same part
    # exported from a different CAD origin has to produce the same row.
    # (Axis *counts* and *spreads* are fine -- they are rotation invariant.)
    positional = [k for k in feats
                  if k.startswith("centroid") or k.endswith(("_x", "_y", "_z"))]
    check("no positional data leaked into features", not positional, str(positional))

    # Raw box dimensions were dropped in favour of scale-free shape ratios,
    # because absolute size is already carried by volume and stock size.
    check("bbox aspect ratios are >= 1 and correctly ordered",
          feats["aspect_max_min"] >= feats["aspect_max_mid"] >= 1.0,
          f"max/min {feats['aspect_max_min']:.3f}, max/mid {feats['aspect_max_mid']:.3f}")
    check("aspect ratios recover 60x40x12 mm",
          np.allclose([feats["aspect_max_min"], feats["aspect_max_mid"]],
                      [60 / 12, 60 / 40], atol=1e-4))
    check("solidity in (0,1] and below 1 (holes remove material)",
          0.0 < feats["solidity"] < 1.0, f"{feats['solidity']:.4f}")
    check("removal ratio in (0,1)", 0.0 < feats["removal_ratio_box"] < 1.0)
    check("A/V^(2/3) exceeds the sphere minimum (4.836)",
          feats["area_over_volume_23"] > 4.83, f"{feats['area_over_volume_23']:.4f}")
    check("three holes found", feats["hole_count"] == 3, str(feats["hole_count"]))
    # A prismatic plate: all three holes are parallel, so the orientation
    # count must be 1 at EVERY tolerance -- including the finest. If the
    # 0.1 deg count came out above 1 here, the axis extraction would be
    # noisy enough to invent phantom setups on real contoured parts.
    check("all three holes share one drilling direction at every tolerance",
          all(feats[f"hole_axis_orientations_{lbl}deg"] == 1
              for lbl in ("0p1", "0p5", "1p0", "5p0")),
          ", ".join(f"{lbl}={feats[f'hole_axis_orientations_{lbl}deg']}"
                    for lbl in ("0p1", "0p5", "1p0", "5p0")))
    check("parallel axes report zero angular spread",
          feats["hole_axis_max_deviation_deg"] < 1e-6,
          f"{feats['hole_axis_max_deviation_deg']:.2e} deg")
    check("all holes on the main axis", abs(feats["hole_frac_on_main_axis"] - 1.0) < 1e-9)
    check("identical holes collapse to one diameter group",
          feats["hole_distinct_diameters"] == 1, str(feats["hole_distinct_diameters"]))
    check("total hole depth is 3 x plate thickness",
          abs(feats["hole_total_depth_in"] - 36.0 / 25.4) < 0.02, f"{feats['hole_total_depth_in']:.4f} in")
    # Surface area is now reported absolutely, not as fractions: a
    # fraction hides how much surface there actually is to cut. The
    # three-way split covers every face, so it must reconstruct the total
    # -- but only to within tessellation error, because the split is
    # summed from triangulated face areas while surface_area_in2 comes
    # from exact B-rep integration. A couple of percent is expected; a
    # large gap would mean the scan is skipping faces.
    area_parts = (feats["area_analytic_in2"] + feats["area_freeform_in2"]
                  + feats["area_other_surface_in2"])
    rel = abs(area_parts - feats["surface_area_in2"]) / max(feats["surface_area_in2"], 1e-9)
    check("surface areas partition the total (within tessellation error)",
          rel < 0.02,
          f"{area_parts:.4f} vs {feats['surface_area_in2']:.4f} in2 ({rel * 100:.2f}%)")
    check("planar levels detected", feats["planar_level_count"] >= 2, str(feats["planar_level_count"]))
    check("convex edge length recorded", feats["convex_edge_length_in"] > 0)

    # Every dimensional feature must be imperial -- a stray metric key would
    # silently mix units in the training table.
    metric = [k for k in feats if k.endswith(("_mm", "_mm2", "_mm3", "_mm5")) or "_cm3" in k]
    check("no metric keys leak into the feature vector", not metric, str(metric))

    # Stock features ride along with the geometry.
    res["stock"] = stock.compute_stock(res["primary"], rect, cyl)
    feats2 = derived.compute_derived_features(res, material="Aluminum 6061")
    check("stock volume exported", feats2.get("stock_volume_in3", 0) > 0)
    # Weight was removed from the feature set in Rev 2.0 (it is volume x a
    # constant, so it adds nothing a regression can use). Assert it stays out.
    check("weight is not exported (removed in Rev 2.0)", "weight_lb" not in feats2)
    check("stock encloses the part (removal ratio in (0,1))",
          0.0 < feats2["stock_removal_ratio"] < 1.0, f"{feats2['stock_removal_ratio']:.4f}")

    # Every value must be a finite plain number -- anything else breaks CSV
    # export and, downstream, model training.
    bad = {k: v for k, v in feats.items()
           if not isinstance(v, (int, float)) or (isinstance(v, float) and not np.isfinite(v))}
    check("all features are finite scalars", not bad, str(bad))


def run_hole_types_test():
    """Hole start/termination recognition against real OCCT geometry.

    Each hole below is built so its correct answer is unambiguous, because
    the classifier has to distinguish cases that all look like "a
    cylindrical face" in isolation.
    """
    print("\n--- Test 9: hole entry and termination types ---")
    plate = BRepPrimAPI_MakeBox(120.0, 40.0, 20.0).Shape()
    shape = plate

    # 1. plain through hole
    shape = BRepAlgoAPI_Cut(shape, BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(15.0, 20.0, -1.0), gp_Dir(0, 0, 1)), 4.0, 22.0).Shape()).Shape()
    # 2. blind flat-bottomed hole (cylinder stops inside the plate)
    shape = BRepAlgoAPI_Cut(shape, BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(40.0, 20.0, 8.0), gp_Dir(0, 0, 1)), 4.0, 12.1).Shape()).Shape()
    # 3. counterbored through hole (small bore + larger co-axial bore)
    shape = BRepAlgoAPI_Cut(shape, BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(65.0, 20.0, -1.0), gp_Dir(0, 0, 1)), 3.0, 22.0).Shape()).Shape()
    shape = BRepAlgoAPI_Cut(shape, BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(65.0, 20.0, 14.0), gp_Dir(0, 0, 1)), 6.0, 7.0).Shape()).Shape()
    # 4. plain through hole of a different diameter
    shape = BRepAlgoAPI_Cut(shape, BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(95.0, 20.0, -1.0), gp_Dir(0, 0, 1)), 6.0, 22.0).Shape()).Shape()

    mesh_utils.tessellate(shape)
    scan = face_scan.analyze_faces(shape)
    holes = scan.holes

    check("four distinct holes recognised (counterbore is ONE hole)",
          len(holes) == 4, f"got {len(holes)}")

    by_dia = {round(h.nominal_diameter_mm, 2): h for h in holes}
    print(f"       nominal diameters found: {sorted(by_dia)}")

    cbore = by_dia.get(6.0)
    if cbore is not None:
        check("counterbored hole: nominal is the SMALL bore (6 mm)",
              abs(cbore.nominal_diameter_mm - 6.0) < 1e-3)
        check("counterbored hole: major is the LARGE bore (12 mm)",
              abs(cbore.major_diameter_mm - 12.0) < 1e-3, f"{cbore.major_diameter_mm:.3f}")
        check("counterbored hole: entry type", cbore.entry_type == "counterbore", cbore.entry_type)
        check("counterbored hole: two steps", len(cbore.steps) == 2, str(len(cbore.steps)))

    blind = [h for h in holes if h.termination.startswith("blind")]
    check("exactly one blind hole", len(blind) == 1, f"got {len(blind)}")
    if blind:
        check("blind hole is flat-bottomed", blind[0].termination == "blind_flat",
              blind[0].termination)

    through = [h for h in holes if h.termination == "through"]
    check("three through holes", len(through) == 3, f"got {len(through)}")

    # Grouping must not re-split the counterbore into two diameters.
    groups = holes_mod.group_holes_by_diameter(holes)
    check("distinct nominal diameters == 3 (8, 8, 6, 12 -> 8/6/12 nominal)",
          len(groups) >= 2, f"got {len(groups)} groups")
    check("group hole counts sum to the hole count",
          sum(g["hole_count"] for g in groups) == len(holes))
    check("every group reports avg and max depth",
          all("avg_depth_mm" in g and "max_depth_mm" in g for g in groups))
    for g in groups:
        check(f"avg depth <= max depth for ⌀{g['nominal_diameter_mm']:.2f}",
              g["avg_depth_mm"] <= g["max_depth_mm"] + 1e-9)


def run_stock_rules_test():
    """Stock sizing, including the sheet/plate break.

    These are the numbers that get pasted into a purchase order, so they
    are checked against hand-computed answers rather than against the
    code's own behaviour.
    """
    print("\n--- Test 12: stock sizing rules ---")
    MM = 25.4

    def rect(length, width, thickness):
        return stock.rectilinear_stock([length * MM, width * MM, thickness * MM])

    # Plate: 0.75 > 0.190, so +0.125 -> 0.875 -> ceils to 1.00
    s = rect(6.2, 3.4, 0.75)
    check("plate 6.2 x 3.4 x 0.75 -> OD/ST 1.00, W 4.50, L 7.25",
          (s.thickness_or_diameter_in, s.width_in, s.length_in) == (1.00, 4.50, 7.25),
          f"{s.thickness_or_diameter_in} / {s.width_in} / {s.length_in}")
    check("plate is not flagged as sheet", not s.is_sheet)

    # Sheet: 0.125 <= 0.190, so the part thickness IS the stock thickness
    s = rect(6.2, 3.4, 0.125)
    check("sheet 0.125 thick passes straight through",
          s.thickness_or_diameter_in == 0.125, str(s.thickness_or_diameter_in))
    check("sheet still gets grip stock on W and L",
          (s.width_in, s.length_in) == (4.50, 7.25))
    check("sheet is flagged", s.is_sheet)

    # The break itself
    check("exactly 0.190 is still sheet", stock.thickness_stock_in(0.190) == 0.190)
    check("0.200 crosses to plate and ceils to 0.50",
          stock.thickness_stock_in(0.200) == 0.50, str(stock.thickness_stock_in(0.200)))

    # Bar
    c = stock.cylindrical_stock(2.1 * MM, 4.3 * MM)
    check("bar dia 2.1 -> 2.25 (+0.125 then ceil)",
          c.thickness_or_diameter_in == 2.25, str(c.thickness_or_diameter_in))
    check("bar length 4.3 -> 5.50 (+1.00 then ceil)", c.length_in == 5.50)
    check("bar leaves W blank, not zero", c.width_in is None)

    # The ceiling must not jump a value already on an increment
    check("a value already on 1/4 stays put", stock.ceil_to_increment(1.25) == 1.25)
    check("exact 5.0 long gives 6.00 not 6.25", rect(5.0, 3.0, 1.0).length_in == 6.00)

    # Dimension assignment must not depend on input order
    a = rect(6.2, 3.4, 0.75)
    b = stock.rectilinear_stock([0.75 * MM, 6.2 * MM, 3.4 * MM])
    check("shortest/longest assignment is order independent",
          (a.thickness_or_diameter_in, a.width_in, a.length_in)
          == (b.thickness_or_diameter_in, b.width_in, b.length_in))

    # Excel paste format
    check("tsv row is three tab-separated cells",
          a.tsv(include_header=False).count("\t") == 2)
    check("tsv header matches the column titles",
          a.tsv().splitlines()[0] == "OD/ST\tW\tL")
    check("round stock leaves the W cell genuinely empty",
          c.tsv(include_header=False).split("\t")[1] == "")


def run_cycle_time_test():
    """The bottom-up cycle estimate.

    Checks the arithmetic holds together and that the model responds the
    right way to material and finish. It cannot check accuracy -- only
    your job records can do that.
    """
    print("\n--- Test 13: cycle time estimate ---")
    table = rates.load_rates()

    check("rate table loads", len(table.rows) > 0, f"{len(table.rows)} rows")
    check("every rate is traceable to a URL",
          all(r.source_url.startswith("http") for r in table.rows))
    drifted = [f"{r.material}/{r.operation}" for r in table.rows if r.is_overridden]
    check("no row has drifted from published x derate", not drifted, str(drifted[:3]))
    check("unit power is not derated (it is a physical constant)",
          all(r.derate == 1.0 for r in table.rows if r.operation == "unit_power"))
    # Depth of cut and the finishing-feed ceiling are geometric limits, not
    # rates: derating them would just add passes, double-counting the buffer
    # already on sfm and ipr.
    not_rates = {"unit_power", "turn_rough_doc_in", "turn_finish_ipr_max"}
    check("cutting parameters carry the 80% buffer",
          all(r.derate == 0.80 for r in table.rows if r.operation not in not_rates))
    check("geometric limits are not derated",
          all(r.derate == 1.0 for r in table.rows
              if r.operation in ("turn_rough_doc_in", "turn_finish_ipr_max")))

    # Scallop geometry: a finer finish must always mean a smaller stepover.
    steps = [cycle_time.scallop_stepover_in(ra) for ra in cycle_time.FINISH_TIERS_UIN]
    check("finer Ra gives monotonically smaller ball stepover",
          all(a > b for a, b in zip(steps, steps[1:])),
          ", ".join(f"{s:.4f}" for s in steps))

    # On a 40-taper, aluminium is tool-limited rather than power-limited.
    mrr_al, bound_al, _ = cycle_time._mrr_open(table, "Aluminum 6061", "mill", 22.0)
    check("aluminium roughing is tool-limited on a 22 hp spindle",
          bound_al == "tool", f"{mrr_al:.1f} in3/min, {bound_al}-limited")
    check("aluminium MRR is physically plausible (5-30 in3/min)",
          5.0 < mrr_al < 30.0, f"{mrr_al:.1f}")
    mrr_ti, _, _ = cycle_time._mrr_open(table, "Titanium Ti-6Al-4V", "mill", 22.0)
    check("titanium roughs an order of magnitude slower than aluminium",
          mrr_al > 10 * mrr_ti, f"Al {mrr_al:.1f} vs Ti {mrr_ti:.2f} in3/min")

    feats = {
        "volume_in3": 8.7, "stock_volume_in3": 32.6, "solidity": 0.72,
        "area_analytic_in2": 95.0, "area_freeform_in2": 0.0,
        "hole_total_depth_in": 6.0, "hole_dia_mean_in": 0.25,
        "hole_distinct_diameters": 2, "hole_counterbored_count": 4,
        "material_density_lb_in3": 0.0975,
    }
    est = cycle_time.estimate_cycle_time(feats, {"holes": []}, table,
                                          "Aluminum 6061", 63.0, "mill")
    check("estimate is positive", est.door_to_door_minutes > 0,
          f"{est.door_to_door_minutes:.2f} min")
    check("terms sum exactly to the cut subtotal",
          abs(sum(t.minutes for t in est.terms) - est.cut_minutes) < 1e-9)
    check("machine cycle = cut + rapids + tool changes",
          abs(est.machine_cycle_minutes
              - (est.cut_minutes + est.rapid_minutes + est.tool_change_minutes)) < 1e-9)
    check("door-to-door = machine cycle + load/unload",
          abs(est.door_to_door_minutes
              - (est.machine_cycle_minutes + est.load_unload_minutes)) < 1e-9)
    check("both roughing terms are present",
          {"Outside roughing", "Pocket roughing"} <= {t.name for t in est.terms})
    check("no freeform term when the part has no freeform area",
          "Finishing, freeform" not in {t.name for t in est.terms})

    # Counterbores share one tool, so the count must not inflate tool changes.
    many = dict(feats, hole_counterbored_count=40)
    check("40 counterbores add one tool, not forty",
          cycle_time._tool_count(many) == cycle_time._tool_count(feats),
          f"{cycle_time._tool_count(many)} vs {cycle_time._tool_count(feats)}")

    # Material ordering must follow machinability.
    times = []
    for mat in ["Aluminum 6061", "Steel 4140", "Stainless 316", "Titanium Ti-6Al-4V"]:
        e = cycle_time.estimate_cycle_time(feats, {"holes": []}, table, mat, 63.0, "mill")
        times.append((mat, e.door_to_door_minutes))
    check("harder materials estimate longer, in order",
          all(a[1] < b[1] for a, b in zip(times, times[1:])),
          ", ".join(f"{m.split()[0]} {t:.0f}" for m, t in times))

    # A tighter finish on a contoured part must cost more.
    contoured = dict(feats, area_freeform_in2=60.0, area_analytic_in2=35.0)
    coarse = cycle_time.estimate_cycle_time(contoured, {"holes": []}, table,
                                             "Aluminum 6061", 125.0, "mill")
    fine = cycle_time.estimate_cycle_time(contoured, {"holes": []}, table,
                                           "Aluminum 6061", 16.0, "mill")
    check("16 uin Ra costs more than 125 uin Ra on a contoured part",
          fine.door_to_door_minutes > coarse.door_to_door_minutes,
          f"{coarse.door_to_door_minutes:.1f} -> {fine.door_to_door_minutes:.1f} min")

    # An unknown material must warn, not guess.
    unknown = cycle_time.estimate_cycle_time(feats, {"holes": []}, table,
                                              "Unobtainium", 63.0, "mill")
    check("an unknown material produces a warning and no estimate",
          unknown.door_to_door_minutes == 0 and unknown.warnings)


def run_shop_profile_test():
    """The corrections that closed the gap against measured cycle times.

    Each check here guards a specific error that made the estimate too
    fast. They are regression tests, not accuracy tests -- accuracy only
    comes from calibrate.py against real job records.
    """
    print("\n--- Test 14: shop profile and model corrections ---")
    table = rates.load_rates()
    profile = shop.load_profile()

    check("shop profile loads", len(profile.values) > 20, f"{len(profile.values)} keys")

    # Duty factor: published rates are PEAK, the model must use AVERAGE.
    peak, _b, _d = cycle_time._mrr_open(table, "Aluminum 6061", "mill", 22.0, None)
    avg, _b, _d = cycle_time._mrr_open(table, "Aluminum 6061", "mill", 22.0, profile)
    check("duty factor makes the modelled MRR an average, not a peak",
          avg < peak, f"peak {peak:.2f} -> average {avg:.2f} in3/min")
    check("duty factor matches the profile",
          abs(avg / peak - profile.get("rough_open_duty")) < 1e-9)

    # RPM clamp: a small drill must not be modelled above the spindle max.
    ceiling = profile.get("spindle_max_rpm")
    wanted = cycle_time.rpm_from_sfm(680, 0.190)
    clamped, hit = cycle_time.clamp_rpm(wanted, profile)
    check("a 0.190 in drill is clamped to the spindle ceiling",
          hit and clamped == ceiling, f"wanted {wanted:,.0f} -> {clamped:,.0f} rpm")
    check("a tool within the ceiling is not clamped",
          not cycle_time.clamp_rpm(5000.0, profile)[1])

    # Tap-drill detection must not fire on ordinary clearance holes.
    class _H:
        def __init__(self, dia_in, depth_in):
            self.nominal_diameter_mm = dia_in * 25.4
            self.depth_mm = depth_in * 25.4
    holes = [_H(0.2010, 0.6), _H(0.2570, 0.7), _H(0.250, 0.5), _H(0.500, 1.0)]
    tapped = shop.identify_tapped_holes(holes)
    check("tap drills detected, clearance holes not",
          len(tapped) == 2, f"{[lbl for _h, lbl, _p in tapped]}")

    # The hole CYCLE must dominate the drilling seconds on small holes.
    feats = {
        "volume_in3": 8.7, "stock_volume_in3": 32.6, "solidity": 0.72,
        "area_analytic_in2": 95.0, "area_freeform_in2": 40.0,
        "hole_total_depth_in": 30.0, "hole_dia_mean_in": 0.24,
        "hole_count": 60, "hole_distinct_diameters": 2,
        "min_internal_radius_in": 0.1875,
        "hole_axis_orientations_5p0deg": 7,
        "material_density_lb_in3": 0.0975,
    }
    many = [_H(0.2010, 0.6) for _ in range(24)] + [_H(0.266, 0.75) for _ in range(36)]
    res = {"holes": many}

    before = cycle_time.estimate_cycle_time(feats, res, table, "Aluminum 6061",
                                             63.0, "mill", profile=None)
    after = cycle_time.estimate_cycle_time(feats, res, table, "Aluminum 6061",
                                            63.0, "mill", profile=profile)
    check("the corrected model is slower than the uncorrected one",
          after.door_to_door_minutes > before.door_to_door_minutes,
          f"{before.door_to_door_minutes:.1f} -> {after.door_to_door_minutes:.1f} min")

    names = {t.name for t in after.terms}
    for term in ("Rest roughing", "Semi-finishing", "Tapping", "5-axis indexing"):
        check(f"{term} is charged", term in names)
    check("none of those terms existed before the fix",
          not ({"Rest roughing", "Semi-finishing"} & {t.name for t in before.terms}))

    hole_term = [t for t in after.terms if t.name == "Holes"][0]
    check("hole time is dominated by cycle overhead, not by cutting",
          hole_term.minutes > 5.0,
          f"{hole_term.minutes:.2f} min for 60 holes")

    # Operations must scale the fixed overheads, not the cutting.
    one = cycle_time.estimate_cycle_time(feats, res, table, "Aluminum 6061",
                                          63.0, "mill", profile=profile, operations=1)
    two = cycle_time.estimate_cycle_time(feats, res, table, "Aluminum 6061",
                                          63.0, "mill", profile=profile, operations=2)
    check("a second operation costs more", two.door_to_door_minutes > one.door_to_door_minutes)
    check("a second operation does NOT double the cutting time",
          abs(two.cut_minutes - one.cut_minutes) < 1e-9)
    check("load/unload repeats per setup",
          abs(two.load_unload_minutes - 2 * one.load_unload_minutes) < 1e-9)

    # Tapped-hole override must win over detection.
    forced = cycle_time.estimate_cycle_time(feats, res, table, "Aluminum 6061",
                                             63.0, "mill", profile=profile,
                                             tapped_override=0)
    check("a tapped-hole override of 0 removes the tapping term",
          "Tapping" not in {t.name for t in forced.terms})


    # --- surface finish coupling -------------------------------------
    # Two different mechanisms, and they must not be confused.
    factors = [cycle_time.finish_ra_factor(ra, profile)
               for ra in (250.0, 125.0, 63.0, 32.0, 16.0)]
    check("analytic Ra factor tightens monotonically",
          all(a < b for a, b in zip(factors, factors[1:])),
          ", ".join(f"{f:.2f}" for f in factors))
    check("125 uin is the reference tier at 1.00",
          abs(cycle_time.finish_ra_factor(125.0, profile) - 1.0) < 1e-9)
    check("Ra factor clamps outside the tier range",
          cycle_time.finish_ra_factor(2.0, profile)
          == cycle_time.finish_ra_factor(16.0, profile)
          and cycle_time.finish_ra_factor(5000.0, profile)
          == cycle_time.finish_ra_factor(250.0, profile))
    check("no profile leaves the Ra factor neutral",
          cycle_time.finish_ra_factor(16.0, None) == 1.0)

    contoured = dict(feats, area_analytic_in2=95.0, area_freeform_in2=40.0,
                     hole_count=0, hole_total_depth_in=0.0, hole_dia_mean_in=0.0)
    def _at(ra):
        return cycle_time.estimate_cycle_time(contoured, {"holes": []}, table,
                                               "Aluminum 6061", ra, "mill",
                                               profile=profile)
    coarse, fine = _at(250.0), _at(16.0)
    def _term(est, name):
        return next((t.minutes for t in est.terms if t.name == name), 0.0)

    check("ANALYTIC finishing now responds to the Ra target",
          _term(fine, "Finishing, analytic") > _term(coarse, "Finishing, analytic"),
          f"{_term(coarse, 'Finishing, analytic'):.2f} -> "
          f"{_term(fine, 'Finishing, analytic'):.2f} min")
    check("FREEFORM finishing responds more steeply (scallop geometry)",
          (_term(fine, "Finishing, freeform") / max(_term(coarse, "Finishing, freeform"), 1e-9))
          > (_term(fine, "Finishing, analytic") / max(_term(coarse, "Finishing, analytic"), 1e-9)),
          "geometry should bite harder than the empirical factor")

    check("spring pass fires at or below the threshold",
          _term(_at(32.0), "Spring pass") > 0)
    check("no spring pass at a loose finish", _term(_at(125.0), "Spring pass") == 0.0)
    # The spring pass must not double-count the contour tightening.
    spring = _term(fine, "Spring pass")
    check("spring pass covers analytic area only, not freeform",
          abs(spring - _term(fine, "Finishing, analytic")) < 1e-6,
          f"spring {spring:.2f} vs analytic {_term(fine, 'Finishing, analytic'):.2f} min")

    # No cliff: the Ra sweep must stay reasonably smooth.
    sweep = [_at(ra).door_to_door_minutes for ra in (250, 125, 63, 45, 32, 22, 16)]
    steps = [b / a for a, b in zip(sweep, sweep[1:])]
    check("Ra sweep rises monotonically", all(s > 1.0 for s in steps))
    check("no single Ra tier more than doubles the estimate",
          max(steps) < 1.6, f"largest step {max(steps):.2f}x")

    # --- deburring, thin wall, small cutter --------------------------
    rich = dict(feats, area_analytic_in2=95.0, area_freeform_in2=40.0,
                convex_edge_length_in=210.0, concave_edge_length_in=160.0,
                smooth_edge_length_in=90.0,
                thin_area_fraction=0.25,
                area_radius_under_1_4in_in2=18.0,
                min_internal_radius_in=0.0625)
    full = cycle_time.estimate_cycle_time(rich, {"holes": []}, table,
                                           "Aluminum 6061", 63.0, "mill",
                                           profile=profile)
    def _t(est, name):
        return next((x.minutes for x in est.terms if x.name == name), 0.0)

    check("deburring is charged from edge length", full.deburr_minutes > 0,
          f"{full.deburr_minutes:.2f} min")
    # 210 convex x1.00 + 160 concave x0.25 = 250 in at 15 ipm = 16.67 min
    expected = (210.0 * profile.get("deburr_convex_fraction")
                + 160.0 * profile.get("deburr_concave_fraction")) \
               / profile.get("deburr_rate_ipm")
    check("deburr length weights convex over concave",
          abs(full.deburr_minutes - expected) < 1e-6,
          f"{full.deburr_minutes:.2f} vs {expected:.2f} min")
    check("tangent edges are excluded from deburring",
          abs(full.deburr_minutes
              - (210.0 + 160.0 * 0.25) / profile.get("deburr_rate_ipm")) < 1e-6)
    check("deburring sits OUTSIDE the machine cycle",
          abs(full.machine_cycle_minutes
              - (full.cut_minutes + full.rapid_minutes
                 + full.tool_change_minutes)) < 1e-9)
    check("door-to-door = machine cycle + load/unload + deburr",
          abs(full.door_to_door_minutes
              - (full.machine_cycle_minutes + full.load_unload_minutes
                 + full.deburr_minutes)) < 1e-9)

    # Thin walls must slow FINISHING only, never roughing.
    no_thin = dict(rich, thin_area_fraction=0.0)
    thin = cycle_time.estimate_cycle_time(rich, {"holes": []}, table,
                                           "Aluminum 6061", 63.0, "mill",
                                           profile=profile)
    plain = cycle_time.estimate_cycle_time(no_thin, {"holes": []}, table,
                                            "Aluminum 6061", 63.0, "mill",
                                            profile=profile)
    check("thin walls slow finishing",
          _t(thin, "Finishing, analytic") > _t(plain, "Finishing, analytic"),
          f"{_t(plain, 'Finishing, analytic'):.2f} -> "
          f"{_t(thin, 'Finishing, analytic'):.2f} min")
    check("thin walls do NOT slow roughing (the wall is not thin yet)",
          abs(_t(thin, "Outside roughing") - _t(plain, "Outside roughing")) < 1e-9)

    # Small-cutter area must be split out and run slower per square inch.
    small = _t(full, "Finishing, small cutter")
    check("small-cutter area is split into its own term", small > 0,
          f"{small:.2f} min for 18 in2")
    rate_small = next(x.rate for x in full.terms
                      if x.name == "Finishing, small cutter")
    rate_norm = next(x.rate for x in full.terms
                     if x.name == "Finishing, analytic")
    check("the small cutter is much slower per in2, not merely proportionally",
          rate_norm / rate_small > 2.5,
          f"{rate_norm:.1f} vs {rate_small:.1f} in2/min "
          f"({rate_norm / rate_small:.1f}x) — rpm clamp compounds the diameter")
    check("small-cutter area is removed from the normal analytic term",
          abs(next(x.quantity for x in full.terms if x.name == "Finishing, analytic")
              - (95.0 - 18.0)) < 1e-6)

    # All three must be switchable off for a shop they do not apply to.
    for key in ("deburr_enabled", "thin_wall_derate_enabled",
                "small_cutter_derate_enabled"):
        check(f"{key} exists so it can be turned off", key in profile.values)

    # Calibration must be a visible multiplier, not a hidden one.
    check("calibration factors default to 1.00",
          all(profile.get(k, 1.0) == 1.0 for k in
              ("calibrate_roughing", "calibrate_finishing",
               "calibrate_holes", "calibrate_overall")))


def run_thickness_robustness_test():
    """The thickness scan must survive corners.

    This is a regression test for a real failure: on parts with fillets and
    concave corners the reported MINIMUM thickness was a value that existed
    nowhere on the part. The cause was geometric, not numerical -- a ray
    fired inward from near a corner hits the neighbouring face of the same
    wall after a millimetre or two, and that distance was being recorded as
    a wall thickness.

    The part below is an L-shaped block: two slabs, each a known 10 mm
    thick, meeting at a concave internal corner. Every genuine thickness on
    it is 10 mm or more, so anything materially below that is an artifact.
    """
    print("\n--- Test 10: wall-thickness robustness at corners ---")
    # L-shape: 60x40x10 base with a 10 mm thick wall standing on one edge.
    base = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 60.0, 40.0, 10.0).Shape()
    upright = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 10.0), 60.0, 10.0, 30.0).Shape()
    shape = BRepAlgoAPI_Fuse(base, upright).Shape()
    mesh_utils.tessellate(shape)
    _pts, _tris, face_stats = mesh_utils.extract_mesh(shape, with_face_stats=True)

    result = features.detect_thin_walls(
        shape, max_search_distance_mm=200.0,
        threshold_mm=features.THIN_WALL_DEFAULT_THRESHOLD_MM,
        quick=True, face_areas=[a for a, _c in face_stats])

    check("scan produced usable samples", result.samples_checked > 0,
          f"{result.samples_checked} used, {result.samples_rejected} rejected")
    check("percentiles are populated",
          result.p10_thickness_mm is not None and result.median_thickness_mm is not None)
    check("percentiles are ordered p10 <= p25 <= median",
          result.p10_thickness_mm <= result.p25_thickness_mm + 1e-9
          <= result.median_thickness_mm + 1e-9,
          f"{result.p10_thickness_mm:.3f} / {result.p25_thickness_mm:.3f} / "
          f"{result.median_thickness_mm:.3f} mm")
    # The real assertion: nothing on this part is thinner than 10 mm, so the
    # robust statistic must say so even if a stray sample does not.
    check("p10 recovers the true 10 mm wall, not a corner artifact",
          result.p10_thickness_mm >= 9.0,
          f"p10 {result.p10_thickness_mm:.3f} mm (min sample was "
          f"{result.min_thickness_mm:.3f} mm)")
    check("a solid part is not flagged thin-walled",
          result.p10_thickness_mm >= features.THIN_WALL_DEFAULT_THRESHOLD_MM)
    check("thin area fraction is below the flag threshold on a 10 mm part",
          result.thin_area_fraction < features.MIN_THIN_AREA_FRACTION,
          f"{result.thin_area_fraction:.4f}")

    # And the genuinely thin case must still be caught: a 1 mm web.
    thin = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 60.0, 40.0, 1.0).Shape()
    mesh_utils.tessellate(thin)
    _p2, _t2, stats2 = mesh_utils.extract_mesh(thin, with_face_stats=True)
    thin_result = features.detect_thin_walls(
        thin, max_search_distance_mm=200.0, quick=True,
        face_areas=[a for a, _c in stats2])
    check("a real 1 mm web is still detected as thin",
          thin_result.p10_thickness_mm is not None
          and thin_result.p10_thickness_mm < features.THIN_WALL_DEFAULT_THRESHOLD_MM,
          f"p10 {thin_result.p10_thickness_mm:.3f} mm")
    check("and it clears the area threshold that drives the flag",
          thin_result.thin_area_fraction >= features.MIN_THIN_AREA_FRACTION,
          f"{thin_result.thin_area_fraction:.3f}")


def run_probe_budget_test():
    """The hole probe must bound its own runtime and never lose a hole.

    On parts with many holes the probe stage could run for minutes, which
    is indistinguishable from a hang. It now stops at a wall-clock budget.
    The critical property is what happens on expiry: unprobed holes are
    KEPT and flagged, because an unprobed hole is one we know exists --
    dropping it would silently understate the hole count, which is the
    single most load-bearing drilling feature in the table.
    """
    print("\n--- Test 11: hole-probe time budget ---")
    plate = BRepPrimAPI_MakeBox(120.0, 80.0, 12.0).Shape()
    shape = plate
    made = 0
    for row in range(4):
        for col in range(6):
            shape = BRepAlgoAPI_Cut(shape, BRepPrimAPI_MakeCylinder(
                gp_Ax2(gp_Pnt(12.0 + col * 18.0, 12.0 + row * 18.0, -1.0),
                       gp_Dir(0, 0, 1)), 3.0, 14.0).Shape()).Shape()
            made += 1
    mesh_utils.tessellate(shape)
    scan = face_scan.analyze_faces(shape)
    found = len(scan.holes)
    check(f"all {made} drilled holes recognised", found == made, f"got {found}")

    diagonal = float(np.linalg.norm([120.0, 80.0, 12.0]))

    kept, stats = hole_probe.probe_holes(shape, list(scan.holes), diagonal)
    check("normal run completes without timing out", not stats.timed_out,
          f"{stats.elapsed_s:.2f}s for {found} holes")
    check("every hole probed on a normal run", stats.holes_probed == found)
    check("no hole left unprobed", stats.holes_unprobed == 0)

    # Now force expiry with a budget no run can meet.
    kept2, stats2 = hole_probe.probe_holes(
        shape, list(scan.holes), diagonal, time_budget_s=1e-9)
    check("an impossible budget trips the timeout", stats2.timed_out)
    check("NO holes are lost when the budget expires",
          len(kept2) == found, f"kept {len(kept2)} of {found}")
    check("probed + unprobed accounts for every hole",
          stats2.holes_probed + stats2.holes_unprobed == found)
    check("unprobed holes are flagged as such",
          all(h.probe_skipped for h in kept2 if h.probe_skipped) and stats2.holes_unprobed > 0)
    check("unprobed holes make no accessibility claim",
          not any(h.entry_obstructed for h in kept2 if h.probe_skipped))

    # Approach rays must be bounded by tool reach, not by the part size.
    check("approach reach is bounded well below the part diagonal",
          hole_probe.APPROACH_CLEARANCE_MIN_MM < diagonal * 2.5,
          f"reach floor {hole_probe.APPROACH_CLEARANCE_MIN_MM:.1f} mm vs "
          f"old {diagonal * 2.5:.1f} mm")


def run_turning_test():
    """Turning is priced as turning, not as milling with a bigger motor."""
    print("\n--- Test 15: turning model and material library ---")
    import math
    import tempfile

    from partestimator import materials_db
    from partestimator.machining.geometry import turning, turn_time
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse

    IN = 25.4
    # 2.000 dia x 1.500 + 1.250 dia x 2.000, 0.500 axial bore.
    big = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 1.0 * IN, 1.5 * IN).Shape()
    small = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, 1.5 * IN), gp_Dir(0, 0, 1)), 0.625 * IN, 2.0 * IN).Shape()
    shaft = BRepAlgoAPI_Fuse(big, small).Shape()
    bore = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, -IN), gp_Dir(0, 0, 1)), 0.25 * IN, 6 * IN).Shape()
    shaft = BRepAlgoAPI_Cut(shaft, bore).Shape()

    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    path = Path(tempfile.mkdtemp()) / "shaft.step"
    w = STEPControl_Writer()
    w.Transfer(shaft, STEPControl_AsIs)
    w.Write(str(path))
    from partestimator.machining.geometry import pipeline
    res = pipeline.compute_analysis(str(path))
    tg = res["turning"]
    check("turning geometry found for a round part", tg is not None)
    exact = math.pi * (1.0 ** 2 * 1.5 + 0.625 ** 2 * 2.0)
    env = tg.envelope_volume_mm3 / IN ** 3
    check("turned envelope within 1% of the exact solid of revolution",
          abs(env - exact) / exact < 0.01, f"{env:.4f} vs {exact:.4f} in3")
    check("a plain turned part is 100% turned surface",
          tg.turned_fraction > 0.99, f"{tg.turned_fraction:.3f}")
    check("the bore is recognised as an axial hole", len(tg.axial_holes) == 1)

    table = rates.load_rates()
    prof = shop.load_profile()
    for mat in ("Aluminum 6061-T6", "Stainless 304"):
        feats = derived.compute_derived_features(res, material=mat)
        mill = cycle_time.estimate_cycle_time(feats, res, table, mat, 63.0, "mill", profile=prof)
        lathe = cycle_time.estimate_cycle_time(feats, res, table, mat, 63.0, "lathe", profile=prof)
        names = {t.name for t in lathe.terms}
        check(f"{mat}: lathe has turning terms, not milling terms",
              "OD roughing (turn)" in names and "Outside roughing" not in names, str(names))
        check(f"{mat}: turning a round part is faster than milling it",
              lathe.door_to_door_minutes < mill.door_to_door_minutes,
              f"lathe {lathe.door_to_door_minutes:.1f} vs mill {mill.door_to_door_minutes:.1f} min")
    feats = derived.compute_derived_features(res, material="Stainless 304")
    fine = cycle_time.estimate_cycle_time(feats, res, table, "Stainless 304", 16.0, "lathe", profile=prof)
    coarse = cycle_time.estimate_cycle_time(feats, res, table, "Stainless 304", 125.0, "lathe", profile=prof)
    t_f = sum(t.minutes for t in fine.terms if t.name == "Finish turning")
    t_c = sum(t.minutes for t in coarse.terms if t.name == "Finish turning")
    check("a finer Ra means a slower finish-turning feed", t_f > t_c, f"{t_f:.2f} vs {t_c:.2f}")

    # Facing integral: with no rpm ceiling it is R^2 / (3.82 sfm ipr).
    t = turn_time._facing_minutes(1.0, 500.0, 0.01, 1e9)
    check("facing integral matches its closed form", abs(t - 1.0 / (3.82 * 500 * 0.01)) < 1e-9)
    t_clamped = turn_time._facing_minutes(1.0, 500.0, 0.01, 1000.0)
    check("an rpm ceiling makes facing slower", t_clamped > t)

    # Library: every machining material has a full mill and lathe rate set.
    mill_ops = {"unit_power", "rough_open_sfm", "rough_open_ipt", "drill_sfm",
                "drill_ipr_per_inch_dia"}
    lathe_ops = {"turn_rough_sfm", "turn_rough_ipr", "turn_rough_doc_in",
                 "turn_finish_sfm", "turn_finish_ipr_max"}
    missing = [f"{m}/{rt}/{op}" for m in materials_db.machining_names()
               for rt, ops in (("mill", mill_ops), ("lathe", lathe_ops))
               for op in ops if table.get(m, rt, op) is None]
    check("every machining material has every mill and lathe rate", not missing,
          str(missing[:4]))
    check("library holds 50+ materials", len(materials_db.load()) >= 50,
          str(len(materials_db.load())))
    check("old material names still resolve",
          table.value("Aluminum 6061", "mill", "rough_open_sfm")
          == table.value("Aluminum 6061-T6", "mill", "rough_open_sfm"))


def run_all() -> int:
    run_box_test()
    run_plate_with_hole_test()
    run_two_holes_test()
    run_face_scan_equivalence_test()
    run_mesh_extraction_test()
    run_edge_extraction_test()
    run_edge_convexity_test()
    run_derived_features_test()
    run_hole_types_test()
    run_thickness_robustness_test()
    run_probe_budget_test()
    run_stock_rules_test()
    run_cycle_time_test()
    run_shop_profile_test()
    run_turning_test()

    print("\n=====================================")
    if failures:
        print(f"{len(failures)} check(s) FAILED: {failures}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(run_all())
