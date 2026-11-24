"""The analysis pipeline as a plain function, free of any Qt dependency.

Kept separate from `ui.workers` so it can run either in a thread or -- the
important case -- inside a separate *process*. Everything it returns must
therefore be picklable: numpy arrays and simple dataclasses only. In
particular it never returns the `TopoDS_Shape`, which is a C++ object and
cannot cross a process boundary. Callers that need the shape afterwards
(the thin-wall scan) re-load the file on their own side.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np
from pathlib import Path

from . import (
    bounding_box,
    derived,
    edge_convexity,
    edges,
    extrusions,
    face_scan,
    features,
    hole_probe,
    holes as holes_mod,
    mesh_utils,
    properties,
    round_surfaces,
    step_io,
    stock as stock_mod,
    turning,
)

DEFAULT_OPTIONS = {
    "oriented_box": True,
    "face_aligned_box": True, # square the stock to the part's main flat faces
    "cylinder": True,
    "undercut": False,
    "edges": True,
    "robust_concavity": False,
    "edge_convexity": True,   # concave-edge metric for the ML feature set
    "extrusion_fit": True,    # open on an extrusion when one beats plate
}


def fit_extrusion(primary, rect_box, cyl, plate_stock, points, triangles,
                  part_volume_in3, profiles=None):
    """(shape, name) of the extrusion to open the part on, or None.

    Only for a part that is not round-bar work, and only when a profile
    holds the whole part and needs less material than the recommended
    plate: otherwise plate is the simpler buy and stays the default.
    """
    if primary != "rectilinear" or rect_box is None or plate_stock is None:
        return None
    pts = extrusions._sample(extrusions.surface_samples(points, triangles))
    pool = extrusions.load_profiles() if profiles is None else profiles
    prof = extrusions.best_profile(pool, rect_box, cyl, part_volume_in3=part_volume_in3,
                                   points_mm=pts)
    if prof is None:
        return None
    s = extrusions.extrusion_stock(prof, rect_box, cyl, part_volume_in3=part_volume_in3,
                                   points_mm=pts)
    if s is None or not (s.fits and s.enough_material):
        return None
    # The profile cannot hold more metal than its own section along the part:
    # a check that does not depend on how the surface happened to be sampled.
    if (part_volume_in3 is not None and prof.area_in2 * s.min_length_in
            < (1.0 - extrusions.VOLUME_SLACK) * part_volume_in3):
        return None
    if s.volume_in3 >= plate_stock.volume_in3:
        return None
    return prof.shape, prof.name


def compute_analysis(path: str, options: dict | None = None, progress=None,
                      should_cancel=None) -> dict:
    """Runs the full load + analysis and returns a picklable result dict.

    `progress(message)` is called at each stage boundary and periodically
    within the face scan. `should_cancel()` is polled where it's possible
    to honour it -- note that it CANNOT interrupt a single long-running
    OCCT call (see the module docstring in ui/proc_workers.py), which is
    why the process-based runner also supports hard termination.
    """
    opt = {**DEFAULT_OPTIONS, **(options or {})}
    if opt.get("stock_allowances") is not None:
        # The shop's own stock allowances (Settings...): a worker process
        # starts on the shipped ones.
        stock_mod.set_allowances(**opt["stock_allowances"])
    timings: dict[str, float] = {}

    def emit(msg):
        if progress is not None:
            progress(msg)

    @contextmanager
    def stage(label: str, message: str):
        emit(message)
        t0 = time.perf_counter()
        yield
        timings[label] = time.perf_counter() - t0

    with stage("STEP parse", f"Reading {Path(path).name}…"):
        shape = step_io.load_step_file(path)

    with stage("Tessellate", "Tessellating…"):
        deflection = mesh_utils.tessellate(shape)
        diagonal = mesh_utils.shape_diagonal_mm(shape)

    with stage("Mesh extract", "Extracting mesh…"):
        # One face map shared with the face scan, and per-face area/centroid
        # computed here while the nodes are already loaded -- the scan would
        # otherwise walk every triangulation a second time.
        face_map = mesh_utils.face_map_of(shape)
        render_points, render_triangles, face_stats = mesh_utils.extract_mesh(
            shape, face_map=face_map, with_face_stats=True,
        )
        vertices = render_points

    with stage("Edge extract", "Extracting edges…"):
        edge_polylines = (edges.extract_edge_polylines(shape, diagonal_mm=diagonal)
                          if opt["edges"] else [])
        # Finer, for the tabbed-fixturing channel round the part's outline.
        outline_edges = edges.extract_outline_edges(shape)

    with stage("Mass props", "Computing mass properties…"):
        shape_props = properties.compute_shape_properties(shape)
        moi_value = properties.spherical_moment_of_inertia(shape_props.inertia_tensor_about_centroid)
        gyration_mm = properties.radius_of_gyration_spherical(moi_value, shape_props.volume_mm3)

    with stage("Convex hull", "Building convex hull…"):
        hull_data = bounding_box.prepare_hull(vertices)

    with stage("Face scan", "Analyzing faces…"):
        def _face_progress(done, total):
            emit(f"Analyzing faces… {done:,} / {total:,}")

        scan = face_scan.analyze_faces(
            shape,
            do_undercut=opt["undercut"],
            robust=opt["robust_concavity"],
            progress=_face_progress,
            should_cancel=should_cancel,
            face_map=face_map,
            face_stats=face_stats,
        )
        # Ray-cast validation (is it a real enclosed bore?) and tool-access
        # probing from each open mouth. Rejected features are dropped.
        def _probe_progress(done, total):
            emit(f"Probing holes… {done:,} / {total:,}")

        all_holes = scan.holes
        validated_holes, probe_stats = hole_probe.probe_holes(
            shape, all_holes, diagonal, should_cancel=should_cancel,
            progress=_probe_progress,
        )
        rejected_hole_count = probe_stats.holes_rejected
        if probe_stats.timed_out:
            emit(f"Hole probing hit its {hole_probe.PROBE_TIME_BUDGET_S:.0f}s budget; "
                 f"{probe_stats.holes_unprobed:,} of {probe_stats.holes_in:,} holes kept unprobed")
        hole_groups = holes_mod.group_holes_by_diameter(validated_holes)
        deep_holes = features.flag_deep_holes(validated_holes)
        smallest_radii = features.smallest_fillet_radii_mm(scan.fillets)

    with stage("Bounding boxes", "Fitting bounding boxes…"):
        rect_box = None
        if opt["oriented_box"] and opt["face_aligned_box"]:
            # Squared to the part's own faces, as it would be set up in the
            # vise; the tightest box only for a part with too little flat.
            rect_box = bounding_box.face_aligned_box(
                vertices, scan.planes, shape_props.surface_area_mm2, hull_data)
        if rect_box is None and opt["oriented_box"]:
            rect_box = bounding_box.minimum_oriented_box(vertices, hull_data)
        elif rect_box is None:
            rect_box = bounding_box.axis_aligned_box(vertices)
        cyl = (bounding_box.minimum_enclosing_cylinder(vertices, hull_data)
               if opt["cylinder"] else None)
        primary = "rectilinear" if cyl is None else bounding_box.select_primary(rect_box, cyl)
        stock = stock_mod.compute_stock(primary, rect_box, cyl)

    with stage("Extrusion fit", "Fitting extrusion profiles…"):
        extrusion = None
        if opt["extrusion_fit"]:
            try:
                extrusion = fit_extrusion(
                    primary, rect_box, cyl, stock, render_points, render_triangles,
                    shape_props.volume_mm3 / stock_mod.MM_PER_INCH ** 3)
            except Exception as exc:  # noqa: BLE001 -- plate stock is a fine fallback
                emit(f"Extrusion fit skipped: {type(exc).__name__}")

    with stage("Turning", "Measuring the turned profile…"):
        try:
            turning_geo = turning.analyze_turning(
                shape, cyl, render_points, render_triangles, validated_holes)
        except Exception as exc:  # noqa: BLE001 -- turning is optional context
            emit(f"Turning analysis skipped: {type(exc).__name__}")
            turning_geo = None

    with stage("Wall thickness", "Checking wall thickness…"):
        # Quick pass on every load: one ray per (non-sliver) face, capped.
        # Enough to answer "does this part have thin walls", which is a real
        # feed/depth-of-cut driver; the thorough grid stays on the button.
        thin_wall = features.detect_thin_walls(
            shape,
            max_search_distance_mm=float(np.linalg.norm(rect_box.extents_mm)) * 1.5,
            threshold_mm=features.THIN_WALL_DEFAULT_THRESHOLD_MM,
            quick=True,
            face_areas=[a for a, _c in face_stats],
            should_cancel=should_cancel,
        )

    with stage("Edge convexity", "Classifying edges…"):
        convexity = (edge_convexity.compute_edge_convexity(shape, should_cancel=should_cancel)
                     if opt["edge_convexity"] else None)

    result = {
        "path": path,
        "deflection_mm": deflection,
        "diagonal_mm": diagonal,
        "timings": timings,
        "options": dict(opt),
        "render_points": render_points,
        "render_triangles": render_triangles,
        "edge_polylines": edge_polylines,
        "outline_edges": outline_edges,
        "shape_props": shape_props,
        "rect_box": rect_box,
        "cylinder": cyl,
        "primary": primary,
        "stock": stock,
        "extrusion": extrusion,
        "moi_value_mm5": moi_value,
        "gyration_mm": gyration_mm,
        "holes": validated_holes,
        "rejected_hole_count": rejected_hole_count,
        "holes_unprobed": probe_stats.holes_unprobed,
        "probe_timed_out": probe_stats.timed_out,
        "fillets": scan.fillets,
        "hole_groups": hole_groups,
        "deep_holes": deep_holes,
        "smallest_fillet_radii_mm": smallest_radii,
        "surface_split": scan.surface_split,
        "undercut": scan.undercut,
        "face_count": scan.face_count,
        "cones": scan.cones,
        "planes": scan.planes,
        "edge_convexity": convexity,
        "hull_volume_mm3": hull_data.hull_volume_mm3,
        "hull_is_approximate": hull_data.hull_is_approximate,
        "thin_wall": thin_wall,
        "turning": turning_geo,
    }

    with stage("Derived features", "Deriving features…"):
        result["derived"] = derived.compute_derived_features(result)

    result["timings"] = timings
    return result


def compute_turning(path: str, cylinder, holes, progress=None):
    """Re-loads the shape and measures its turned profile about `cylinder`'s
    axis -- for a bar the user has set a different way round than the
    automatic fit. Re-loaded for the same reason as compute_thin_wall: the
    face classification needs the shape, which cannot cross processes. The
    mesh is re-extracted the way compute_analysis does it, so it is the same
    one the window holds."""
    if progress is not None:
        progress(f"Reading {Path(path).name}…")
    shape = step_io.load_step_file(path)
    mesh_utils.tessellate(shape)
    if progress is not None:
        progress("Measuring the turned profile about the new axis…")
    points, triangles = mesh_utils.extract_mesh(shape, face_map=mesh_utils.face_map_of(shape))
    return turning.analyze_turning(shape, cylinder, points, triangles, holes)


def compute_thin_wall(path: str, threshold_mm: float, max_search_distance_mm: float,
                       progress=None):
    """Re-loads the shape and runs the thin-wall scan.

    Re-reading the STEP file looks wasteful, but a TopoDS_Shape can't be
    sent between processes, and this scan is opt-in and already the slowest
    operation -- the parse is small beside it, and running it out-of-process
    is what keeps the window alive.
    """
    if progress is not None:
        progress(f"Reading {Path(path).name}…")
    shape = step_io.load_step_file(path)
    mesh_utils.tessellate(shape)
    if progress is not None:
        progress("Running thin-wall scan…")
    # Face areas are needed even in thorough mode: without them every
    # sample carries equal weight and the percentiles revert to being
    # dominated by whichever faces happen to be most numerous, which on a
    # blended part means the fillets.
    _pts, _tris, face_stats = mesh_utils.extract_mesh(shape, with_face_stats=True)
    return features.detect_thin_walls(
        shape,
        max_search_distance_mm=max_search_distance_mm,
        threshold_mm=threshold_mm,
        face_areas=[a for a, _c in face_stats],
    )
