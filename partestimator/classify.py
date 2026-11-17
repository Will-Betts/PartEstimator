"""Is this STEP part sheet metal or a machined part?

The rule (chosen by the user, and editable):

    1. BENT + uniform thickness          -> sheet metal, any thickness
    2. FLAT + uniform thickness <= cutoff -> sheet metal (a laser/punch blank)
    3. anything else                     -> machined

with the cutoff defaulting to 0.190 in -- the same figure the machining side
already uses for "sheet bought to gauge, never faced".

How "uniform thickness" is measured
-----------------------------------
Rays. Points are sampled over every face in proportion to its area; from each
one a ray is fired straight into the material along the inward normal, and the
distance to where it comes out is recorded -- but only when it comes out
through a face that genuinely faces back along the ray (within 10 degrees). A
ray that grazes the far side of a fillet or the wall of a corner measures
nothing useful and is dropped.

On sheet metal almost every surviving sample is the sheet thickness: skin to
skin across a flange, and inner to outer cylinder across a bend. Two kinds of
face give a different answer and are set aside before judging:

  * EDGE BANDS -- the cut edges of the blank and the walls of holes through
    it. They are one thickness tall, and a ray fired from them runs across the
    part, not through the sheet. A face whose own narrow dimension equals the
    thickness is a band.

Then, area-weighted:

  * at least 80% of what remains must sit within +/-3% of the thickness, and
  * no more than 1% may be THINNER than it -- a pocket, a step or a spot face
    is exactly a thinner region, and it is the signature of a machined part
    that happens to be made from plate.

"Flat" means every sheet-skin normal is parallel to one direction. "Bent"
means the sheet metal tool's own bend detector found at least one brake bend
-- the same code that will then develop it, so the two cannot disagree.

All lengths internally are millimetres; the result reports inches.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np

MM_PER_IN = 25.4
DEFAULT_CUTOFF_IN = 0.190

SHEET = "sheet_metal"
MACHINED = "machined"

UNIFORM_MIN_FRACTION = 0.80
THIN_MAX_FRACTION = 0.01
THICKNESS_REL_TOL = 0.03
THICKNESS_ABS_TOL_MM = 0.05
OPPOSING_MIN_COS = math.cos(math.radians(10.0))
FLAT_MAX_ANGLE_DEG = 1.0
TARGET_SAMPLES = 2500
MAX_SAMPLES_PER_FACE = 36
TIME_BUDGET_S = 20.0


@dataclass
class Classification:
    kind: str                         # SHEET or MACHINED
    reason: str                       # one plain sentence for the banner
    cutoff_in: float
    thickness_in: float | None = None
    bent: bool = False
    flat: bool = False
    bend_count: int = 0
    uniform_fraction: float = 0.0
    thin_fraction: float = 0.0
    samples: int = 0
    solids: int = 1
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def is_sheet(self) -> bool:
        return self.kind == SHEET

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# the decision, kept free of the kernel so it can be tested on numbers alone
# ---------------------------------------------------------------------------


def decide(*, uniform: bool, flat: bool, bend_count: int, thickness_in: float | None,
           cutoff_in: float, solids: int = 1, uniform_fraction: float = 0.0,
           thin_fraction: float = 0.0, thickness_range_in=None) -> tuple[str, str]:
    """The rule. Returns (kind, reason)."""
    t = f'{thickness_in:.3f}"' if thickness_in else "?"
    if solids != 1:
        return MACHINED, f"{solids} separate bodies in the file -- treated as machined."
    if not uniform:
        if thin_fraction > THIN_MAX_FRACTION:
            return MACHINED, (
                f"Thickness is not uniform: {thin_fraction:.0%} of the surface is thinner "
                f"than the main {t} wall (pockets or steps) -- machined.")
        span = ""
        if thickness_range_in:
            lo, hi = thickness_range_in
            span = f' (measured {lo:.3f}" to {hi:.3f}")'
        return MACHINED, f"Thickness is not uniform{span} -- machined."
    if bend_count > 0:
        return SHEET, (f"Sheet metal: {bend_count} bend{'s' if bend_count != 1 else ''}, "
                       f"uniform {t} thick.")
    if not flat:
        if thickness_in is not None and thickness_in > cutoff_in:
            return MACHINED, f"No brake bends and {t} through -- machined."
        return MACHINED, (f"Uniform {t} wall, but not flat and no brake bends found "
                          f"(tube or rolled form?) -- machined.")
    if thickness_in is not None and thickness_in <= cutoff_in + 1e-9:
        return SHEET, f'Sheet metal: flat blank, {t} thick (cutoff {cutoff_in:.3f}").'
    return MACHINED, (f'Flat, uniform {t} thick -- above the {cutoff_in:.3f}" sheet cutoff, '
                      f"so machined plate.")


# ---------------------------------------------------------------------------
# kernel side
# ---------------------------------------------------------------------------


def classify_file(path, cutoff_in: float = DEFAULT_CUTOFF_IN) -> Classification:
    from .smt.occ.load import load_step

    return classify_shape(load_step(path), cutoff_in)


def classify_shape(shape, cutoff_in: float = DEFAULT_CUTOFF_IN) -> Classification:
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    t0 = time.perf_counter()
    solids = 0
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        solids += 1
        exp.Next()

    if solids != 1:
        kind, reason = decide(uniform=False, flat=False, bend_count=0, thickness_in=None,
                              cutoff_in=cutoff_in, solids=solids)
        return Classification(kind, reason, cutoff_in, solids=solids,
                              seconds=time.perf_counter() - t0)

    samples, timed_out = _thickness_samples(shape, t0)
    c = _judge(samples, cutoff_in, shape)
    c.solids = solids
    if timed_out:
        c.notes.append("thickness scan hit its time budget; judged on a partial sample")
    c.seconds = time.perf_counter() - t0
    return c


def _judge(samples, cutoff_in, shape) -> Classification:
    if not samples:
        kind, reason = MACHINED, "No through-thickness measurement was possible -- machined."
        return Classification(kind, reason, cutoff_in)

    d = np.array([s[0] for s in samples])
    w = np.array([s[1] for s in samples])
    normals = np.array([s[2] for s in samples])
    face_ids = np.array([s[3] for s in samples])
    band_width = {s[3]: s[4] for s in samples}

    t = _weighted_quantile(d, w, 0.5)
    tol = max(THICKNESS_REL_TOL * t, THICKNESS_ABS_TOL_MM)

    # Set aside edge bands: faces whose own narrow dimension is the thickness.
    is_band = np.array([abs((band_width.get(f) or -1e9) - t) <= tol for f in face_ids])
    keep = ~is_band
    if not keep.any():
        keep = np.ones_like(is_band)
    dk, wk, nk = d[keep], w[keep], normals[keep]
    t = _weighted_quantile(dk, wk, 0.5)
    tol = max(THICKNESS_REL_TOL * t, THICKNESS_ABS_TOL_MM)

    total = float(wk.sum()) or 1.0
    on_t = np.abs(dk - t) <= tol
    uniform_fraction = float(wk[on_t].sum()) / total
    thin_fraction = float(wk[dk < t - tol].sum()) / total
    uniform = uniform_fraction >= UNIFORM_MIN_FRACTION and thin_fraction <= THIN_MAX_FRACTION

    flat = False
    if on_t.any():
        n = nk[on_t]
        ref = n[int(np.argmax(wk[on_t]))]
        cosines = np.abs(n @ ref)
        flat = bool(np.all(cosines >= math.cos(math.radians(FLAT_MAX_ANGLE_DEG))))

    bend_count = 0
    notes: list[str] = []
    if uniform and not flat:
        bend_count, note = _bend_count(shape)
        if note:
            notes.append(note)

    t_in = t / MM_PER_IN
    lo, hi = _weighted_quantile(dk, wk, 0.05), _weighted_quantile(dk, wk, 0.95)
    kind, reason = decide(
        uniform=uniform, flat=flat, bend_count=bend_count, thickness_in=t_in,
        cutoff_in=cutoff_in, uniform_fraction=uniform_fraction,
        thin_fraction=thin_fraction, thickness_range_in=(lo / MM_PER_IN, hi / MM_PER_IN),
    )
    return Classification(
        kind=kind, reason=reason, cutoff_in=cutoff_in, thickness_in=t_in,
        bent=bend_count > 0, flat=flat, bend_count=bend_count,
        uniform_fraction=uniform_fraction, thin_fraction=thin_fraction,
        samples=len(samples), notes=notes,
    )


def _bend_count(shape) -> tuple[int, str]:
    """Ask the sheet metal tool's own detector. 0 if it cannot make sense of it."""
    try:
        from .smt.config import UnfoldOptions
        from .smt.occ.detect import detect

        part = detect(shape, UnfoldOptions())
        return len(part.bends), ""
    except Exception as exc:  # noqa: BLE001 -- not developable is an answer
        return 0, f"bend detector declined the part: {type(exc).__name__}"


def _weighted_quantile(values, weights, q: float) -> float:
    order = np.argsort(values)
    v = np.asarray(values, float)[order]
    w = np.asarray(weights, float)[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return float(np.median(v))
    return float(v[min(np.searchsorted(cw, q * cw[-1]), len(v) - 1)])


def _thickness_samples(shape, t0):
    """[(distance_mm, weight, skin_normal, face_index, band_width_mm|None)], timed_out."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepGProp import BRepGProp
    from OCP.BRepLProp import BRepLProp_SLProps
    from OCP.BRepTopAdaptor import BRepTopAdaptor_FClass2d
    from OCP.GProp import GProp_GProps
    from OCP.gp import gp_Dir, gp_Lin, gp_Pnt, gp_Pnt2d
    from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector
    from OCP.TopAbs import TopAbs_FACE, TopAbs_IN, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    faces, areas = [], []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        f = TopoDS.Face_s(exp.Current())
        g = GProp_GProps()
        BRepGProp.SurfaceProperties_s(f, g)
        faces.append(f)
        areas.append(abs(g.Mass()))
        exp.Next()
    total_area = sum(areas) or 1.0

    inter = IntCurvesFace_ShapeIntersector()
    inter.Load(shape, 1e-6)

    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    x0, y0, z0, x1, y1, z1 = box.Get()
    reach = math.dist((x0, y0, z0), (x1, y1, z1)) * 1.1 + 1.0

    def outward(face, adaptor, u, v):
        props = BRepLProp_SLProps(adaptor, u, v, 1, 1e-6)
        if not props.IsNormalDefined():
            return None
        n = props.Normal()
        vec = np.array([n.X(), n.Y(), n.Z()])
        if face.Orientation() == TopAbs_REVERSED:
            vec = -vec
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-12 else None

    out = []
    for fi, (face, area) in enumerate(zip(faces, areas)):
        if time.perf_counter() - t0 > TIME_BUDGET_S:
            return out, True
        if area < 1e-5 * total_area:
            continue
        n_target = int(min(MAX_SAMPLES_PER_FACE, max(1, round(TARGET_SAMPLES * area / total_area))))
        grid = max(1, int(math.ceil(math.sqrt(n_target))))
        adaptor = BRepAdaptor_Surface(face, True)
        u0, u1 = adaptor.FirstUParameter(), adaptor.LastUParameter()
        v0, v1 = adaptor.FirstVParameter(), adaptor.LastVParameter()
        fclass = BRepTopAdaptor_FClass2d(face, 1e-7)
        band = _band_width(face, adaptor)

        face_samples = []
        # Interior grid, offset so a 1x1 grid lands in the middle. Several
        # offsets are tried for faces whose middle is a hole.
        for shift in (0.5, 0.27, 0.73):
            for i in range(grid):
                for j in range(grid):
                    u = u0 + (u1 - u0) * ((i + shift) / grid)
                    v = v0 + (v1 - v0) * ((j + shift) / grid)
                    if fclass.Perform(gp_Pnt2d(u, v)) != TopAbs_IN:
                        continue
                    n = outward(face, adaptor, u, v)
                    if n is None:
                        continue
                    p = adaptor.Value(u, v)
                    ray = -n
                    inter.PerformNearest(
                        gp_Lin(gp_Pnt(p.X(), p.Y(), p.Z()), gp_Dir(*ray)), 1e-4, reach)
                    if not (inter.IsDone() and inter.NbPnt() > 0):
                        continue
                    dist = float(inter.WParameter(1))
                    if dist <= 1e-3:
                        continue
                    try:
                        hf = inter.Face(1)
                        hn = outward(hf, BRepAdaptor_Surface(hf, True),
                                     inter.UParameter(1), inter.VParameter(1))
                    except Exception:  # noqa: BLE001
                        hn = None
                    if hn is None or float(hn @ ray) < OPPOSING_MIN_COS:
                        continue
                    face_samples.append((dist, n))
            if face_samples:
                break
        if face_samples:
            wgt = area / len(face_samples)
            out.extend((d, wgt, n, fi, band) for d, n in face_samples)
    return out, False


def _band_width(face, adaptor) -> float | None:
    """The narrow dimension of a face, for spotting edge bands and hole walls.

    Planar: minimum width of its boundary over directions in its plane.
    Cylinder / cone: extent of its boundary along the axis.
    Anything else: None (never treated as a band).
    """
    from OCP.GeomAbs import GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Plane

    kind = adaptor.GetType()
    pts = _boundary_points(face)
    if len(pts) < 3:
        return None
    if kind == GeomAbs_Plane:
        pl = adaptor.Plane()
        o, xd, yd = pl.Location(), pl.XAxis().Direction(), pl.YAxis().Direction()
        org = np.array([o.X(), o.Y(), o.Z()])
        ex = np.array([xd.X(), xd.Y(), xd.Z()])
        ey = np.array([yd.X(), yd.Y(), yd.Z()])
        q = np.c_[(pts - org) @ ex, (pts - org) @ ey]
        best = math.inf
        for a in np.radians(np.arange(0.0, 180.0, 1.0)):
            proj = q @ np.array([math.cos(a), math.sin(a)])
            best = min(best, float(proj.max() - proj.min()))
        return best
    if kind in (GeomAbs_Cylinder, GeomAbs_Cone):
        ax = (adaptor.Cylinder() if kind == GeomAbs_Cylinder else adaptor.Cone()).Axis()
        d = ax.Direction()
        proj = pts @ np.array([d.X(), d.Y(), d.Z()])
        return float(proj.max() - proj.min())
    return None


def _boundary_points(face) -> np.ndarray:
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    from .smt.occ.topo import edge_has_curve

    pts = []
    exp = TopExp_Explorer(face, TopAbs_EDGE)
    while exp.More():
        e = TopoDS.Edge_s(exp.Current())
        exp.Next()
        if not edge_has_curve(e):
            continue
        c = BRepAdaptor_Curve(e)
        a, b = c.FirstParameter(), c.LastParameter()
        for k in range(9):
            p = c.Value(a + (b - a) * k / 8)
            pts.append((p.X(), p.Y(), p.Z()))
    return np.array(pts) if pts else np.zeros((0, 3))


def classify_to_dict(path: str, cutoff_in: float) -> dict:
    """Process-pool entry point: plain data in, plain data out."""
    return classify_file(path, cutoff_in).to_dict()


def from_dict(d: dict) -> Classification:
    return Classification(**d)
