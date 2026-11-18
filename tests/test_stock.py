"""Stock choices (issue #2) and extrusion profiles (issue #4).

Pure Python: boxes and cylinders are plain stand-ins for the kernel's fits,
so this runs on the CI runner with the timing tests.
"""
from __future__ import annotations

import math
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from partestimator.machining.geometry import extrusions as ex
from partestimator.machining.geometry import stock as st

MM = 25.4
X, Y, Z = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)


def box(l_in, w_in, t_in, axes=(X, Y, Z)):
    return SimpleNamespace(extents_mm=[l_in * MM, w_in * MM, t_in * MM],
                           axes=[list(a) for a in axes], center_mm=[1.0, 2.0, 3.0])


def cyl(d_in, h_in, axis=Z):
    return SimpleNamespace(radius_mm=d_in / 2 * MM, height_mm=h_in * MM,
                           axis_direction=list(axis), axis_point_mm=[0.0, 0.0, 0.0])


# --------------------------------------------------------------------------
# issue #2: choosing the stock
# --------------------------------------------------------------------------


class TestMinimumAndCustom(unittest.TestCase):
    def test_minimum_is_the_bare_envelope(self):
        s = st.minimum_rectilinear(box(4.0, 2.5, 0.75).extents_mm)
        self.assertEqual(s.basis, st.MINIMUM)
        self.assertAlmostEqual(s.length_in, 4.0)
        self.assertAlmostEqual(s.width_in, 2.5)
        self.assertAlmostEqual(s.thickness_or_diameter_in, 0.75)
        self.assertAlmostEqual(s.volume_in3, 4.0 * 2.5 * 0.75)

    def test_minimum_is_smaller_than_recommended(self):
        b = box(4.0, 2.5, 0.75)
        self.assertLess(st.minimum_rectilinear(b.extents_mm).volume_in3,
                        st.rectilinear_stock(b.extents_mm).volume_in3)

    def test_custom_sorts_its_dimensions(self):
        s = st.custom_rectilinear(6.0, 0.5, 3.0)
        self.assertEqual((s.thickness_or_diameter_in, s.width_in, s.length_in), (0.5, 3.0, 6.0))
        self.assertEqual(s.basis, st.CUSTOM)

    def test_custom_thin_plate_counts_as_sheet(self):
        self.assertTrue(st.custom_rectilinear(0.125, 4, 6).is_sheet)
        self.assertFalse(st.custom_rectilinear(0.25, 4, 6).is_sheet)

    def test_round_bar(self):
        s = st.custom_cylindrical(2.0, 5.0)
        self.assertTrue(s.is_round)
        self.assertIsNone(s.width_in)
        self.assertAlmostEqual(s.volume_in3, math.pi * 5.0)

    def test_minimum_bar_is_the_cylinder_fit(self):
        s = st.snap("cylindrical", st.MINIMUM, None, cyl(1.8, 4.2))
        self.assertAlmostEqual(s.thickness_or_diameter_in, 1.8)
        self.assertAlmostEqual(s.length_in, 4.2)

    def test_snap_without_a_fit_is_none(self):
        self.assertIsNone(st.snap("cylindrical", st.RECOMMENDED, box(1, 1, 1), None))
        self.assertIsNone(st.snap("rectilinear", st.MINIMUM, None, cyl(1, 1)))


class TestBasis(unittest.TestCase):
    def setUp(self):
        self.b = box(4.0, 2.5, 0.75)

    def test_typing_the_recommended_size_reads_as_recommended(self):
        rec = st.rectilinear_stock(self.b.extents_mm)
        typed = st.custom_rectilinear(rec.thickness_or_diameter_in, rec.width_in, rec.length_in)
        self.assertEqual(st.classify_basis(typed, self.b, None).basis, st.RECOMMENDED)

    def test_typing_the_part_envelope_reads_as_minimum(self):
        typed = st.custom_rectilinear(0.75, 2.5, 4.0)
        self.assertEqual(st.classify_basis(typed, self.b, None).basis, st.MINIMUM)

    def test_anything_else_is_custom(self):
        typed = st.custom_rectilinear(1.0, 3.0, 5.0)
        self.assertEqual(st.classify_basis(typed, self.b, None).basis, st.CUSTOM)


class TestFitWarnings(unittest.TestCase):
    def test_enclosing_stock_is_fine(self):
        b = box(4.0, 2.5, 0.75)
        self.assertEqual(st.fit_warnings(st.rectilinear_stock(b.extents_mm), b, None), [])
        self.assertEqual(st.fit_warnings(st.minimum_rectilinear(b.extents_mm), b, None), [])

    def test_stock_smaller_than_the_part_is_flagged(self):
        b = box(4.0, 2.5, 0.75)
        warn = st.fit_warnings(st.custom_rectilinear(0.5, 2.5, 4.0), b, None)
        self.assertEqual(len(warn), 1)
        self.assertIn("does not enclose", warn[0])

    def test_short_bar_is_flagged(self):
        warn = st.fit_warnings(st.custom_cylindrical(2.0, 3.0), None, cyl(1.8, 4.2))
        self.assertTrue(any("shorter" in w for w in warn))


# --------------------------------------------------------------------------
# issue #4: extrusion profiles
# --------------------------------------------------------------------------


class TestAreas(unittest.TestCase):
    def area(self, shape, a, b=0.0, t=0.0, t2=0.0):
        return ex.ExtrusionProfile("x", shape, a, b, t, t2).area_in2

    def test_every_shape_against_hand_arithmetic(self):
        self.assertAlmostEqual(self.area("angle", 2, 2, 0.25), 0.25 * 2 + 0.25 * 1.75)
        self.assertAlmostEqual(self.area("channel", 4, 2, 0.25, 0.25), 4 * 0.25 + 2 * 1.75 * 0.25)
        self.assertAlmostEqual(self.area("i_beam", 4, 3, 0.25, 0.25), 2 * 3 * 0.25 + 3.5 * 0.25)
        self.assertAlmostEqual(self.area("tee", 2, 2, 0.25, 0.25), 2 * 0.25 + 1.75 * 0.25)
        self.assertAlmostEqual(self.area("rect_tube", 2, 1, 0.125), 2 - 1.75 * 0.75)
        self.assertAlmostEqual(self.area("round_tube", 2, 0, 0.25), math.pi / 4 * (4 - 2.25))
        self.assertAlmostEqual(self.area("flat_bar", 3, 0.5), 1.5)
        self.assertAlmostEqual(self.area("hex_bar", 1), math.sqrt(3) / 2)

    def test_outline_area_matches_the_formula(self):
        """The drawn outline and the priced area are the same section."""
        def shoelace(loop):
            return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2)
                           in zip(loop, loop[1:] + loop[:1]))) / 2.0

        for p in ex.load_profiles():
            loops = p.outline_loops()
            area = shoelace(loops[0]) - sum(shoelace(lp) for lp in loops[1:])
            tol = 0.01 * p.section_area_in2 if p.shape == "round_tube" else 1e-9
            with self.subTest(profile=p.describe()):
                self.assertAlmostEqual(area, p.section_area_in2, delta=tol)

    def test_new_shapes_against_hand_arithmetic(self):
        zee = ex.ExtrusionProfile("z", "zee", 1.0, 0.75, 0.0625, c=0.5)
        self.assertAlmostEqual(zee.section_area_in2,
                               1.0 * 0.0625 + (0.75 - 0.0625) * 0.0625 + (0.5 - 0.0625) * 0.0625)
        self.assertEqual(zee.envelope_in, (0.75 + 0.5 - 0.0625, 1.0))
        # A bulb angle is the plain angle plus the part of the bulb off the leg.
        a, b, t, r = 0.5, 0.5, 0.04, 0.06
        bulb = ex.ExtrusionProfile("b", "bulb_angle", a, b, t, r=r).section_area_in2
        d = r - t
        under_leg = r * r * math.acos(d / r) - d * math.sqrt(r * r - d * d)
        self.assertAlmostEqual(bulb, t * (a + b - t) + math.pi * r * r - under_leg, delta=2e-4)


class TestStandardProfiles(unittest.TestCase):
    """The AND / AMS library (standard_extrusions.csv)."""

    def test_every_standard_loads(self):
        std = ex.load_standard()
        self.assertEqual({p.spec for p in std}, set(ex.STANDARDS))
        self.assertGreater(len(std), 400)

    def test_listed_areas_agree_with_the_dimensions(self):
        """A mistyped dimension shows up as an area far from the listed one;
        the standards' radii only move it a few percent."""
        for p in ex.load_standard():
            if not p.listed_area_in2:
                continue
            with self.subTest(profile=p.name):
                self.assertAlmostEqual(p.section_area_in2, p.listed_area_in2,
                                       delta=0.07 * p.listed_area_in2)

    def test_part_number_lookup_either_standard(self):
        p = ex.find_part_number("AND10134-1205")
        self.assertEqual((p.shape, p.a, p.b, p.t), ("angle", 1.25, 1.0, 0.094))
        self.assertAlmostEqual(p.area_in2, 0.202)       # the standard's own area
        self.assertEqual(p.alias, "AMS10134-1205")
        for text in ("AMS10134-1205", "ams 10134 1205", "AND10134 - 1205"):
            self.assertIs(ex.find_part_number(text), p)
        self.assertIsNone(ex.find_part_number("AND10134-9999"))
        self.assertIsNone(ex.find_part_number("angle 2 x 2"))

    def test_stock_volume_uses_the_listed_area(self):
        p = ex.find_part_number("AND10137-2005")          # channel 2 x 1 x 1/8
        s = ex.extrusion_stock(p, box(6.0, 1.9, 0.9), length_in=10.0)
        self.assertAlmostEqual(s.volume_in3, 0.469 * 10.0)

    def test_bulb_shapes_draw_inside_their_envelope(self):
        for name in ("AND10135-1603", "AND10141-2002"):
            p = ex.find_part_number(name)
            ex_, ey = p.envelope_in
            xs = [x for x, _ in p.outline_loops()[0]]
            ys = [y for _, y in p.outline_loops()[0]]
            with self.subTest(profile=name):
                self.assertAlmostEqual(max(xs) - min(xs), ex_, places=6)
                self.assertAlmostEqual(max(ys) - min(ys), ey, places=6)
                self.assertAlmostEqual(max(xs), -min(xs), places=9)


class TestShippedProfiles(unittest.TestCase):
    def test_the_file_loads_and_covers_every_shape(self):
        profiles = ex.load_profiles()
        self.assertGreater(len(profiles), 20)
        self.assertEqual({p.shape for p in profiles}, set(ex.SHAPES))
        for p in profiles:
            with self.subTest(profile=p.describe()):
                self.assertGreater(p.area_in2, 0)
                self.assertIn("Verify", p.notes)

    def test_impossible_rows_are_rejected(self):
        with self.assertRaises(ValueError):
            ex.validate(ex.ExtrusionProfile("bad", "rect_tube", 1.0, 1.0, 0.6))
        with self.assertRaises(ValueError):
            ex.validate(ex.ExtrusionProfile("bad", "pentagon", 1.0))


class TestFitting(unittest.TestCase):
    def test_runs_along_the_axis_whose_section_fits(self):
        """A 6 x 0.9 x 0.9 part in 1 x 1 angle runs along its 6 inch axis."""
        s = ex.extrusion_stock(ex.find("angle", "1 x 1 x 1/8"), box(6.0, 0.9, 0.9))
        self.assertTrue(s.fits)
        self.assertAlmostEqual(s.min_length_in, 6.0)
        self.assertEqual(s.placement[1], X)
        self.assertEqual(s.basis, st.EXTRUSION)

    def test_recommended_length_follows_the_l_rule(self):
        s = ex.extrusion_stock(ex.find("flat_bar", "2 x 1/2"), box(5.3, 1.9, 0.4))
        self.assertAlmostEqual(s.recommended_length_in, 6.5)    # 5.3 + 1.0 -> 6.5
        self.assertAlmostEqual(s.length_in, 6.5)
        self.assertAlmostEqual(s.volume_in3, 1.0 * 6.5)

    def test_the_shortest_fitting_cut_wins(self):
        """A 3 x 3 x 1 part in 3 x 3 bar is a 1 inch slice, not 3 inches."""
        s = ex.extrusion_stock(ex.find("flat_bar", "3 x 3 square"), box(3.0, 3.0, 1.0))
        self.assertAlmostEqual(s.min_length_in, 1.0)
        self.assertEqual(s.placement[1], Z)

    def test_a_custom_length_is_kept(self):
        s = ex.extrusion_stock(ex.find("flat_bar", "2 x 1/2"), box(5.3, 1.9, 0.4), length_in=12.0)
        self.assertAlmostEqual(s.length_in, 12.0)
        self.assertAlmostEqual(s.volume_in3, 12.0)
        self.assertAlmostEqual(s.min_length_in, 5.3)

    def test_too_big_a_section_is_flagged(self):
        b = box(6.0, 3.0, 2.0)
        s = ex.extrusion_stock(ex.find("angle", "1 x 1 x 1/8"), b)
        self.assertFalse(s.fits)
        self.assertAlmostEqual(s.min_length_in, 6.0)   # falls back to the longest axis
        self.assertTrue(st.fit_warnings(s, b, None))

    def test_round_part_in_round_tube_uses_the_cylinder(self):
        """The box corners of a round part would not fit; the circle does."""
        b = box(1.9, 1.9, 4.0)
        s = ex.extrusion_stock(ex.find("round_tube", "2 OD x 0.250 wall"), b, cyl(1.9, 4.0))
        self.assertTrue(s.fits)
        self.assertAlmostEqual(s.min_length_in, 4.0)
        self.assertIsNone(s.width_in)

    def test_a_solid_part_does_not_fit_an_open_profile(self):
        """A solid 6 x 0.9 x 0.9 bar fits the 1 x 1 angle's envelope, but the
        angle holds far less than the part's 4.86 in^3."""
        b = box(6.0, 0.9, 0.9)
        angle = ex.extrusion_stock(ex.find("angle", "1 x 1 x 1/8"), b,
                                   part_volume_in3=6.0 * 0.9 * 0.9)
        self.assertTrue(angle.fits)
        self.assertFalse(angle.enough_material)
        self.assertTrue(any("less material" in w for w in st.fit_warnings(angle, b, None)))
        bar = ex.extrusion_stock(ex.find("flat_bar", "1 x 1 square"), b,
                                 part_volume_in3=6.0 * 0.9 * 0.9)
        self.assertTrue(bar.enough_material)
        self.assertEqual(st.fit_warnings(bar, b, None), [])

    def test_an_angle_bracket_does_fit_an_angle(self):
        """An L-bracket cut from 1 x 1 x 1/8 angle has about the angle's volume,
        a little more with its root fillet -- within the slack."""
        b = box(6.0, 1.0, 1.0)
        part = 0.2344 * 6.0 * 1.03
        s = ex.extrusion_stock(ex.find("angle", "1 x 1 x 1/8"), b, part_volume_in3=part)
        self.assertTrue(s.fits and s.enough_material)

    def test_hex_bar_fit(self):
        hexbar = ex.ExtrusionProfile("1", "hex_bar", 1.0)
        self.assertTrue(hexbar.fits_rect(0.9, 0.5))
        self.assertFalse(hexbar.fits_rect(1.1, 1.1))
        self.assertTrue(hexbar.fits_circle(1.0))

    def test_short_length_is_flagged(self):
        b = box(6.0, 0.9, 0.9)
        s = ex.extrusion_stock(ex.find("angle", "1 x 1 x 1/8"), b, length_in=5.0)
        self.assertTrue(any("shorter" in w for w in st.fit_warnings(s, b, None)))


class TestFittingToTheMesh(unittest.TestCase):
    """With the part's vertices, the profile is checked against the real
    section and turned to match it."""

    @staticmethod
    def bracket_points(corner_y, corner_z):
        """A 6 in long L-bracket, legs 2 x 2 x 1/4, its corner at the given
        (y, z) corner of the 2 x 2 section box centred on the origin."""
        sy, sz = (1 if corner_y < 0 else -1), (1 if corner_z < 0 else -1)
        y0, z0 = corner_y, corner_z
        section = [(y0, z0), (y0 + sy * 2, z0), (y0 + sy * 2, z0 + sz * 0.25),
                   (y0 + sy * 0.25, z0 + sz * 0.25), (y0 + sy * 0.25, z0 + sz * 2),
                   (y0, z0 + sz * 2)]
        return [((x - 3.0) * MM, y * MM, z * MM) for x in (0, 1.5, 3, 4.5, 6)
                for y, z in section]

    @staticmethod
    def corner_of(stock):
        """Where the profile's own corner (its first outline point) lands."""
        centre, _d, u, v = stock.placement
        x, y = stock.profile.outline_loops()[0][0]
        return [(c + (x * a + y * b) * MM) / MM for c, a, b in zip(centre, u, v)]

    def test_the_angle_is_turned_to_match_the_bracket(self):
        b = SimpleNamespace(extents_mm=[6 * MM, 2 * MM, 2 * MM], axes=[X, Y, Z],
                            center_mm=[0.0, 0.0, 0.0])
        for corner in ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)):
            with self.subTest(corner=corner):
                pts = self.bracket_points(*corner)
                s = ex.extrusion_stock(ex.find("angle", "2 x 2 x 1/4"), b, points_mm=pts)
                self.assertTrue(s.fits and s.enough_material)
                self.assertGreaterEqual(s.section_inside, 0.999)
                _, y, z = self.corner_of(s)
                self.assertAlmostEqual(y, corner[0], places=6)
                self.assertAlmostEqual(z, corner[1], places=6)

    def test_a_solid_bar_is_not_cut_from_an_angle(self):
        b = box(6.0, 0.9, 0.9)
        cx, cy, cz = b.center_mm                      # millimetres
        pts = [((x - 3) * MM + cx, (y - 0.45) * MM + cy, (z - 0.45) * MM + cz)
               for x in (0, 3, 6) for y in (0, 0.45, 0.9) for z in (0, 0.45, 0.9)]
        angle = ex.extrusion_stock(ex.find("angle", "1 x 1 x 1/8"), b, points_mm=pts)
        self.assertFalse(angle.enough_material)
        self.assertTrue(any("only" in w for w in st.fit_warnings(angle, b, None)))
        bar = ex.extrusion_stock(ex.find("flat_bar", "1 x 1 square"), b, points_mm=pts)
        self.assertTrue(bar.enough_material)

    def test_a_small_part_slides_into_the_profile_corner(self):
        """A 1.5 x 1.5 bracket of 1/4 legs sits in the corner of 2 x 2 angle."""
        b = SimpleNamespace(extents_mm=[6 * MM, 1.5 * MM, 1.5 * MM], axes=[X, Y, Z],
                            center_mm=[0.0, 0.0, 0.0])
        section = [(-0.75, -0.75), (0.75, -0.75), (0.75, -0.5), (-0.5, -0.5),
                   (-0.5, 0.75), (-0.75, 0.75)]
        pts = [((x - 3) * MM, y * MM, z * MM) for x in (0, 3, 6) for y, z in section]
        s = ex.extrusion_stock(ex.find("angle", "2 x 2 x 1/4"), b, points_mm=pts)
        self.assertTrue(s.enough_material)

    def test_inside_fraction_counts_a_tube_bore_as_outside(self):
        loops = ex.find("rect_tube", "2 x 2 x 1/8").outline_loops()
        import numpy as np

        frac = ex._inside_fraction(loops, np.array([[0.0, 0.0], [0.95, 0.0]]))
        self.assertAlmostEqual(frac, 0.5)


def prism_mesh(loops_in, length_in):
    """Triangle mesh (mm) of 2D loops (inches, in the YZ plane) swept along X
    from -length/2 to +length/2. One loop: a solid prism with fan-capped ends.
    Two loops of equal size: a tube with ring-capped ends."""
    pts, tris = [], []
    h = length_in / 2.0
    rings = []
    for loop in loops_in:
        base = len(pts)
        n = len(loop)
        pts += [(-h * MM, y * MM, z * MM) for y, z in loop]
        pts += [(h * MM, y * MM, z * MM) for y, z in loop]
        rings.append((base, n))
        for k in range(n):
            k2 = (k + 1) % n
            tris += [(base + k, base + k2, base + n + k2), (base + k, base + n + k2, base + n + k)]
    if len(rings) == 1:
        base, n = rings[0]
        tris += [(base, base + k, base + k + 1) for k in range(1, n - 1)]
        tris += [(base + n, base + n + k, base + n + k + 1) for k in range(1, n - 1)]
    else:
        (o, n), (i, _) = rings
        for end in (0, n):
            for k in range(n):
                k2 = (k + 1) % n
                tris += [(o + end + k, o + end + k2, i + end + k2), (o + end + k, i + end + k2, i + end + k)]
    return pts, tris


class TestSurfaceSampling(unittest.TestCase):
    """Vertices alone sit only on a prism's end faces; the fit has to use the
    whole surface."""

    def test_a_hex_standoff_is_not_cut_from_a_channel(self):
        hexbar = ex.ExtrusionProfile("1", "hex_bar", 1.0)
        pts, tris = prism_mesh(hexbar.outline_loops(), 2.0)
        b = SimpleNamespace(extents_mm=[2.0 * MM, 2 / 3 ** 0.5 * MM, 1.0 * MM],
                            axes=[X, Y, Z], center_mm=[0.0, 0.0, 0.0])
        samples = ex.surface_samples(pts, tris)
        channel = ex.extrusion_stock(ex.find("channel", "2 x 1 x 1/8"), b, points_mm=samples)
        self.assertFalse(channel.fits and channel.enough_material)
        hexed = ex.extrusion_stock(ex.find("hex_bar", "1 A/F"), b, points_mm=samples)
        self.assertTrue(hexed.fits and hexed.enough_material)
        self.assertAlmostEqual(hexed.min_length_in, 2.0)

    def test_a_tube_bushing_fits_its_tube_despite_a_generous_cylinder_fit(self):
        tube = ex.find("round_tube", "2 OD x 0.250 wall")
        pts, tris = prism_mesh(tube.outline_loops(), 1.5)
        b = SimpleNamespace(extents_mm=[1.5 * MM, 2.0 * MM, 2.0 * MM],
                            axes=[X, Y, Z], center_mm=[0.0, 0.0, 0.0])
        fit = SimpleNamespace(radius_mm=2.018 / 2 * MM, height_mm=1.5 * MM,
                              axis_direction=[1, 0, 0], axis_point_mm=[0, 0, 0])
        s = ex.extrusion_stock(tube, b, fit, points_mm=ex.surface_samples(pts, tris))
        self.assertTrue(s.fits and s.enough_material)
        self.assertAlmostEqual(s.min_length_in, 1.5)

    def test_sampling_is_repeatable(self):
        pts, tris = prism_mesh([[(0, 0), (1, 0), (1, 1), (0, 1)]], 3.0)
        a, b = ex.surface_samples(pts, tris), ex.surface_samples(pts, tris)
        self.assertTrue((a == b).all())
        self.assertEqual(len(a), len(pts) + ex.MAX_POINTS)


class TestCustomLibrary(unittest.TestCase):
    """custom_extrusions.csv: the user's own dies, beside the launcher."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="custom_ex_"))
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.path = self.dir / "custom_extrusions.csv"

    def write(self, *lines: str) -> ex.CustomLibrary:
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return ex.read_custom(self.path)

    def test_no_file_is_an_empty_library(self):
        lib = ex.read_custom(self.path)
        self.assertEqual((lib.profiles, lib.errors, lib.exists), ((), (), False))

    def test_the_template_loads_clean_and_its_examples_work(self):
        ex.ensure_custom_file(self.path)
        lib = ex.read_custom(self.path)
        self.assertEqual((lib.profiles, lib.errors), ((), ()))
        # Uncommenting the examples is all it takes to use them.
        text = self.path.read_text(encoding="utf-8")
        text = text.replace("# ACME-2210", "ACME-2210").replace("# DIE-0457", "DIE-0457")
        lib = self.write(text)
        self.assertEqual([p.name for p in lib.profiles], ["ACME-2210", "DIE-0457"])
        self.assertEqual(lib.errors, ())

    def test_ensure_never_overwrites(self):
        self.write("name,shape,a_in", "MINE,hex_bar,1")
        ex.ensure_custom_file(self.path)
        self.assertIn("MINE", self.path.read_text(encoding="utf-8"))

    def test_rows_load_with_their_area(self):
        lib = self.write(
            "# my dies",
            "name,shape,a_in,b_in,c_in,t_in,t2_in,r_in,area_in2,notes",
            "ACME-1,angle,2,1.5,,0.125,,,0.430,from the drawing",
            "ACME-2,zee,1.25,0.75,0.5,0.0625,,,,")
        a, z = lib.profiles
        self.assertTrue(a.custom and z.custom)
        self.assertAlmostEqual(a.area_in2, 0.430)                  # as given
        self.assertAlmostEqual(z.area_in2, z.section_area_in2)    # computed
        self.assertIn("(custom)", a.describe())

    def test_bad_rows_are_skipped_and_reported_not_raised(self):
        lib = self.write(
            "name,shape,a_in,b_in,t_in",
            "GOOD,flat_bar,2,0.5,",
            "TYPO,angle,2,1.5,abc",                # not a number
            "THICK,rect_tube,1,1,0.6",             # wall too thick
            "ODD,pentagon,1,,",                    # unknown shape
            ",flat_bar,1,1,",                      # no name
            "GOOD,flat_bar,3,0.5,",                # listed twice
            "AND10134-1205,angle,1.25,1,0.094")    # shadows a built-in
        self.assertEqual([p.name for p in lib.profiles], ["GOOD"])
        self.assertEqual(len(lib.errors), 6)
        self.assertTrue(lib.errors[0].startswith("line 3:"))
        self.assertIn("built-in", lib.errors[-1])

    def test_a_bad_header_is_reported(self):
        lib = self.write("part,kind", "X,angle")
        self.assertEqual(lib.profiles, ())
        self.assertIn("name", lib.errors[0])

    def test_offered_with_everything_else(self):
        self.write("name,shape,a_in,b_in", "MY-BAR,flat_bar,0.95,0.95")
        saved = ex.CUSTOM_PATH
        ex.CUSTOM_PATH = self.path
        self.addCleanup(lambda: (setattr(ex, "CUSTOM_PATH", saved), ex.custom_library(True)))
        self.assertIn("MY-BAR", [p.name for p in ex.load_profiles()])
        # The tightest solid bar for a 0.9 square part is now the user's own.
        best = ex.best_profile(ex.load_profiles(), box(6.0, 0.9, 0.9),
                               part_volume_in3=6.0 * 0.81)
        self.assertEqual(best.name, "MY-BAR")


if __name__ == "__main__":
    unittest.main()
