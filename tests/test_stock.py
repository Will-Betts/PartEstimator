"""Stock choices (issue #2) and extrusion profiles (issue #4).

Pure Python: boxes and cylinders are plain stand-ins for the kernel's fits,
so this runs on the CI runner with the timing tests.
"""
from __future__ import annotations

import math
import unittest
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
            tol = 0.01 * p.area_in2 if p.shape == "round_tube" else 1e-9
            with self.subTest(profile=p.describe()):
                self.assertAlmostEqual(area, p.area_in2, delta=tol)


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


if __name__ == "__main__":
    unittest.main()
