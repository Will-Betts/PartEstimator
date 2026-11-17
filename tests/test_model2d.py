"""The kernel-free 2D flat geometry that drives the live preview.

The whole point of `model2d` is that it is a fast path around the CAD kernel.
A fast path that disagrees with the exact path is worse than no fast path at
all, so these tests pin it against the SAME hand calculation used for the
transform solver: a U-channel whose flat length is 153.0477868 mm.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from partestimator.smt.config import UnfoldOptions
from partestimator.smt.model2d import Flat2D, build, recompute

from tests import synthetic
from tests.synthetic import BA, EXPECTED_FLAT, K, T, W


def flat_for(part, **kw):
    opts = UnfoldOptions(k_factor=kw.pop("k_factor", K), **kw)
    flat, result = recompute(part, opts)
    return flat, result, opts


class TestUChannel2D(unittest.TestCase):
    def setUp(self):
        self.flat, self.result, self.opts = flat_for(synthetic.u_channel())

    def test_blank_length_matches_hand_calc(self):
        """The preview's blank length must equal the exact path's answer."""
        self.assertAlmostEqual(self.flat.blank_length, EXPECTED_FLAT, places=6)

    def test_blank_width_is_the_part_width(self):
        self.assertAlmostEqual(self.flat.blank_width, W, places=9)

    def test_reported_dims_are_orientation_independent(self):
        """The flat frame's in-plane axes come from an arbitrary seed, so the
        long side may land on either axis. blank_length/blank_width must not
        care -- otherwise the same part reports 153x50 or 50x153 depending on
        how it was oriented in CAD."""
        self.assertGreaterEqual(self.flat.blank_length, self.flat.blank_width)
        self.assertEqual(
            sorted([self.flat.width, self.flat.height]),
            sorted([self.flat.blank_width, self.flat.blank_length]),
        )

    def test_bbox_area(self):
        self.assertAlmostEqual(
            self.flat.bbox_area, EXPECTED_FLAT * W, places=4
        )

    def test_piece_count(self):
        """Three flange footprints and two bend strips."""
        self.assertEqual(len(self.flat.flange_loops), 3)
        self.assertEqual(len(self.flat.strips), 2)
        self.assertEqual(len(self.flat.hole_loops), 0)

    def test_strips_are_ba_wide(self):
        for s in self.flat.strips:
            # Rectangle corners: a, b, b+u, a+u. The offset edge is BA long.
            width = np.linalg.norm(s[3] - s[0])
            self.assertAlmostEqual(float(width), BA, places=9)

    def test_polygon_area_equals_flat_length_times_width(self):
        """Flanges and strips must tile the blank without gaps or overlaps."""
        self.assertAlmostEqual(
            self.flat.polygon_area(), EXPECTED_FLAT * W, places=4
        )

    def test_bend_lines_sit_mid_strip(self):
        for (a, b, bid, _d) in self.flat.bend_lines:
            self.assertAlmostEqual(float(np.linalg.norm(b - a)), W, places=9)
            # Each bend line should be BA/2 from its strip's leading edge.
            strip = self.flat.strips[bid]
            lead_mid = (strip[0] + strip[1]) / 2.0
            line_mid = (a + b) / 2.0
            self.assertAlmostEqual(
                float(np.linalg.norm(line_mid - lead_mid)), BA / 2.0, places=9
            )

    def test_min_area_rect_matches_axis_aligned_here(self):
        """A rectangular blank: the tightest rectangle is the bounding box."""
        w, h, _ang = self.flat.min_area_rect()
        self.assertAlmostEqual(max(w, h), EXPECTED_FLAT, places=4)
        self.assertAlmostEqual(min(w, h), W, places=4)


class TestDevelopmentPathEndToEnd(unittest.TestCase):
    """The developed bend must give the same blank the rectangle did.

    Developing the real cylindrical face is a generalisation, not a
    replacement. Where the old rectangular strip was correct it has to stay
    correct to the last decimal, or every part that already worked would shift.
    """

    def setUp(self):
        self.part = synthetic.u_channel()
        self.flat, self.result, self.opts = flat_for(self.part)

    def _developed_count(self):
        from partestimator.smt import geom
        from partestimator.smt.model2d import _develop_bend, flat_frame

        base = self.result.sides[self.result.base]
        M = flat_frame(base[1], base[0])
        n = 0
        for fb in self.result.flat_bends:
            bend = self.part.bends[fb.bend_id]
            a = geom.apply_point(M, fb.line_start)[:2]
            u = geom.apply_vector(M, fb.step_dir)[:2]
            u = u / np.linalg.norm(u)
            if _develop_bend(bend, fb, M, a, u):
                n += 1
        return n

    def test_the_development_path_is_actually_being_used(self):
        """Guards against silently falling back and testing nothing."""
        self.assertEqual(self._developed_count(), 2)

    def test_developed_blank_matches_the_hand_calculation(self):
        self.assertAlmostEqual(
            self.flat.blank_length, EXPECTED_FLAT, places=6
        )
        self.assertAlmostEqual(self.flat.blank_width, W, places=9)

    def test_developed_area_still_tiles_exactly(self):
        self.assertAlmostEqual(
            self.flat.polygon_area(), EXPECTED_FLAT * W, places=3
        )

    def test_a_bend_with_no_captured_face_falls_back_cleanly(self):
        part = synthetic.u_channel()
        for b in part.bends.values():
            b.inner_loops = None
        flat, _r, _o = flat_for(part)
        self.assertAlmostEqual(flat.blank_length, EXPECTED_FLAT, places=6)

    def test_k_still_moves_the_developed_blank_correctly(self):
        """Through the development, BA is rho*theta -- so dK must still bite."""
        lo, _r, _o = flat_for(synthetic.u_channel(), k_factor=0.33)
        hi, _r, _o = flat_for(synthetic.u_channel(), k_factor=0.50)
        delta = hi.blank_length - lo.blank_length
        self.assertAlmostEqual(delta, 2 * (math.pi / 2) * T * 0.17, places=6)


class TestZSection2D(unittest.TestCase):
    def test_same_blank_as_u_channel(self):
        flat, _r, _o = flat_for(synthetic.z_section())
        self.assertAlmostEqual(flat.blank_length, EXPECTED_FLAT, places=6)
        self.assertAlmostEqual(flat.blank_width, W, places=9)

    def test_down_bend_uses_far_side_loops(self):
        """Crossing a down-bend, the preview must draw the OTHER face's loop.

        Flange 2's primary loop is the underside; referencing it here would put
        the footprint one thickness out of position in the flat.
        """
        flat, result, _o = flat_for(synthetic.z_section())
        _n, _o2, face, side = result.sides[2]
        part = synthetic.z_section()
        self.assertIs(face, part.flanges[2].partner_face)


class TestKSensitivity2D(unittest.TestCase):
    """The slider's whole purpose: K must move the blank, predictably."""

    def _width(self, k):
        flat, _r, _o = flat_for(synthetic.u_channel(), k_factor=k)
        return flat.blank_length

    def test_higher_k_gives_longer_blank(self):
        self.assertGreater(self._width(0.50), self._width(0.33))

    def test_magnitude_matches_formula(self):
        """delta blank == n_bends * theta * t * delta_K."""
        delta = self._width(0.50) - self._width(0.33)
        self.assertAlmostEqual(delta, 2 * (math.pi / 2) * T * 0.17, places=6)

    def test_recompute_is_stable(self):
        """Recomputing repeatedly must not drift -- the slider does this a lot."""
        part = synthetic.u_channel()
        widths = []
        for k in (0.33, 0.50, 0.40, 0.33, 0.44, 0.33):
            opts = UnfoldOptions(k_factor=k)
            flat, _r = recompute(part, opts)
            widths.append(flat.blank_length)
        self.assertAlmostEqual(widths[0], widths[3], places=12)
        self.assertAlmostEqual(widths[0], widths[5], places=12)


class TestFrameAgreement(unittest.TestCase):
    def test_matches_occ_flat_frame(self):
        """model2d and occ.flatten must choose the same in-plane axes.

        If they drift apart, the preview and the exported DXF come out rotated
        relative to each other -- a confusing failure that no other test here
        would catch, since each path is self-consistent.
        """
        import inspect

        from partestimator.smt import model2d

        src2d = inspect.getsource(model2d.flat_frame)
        # occ.flatten cannot be imported without a kernel; read the source.
        import pathlib

        occ_src = pathlib.Path(
            pathlib.Path(model2d.__file__).parent / "occ" / "flatten.py"
        ).read_text()
        body = occ_src.split("def flat_frame(")[1].split("\ndef ")[0]

        for token in ("seed = np.array([1.0, 0.0, 0.0])", "> 0.9",
                      "np.cross(seed, n)", "np.cross(n, ex)",
                      "np.vstack([ex, ey, n])"):
            self.assertIn(token, src2d, f"model2d.flat_frame lost {token!r}")
            self.assertIn(token, body, f"occ.flatten.flat_frame lost {token!r}")


class TestEmptyAndDegenerate(unittest.TestCase):
    def test_flange_without_loops_is_skipped_not_crashed(self):
        part = synthetic.u_channel()
        part.flanges[1].loops_a = None
        part.flanges[1].loops_b = None
        flat, _r, _o = flat_for(part)
        self.assertEqual(len(flat.flange_loops), 2)

    def test_empty_flat_has_zero_size(self):
        f = Flat2D()
        self.assertEqual(f.width, 0.0)
        self.assertEqual(f.height, 0.0)
        self.assertEqual(len(f.all_points()), 0)


if __name__ == "__main__":
    unittest.main()
