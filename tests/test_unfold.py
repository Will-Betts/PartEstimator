"""Transform solver tests, checked against hand arithmetic.

Run with:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from partestimator.smt import geom
from partestimator.smt.allowance import (
    bend_allowance,
    bend_deduction,
    check_consistency,
    k_factor,
    outside_setback,
)
from partestimator.smt.config import UnfoldOptions
from partestimator.smt.graph import UnflattenableError
from partestimator.smt.unfold import assign_allowances, refold_residual, solve

from tests import synthetic
from tests.synthetic import BA, BD, EXPECTED_FLAT, K, R, T, W

X = np.array([1.0, 0.0, 0.0])

# Probe points on the reference (inner) surfaces of each flange, and on the
# opposite (outer) surfaces. Transforming either set must give the same flat
# extent -- the unfold transform is a rigid motion of the whole body, so it
# cannot care which surface you measure from.
U_INNER = {
    0: [(0, 0, 0), (92, 0, 0), (92, W, 0), (0, W, 0)],
    1: [(94, 0, 2), (94, W, 2), (94, 0, 28), (94, W, 28)],
    2: [(-2, 0, 2), (-2, W, 2), (-2, 0, 28), (-2, W, 28)],
}
U_OUTER = {
    0: [(0, 0, -2), (92, 0, -2), (92, W, -2), (0, W, -2)],
    1: [(96, 0, 2), (96, W, 2), (96, 0, 28), (96, W, 28)],
    2: [(-4, 0, 2), (-4, W, 2), (-4, 0, 28), (-4, W, 28)],
}
Z_INNER = {
    0: [(0, 0, 0), (92, 0, 0), (92, W, 0), (0, W, 0)],
    1: [(94, 0, 2), (94, W, 2), (94, 0, 28), (94, W, 28)],
    2: [(-2, 0, -4), (-2, W, -4), (-2, 0, -30), (-2, W, -30)],
}


def flat_extent(result, probes, axis=X) -> float:
    vals = []
    for fid, pts in probes.items():
        M = result.transforms[fid]
        for p in pts:
            vals.append(float(np.dot(geom.apply_point(M, p), axis)))
    return max(vals) - min(vals)


def solved(part):
    opts = UnfoldOptions(k_factor=K, material="CRS")
    assign_allowances(part, opts)
    return solve(part, opts), opts


# ---------------------------------------------------------------------------


class TestAllowanceArithmetic(unittest.TestCase):
    def test_hand_values(self):
        ba = bend_allowance(math.pi / 2, R, T, K)
        self.assertAlmostEqual(ba, 4.5238934, places=6)
        self.assertAlmostEqual(outside_setback(math.pi / 2, R, T), 4.0, places=9)
        self.assertAlmostEqual(bend_deduction(math.pi / 2, R, T, K), 3.4761066, places=6)

    def test_two_formulations_agree(self):
        """legs-to-tangent + BA must equal outside dims - BD."""
        bends = [(math.pi / 2, R, T, K), (math.pi / 2, R, T, K)]
        self.assertTrue(
            check_consistency(
                legs_to_tangent=[92.0, 26.0, 26.0],
                outside_dims=[100.0, 30.0, 30.0],
                bends=bends,
            )
        )

    def test_expected_flat_length(self):
        self.assertAlmostEqual(EXPECTED_FLAT, 153.0477868, places=6)
        self.assertAlmostEqual(92 + 26 + 26 + 2 * BA, EXPECTED_FLAT, places=9)

    def test_k_rises_with_r_over_t(self):
        """K is not a constant; it must increase with r/t and stay under 0.5."""
        ks = [k_factor(r, 1.0, "CRS", "air") for r in (0.5, 1.0, 2.0, 4.0, 8.0)]
        self.assertTrue(all(b >= a for a, b in zip(ks, ks[1:])), ks)
        self.assertTrue(all(0.30 <= k <= 0.50 for k in ks), ks)

    def test_setback_diverges_at_180(self):
        self.assertTrue(math.isinf(outside_setback(math.pi, R, T)))
        self.assertTrue(math.isinf(bend_deduction(math.pi, R, T, K)))


class TestGeom(unittest.TestCase):
    def test_rotation_is_rigid(self):
        M = geom.rotation_about_axis([1, 2, 3], [0, 1, 1], 0.7)
        R3 = M[:3, :3]
        np.testing.assert_allclose(R3 @ R3.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(R3)), 1.0, places=12)

    def test_rotation_fixes_its_axis(self):
        p = np.array([1.0, 2.0, 3.0])
        d = geom.unit([0.0, 1.0, 0.0])
        M = geom.rotation_about_axis(p, d, 1.234)
        on_axis = p + 5.0 * d
        np.testing.assert_allclose(geom.apply_point(M, on_axis), on_axis, atol=1e-12)

    def test_invert_rigid(self):
        M = geom.translation([3, -1, 2]) @ geom.rotation_about_axis([1, 0, 0], [1, 1, 0], 0.4)
        np.testing.assert_allclose(geom.invert_rigid(M) @ M, np.eye(4), atol=1e-12)

    def test_cylinder_plane_tangency(self):
        # Axis along y at z = R above the plane z = 0 -> tangent.
        self.assertTrue(
            geom.cylinder_tangent_to_plane([0, 0, 2], [0, 1, 0], 2.0, [0, 0, 0], [0, 0, 1])
        )
        # Same axis, wrong radius -> not tangent.
        self.assertFalse(
            geom.cylinder_tangent_to_plane([0, 0, 2], [0, 1, 0], 3.0, [0, 0, 0], [0, 0, 1])
        )
        # A hole: axis normal to the plane -> never tangent.
        self.assertFalse(
            geom.cylinder_tangent_to_plane([0, 0, 2], [0, 0, 1], 2.0, [0, 0, 0], [0, 0, 1])
        )


class TestUChannel(unittest.TestCase):
    def setUp(self):
        self.part = synthetic.u_channel()
        self.result, self.opts = solved(self.part)

    def test_allowance_assigned(self):
        for b in self.part.bends.values():
            self.assertAlmostEqual(b.bend_allowance, BA, places=9)
            self.assertAlmostEqual(b.bend_deduction, BD, places=9)

    def test_flat_length_matches_hand_calc(self):
        self.assertAlmostEqual(
            flat_extent(self.result, U_INNER), EXPECTED_FLAT, places=6
        )

    def test_reference_surface_independence(self):
        """Measuring from the outer surface must give the same answer."""
        self.assertAlmostEqual(
            flat_extent(self.result, U_OUTER), EXPECTED_FLAT, places=6
        )

    def test_flanges_end_up_coplanar(self):
        """Every flange normal must align with the base normal after unfolding."""
        n_base = self.result.base_normal
        for fid, (n, _, _, _) in self.result.sides.items():
            n_flat = geom.apply_vector(self.result.transforms[fid], n)
            self.assertAlmostEqual(float(np.dot(n_flat, n_base)), 1.0, places=9)

    def test_width_is_preserved(self):
        """Bending does not change the dimension along the bend axis."""
        y = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(flat_extent(self.result, U_INNER, y), W, places=9)

    def test_both_bends_read_as_up(self):
        self.assertEqual(
            [b.direction for b in self.part.bends.values()], [1, 1]
        )

    def test_bend_lines_span_the_part(self):
        for fb in self.result.flat_bends:
            self.assertAlmostEqual(
                float(np.linalg.norm(fb.line_end - fb.line_start)), W, places=9
            )
            self.assertAlmostEqual(fb.width, BA, places=9)

    def test_refold_residual_is_zero(self):
        self.assertLess(refold_residual(self.part, self.result), 1e-9)


class TestZSection(unittest.TestCase):
    """One up-bend, one down-bend: exercises side propagation across the sheet."""

    def setUp(self):
        self.part = synthetic.z_section()
        self.result, self.opts = solved(self.part)

    def test_flat_length_matches_u_channel(self):
        self.assertAlmostEqual(
            flat_extent(self.result, Z_INNER), EXPECTED_FLAT, places=6
        )

    def test_directions_are_opposite(self):
        dirs = sorted(b.direction for b in self.part.bends.values())
        self.assertEqual(dirs, [-1, 1])

    def test_flanges_end_up_coplanar(self):
        n_base = self.result.base_normal
        for fid, (n, _, _, _) in self.result.sides.items():
            n_flat = geom.apply_vector(self.result.transforms[fid], n)
            self.assertAlmostEqual(float(np.dot(n_flat, n_base)), 1.0, places=9)

    def test_down_bend_references_far_side(self):
        """The down-bend's child must be referenced from the opposite surface.

        Flange 2's primary face is the underside (normal +x at x = -2); crossing
        a down-bend from a top-referenced base, the solver must instead pick the
        top-side face at x = -4 with normal -x.
        """
        n, o, _f, _s = self.result.sides[2]
        np.testing.assert_allclose(n, [-1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(o, [-4.0, 0.0, -4.0], atol=1e-12)


class TestUnflattenable(unittest.TestCase):
    def test_closed_section_is_rejected(self):
        part = synthetic.closed_box()
        opts = UnfoldOptions(k_factor=K)
        assign_allowances(part, opts)
        with self.assertRaises(UnflattenableError) as ctx:
            solve(part, opts)
        self.assertIn("closed section", str(ctx.exception))


class TestKFactorSensitivity(unittest.TestCase):
    """Flat length must move the way the physics says when K changes."""

    def _flat_for_k(self, k):
        part = synthetic.u_channel()
        opts = UnfoldOptions(k_factor=k)
        assign_allowances(part, opts)
        return flat_extent(solve(part, opts), U_INNER)

    def test_higher_k_gives_longer_blank(self):
        lo = self._flat_for_k(0.33)
        hi = self._flat_for_k(0.50)
        self.assertGreater(hi, lo)

    def test_magnitude_matches_formula(self):
        """Delta flat == 2 bends * theta * t * delta_K."""
        delta = self._flat_for_k(0.50) - self._flat_for_k(0.33)
        expected = 2 * (math.pi / 2) * T * (0.50 - 0.33)
        self.assertAlmostEqual(delta, expected, places=6)


if __name__ == "__main__":
    unittest.main()
