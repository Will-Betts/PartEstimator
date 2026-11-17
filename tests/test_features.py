"""Feature classification: the rules that let a dirty STEP file through.

Every part in a real job carries countersunk holes, chamfered edges, slots and
corner reliefs. Up to v5 those were unrecognised geometry and the tool said so
in a way that made a correct answer look wrong. These tests pin the rules that
tell each of them apart -- and, more importantly, pin the cases that used to
break, so they cannot come back.

All of it runs on plain numpy. That is the point of keeping `smt.features`
kernel-free: these rules can be checked on any machine, not just one with a
500 MB OpenCASCADE install.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from partestimator.smt.config import Tolerance
from partestimator.smt.features import (
    FaceDesc,
    Role,
    classify,
    counts_by_role,
    group_features,
    local_normal,
    sheet_plane_pairs,
    thickness_candidates,
    undevelopable,
)

T = 2.0            # sheet thickness, mm
UP = np.array([0.0, 0.0, 1.0])
DOWN = -UP


def plane(index, normal, point, centroid, area=1000.0):
    return FaceDesc(
        index=index, kind="plane", area=area,
        centroid=np.asarray(centroid, float),
        normal=np.asarray(normal, float), point=np.asarray(point, float),
    )


def cylinder(index, axis_dir, axis_point, radius, sweep, axial, convex,
             area=None, centroid=(0, 0, 0)):
    area = area if area is not None else radius * sweep * axial
    return FaceDesc(
        index=index, kind="cylinder", area=area,
        centroid=np.asarray(centroid, float),
        axis_dir=np.asarray(axis_dir, float),
        axis_point=np.asarray(axis_point, float),
        radius=radius, sweep=sweep, axial_length=axial, convex=convex,
    )


def cone(index, axis_dir, half_angle_deg, radius, axial, convex, area=50.0,
         centroid=(0, 0, 0)):
    return FaceDesc(
        index=index, kind="cone", area=area,
        centroid=np.asarray(centroid, float),
        axis_dir=np.asarray(axis_dir, float),
        axis_point=np.zeros(3),
        radius=radius, half_angle=math.radians(half_angle_deg),
        axial_length=axial, convex=convex,
    )


def sheet(area=10000.0):
    """The two skins of a 100 x 100 plate, `T` apart, top face outward +Z."""
    return [
        plane(0, UP, [0, 0, T], [50, 50, T], area),
        plane(1, DOWN, [0, 0, 0], [50, 50, 0], area),
    ]


class TestSheetPairing(unittest.TestCase):
    """Which faces are the sheet, which are something else entirely."""

    def setUp(self):
        self.tol = Tolerance()

    def test_the_two_skins_pair(self):
        pairs = sheet_plane_pairs(sheet(), T, self.tol)
        self.assertEqual(pairs, [(0, 1)])

    def test_slot_walls_do_not_pair(self):
        """A slot narrower than the sheet is thick must not look like sheet.

        The two walls of a slot are antiparallel and a fixed distance apart,
        which is every condition the naive test checks. What rules them out is
        that the material is OUTSIDE the pair, not between it.
        """
        faces = sheet() + [
            plane(2, [1, 0, 0], [30, 0, 0], [30, 50, 1.0], 60.0),
            plane(3, [-1, 0, 0], [32, 0, 0], [32, 50, 1.0], 60.0),
        ]
        pairs = sheet_plane_pairs(faces, T, self.tol)
        self.assertEqual(pairs, [(0, 1)])

    def test_a_flange_folded_back_over_the_base_does_not_pair_with_it(self):
        """The bug that produced a flange which did not exist.

        A hem folded back over the base plate is antiparallel to it and can sit
        exactly one thickness away. Distance alone matched them. What separates
        them is that they do not overlap: their centroids are far apart once the
        thickness direction is projected out.
        """
        faces = sheet() + [
            # a small tab sitting one thickness above the top skin, but way off
            # to one side rather than over it
            plane(2, DOWN, [0, 0, T + T], [400, 50, T + T], 400.0),
        ]
        pairs = sheet_plane_pairs(faces, T, self.tol)
        self.assertEqual(pairs, [(0, 1)])

    def test_partner_must_be_on_the_inward_side(self):
        """Two plates stacked face to face are two parts, not one sheet."""
        faces = [
            plane(0, UP, [0, 0, T], [50, 50, T], 10000.0),
            # directly above, facing DOWN at it -- material outside the pair
            plane(1, DOWN, [0, 0, T], [50, 50, T], 10000.0),
        ]
        self.assertEqual(sheet_plane_pairs(faces, T, Tolerance()), [])


class TestThicknessWithoutBends(unittest.TestCase):
    """A flat blank has no bends, so the thickness comes from the skins."""

    def test_slot_does_not_win_over_the_real_thickness(self):
        faces = sheet() + [
            plane(2, [1, 0, 0], [30, 0, 0], [30, 50, 1.0], 60.0),
            plane(3, [-1, 0, 0], [31, 0, 0], [31, 50, 1.0], 60.0),
        ]
        cands = thickness_candidates(faces, Tolerance())
        self.assertTrue(cands, "no thickness candidate found at all")
        self.assertAlmostEqual(cands[0][0], T, places=6)

    def test_support_is_ranked_by_area(self):
        """The biggest faces decide it, not the first pair encountered."""
        faces = sheet(area=10000.0) + [
            plane(2, UP, [0, 0, 9.0], [50, 50, 9.0], 12.0),
            plane(3, DOWN, [0, 0, 6.0], [50, 50, 6.0], 12.0),
        ]
        cands = thickness_candidates(faces, Tolerance())
        self.assertAlmostEqual(cands[0][0], T, places=6)


class TestCountersinks(unittest.TestCase):
    """The one that made a correct answer look wrong."""

    def setUp(self):
        self.tol = Tolerance()

    def _classify(self, extra, adj):
        faces = sheet() + extra
        return classify(faces, adj, T, self.tol)

    def test_a_countersunk_hole_is_a_countersink_not_a_formed_feature(self):
        hole = cylinder(2, UP, [10, 10, 0], 2.5, 2 * math.pi, T, convex=False,
                        centroid=(10, 10, 1))
        csk = cone(3, UP, 45.0, 5.0, 1.2, convex=False, centroid=(10, 10, 1.8))
        adj = {0: {1, 2, 3}, 1: {0, 2}, 2: {0, 1, 3}, 3: {0, 2}}
        out = self._classify([hole, csk], adj)

        self.assertIs(out[2].role, Role.HOLE)
        self.assertIs(out[3].role, Role.COUNTERSINK)
        self.assertAlmostEqual(out[3].included_angle, 90.0, places=3)

    def test_a_countersink_does_not_invalidate_the_flat_pattern(self):
        csk = cone(2, UP, 41.0, 5.0, 1.2, convex=False, centroid=(10, 10, 1.8))
        adj = {0: {1, 2}, 1: {0}, 2: {0}}
        out = self._classify([csk], adj)
        self.assertEqual(undevelopable(out), [])

    def test_an_82_degree_countersink_reports_82(self):
        csk = cone(2, UP, 41.0, 5.0, 1.2, convex=False, centroid=(10, 10, 1.8))
        adj = {0: {1, 2}, 1: {0}, 2: {0}}
        out = self._classify([csk], adj)
        self.assertAlmostEqual(out[2].included_angle, 82.0, places=3)

    def test_a_sphere_IS_undevelopable_and_still_warns(self):
        """The blanket warning was wrong; no warning at all would also be wrong."""
        dome = FaceDesc(index=2, kind="sphere", area=300.0,
                        centroid=np.array([10.0, 10.0, 4.0]))
        adj = {0: {1, 2}, 1: {0}, 2: {0}}
        out = self._classify([dome], adj)
        self.assertIs(out[2].role, Role.FORMED)
        self.assertEqual(len(undevelopable(out)), 1)


class TestRoundFeatures(unittest.TestCase):
    """Hole, slot end and corner radius are all partial cylinders."""

    def setUp(self):
        self.tol = Tolerance()
        self.adj = {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}

    def _role(self, face):
        return classify(sheet() + [face], self.adj, T, self.tol)[2].role

    def test_full_cylinder_through_the_sheet_is_a_hole(self):
        self.assertIs(
            self._role(cylinder(2, UP, [10, 10, 0], 3.0, 2 * math.pi, T,
                                convex=False, centroid=(10, 10, 1))),
            Role.HOLE,
        )

    def test_half_cylinder_concave_is_a_slot_end(self):
        self.assertIs(
            self._role(cylinder(2, UP, [10, 10, 0], 3.0, math.pi, T,
                                convex=False, centroid=(10, 10, 1))),
            Role.SLOT_END,
        )

    def test_quarter_cylinder_convex_is_a_corner_radius(self):
        """Same shape as a slot end. Only the side the material is on differs."""
        self.assertIs(
            self._role(cylinder(2, UP, [0, 0, 0], 6.0, math.pi / 2, T,
                                convex=True, centroid=(3, 3, 1))),
            Role.CORNER_RADIUS,
        )

    def test_an_unclaimed_cylinder_in_the_sheet_plane_is_flagged_loudly(self):
        """A bend that failed a gate must not be silently absorbed.

        This is the only misclassification that changes the blank length, so it
        is reported as unrecognised rather than filed under something harmless.
        """
        f = classify(
            sheet() + [cylinder(2, [0, 1, 0], [0, 0, 0], 4.0, math.pi / 2, 50.0,
                                convex=False, centroid=(0, 25, 2))],
            self.adj, T, self.tol,
        )[2]
        self.assertIs(f.role, Role.UNKNOWN)
        self.assertIn("unrecognised bend", f.reason)


class TestEdgeTreatments(unittest.TestCase):
    def setUp(self):
        self.tol = Tolerance()
        self.adj = {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}

    def _role(self, face):
        return classify(sheet() + [face], self.adj, T, self.tol)[2].role

    def test_perimeter_wall_is_a_cut_edge(self):
        self.assertIs(
            self._role(plane(2, [1, 0, 0], [100, 0, 0], [100, 50, 1.0], 200.0)),
            Role.CUT_EDGE,
        )

    def test_a_45_degree_bevel_is_a_chamfer(self):
        n = np.array([1.0, 0.0, 1.0]) / math.sqrt(2.0)
        self.assertIs(
            self._role(plane(2, n, [100, 0, T], [100, 50, T], 80.0)),
            Role.CHAMFER,
        )

    def test_a_conical_chamfer_round_a_boss_is_a_chamfer_not_a_countersink(self):
        """Same surface type, same axis. Only convexity separates them."""
        out = classify(
            sheet() + [cone(2, UP, 45.0, 5.0, 1.0, convex=True,
                            centroid=(10, 10, 2.5))],
            self.adj, T, self.tol,
        )
        self.assertIs(out[2].role, Role.CHAMFER)


class TestBendVerdictWins(unittest.TestCase):
    def test_the_detector_overrides_the_classifier(self):
        """The bend test is stricter, so where it has an answer it wins.

        It requires a concentric mate one thickness away AND tangency to two
        distinct planes; nothing in this module tests either.
        """
        bend = cylinder(2, [0, 1, 0], [0, 0, 0], 3.0, math.pi / 2, 50.0,
                        convex=False, centroid=(0, 25, 2))
        out = classify(
            sheet() + [bend], {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}, T,
            Tolerance(), bend_faces={2: Role.BEND_INNER},
        )
        self.assertIs(out[2].role, Role.BEND_INNER)


class TestLocalNormal(unittest.TestCase):
    def test_it_finds_the_sheet_two_rings_away(self):
        """A chamfer touches a cut edge, which touches the skin. Two hops."""
        faces = sheet() + [
            plane(2, [1, 0, 0], [100, 0, 0], [100, 50, 0.5], 90.0),   # cut edge
            plane(3, [1, 0, 1], [100, 0, T], [100, 50, T], 40.0),     # chamfer
        ]
        by_index = {f.index: f for f in faces}
        adj = {0: {1, 3}, 1: {0, 2}, 2: {1, 3}, 3: {0, 2}}
        n = local_normal(2, by_index, adj, sheet={0, 1})
        self.assertIsNotNone(n)
        self.assertAlmostEqual(abs(float(np.dot(n, UP))), 1.0, places=9)


class TestGrouping(unittest.TestCase):
    def test_identical_holes_collapse_to_one_line(self):
        adj = {0: {1}, 1: {0}}
        faces = sheet()
        for i in range(12):
            faces.append(
                cylinder(2 + i, UP, [i * 5.0, 10, 0], 3.25, 2 * math.pi, T,
                         convex=False, centroid=(i * 5.0, 10, 1))
            )
            adj[2 + i] = {0, 1}
            adj[0].add(2 + i)

        out = classify(faces, adj, T, Tolerance())
        groups = group_features(out, roles=(Role.HOLE,))
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].count, 12)
        self.assertAlmostEqual(groups[0].diameter, 6.5, places=6)

    def test_counts_cover_every_face(self):
        adj = {0: {1}, 1: {0}}
        out = classify(sheet(), adj, T, Tolerance())
        self.assertEqual(sum(counts_by_role(out).values()), 2)


if __name__ == "__main__":
    unittest.main()


class TestThicknessPlausibility(unittest.TestCase):
    """Sheet metal is thin, and that bound is what stops a wrong sheet.

    Without it, a part whose real skins are partly hidden -- behind a boss, a
    weld prep, a mounting pad -- can have some other pair of parallel faces
    measured as the material. Every downstream number is then confidently wrong
    rather than obviously wrong, which is the worse of the two failures.
    """

    def setUp(self):
        from partestimator.smt.features import thickness_plausible

        self.ok = lambda v: thickness_plausible(v, Tolerance())

    def test_normal_gauges_are_accepted(self):
        for mm in (0.5, 0.9, 1.5, 1.905, 3.0, 6.35, 12.7):
            with self.subTest(mm=mm):
                self.assertTrue(self.ok(mm), f"{mm} mm should be sheet")

    def test_half_an_inch_is_the_boundary_and_is_included(self):
        self.assertTrue(self.ok(12.7))
        self.assertFalse(self.ok(12.8))

    def test_plate_is_rejected(self):
        for mm in (15.0, 20.0, 25.4, 50.0):
            with self.subTest(mm=mm):
                self.assertFalse(self.ok(mm), f"{mm} mm is plate, not sheet")

    def test_noise_is_rejected(self):
        self.assertFalse(self.ok(0.001))
        self.assertFalse(self.ok(0.0))


class TestPickingTheSheetFaceByHand(unittest.TestCase):
    """The manual override, for parts where the geometry is genuinely ambiguous."""

    def setUp(self):
        self.tol = Tolerance()
        # A 2 mm sheet carrying a 20 mm boss. The boss gives a parallel pair
        # too, and on a part with several of them it can out-vote the real one.
        self.faces = [
            plane(0, UP, [0, 0, 2], [50, 50, 2], 9000.0),
            plane(1, DOWN, [0, 0, 0], [50, 50, 0], 9000.0),
            plane(2, UP, [0, 0, 22], [50, 50, 22], 400.0),
            plane(3, DOWN, [0, 0, 2], [50, 50, 2], 400.0),
        ]

    def test_picking_the_sheet_face_measures_the_sheet(self):
        from partestimator.smt.features import thickness_from_face

        got = thickness_from_face(0, self.faces, self.tol)
        self.assertIsNotNone(got)
        thickness, partner = got
        self.assertAlmostEqual(thickness, 2.0, places=9)
        self.assertEqual(partner, 1)

    def test_the_cap_still_applies_by_default(self):
        """Picking the boss top gives 20 mm, which is not sheet."""
        from partestimator.smt.features import thickness_from_face

        self.assertIsNone(thickness_from_face(2, self.faces, self.tol))

    def test_the_cap_can_be_overridden_because_the_user_can_see_the_model(self):
        """A person pointing at two faces has better information than a rule.

        Refusing them would make the override useless in the one situation it
        exists for, so the bound is advisory here and enforced everywhere else.
        """
        from partestimator.smt.features import thickness_from_face

        got = thickness_from_face(2, self.faces, self.tol, cap=False)
        self.assertIsNotNone(got)
        self.assertAlmostEqual(got[0], 20.0, places=9)

    def test_a_face_with_nothing_behind_it_returns_nothing(self):
        from partestimator.smt.features import thickness_from_face

        lonely = [plane(0, UP, [0, 0, 0], [0, 0, 0], 100.0)]
        self.assertIsNone(thickness_from_face(0, lonely, self.tol))

    def test_the_nearest_qualifying_partner_wins(self):
        """Anchored to an asserted face, so nearest wins outright -- no vote."""
        from partestimator.smt.features import thickness_from_face

        faces = [
            plane(0, UP, [0, 0, 10], [0, 0, 10], 1000.0),
            plane(1, DOWN, [0, 0, 8], [0, 0, 8], 1000.0),    # 2 mm behind
            plane(2, DOWN, [0, 0, 4], [0, 0, 4], 5000.0),    # 6 mm, bigger
        ]
        thickness, partner = thickness_from_face(0, faces, self.tol)
        self.assertAlmostEqual(thickness, 2.0, places=9)
        self.assertEqual(partner, 1)
