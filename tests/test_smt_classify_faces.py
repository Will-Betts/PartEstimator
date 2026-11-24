"""The sheet metal face classifier on hand-made faces: cylinders and cones
in every orientation the part can present them, slivers, and the face
description's properties. Plain Python.
"""
from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np

from partestimator.smt import features as F

Z = np.array([0, 0, 1.0])
X = np.array([1.0, 0, 0])
T = 1.5


def face(**kw):
    base = dict(index=0, radius=3.0, axis_dir=Z, axial_length=T, sweep=2 * math.pi,
                convex=False, half_angle=math.radians(41), area=10.0)
    base.update(kw)
    return SimpleNamespace(**base)


def cyl(d, n=Z):
    return F._classify_cylinder(d, n, T, F.Feature(index=0, role=F.Role.UNKNOWN))


def cone(d, n=Z):
    return F._classify_cone(d, n, F.Feature(index=0, role=F.Role.UNKNOWN))


class TestCylinders(unittest.TestCase):
    def test_every_orientation(self):
        self.assertEqual(cyl(face(), n=None).role, F.Role.UNKNOWN)
        self.assertEqual(cyl(face()).role, F.Role.HOLE)
        self.assertEqual(cyl(face(sweep=math.pi)).role, F.Role.SLOT_END)
        self.assertEqual(cyl(face(sweep=math.pi / 2, convex=True)).role, F.Role.CORNER_RADIUS)
        # Axis in the sheet plane: an edge break, or a bend that failed a gate.
        self.assertEqual(cyl(face(axis_dir=X, radius=0.5, axial_length=0.1)).role,
                         F.Role.FILLET)
        lost = cyl(face(axis_dir=X, radius=5.0))
        self.assertEqual(lost.role, F.Role.UNKNOWN)
        self.assertIn("unrecognised bend", lost.reason)
        # Neither through the sheet nor in its plane: a blend.
        tilted = np.array([0.6, 0, 0.8])
        self.assertEqual(cyl(face(axis_dir=tilted)).role, F.Role.FILLET)


class TestCones(unittest.TestCase):
    def test_every_orientation(self):
        self.assertEqual(cone(face(), n=None).role, F.Role.UNKNOWN)
        csk = cone(face())
        self.assertEqual(csk.role, F.Role.COUNTERSINK)
        self.assertAlmostEqual(csk.included_angle, 82.0)
        self.assertEqual(cone(face(convex=True)).role, F.Role.CHAMFER)
        tapered = cone(face(axis_dir=X))
        self.assertEqual(tapered.role, F.Role.UNKNOWN)
        self.assertIn("tapered bend", tapered.reason)
        self.assertEqual(cone(face(axis_dir=np.array([0.6, 0, 0.8]))).role, F.Role.CHAMFER)


class TestDescAndSlivers(unittest.TestCase):
    def test_properties_and_a_sliver(self):
        from partestimator.smt.config import Tolerance

        plane = F.FaceDesc(index=0, kind="plane", area=1.0)
        cone_desc = F.FaceDesc(index=1, kind="cone", area=1.0)
        torus = F.FaceDesc(index=2, kind="torus", area=1.0)
        self.assertFalse(plane.is_curved)
        self.assertTrue(cone_desc.developable)
        self.assertFalse(torus.developable)
        sliver = F._classify_one(F.FaceDesc(index=5, kind="plane", area=0.0), Z, T,
                                 Tolerance(), {}, set())
        self.assertEqual((sliver.role, sliver.reason), (F.Role.UNKNOWN, "sliver"))


if __name__ == "__main__":
    unittest.main()
