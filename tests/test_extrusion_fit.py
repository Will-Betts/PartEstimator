"""Opening a machined part on an extrusion when one beats plate.

`pipeline.fit_extrusion` decides it from the part's box and mesh; no STEP
file is needed, but the pipeline module imports the CAD kernel.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

try:
    from partestimator.machining.geometry import pipeline
    from partestimator.machining.geometry import stock as st

    HAVE = True
except ImportError:  # no CAD kernel in the light CI environment
    HAVE = False

MM = 25.4


def box(l_in, w_in, t_in):
    return SimpleNamespace(extents_mm=np.array([l_in, w_in, t_in]) * MM,
                           axes=np.eye(3), center_mm=np.zeros(3),
                           volume_mm3=l_in * w_in * t_in * MM ** 3)


def prism(section_in, length_in):
    """Vertices and side triangles of a prism along X with this section,
    centred like `box`."""
    sec = np.asarray(section_in, dtype=float)
    sec = sec - (sec.min(axis=0) + sec.max(axis=0)) / 2.0
    n = len(sec)
    pts = [(x, y, z) for x in (-length_in / 2, length_in / 2) for y, z in sec]
    tris = []
    for i in range(n):
        j = (i + 1) % n
        tris += [(i, j, n + i), (j, n + j, n + i)]
    return np.asarray(pts) * MM, np.asarray(tris)


L_SECTION = [(0, 0), (2, 0), (2, 0.25), (0.25, 0.25), (0.25, 2), (0, 2)]
SQUARE = [(0, 0), (2, 0), (2, 2), (0, 2)]


@unittest.skipUnless(HAVE, "needs the CAD kernel")
class TestFitExtrusion(unittest.TestCase):
    def fit(self, section, primary="rectilinear"):
        b = box(6.0, 2.0, 2.0)
        pts, tris = prism(section, 6.0)
        area = abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2)
                       in zip(section, section[1:] + section[:1]))) / 2.0
        plate = st.rectilinear_stock(b.extents_mm)
        return pipeline.fit_extrusion(primary, b, None, plate, pts, tris, area * 6.0)

    def test_an_angle_bracket_opens_on_angle(self):
        shape, name = self.fit(L_SECTION)
        self.assertEqual(shape, "angle")

    def test_a_solid_block_stays_on_plate(self):
        self.assertIsNone(self.fit(SQUARE))

    def test_round_work_is_left_alone(self):
        self.assertIsNone(self.fit(L_SECTION, primary="cylindrical"))

    def test_a_flat_strip_is_not_put_in_an_angle(self):
        # It fits in one leg of a 2 x 2 angle with less metal than plate,
        # but nobody buys an angle to make a flat part.
        strip = [(0, 0), (2, 0), (2, 0.25), (0, 0.25)]
        b = box(6.0, 2.0, 0.25)
        pts, tris = prism(strip, 6.0)
        plate = st.rectilinear_stock(b.extents_mm)
        self.assertIsNone(pipeline.fit_extrusion("rectilinear", b, None, plate,
                                                 pts, tris, 0.5 * 6.0))

    def test_a_profile_that_needs_more_than_plate_is_not_used(self):
        b = box(6.0, 2.0, 2.0)
        pts, tris = prism(L_SECTION, 6.0)
        tiny_plate = st.custom_rectilinear(0.1, 0.1, 0.1)
        self.assertIsNone(pipeline.fit_extrusion("rectilinear", b, None, tiny_plate,
                                                 pts, tris, 0.9375 * 6.0))


if __name__ == "__main__":
    unittest.main()
