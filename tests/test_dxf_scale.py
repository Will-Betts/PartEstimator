"""The DXF is the thing that gets cut. Read every export BACK and check it.

Each test builds a real part with known dimensions in the CAD kernel, runs it
through the same pipeline the window uses, writes the DXF, re-opens it with
ezdxf and compares what is in the file -- not what the code meant to write --
against the developed blank:

  * the file is inches through and through ($INSUNITS, $MEASUREMENT, the
    model-space block record);
  * header extents are real, not the +/-1e20 placeholders;
  * the profile's extents equal the developed blank, in inches, to 0.0001";
  * hole diameters and bend-line positions land where the geometry says;
  * nothing but outline, holes and bend lines is in the file -- no text;
  * bend UP is DASHED and bend DOWN is PHANTOM, on one BEND layer.
"""
from __future__ import annotations

import math
import shutil
import tempfile
import unittest
from pathlib import Path

try:
    import OCP  # noqa: F401
    import ezdxf
    from ezdxf import bbox

    HAVE = True
except Exception:  # noqa: BLE001
    HAVE = False

IN = 25.4
TOL = 1e-4  # inches


@unittest.skipUnless(HAVE, "needs OCP and ezdxf")
class TestDxfScale(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests import occparts

        import os

        cls.tmp = Path(tempfile.mkdtemp(prefix="dxfscale_"))
        devnull = os.open(os.devnull, os.O_WRONLY)  # STEP writer banner -> fd 1
        saved = os.dup(1)
        os.dup2(devnull, 1)
        try:
            cls.parts = {
                "l": occparts.l_bracket(cls.tmp),
                "z": occparts.z_section(cls.tmp),
                "flat": occparts.flat_blank(cls.tmp),
            }
        finally:
            os.dup2(saved, 1)
            os.close(devnull)
            os.close(saved)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _export(self, key, display_units="mm"):
        from partestimator.smt.config import UnfoldOptions
        from partestimator.smt.dxf import export_dxf
        from partestimator.smt.pipeline import unfold_step

        opts = UnfoldOptions()
        if hasattr(opts, "display_units"):
            opts.display_units = display_units
        part = unfold_step(self.parts[key], opts)
        out = self.tmp / f"{key}_{display_units}.dxf"
        # Pass a metric unit on purpose: it must be ignored.
        from partestimator.smt.units import MM

        export_dxf(part, out, unit=MM)
        return part, ezdxf.readfile(out)

    # -- the file is inches -------------------------------------------------

    def _assert_inch_document(self, doc):
        self.assertEqual(doc.header["$INSUNITS"], 1)
        self.assertEqual(doc.header["$MEASUREMENT"], 0)
        self.assertEqual(doc.modelspace().block_record.dxf.get("units", 1), 1)
        lo = doc.header["$EXTMIN"]
        hi = doc.header["$EXTMAX"]
        self.assertLess(abs(lo[0]), 1e6, "header extents left at placeholder")
        self.assertLess(abs(hi[0]), 1e6, "header extents left at placeholder")

    def _assert_geometry_only(self, doc):
        msp = doc.modelspace()
        kinds = {e.dxftype() for e in msp}
        self.assertTrue(kinds <= {"LINE", "ARC", "CIRCLE", "LWPOLYLINE"}, kinds)
        layers = {e.dxf.layer for e in msp}
        self.assertTrue(layers <= {"OUTLINE", "HOLES", "BEND"}, layers)
        self.assertNotIn("NOTES", [lay.dxf.name for lay in doc.layers])

    def _profile_size(self, doc):
        msp = doc.modelspace()
        box = bbox.extents([e for e in msp if e.dxf.layer == "OUTLINE"])
        return box.size.x, box.size.y, box

    # -- tests --------------------------------------------------------------

    def test_l_bracket_is_true_size_in_inches(self):
        part, doc = self._export("l")
        self._assert_inch_document(doc)
        self._assert_geometry_only(doc)

        (bend,) = part.bends.values()
        ba_in = bend.bend_allowance / IN
        # Straight mid-surface lengths: (3 - t/2 - rmid) + (2 - t/2 - rmid)
        expected_len = (3.0 - 0.03 - 0.09) + (2.0 - 0.03 - 0.09) + ba_in
        sx, sy, _ = self._profile_size(doc)
        self.assertAlmostEqual(max(sx, sy), expected_len, delta=TOL * 10)
        self.assertAlmostEqual(min(sx, sy), 4.000, delta=TOL)

        # The DXF and the reported blank must agree exactly.
        w, h = part.flat.bbox
        self.assertAlmostEqual(sx, w / IN, delta=TOL)
        self.assertAlmostEqual(sy, h / IN, delta=TOL)

        circles = [e for e in doc.modelspace() if e.dxftype() == "CIRCLE"]
        self.assertEqual(len(circles), 1)
        self.assertAlmostEqual(circles[0].dxf.radius * 2, 0.250, delta=TOL)
        # The hole sits 1.000 from the free end of the base leg and on the
        # centreline of the 4.000 width -- in the flat, along the long axis.
        _sx, _sy, box = self._profile_size(doc)
        c = circles[0].dxf.center
        long_x = sx >= sy
        along = (c.x - box.extmin.x, box.extmax.x - c.x) if long_x else \
                (c.y - box.extmin.y, box.extmax.y - c.y)
        across = (c.y - box.extmin.y) if long_x else (c.x - box.extmin.x)
        self.assertAlmostEqual(min(along), 1.000, delta=TOL * 10)
        self.assertAlmostEqual(across, 2.000, delta=TOL)

        bends = [e for e in doc.modelspace() if e.dxf.layer == "BEND"]
        self.assertEqual(len(bends), 1)
        self.assertEqual(bends[0].dxf.linetype, "DASHED")
        s, e = bends[0].dxf.start, bends[0].dxf.end
        length = math.hypot(e.x - s.x, e.y - s.y)
        self.assertAlmostEqual(length, 4.000, delta=TOL)

    def test_inch_output_regardless_of_display_unit(self):
        _p1, d_mm = self._export("l", "mm")
        _p2, d_in = self._export("l", "in")
        self.assertEqual(d_mm.header["$INSUNITS"], 1)
        self.assertEqual(d_in.header["$INSUNITS"], 1)
        a = self._profile_size(d_mm)[:2]
        b = self._profile_size(d_in)[:2]
        self.assertAlmostEqual(a[0], b[0], delta=TOL)
        self.assertAlmostEqual(a[1], b[1], delta=TOL)

    def test_up_and_down_bends_use_different_linetypes(self):
        part, doc = self._export("z")
        self._assert_inch_document(doc)
        self._assert_geometry_only(doc)
        bends = [e for e in doc.modelspace() if e.dxf.layer == "BEND"]
        self.assertEqual(sorted(e.dxf.linetype for e in bends), ["DASHED", "PHANTOM"])
        for e in bends:
            self.assertIn(e.dxf.linetype, doc.linetypes)
        sx, sy, _ = self._profile_size(doc)
        self.assertAlmostEqual(min(sx, sy), 3.000, delta=TOL)
        w, h = part.flat.bbox
        self.assertAlmostEqual(sx, w / IN, delta=TOL)

    def test_flat_blank_is_its_own_size(self):
        _part, doc = self._export("flat")
        self._assert_inch_document(doc)
        self._assert_geometry_only(doc)
        sx, sy, _ = self._profile_size(doc)
        self.assertAlmostEqual(max(sx, sy), 6.000, delta=TOL)
        self.assertAlmostEqual(min(sx, sy), 4.000, delta=TOL)
        dias = sorted(round(e.dxf.radius * 2, 4) for e in doc.modelspace()
                      if e.dxftype() == "CIRCLE")
        self.assertEqual(dias, [0.375, 0.5])
        self.assertFalse([e for e in doc.modelspace() if e.dxf.layer == "BEND"])

    def test_header_extents_match_content(self):
        _part, doc = self._export("l")
        box = bbox.extents(doc.modelspace())
        lo, hi = doc.header["$EXTMIN"], doc.header["$EXTMAX"]
        self.assertAlmostEqual(lo[0], box.extmin.x, delta=TOL)
        self.assertAlmostEqual(hi[0], box.extmax.x, delta=TOL)
        self.assertAlmostEqual(hi[1], box.extmax.y, delta=TOL)


if __name__ == "__main__":
    unittest.main()
