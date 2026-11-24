"""Every sheet metal feature on one bracket (occparts.sheet_feature_bracket):
countersink, counterbore, slot, a corner radius, a chamfered edge and a hole
inside the bend's deformation zone -- recognised, reported, and exported.
Plus the report's other modes (inches, grain, cost, a flat blank) and the
feature labels for every role.

Also pins down a known limit: on a one-bend part a counterbore's two
concentric bores look like a bend of another thickness, so the thickness
is ambiguous and must be given.
"""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

try:
    import OCP  # noqa: F401

    HAVE = True
except Exception:  # noqa: BLE001
    HAVE = False

T_MM = 3.175


@unittest.skipUnless(HAVE, "needs the CAD kernel")
class TestFeatureBracket(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from partestimator.smt.config import UnfoldOptions
        from partestimator.smt.pipeline import unfold_step
        from tests import occparts

        cls.tmp = Path(tempfile.mkdtemp())
        with contextlib.redirect_stdout(io.StringIO()):
            cls.path = occparts.sheet_feature_bracket(cls.tmp)
        cls.opts = UnfoldOptions(simple_mode=False, thickness=T_MM)
        cls.part = unfold_step(cls.path, cls.opts)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_every_feature_is_recognised(self):
        roles = Counter(f.role.value for f in self.part.features.values())
        for role in ("countersink", "counterbore", "slot_end", "corner_radius", "chamfer",
                     "hole", "bend_inner", "bend_outer"):
            self.assertGreaterEqual(roles[role], 1, role)
        self.assertEqual(len(self.part.bends), 1)

    def test_the_report(self):
        from partestimator.smt.report import build_report, manufacturability_warnings

        warnings = manufacturability_warnings(self.part, self.opts)
        self.assertTrue(any("deformation zone" in w for w in warnings))
        text = build_report(self.part, self.opts).to_text()
        for want in ("FEATURES", "countersink", "counterbore", "slot end", "corner R",
                     "chamfer at 45 deg"):
            self.assertIn(want, text)

    def test_report_modes(self):
        from partestimator.smt.config import UnfoldOptions
        from partestimator.smt.report import build_report, material_summary

        for kw in ({"display_units": "in"}, {"grain_deg": 90, "cost_per_kg": 2.5},
                   {"grain_deg": 0}):
            with self.subTest(**kw):
                opts = UnfoldOptions(simple_mode=False, thickness=T_MM, **kw)
                text = build_report(self.part, opts).to_text()
                self.assertIn("Flat blank", text)
                if kw.get("display_units") == "in":
                    self.assertIn("t = 0.1250 in", text)
        flat, self.part.flat = self.part.flat, None
        try:
            with self.assertRaises(ValueError):
                material_summary(self.part, self.opts)
        finally:
            self.part.flat = flat

    def test_exports(self):
        from partestimator.smt.pipeline import write_outputs

        out = self.tmp / "out"
        written = write_outputs(self.part, self.opts, out, "bracket")
        self.assertEqual(set(written), {"report", "json", "dxf", "step"})
        data = json.loads(written["json"].read_text(encoding="utf-8"))
        self.assertEqual(len(data["bends"]), 1)
        none = write_outputs(self.part, self.opts, self.tmp / "none", "b", want_dxf=False,
                             want_step=False, want_json=False)
        self.assertEqual(set(none), {"report"})

    def test_a_counterbore_makes_a_one_bend_part_ambiguous(self):
        from partestimator.smt.config import UnfoldOptions
        from partestimator.smt.occ.detect import DetectionError
        from partestimator.smt.pipeline import unfold_step

        with self.assertRaisesRegex(DetectionError, "ambiguous thickness"):
            unfold_step(self.path, UnfoldOptions(simple_mode=False))


class TestLabels(unittest.TestCase):
    def test_feature_labels_for_every_role(self):
        from partestimator.smt.features import Feature, Role

        def f(role, **kw):
            feat = Feature(index=0, role=role)
            for k, v in kw.items():
                setattr(feat, k, v)
            return feat.label(decimals=2)

        self.assertEqual(f(Role.HOLE, diameter=5.0), "hole 5.00 mm dia")
        self.assertEqual(f(Role.COUNTERSINK, included_angle=82.0, diameter=8.0),
                         "countersink 82 deg 8.00 mm dia")
        self.assertEqual(f(Role.COUNTERSINK), "countersink")
        self.assertEqual(f(Role.COUNTERBORE, diameter=9.0), "counterbore 9.00 mm dia")
        self.assertEqual(f(Role.SLOT_END, diameter=4.0), "slot end R2.00 mm")
        self.assertEqual(f(Role.CORNER_RADIUS, diameter=4.0), "corner R2.00 mm")
        self.assertEqual(f(Role.CHAMFER, width=1.0, included_angle=45.0),
                         "chamfer 1.00 mm at 45 deg")
        self.assertEqual(f(Role.FILLET, diameter=2.0), "fillet R1.00 mm")
        self.assertEqual(f(Role.FORMED), "formed")
        self.assertTrue(Role.BEND_OUTER.is_bend)
        self.assertFalse(Role.HOLE.is_bend)

    def test_report_group_labels(self):
        from partestimator.smt.features import Role
        from partestimator.smt.report import _group_label as _feature_label
        from partestimator.smt.units import choose_display_unit

        unit = choose_display_unit("mm", "mm")
        g = lambda role, **kw: SimpleNamespace(role=role, diameter=kw.get("d", 4.0),  # noqa: E731
                                               included_angle=kw.get("a"))
        self.assertIn("countersink 90 deg", _feature_label(g(Role.COUNTERSINK, a=90.0), unit))
        self.assertIn("countersink on", _feature_label(g(Role.COUNTERSINK), unit))
        self.assertIn("counterbore", _feature_label(g(Role.COUNTERBORE), unit))
        self.assertIn("slot end R", _feature_label(g(Role.SLOT_END), unit))
        self.assertIn("corner R", _feature_label(g(Role.CORNER_RADIUS), unit))
        self.assertIn("edge fillet R", _feature_label(g(Role.FILLET), unit))
        self.assertEqual(_feature_label(g(Role.CHAMFER), unit), "chamfer")
        self.assertEqual(_feature_label(g(Role.CHAMFER, a=30.0), unit), "chamfer at 30 deg")
        self.assertEqual(_feature_label(g(Role.UNKNOWN), unit), "unknown")


if __name__ == "__main__":
    unittest.main()
