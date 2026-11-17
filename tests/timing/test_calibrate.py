"""calibrate.py: fitting shop factors to measured cycle times."""
from __future__ import annotations

import contextlib
import csv
import io
import shutil
import tempfile
import unittest
from pathlib import Path

from partestimator.machining import calibrate

COLS = ["part_name", "cycle_time_min", "estimated_cycle_time_min",
        "est_rough_outside_min", "est_rough_pocket_min",
        "est_finish_analytic_min", "est_finish_freeform_min", "est_hole_min"]


class Calibrate(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def write(self, rows) -> Path:
        path = self.dir / "training.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return path

    def report(self, path):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = calibrate.report(path)
        return rc, out.getvalue()

    def test_missing_file(self):
        rc, text = self.report(self.dir / "nope.csv")
        self.assertEqual(rc, 1)
        self.assertIn("not found", text)

    def test_no_measured_rows(self):
        path = self.write([{"part_name": "a", "cycle_time_min": "",
                            "estimated_cycle_time_min": 5}])
        rc, text = self.report(path)
        self.assertEqual(rc, 1)
        self.assertIn("Nothing to fit", text)

    def test_load_pairs_filters_unusable_rows(self):
        path = self.write([
            {"part_name": "ok", "cycle_time_min": 6, "estimated_cycle_time_min": 5},
            {"part_name": "no measured", "cycle_time_min": "", "estimated_cycle_time_min": 5},
            {"part_name": "junk", "cycle_time_min": "n/a", "estimated_cycle_time_min": 5},
            {"part_name": "zero est", "cycle_time_min": 6, "estimated_cycle_time_min": 0},
        ])
        all_rows, usable = calibrate.load_pairs(path)
        self.assertEqual(len(all_rows), 4)
        self.assertEqual([r["part_name"] for r in usable], ["ok"])

    def test_small_sample_fits_overall_median_only(self):
        ratios = [1.2, 1.5, 1.4]                       # median 1.4
        rows = [{"part_name": f"p{i}", "cycle_time_min": 10 * k,
                 "estimated_cycle_time_min": 10} for i, k in enumerate(ratios)]
        rc, text = self.report(self.write(rows))
        self.assertEqual(rc, 0)
        self.assertIn("calibrate_overall,1.40", text)
        self.assertNotIn("calibrate_roughing", text)

    def test_wide_spread_warns(self):
        rows = [{"part_name": "a", "cycle_time_min": 5, "estimated_cycle_time_min": 10},
                {"part_name": "b", "cycle_time_min": 40, "estimated_cycle_time_min": 10}]
        _, text = self.report(self.write(rows))
        self.assertIn("varies more than 2.5x", text)

    def test_least_squares_recovers_known_factors(self):
        # measured = 2.0*rough + 1.5*finish + 3.0*holes + 1.25, exactly.
        rows = []
        for i in range(12):
            rough_o, rough_p = 1.0 + i, 0.5 * (i % 3)
            finish_a, finish_f = 2.0 + 0.7 * ((i * 5) % 7), 0.3 * (i % 4)
            holes = 0.2 + 0.4 * ((i * 3) % 5)
            measured = (2.0 * (rough_o + rough_p) + 1.5 * (finish_a + finish_f)
                        + 3.0 * holes + 1.25)
            est = rough_o + rough_p + finish_a + finish_f + holes
            rows.append({"part_name": f"p{i}", "cycle_time_min": measured,
                         "estimated_cycle_time_min": est,
                         "est_rough_outside_min": rough_o, "est_rough_pocket_min": rough_p,
                         "est_finish_analytic_min": finish_a,
                         "est_finish_freeform_min": finish_f, "est_hole_min": holes})
        rc, text = self.report(self.write(rows))
        self.assertEqual(rc, 0)
        self.assertIn("calibrate_roughing,2.00", text)
        self.assertIn("calibrate_finishing,1.50", text)
        self.assertIn("calibrate_holes,3.00", text)
        self.assertIn("fixed         1.25", text)
        self.assertIn("RMS error 0.00", text)

    def test_non_physical_fit_falls_back_to_overall(self):
        # Measured time FALLS as finishing rises: the fit goes negative.
        rows = []
        for i in range(10):
            finish = 1.0 + i
            measured = 30.0 - 2.0 * finish + 0.5 * (i % 2)
            rows.append({"part_name": f"p{i}", "cycle_time_min": measured,
                         "estimated_cycle_time_min": 10.0,
                         "est_rough_outside_min": 5.0 + (i % 3),
                         "est_finish_analytic_min": finish, "est_hole_min": 1.0 + (i % 2)})
        rc, text = self.report(self.write(rows))
        self.assertEqual(rc, 0)
        self.assertIn("physically meaningless", text)
        self.assertIn("calibrate_overall", text)


if __name__ == "__main__":
    unittest.main()
