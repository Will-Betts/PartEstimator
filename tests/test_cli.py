"""The `part-estimator` / `python -m partestimator` command line.

Pure Python: the commands are dispatched to stubs, so neither the CAD kernel
nor Qt is needed and this runs on the bare CI runner with the timing tests.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from partestimator import __main__ as cli
from partestimator import __version__

HERE = Path(__file__).resolve()


def run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = cli.main(argv)
        except SystemExit as exc:  # argparse exits for --help / --version / errors
            rc = exc.code
    return rc, out.getvalue(), err.getvalue()


class TestNormalise(unittest.TestCase):
    def test_no_arguments_opens_the_window(self):
        self.assertEqual(cli._normalise([]), ["gui"])

    def test_a_bare_file_opens_the_window_on_it(self):
        self.assertEqual(cli._normalise(["part.step"]), ["gui", "part.step"])

    def test_legacy_flags_map_to_subcommands(self):
        self.assertEqual(cli._normalise(["--classify", "a.step"]), ["classify", "a.step"])
        self.assertEqual(cli._normalise(["--sheet-cli", "a.step", "-q"]), ["sheet", "a.step", "-q"])

    def test_subcommands_and_options_are_left_alone(self):
        self.assertEqual(cli._normalise(["classify", "a.step"]), ["classify", "a.step"])
        self.assertEqual(cli._normalise(["--version"]), ["--version"])


class TestTopLevel(unittest.TestCase):
    def test_version(self):
        rc, out, _ = run(["--version"])
        self.assertEqual(rc, 0)
        self.assertIn(__version__, out)

    def test_help_lists_every_command(self):
        rc, out, _ = run(["--help"])
        self.assertEqual(rc, 0)
        for cmd in cli.COMMANDS:
            with self.subTest(cmd=cmd):
                self.assertIn(cmd, out)

    def test_bad_option_is_a_usage_error(self):
        rc, _, err = run(["classify"])
        self.assertEqual(rc, 2)
        self.assertIn("usage:", err)


class FakeClassification:
    def __init__(self, kind):
        self.kind = kind
        self.reason = f"because {kind}"

    def to_dict(self):
        return {"kind": self.kind, "reason": self.reason}


class TestClassify(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def classify_file(path, cutoff):
            self.calls.append((Path(path).name, cutoff))
            if "broken" in path:
                raise ValueError("no solids")
            return FakeClassification("sheet_metal")

        fake = types.ModuleType("partestimator.classify")
        fake.classify_file = classify_file
        fake.DEFAULT_CUTOFF_IN = 0.190
        patcher = mock.patch.dict(sys.modules, {"partestimator.classify": fake})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_text_output(self):
        rc, out, _ = run(["classify", str(HERE), "--cutoff", "0.25"])
        self.assertEqual(rc, 0)
        self.assertEqual(out.strip(), f"{HERE.name}: sheet_metal  -- because sheet_metal")
        self.assertEqual(self.calls, [(HERE.name, 0.25)])

    def test_json_output_is_one_parseable_array(self):
        rc, out, _ = run(["classify", str(HERE), str(HERE), "--json", "--cutoff", "0.19"])
        self.assertEqual(rc, 0)
        data = json.loads(out)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["file"], str(HERE))
        self.assertEqual(data[0]["kind"], "sheet_metal")

    def test_wildcards_are_expanded(self):
        """PowerShell hands `*.step` to a native program unexpanded."""
        rc, out, _ = run(["classify", str(HERE.parent / "test_cl?.py"), "--cutoff", "0.19"])
        self.assertEqual(rc, 0)
        self.assertEqual(self.calls, [(HERE.name, 0.19)])

    def test_missing_file_fails_but_the_rest_still_run(self):
        rc, out, err = run(["classify", "nope.step", str(HERE), "--cutoff", "0.19"])
        self.assertEqual(rc, 1)
        self.assertIn("nope.step", err)
        self.assertIn(HERE.name, out)

    def test_kernel_error_is_reported_not_raised(self):
        broken = HERE.with_name("broken.step")
        with mock.patch.object(Path, "is_file", return_value=True):
            rc, _, err = run(["classify", str(broken), "--cutoff", "0.19"])
        self.assertEqual(rc, 1)
        self.assertIn("ValueError: no solids", err)


class TestDispatch(unittest.TestCase):
    def test_sheet_passes_every_argument_through(self):
        fake = types.ModuleType("partestimator.smt.cli")
        fake.main = mock.Mock(return_value=0)
        with mock.patch.dict(sys.modules, {"partestimator.smt.cli": fake}):
            rc, _, _ = run(["sheet", "a.step", "-o", "out", "-q", "--k", "0.4"])
        self.assertEqual(rc, 0)
        fake.main.assert_called_once_with(["a.step", "-o", "out", "-q", "--k", "0.4"],
                                          prog="part-estimator sheet")

    def test_calibrate_returns_the_report_exit_code(self):
        fake = types.ModuleType("partestimator.machining.calibrate")
        fake.report = mock.Mock(return_value=1)
        with mock.patch.dict(sys.modules, {"partestimator.machining.calibrate": fake}):
            rc, _, _ = run(["calibrate", "times.csv"])
        self.assertEqual(rc, 1)
        fake.report.assert_called_once_with(Path("times.csv"))

    def test_gui_gets_the_file(self):
        with mock.patch.object(cli, "cmd_gui", return_value=0) as gui:
            with mock.patch.dict(cli.HANDLERS, {"gui": gui}):
                rc, _, _ = run(["part.step"])
        self.assertEqual(rc, 0)
        self.assertEqual(gui.call_args.args[0].step, Path("part.step"))


class TestStartupFailure(unittest.TestCase):
    def test_goes_to_the_console_when_there_is_one(self):
        err = io.StringIO()
        with mock.patch.object(cli, "_tool_dir", return_value=HERE.parent / "_no_such_dir"), \
                mock.patch.object(sys, "stderr", err):
            cli._report_startup_failure(ImportError("no module named OCP"))
        text = err.getvalue()
        self.assertIn("could not start", text)
        self.assertIn("pe.ps1", text)


if __name__ == "__main__":
    unittest.main()
