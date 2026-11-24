"""Choosing the 3D viewer's OpenGL (partestimator/opengl.py).

The decision is tested with the probe and the DLL load stubbed out, so this is
pure Python and runs on the CI runner. The real probe needs a Windows desktop
and runs only there.
"""
from __future__ import annotations

import sys
import unittest
from unittest import mock

from partestimator import opengl


class TestParseVersion(unittest.TestCase):
    def test_versions_drivers_report(self):
        cases = {
            "1.1.0": (1, 1),                          # Windows' GDI Generic
            "4.6.0 NVIDIA 552.22": (4, 6),
            "3.3 (Core Profile) Mesa 11.2.2": (3, 3),
            "4.5.0 - Build 31.0.101.4502": (4, 5),    # Intel
            "": None,
            "garbage": None,
        }
        for text, want in cases.items():
            with self.subTest(text=text):
                self.assertEqual(opengl.parse_version(text), want)


@unittest.skipUnless(sys.platform == "win32", "the choice is Windows-only")
class TestChoose(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(opengl, "load_software_gl", return_value="software")
        self.load = p.start()
        self.addCleanup(p.stop)

    def probe(self, result):
        return mock.patch.object(opengl, "probe_driver", return_value=result)

    def test_a_capable_driver_is_used(self):
        with self.probe(((4, 6), "4.6.0 (NVIDIA)")):
            self.assertIn("driver OpenGL 4.6.0", opengl.choose("auto"))
        self.load.assert_not_called()

    def test_exactly_the_minimum_is_enough(self):
        with self.probe(((3, 2), "3.2")):
            opengl.choose("auto")
        self.load.assert_not_called()

    def test_gdi_generic_falls_back_to_software(self):
        with self.probe(((1, 1), "1.1.0 (GDI Generic)")):
            line = opengl.choose("auto")
        self.load.assert_called_once()
        self.assertIn("GDI Generic", line)

    def test_a_crashed_probe_falls_back_to_software(self):
        with self.probe((None, "probe failed (exit code -1073741819)")):
            opengl.choose("auto")
        self.load.assert_called_once()

    def test_forcing(self):
        with self.probe(((4, 6), "4.6")) as probe:
            self.assertIn("forced", opengl.choose("software"))
            self.load.assert_called_once()
            self.assertIn("forced", opengl.choose("hardware"))
            probe.assert_not_called()

    def test_a_failed_fallback_never_stops_the_app(self):
        self.load.side_effect = FileNotFoundError("no opengl32sw.dll")
        with self.probe(((1, 1), "1.1.0")):
            line = opengl.choose("auto")
        self.assertIn("software fallback unavailable", line)


class TestOtherPlatforms(unittest.TestCase):
    def test_nothing_happens_off_windows(self):
        with mock.patch.object(sys, "platform", "linux"), \
                mock.patch.object(opengl, "probe_driver") as probe:
            self.assertEqual(opengl.choose("auto"), "driver OpenGL")
        probe.assert_not_called()


class TestGuiOption(unittest.TestCase):
    def test_gl_choices(self):
        from partestimator.__main__ import build_parser

        p = build_parser()
        self.assertEqual(p.parse_args(["gui"]).gl, "auto")
        self.assertEqual(p.parse_args(["gui", "a.step", "--gl", "software"]).gl, "software")
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            p.parse_args(["gui", "--gl", "opengl"])


class TestProbe(unittest.TestCase):
    """probe_driver runs this module as a child; its failures are reasons,
    never exceptions."""

    def test_a_good_probe(self):
        done = mock.Mock(returncode=0, stdout="4.6.0 NVIDIA 551.23|NVIDIA RTX A2000\n")
        with mock.patch.object(opengl.subprocess, "run", return_value=done):
            version, detail = opengl.probe_driver()
        self.assertEqual(version, (4, 6))
        self.assertEqual(detail, "4.6.0 NVIDIA 551.23 (NVIDIA RTX A2000)")

    def test_a_probe_that_exits_badly(self):
        with mock.patch.object(opengl.subprocess, "run",
                               return_value=mock.Mock(returncode=3, stdout="")):
            self.assertEqual(opengl.probe_driver(), (None, "probe failed (exit code 3)"))

    def test_a_probe_that_hangs(self):
        with mock.patch.object(opengl.subprocess, "run",
                               side_effect=opengl.subprocess.TimeoutExpired("x", 1)):
            self.assertEqual(opengl.probe_driver(), (None, "probe failed (TimeoutExpired)"))

    @unittest.skipUnless(sys.platform == "win32", "WGL is Windows only")
    def test_the_real_driver_answers(self):
        # In this process: whichever opengl32 is loaded (the driver's, or the
        # software one the test run chose) reports a version and a renderer.
        version, renderer = opengl._query_driver()
        self.assertIsNotNone(opengl.parse_version(version), version)
        self.assertTrue(renderer)

    @unittest.skipUnless(sys.platform == "win32", "WGL is Windows only")
    def test_run_as_the_probe_it_prints_version_bar_renderer(self):
        import contextlib
        import io
        import runpy

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            runpy.run_path(opengl.__file__, run_name="__main__")
        self.assertIn("|", out.getvalue())


class TestSoftwareCopy(unittest.TestCase):
    def setUp(self):
        import shutil
        import tempfile
        from pathlib import Path

        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_the_copy_goes_in_the_venv_or_the_scratch_folder(self):
        with mock.patch.object(sys, "prefix", "C:/venv"), \
                mock.patch.object(sys, "base_prefix", "C:/python"):
            self.assertEqual(opengl._copy_dir().as_posix(), "C:/venv/swgl")
        with mock.patch.object(sys, "prefix", "C:/python"), \
                mock.patch.object(sys, "base_prefix", "C:/python"):
            self.assertEqual(opengl._copy_dir(), opengl._tool_dir() / "scratch" / "swgl")

    def test_no_bundled_renderer(self):
        with mock.patch("importlib.util.find_spec", return_value=None):
            self.assertIsNone(opengl.software_dll())
        with mock.patch.object(opengl, "software_dll", return_value=None):
            with self.assertRaises(FileNotFoundError):
                opengl.load_software_gl()

    @unittest.skipUnless(sys.platform == "win32", "Windows only")
    def test_copied_once_and_refused_after_qt_loaded_opengl(self):
        src = self.tmp / "opengl32sw.dll"
        src.write_bytes(b"x" * 64)
        dest = self.tmp / "copy"
        loaded = mock.Mock()
        loaded.GetModuleHandleW.return_value = 1234      # opengl32.dll already loaded
        with mock.patch.object(opengl, "software_dll", return_value=src), \
                mock.patch.object(opengl, "_copy_dir", return_value=dest), \
                mock.patch("ctypes.WinDLL", return_value=loaded):
            with self.assertRaises(RuntimeError):
                opengl.load_software_gl()
            self.assertEqual((dest / "opengl32.dll").read_bytes(), src.read_bytes())
            loaded.GetModuleHandleW.return_value = 0
            self.assertIn("software OpenGL", opengl.load_software_gl())


if __name__ == "__main__":
    unittest.main()
