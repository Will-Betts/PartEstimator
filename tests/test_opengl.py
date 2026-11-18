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


if __name__ == "__main__":
    unittest.main()
