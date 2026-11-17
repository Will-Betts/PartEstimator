"""The launcher's contract with the machine it lands on.

`Run Part Estimator.bat` cannot be executed here (it needs cmd.exe), so its
TEXT is checked: which Pythons it looks for and in what order, that the venv
lives in this folder, that it starts the package windowless, and that it
reinstalls when requirements.txt changes.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BAT = (ROOT / "Run Part Estimator.bat").read_text(encoding="ascii")
REQS = (ROOT / "requirements.txt").read_text(encoding="utf-8")

#: Versions every pinned dependency was verified to have a win_amd64 wheel for.
SUPPORTED = ("3.12", "3.13", "3.11")


class TestPythonDiscovery(unittest.TestCase):
    def _probe(self):
        m = re.search(r"for %%V in \(([^)]*)\)", BAT)
        self.assertIsNotNone(m, "no version probe loop")
        return m.group(1).split()

    def test_every_supported_version_is_probed(self):
        self.assertEqual(sorted(self._probe()), sorted(SUPPORTED))

    def test_312_is_preferred(self):
        self.assertEqual(self._probe()[0], "3.12")

    def test_path_fallback_matches_the_range(self):
        self.assertIn("(3,11) <= sys.version_info < (3,14)", BAT)

    def test_escape_hatches(self):
        self.assertIn("PE_PYTHON", BAT)
        self.assertIn("python-path.txt", BAT)


class TestEnvironment(unittest.TestCase):
    def test_venv_in_folder(self):
        self.assertIn('set "VENV=%TOOLDIR%\\.venv"', BAT)

    def test_wheels_only(self):
        self.assertIn("--only-binary=:all:", BAT)

    def test_lock_written_and_preferred(self):
        self.assertIn('pip freeze > "%LOCK%"', BAT)
        self.assertIn('set "REQFILE=%LOCK%"', BAT)

    def test_requirements_change_triggers_reinstall(self):
        self.assertIn("certutil -hashfile", BAT)
        self.assertIn("goto :reinstall", BAT)
        # ...and the stale lock is discarded so it cannot win.
        block = BAT[BAT.index("\n:reinstall"):BAT.index("\n:setup")]
        self.assertIn('del /q "%LOCK%"', block)

    def test_user_site_packages_ignored(self):
        self.assertIn("PYTHONNOUSERSITE=1", BAT)

    def test_starts_the_package_windowless(self):
        self.assertIn('"%VPYW%" -m partestimator %*', BAT)


class TestPins(unittest.TestCase):
    def test_kernel_is_novtk_build(self):
        self.assertIn("cadquery-ocp-novtk==", REQS)
        self.assertNotRegex(REQS, r"(?m)^cadquery-ocp==")

    def test_qt_is_essentials_only(self):
        self.assertIn("PySide6-Essentials==", REQS)
        self.assertNotRegex(REQS, r"(?m)^PySide6==")

    def test_viewer_stack_pinned(self):
        for pkg in ("vtk==", "pyvista==", "pyvistaqt==", "qtpy==", "scipy==", "ezdxf=="):
            with self.subTest(pkg=pkg):
                self.assertIn(pkg, REQS)


if __name__ == "__main__":
    unittest.main()
