"""The launcher's contract with the machine it lands on.

`pe.ps1` needs PowerShell and network access to exercise for real, so its
TEXT is checked: which Pythons it looks for and in what order, that the venv
lives in this folder, that installs are wheels-only and locked, that it
reinstalls when requirements.txt changes, and that Python's options pass
through it untouched.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PS1 = (ROOT / "pe.ps1").read_text(encoding="ascii")
REQS = (ROOT / "requirements.txt").read_text(encoding="utf-8")

#: Versions every pinned dependency was verified to have a win_amd64 wheel for.
SUPPORTED = ("3.12", "3.13", "3.11")


def code_lines() -> list[str]:
    """pe.ps1 without its comment-based help block and # comments."""
    text = re.sub(r"<#.*?#>", "", PS1, flags=re.S)
    out = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        out.append(line)
    return out


class TestPythonDiscovery(unittest.TestCase):
    def _probe(self):
        m = re.search(r"\$SupportedVersions = @\(([^)]*)\)", PS1)
        self.assertIsNotNone(m, "no version probe list")
        return re.findall(r"'([\d.]+)'", m.group(1))

    def test_every_supported_version_is_probed(self):
        self.assertEqual(sorted(self._probe()), sorted(SUPPORTED))

    def test_312_is_preferred(self):
        self.assertEqual(self._probe()[0], "3.12")

    def test_path_fallback_matches_the_range(self):
        self.assertIn("(3,11) <= sys.version_info < (3,14)", PS1)

    def test_only_64_bit(self):
        self.assertIn("sys.maxsize > 2**32", PS1)

    def test_escape_hatches(self):
        self.assertIn("PE_PYTHON", PS1)
        self.assertIn("python-path.txt", PS1)


class TestEnvironment(unittest.TestCase):
    def test_venv_in_folder(self):
        self.assertIn("$Venv      = Join-Path $ToolDir '.venv'", PS1)

    def test_wheels_only(self):
        self.assertIn("'--only-binary=:all:'", PS1)

    def test_lock_written_and_preferred(self):
        self.assertIn("pip freeze --exclude-editable", PS1)
        self.assertIn("$reqFile = if ($fromLock) { $Lock } else { $Req }", PS1)

    def test_requirements_change_triggers_reinstall(self):
        self.assertIn("Get-FileHash -LiteralPath $Req -Algorithm SHA256", PS1)
        # ...and the stale lock is discarded so it cannot win.
        start = PS1.index("Write-Step 'requirements.txt has changed")
        block = PS1[start:start + 400]
        self.assertIn("Remove-Item -LiteralPath $Lock", block)

    def test_user_site_packages_ignored(self):
        self.assertIn("PYTHONNOUSERSITE              = '1'", PS1)

    def test_environment_changes_are_undone(self):
        """Running the script must not leave PIP_* set in the caller's shell."""
        self.assertIn("[Environment]::SetEnvironmentVariable($k, $saved[$k], 'Process')", PS1)

    def test_package_is_installed_editable(self):
        self.assertIn("'--no-deps'", PS1)
        self.assertIn("'-e', $ToolDir", PS1)

    def test_gui_starts_windowless(self):
        self.assertIn("$VPyW      = Join-Path $Venv 'Scripts\\pythonw.exe'", PS1)
        self.assertIn("@('-m', 'partestimator', 'gui')", PS1)


class TestArgumentsPassThrough(unittest.TestCase):
    def test_no_param_block(self):
        """A param() block would prefix-match Python's -o/-q/-k as PowerShell
        parameters; every argument must arrive in $args instead."""
        for line in code_lines():
            self.assertNotRegex(line.strip().lower(), r"^(\[cmdletbinding.*\]\s*)?param\s*\(")

    def test_every_cli_command_is_forwarded(self):
        from partestimator.__main__ import COMMANDS

        for cmd in COMMANDS:
            with self.subTest(cmd=cmd):
                self.assertIn(f"'{cmd}'", PS1)


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
