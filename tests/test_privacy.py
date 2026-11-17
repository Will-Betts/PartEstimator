"""The guarantees about what leaves this machine, enforced rather than asserted.

Two promises are made about this tool:

  * it makes no network connection, ever, after installation;
  * it writes nothing outside its own folder, and its logs carry operations and
    counts but never file contents or full paths.

Both are the kind of promise that decays quietly. A future edit adds a
convenience import, or logs a path while debugging and leaves it in, and the
promise is broken with nothing to notice. These tests are the thing that
notices. `tests/test_imports.py` covers the network half; this file covers
what gets written and what gets recorded.
"""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

from partestimator.smt.diag import name_only, redact

ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = "Run Part Estimator.bat"
BATS = (LAUNCHER, "Run Self-Test.bat", "Calibrate To My Times.bat", "Reset Environment.bat")
PKG = ROOT / "partestimator" / "smt"


def modules():
    return sorted(PKG.rglob("*.py"))


def code_lines(path: Path):
    """(line number, source) for EXECUTABLE lines only.

    Comments and docstrings are stripped first, because this file's whole
    purpose is to check what the code does -- and a docstring that says "this
    deliberately does NOT write to %APPDATA%" would otherwise fail the very
    test it is describing. Documenting a rule must not violate it.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    skip: set[int] = set()
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            end = first.value.end_lineno or first.value.lineno
            skip.update(range(first.value.lineno, end + 1))

    out = []
    for i, line in enumerate(lines, 1):
        if i in skip:
            continue
        # Approximate comment stripping. Good enough here: a forbidden name
        # inside a string literal on a code line is still worth flagging.
        code = line.split("#", 1)[0]
        if code.strip():
            out.append((i, code))
    return out


class TestPathRedaction(unittest.TestCase):
    """A path is not the part, but it names the customer and the programme."""

    def test_a_windows_path_with_spaces_is_reduced_to_the_file_name(self):
        line = (
            r"reading C:\Users\First Last\Jobs\ACME-2291\rev C\bracket.step "
            r"took 4.2s"
        )
        out = redact(line)
        self.assertNotIn("ACME-2291", out)
        self.assertNotIn("First Last", out)
        self.assertIn("bracket.step", out)

    def test_a_unc_network_share_is_reduced_too(self):
        out = redact(r"wrote \\fileserver\engineering\2291\flat.dxf")
        self.assertNotIn("fileserver", out)
        self.assertNotIn("engineering", out)
        self.assertIn("flat.dxf", out)

    def test_a_posix_path_is_reduced(self):
        out = redact("/mnt/share/customer/project/part.STEP loaded")
        self.assertNotIn("customer", out)
        self.assertIn("part.STEP", out)

    def test_ordinary_prose_is_left_alone(self):
        """A filter that mangles normal messages is one somebody switches off."""
        for line in (
            "ratio area/thickness is 1.02 and/or fine",
            "feed 3.5 mm/s at 40 mm/rev",
            "detected 4 bends in bracket.step",
            "ratio a/b and c/d",
            "K = f(r/t) rising 0.33 -> 0.5",
        ):
            with self.subTest(line=line):
                self.assertEqual(redact(line), line)

    def test_name_only_handles_windows_paths_on_any_platform(self):
        """`Path` follows the rules of the platform it runs on.

        A Windows path handed to a POSIX interpreter comes back whole, which is
        exactly the case that has to work when a log written on Windows is read
        somewhere else.
        """
        self.assertEqual(
            name_only(r"C:\Users\First Last\rev C\part.step"), "part.step"
        )
        self.assertEqual(name_only("/mnt/share/part.step"), "part.step")
        self.assertEqual(name_only("part.step"), "part.step")

    def test_the_filter_is_attached_to_every_handler(self):
        import logging

        from partestimator.smt.diag import LOGGER, RedactPaths, setup_logging

        setup_logging(to_file=False)
        handlers = [h for h in LOGGER.handlers]
        self.assertTrue(handlers, "logging was never configured")
        for h in handlers:
            with self.subTest(handler=type(h).__name__):
                self.assertTrue(
                    any(isinstance(f, RedactPaths) for f in h.filters),
                    "a log handler has no path redaction",
                )


class TestNothingIsWrittenOutsideTheToolFolder(unittest.TestCase):
    """No %TEMP%, no %APPDATA%, no registry. Delete the folder, it is gone."""

    FORBIDDEN = (
        "tempfile", "TemporaryDirectory", "NamedTemporaryFile", "mkdtemp",
        "mkstemp", "gettempdir",
        "APPDATA", "LOCALAPPDATA", "TEMP", "USERPROFILE", "XDG_CACHE_HOME",
        "expanduser", "QSettings", "winreg", "QStandardPaths",
        "environ", "getenv",
    )

    def test_no_module_reaches_for_a_user_or_temp_location(self):
        # Whole identifiers only. A substring match flags `_log_environment`
        # for containing "environ", which is a function that logs the numpy
        # version and touches nothing.
        patterns = {
            bad: re.compile(rf"(?<![A-Za-z0-9_]){re.escape(bad)}(?![A-Za-z0-9_])")
            for bad in self.FORBIDDEN
        }
        offenders = []
        for path in modules():
            for i, line in code_lines(path):
                for bad, rx in patterns.items():
                    if rx.search(line):
                        offenders.append(
                            f"{path.relative_to(ROOT)}:{i}  {bad}"
                        )
        self.assertEqual(
            offenders, [],
            "these write outside the tool folder:\n" + "\n".join(offenders),
        )

    def test_the_launcher_does_not_either(self):
        text = (ROOT / "partestimator" / "__main__.py").read_text(encoding="utf-8")
        for bad in ("tempfile", "APPDATA", "expanduser", "winreg"):
            self.assertNotIn(bad, text, f"__main__.py reaches for {bad}")

    def test_the_batch_launcher_keeps_pips_cache_inside_the_folder(self):
        """pip caches hundreds of megabytes in %LOCALAPPDATA% left alone."""
        bat = (ROOT / LAUNCHER).read_text(encoding="ascii")
        self.assertIn("PIP_CACHE_DIR", bat)
        self.assertIn("TOOLDIR%\\scratch", bat)

    def test_the_batch_launcher_disables_pips_version_check(self):
        """pip contacts PyPI on every run to see if a newer pip exists."""
        bat = (ROOT / LAUNCHER).read_text(encoding="ascii")
        self.assertIn("PIP_DISABLE_PIP_VERSION_CHECK", bat)

    def test_the_batch_launcher_refuses_source_distributions(self):
        """A sdist runs its own setup.py at install time. Wheels only."""
        bat = (ROOT / LAUNCHER).read_text(encoding="ascii")
        self.assertIn("--only-binary=:all:", bat)

    def test_the_venv_is_inside_the_tool_folder(self):
        bat = (ROOT / LAUNCHER).read_text(encoding="ascii")
        self.assertIn('set "VENV=%TOOLDIR%\\.venv"', bat)
        # Only executable lines. The comments explain that %LOCALAPPDATA% is
        # deliberately avoided, and saying so must not fail the check.
        for line in bat.splitlines():
            stripped = line.strip().lower()
            if stripped.startswith("rem") or stripped.startswith("echo"):
                continue
            self.assertNotIn(
                "localappdata", stripped, f"launcher line uses it: {line}"
            )


class TestLauncherIsWindowsSafe(unittest.TestCase):
    def test_every_line_ends_crlf(self):
        for name in BATS:
            with self.subTest(name=name):
                raw = (ROOT / name).read_bytes()
                self.assertNotIn(
                    b"\n", raw.replace(b"\r\n", b""),
                    f"{name} has a bare LF; CMD mis-parses those",
                )

    def test_paths_are_quoted(self):
        """This will live under C:\\Users\\First Last\\... sooner or later."""
        bat = (ROOT / LAUNCHER).read_text(encoding="ascii")
        for fragment in ('"%VPY%"', '"%VPYW%"', '"%VENV%"', '"%REQ%"', '"%LOCK%"'):
            self.assertIn(fragment, bat, f"{fragment} is unquoted")

    def test_it_is_pure_ascii(self):
        """CMD in a non-UTF-8 code page mangles anything else."""
        for name in BATS:
            with self.subTest(name=name):
                (ROOT / name).read_bytes().decode("ascii")


class TestDependenciesArePinnedExactly(unittest.TestCase):
    def test_every_requirement_is_an_exact_pin(self):
        text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        lines = [
            l.strip() for l in text.splitlines()
            if l.strip() and not l.strip().startswith("#")
        ]
        self.assertTrue(lines, "requirements.txt is empty")
        for line in lines:
            with self.subTest(requirement=line):
                self.assertIn("==", line, f"{line!r} is not an exact pin")
                self.assertNotIn(">", line)
                self.assertNotIn("<", line)
                self.assertNotIn("~", line)
                self.assertNotIn("*", line)

    def test_every_pinned_package_has_a_licence_entry(self):
        """The licence manifest has to keep up with the pins, not lag them."""
        reqs = [
            l.split("==")[0].strip().lower()
            for l in (ROOT / "requirements.txt").read_text().splitlines()
            if l.strip() and not l.strip().startswith("#")
        ]
        licences = (ROOT / "THIRD_PARTY_LICENSES.txt").read_text().lower()
        missing = [r for r in reqs if r not in licences]
        self.assertEqual(
            missing, [], f"not listed in THIRD_PARTY_LICENSES.txt: {missing}"
        )


class TestNoSubprocessSpawning(unittest.TestCase):
    """Nothing here starts another program."""

    def test_no_module_spawns_a_process(self):
        offenders = []
        for path in modules():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module.split(".")[0]]
                for n in names:
                    if n in ("subprocess", "multiprocessing", "pty"):
                        offenders.append(f"{path.relative_to(ROOT)}: {n}")
                if isinstance(node, ast.Attribute) and node.attr in (
                    "system", "popen", "execv", "spawnv", "startfile",
                ):
                    offenders.append(
                        f"{path.relative_to(ROOT)}:{node.lineno} os.{node.attr}"
                    )
        self.assertEqual(offenders, [], "process spawning found: " + str(offenders))


if __name__ == "__main__":
    unittest.main()
