"""Static guard against NameErrors in code this suite cannot import.

WHY THIS EXISTS
---------------
`smt/occ/*` imports OpenCASCADE and `smt/gui/*` imports Qt, so neither can be
imported on a machine without those installed -- which is deliberate, and is
what lets the rest of the suite run anywhere. The cost is that an ordinary typo
in those modules goes undetected until the code runs on a machine that has the
kernel.

That cost came due: `topo.py` used `TopExp_Explorer` while importing only
`TopExp`, and the failure surfaced as `name 'TopExp_Explorer' is not defined`
in the middle of opening a real file.

These tests read the source with `ast` and check name resolution without
executing anything, so the guard works on a bare Python. Two rules:

  1. Every OCP symbol a module uses must be imported by that module.
  2. Every Qt symbol a GUI module uses must be imported by that module.

Both are checked against the vocabulary the package itself imports, so a
genuinely new symbol is picked up automatically the first time it is used.
"""

from __future__ import annotations

import ast
import builtins
import re
import unittest
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent / "partestimator" / "smt"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def modules() -> list[Path]:
    return sorted(PKG.rglob("*.py"))


_PARSED: dict[Path, ast.Module] = {}


def parse(path: Path) -> ast.Module:
    """Cached parse -- these helpers walk every module several times over."""
    tree = _PARSED.get(path)
    if tree is None:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        _PARSED[path] = tree
    return tree


def vocabulary(prefix: str) -> set[str]:
    """Every symbol the package imports from modules starting with `prefix`."""
    names: set[str] = set()
    for path in modules():
        for node in ast.walk(parse(path)):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module == prefix or node.module.startswith(prefix + "."):
                    for a in node.names:
                        names.add(a.asname or a.name)
    return names


def imported_names(tree: ast.AST) -> set[str]:
    """Names bound by import statements directly inside this node's body."""
    out: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                out.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, (ast.If, ast.Try, ast.With)):
            # Imports guarded by try/except or a conditional still bind here.
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    for a in sub.names:
                        out.add(a.asname or a.name.split(".")[0])
    return out


def function_scopes(tree: ast.Module):
    """(function node, names imported inside it) for every function."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            local: set[str] = set()
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    for a in sub.names:
                        local.add(a.asname or a.name.split(".")[0])
            yield node, local


def used_names(tree: ast.AST) -> set[str]:
    return {
        n.id for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }


def check_module(path: Path, vocab: set[str]) -> list[str]:
    """Report vocabulary symbols used in `path` but not imported there."""
    tree = parse(path)
    module_level = imported_names(tree)

    # A name is fine if imported at module level, or inside the function that
    # uses it. Anything else in the vocabulary is a NameError waiting to happen.
    problems: list[str] = []

    scopes = list(function_scopes(tree))
    in_functions: set[int] = set()
    for fn, local in scopes:
        for sub in ast.walk(fn):
            in_functions.add(id(sub))
        for name in used_names(fn) & vocab:
            if name not in module_level and name not in local:
                problems.append(
                    f"{path.relative_to(PKG.parent)}:{fn.lineno} "
                    f"function {fn.name}() uses {name!r}, not imported"
                )

    for node in ast.walk(tree):
        if id(node) in in_functions:
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in vocab and node.id not in module_level:
                problems.append(
                    f"{path.relative_to(PKG.parent)}:{node.lineno} "
                    f"module level uses {node.id!r}, not imported"
                )
    return problems


# ---------------------------------------------------------------------------


class TestOcpImports(unittest.TestCase):
    """The regression: TopExp_Explorer used without being imported."""

    def setUp(self):
        self.vocab = vocabulary("OCP")

    def test_vocabulary_is_not_empty(self):
        """Sanity: if this collapses, the tests below pass vacuously."""
        self.assertGreater(len(self.vocab), 30, "OCP vocabulary looks wrong")
        self.assertIn("TopExp_Explorer", self.vocab)
        self.assertIn("BRepAdaptor_Surface", self.vocab)

    def test_every_ocp_symbol_used_is_imported(self):
        problems: list[str] = []
        for path in modules():
            problems.extend(check_module(path, self.vocab))
        self.assertEqual(problems, [], "\n" + "\n".join(problems))

    def test_topexp_explorer_specifically(self):
        """Pin the exact symbol that failed, by name."""
        for path in modules():
            tree = parse(path)
            uses = used_names(tree)
            if "TopExp_Explorer" not in uses:
                continue
            available = imported_names(tree)
            for fn, local in function_scopes(tree):
                available |= local
            self.assertIn(
                "TopExp_Explorer", available,
                f"{path.name} uses TopExp_Explorer without importing it",
            )


class TestQtImports(unittest.TestCase):
    def setUp(self):
        self.vocab = vocabulary("PySide6")

    def test_vocabulary_is_not_empty(self):
        self.assertGreater(len(self.vocab), 15, "Qt vocabulary looks wrong")

    def test_every_qt_symbol_used_is_imported(self):
        problems: list[str] = []
        for path in modules():
            problems.extend(check_module(path, self.vocab))
        self.assertEqual(problems, [], "\n" + "\n".join(problems))


class TestNamingConventionCatchesUnknownSymbols(unittest.TestCase):
    """The gap the vocabulary check cannot close, closed by naming convention.

    `vocabulary()` is built from what the package already imports, so it
    catches a symbol used here that is imported THERE -- which is the common
    typo, and is exactly how `TopExp_Explorer` was found.

    It cannot catch a symbol that is imported NOWHERE. A GUI module that calls
    `QRadialGradient` without importing it anywhere in the package produces a
    vocabulary that has never heard of `QRadialGradient`, so nothing looks for
    it. Verified by introducing exactly that: the OCP check caught its half,
    the Qt check did not catch this one.

    Qt's naming convention closes it. Every Qt class is `Q` followed by a
    capital, and nothing else in this codebase is named that way, so any such
    name that is neither imported nor defined locally is a NameError waiting
    for the machine that has Qt installed.
    """

    QT_CLASS = re.compile(r"^Q[A-Z][A-Za-z0-9_]*$")

    def test_no_gui_module_uses_an_unimported_qt_class(self):
        problems: list[str] = []

        for path in modules():
            tree = parse(path)
            available = imported_names(tree) | _defined_names(tree)

            local_by_fn = {}
            for fn, local in function_scopes(tree):
                local_by_fn[id(fn)] = local

            for node in ast.walk(tree):
                if not (isinstance(node, ast.Name)
                        and isinstance(node.ctx, ast.Load)):
                    continue
                if not self.QT_CLASS.match(node.id):
                    continue
                if node.id in available:
                    continue
                if any(node.id in local for local in local_by_fn.values()):
                    continue
                problems.append(
                    f"{path.relative_to(PKG.parent)}:{node.lineno} "
                    f"uses {node.id!r}, which is imported nowhere"
                )

        self.assertEqual(problems, [], "\n" + "\n".join(problems))


def _defined_names(tree: ast.Module) -> set[str]:
    """Anything the module defines itself: classes, functions, assignments."""
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            out.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            out.add(node.id)
    return out


class TestNoAccidentalShadowing(unittest.TestCase):
    def test_no_module_imports_itself(self):
        for path in modules():
            for node in ast.walk(parse(path)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.assertNotEqual(
                        node.module.split(".")[-1], path.stem,
                        f"{path.name} imports from itself",
                    )

    # Modules outside smt/occ that may touch OCP, and why. Both import it
    # LAZILY inside a function and both tolerate its absence, so neither breaks
    # the rule that the package imports cleanly on a machine with no kernel.
    KERNEL_EXCEPTIONS = {
        # Exact curve export: emits true LINE and ARC entities off the kernel
        # face, wrapped in try/except that falls back to polylines.
        "dxf.py",
        # Logs the OCCT version at startup, inside try/except.
        "diag.py",
    }

    def test_kernel_stays_quarantined(self):
        """Only smt/occ may import OpenCASCADE, bar two sanctioned exceptions.

        This is the architectural rule that keeps the whole suite runnable
        without a CAD kernel. If it erodes, the tests stop being portable and
        the next typo hides until it reaches a machine that has the kernel --
        which is exactly how `TopExp_Explorer` got out.
        """
        for path in modules():
            rel = path.relative_to(PKG)
            if rel.parts and rel.parts[0] == "occ":
                continue
            for node in ast.walk(parse(path)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.startswith("OCP"):
                        self.assertIn(
                            path.name, self.KERNEL_EXCEPTIONS,
                            f"{rel} imports OCP outside smt/occ/. If that is "
                            f"deliberate, import it lazily and add it to "
                            f"KERNEL_EXCEPTIONS with a reason.",
                        )

    def test_exceptions_import_the_kernel_lazily(self):
        """The sanctioned exceptions must not import OCP at module level."""
        for path in modules():
            if path.name not in self.KERNEL_EXCEPTIONS:
                continue
            top_level = imported_names(parse(path))
            ocp_vocab = vocabulary("OCP")
            leaked = top_level & ocp_vocab
            self.assertEqual(
                leaked, set(),
                f"{path.name} imports {sorted(leaked)} at module level; it must "
                f"be lazy or the package stops importing without a kernel",
            )

    def test_package_imports_without_a_kernel(self):
        """The end-to-end guarantee, stated as a test.

        If this ever fails, `python main.py --test` stops working on a machine
        that has not finished installing OpenCASCADE -- which is the one moment
        you most want the maths tests to run.
        """
        import importlib

        for name in ("partestimator.smt", "partestimator.smt.geom", "partestimator.smt.allowance", "partestimator.smt.model2d",
                     "partestimator.smt.report", "partestimator.smt.units", "partestimator.smt.diag", "partestimator.smt.cli"):
            importlib.import_module(name)

    def test_qt_stays_quarantined(self):
        for path in modules():
            rel = path.relative_to(PKG)
            if rel.parts and rel.parts[0] == "gui":
                continue
            for node in ast.walk(parse(path)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.assertFalse(
                        node.module.startswith("PySide6"),
                        f"{rel} imports Qt outside smt/gui/",
                    )


class TestOfflineByConstruction(unittest.TestCase):
    """Your STEP files never leave the machine, and this enforces it.

    Confidential geometry is the normal case in a job shop, so "it's local"
    should be a property the code cannot quietly lose, not an assurance in a
    README. Nothing in the package may import a module capable of opening a
    network connection.

    The only network activity in the whole project is pip fetching
    dependencies during the one-time install, which is the launcher's job and
    happens before any part file is ever opened.
    """

    NETWORK_MODULES = {
        "socket", "ssl", "http", "urllib", "urllib2", "urllib3", "requests",
        "httpx", "aiohttp", "ftplib", "smtplib", "poplib", "imaplib",
        "telnetlib", "xmlrpc", "webbrowser", "asyncio", "websockets",
        "boto3", "paramiko", "pycurl",
    }

    def _imports(self, tree: ast.AST) -> set[str]:
        found: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    found.add(a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module.split(".")[0])
        return found

    def test_no_network_capable_imports(self):
        offenders = []
        for path in modules():
            hits = self._imports(parse(path)) & self.NETWORK_MODULES
            if hits:
                offenders.append(f"{path.relative_to(PKG.parent)}: {sorted(hits)}")
        self.assertEqual(
            offenders, [],
            "the package must not be able to reach the network:\n"
            + "\n".join(offenders),
        )

    def test_launcher_is_also_offline(self):
        launcher = PKG.parent / "__main__.py"
        hits = self._imports(parse(launcher)) & self.NETWORK_MODULES
        self.assertEqual(hits, set(), f"__main__.py imports {sorted(hits)}")

    def test_no_url_literals(self):
        """No endpoint can be reached if none is written down."""
        import re

        pattern = re.compile(r"https?://")
        offenders = []
        for path in modules():
            for i, line in enumerate(path.read_text().splitlines(), 1):
                if pattern.search(line):
                    offenders.append(f"{path.relative_to(PKG.parent)}:{i}")
        self.assertEqual(offenders, [], "URL literals found: " + str(offenders))

    def test_subprocess_use_is_limited_to_opening_the_log(self):
        """One sanctioned shell-out: showing you smt.log in your own viewer."""
        allowed = {"app.py"}
        offenders = []
        for path in modules():
            src = path.read_text()
            if "subprocess" in src or "os.system" in src or "os.popen" in src:
                if path.name not in allowed:
                    offenders.append(str(path.relative_to(PKG.parent)))
        self.assertEqual(offenders, [], f"unexpected process spawning: {offenders}")


if __name__ == "__main__":
    unittest.main()
