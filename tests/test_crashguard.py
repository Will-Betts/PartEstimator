"""The guard that stops a malformed edge killing the process.

Exit code -1073741819 is 0xC0000005, an access violation. It arrived twice:
once against v6 and again, unchanged, against v8 -- because the fix in v7 was
put in `smt/occ/tess.py`, the DISPLAY code, while the crash was on a path that
never touches it.

What proved that was the command line: `--diagnose` loads and analyses a file
with no Qt, no window and no tessellation whatsoever, and it crashed exactly the
same way. `topo.discretise_edge` -- called from `face_loops`, which detection
runs on every flange and every bend face, and from the DXF writer -- had no
guard at all.

These tests are structural. They cannot construct a malformed edge without a CAD
kernel, so instead they assert the property that actually failed: that every
place an edge is sampled goes through the check, and that the check lives
somewhere shared rather than being copied into one module and forgotten in the
others.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "partestimator" / "smt"

#: Constructing either of these on an edge with no 3D curve is undefined
#: behaviour in OCCT, which means an access violation rather than an exception.
DANGEROUS = ("BRepAdaptor_Curve", "GCPnts_QuasiUniformDeflection")


def functions_using(name: str):
    """(module, function) for every function that mentions `name`."""
    out = []
    for path in sorted(PKG.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and sub.id == name:
                    out.append((path.relative_to(ROOT), node))
                    break
    return out


def guards(node: ast.AST) -> bool:
    """Does this function call the shared curve check, or bail out first?"""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in (
            "edge_has_curve", "discretise_edge",
        ):
            return True
    return False


class TestEveryEdgeSamplerIsGuarded(unittest.TestCase):
    def test_no_unguarded_curve_construction_anywhere(self):
        offenders = []
        for name in DANGEROUS:
            for path, fn in functions_using(name):
                if not guards(fn):
                    offenders.append(f"{path}:{fn.lineno} {fn.name}() uses {name}")
        self.assertEqual(
            offenders, [],
            "these sample an edge without checking it has a curve:\n"
            + "\n".join(offenders),
        )

    def test_the_check_is_shared_not_copied(self):
        """One definition. A copy in one module is how this bug survived v7.

        The v7 fix defined a private `_edge_has_curve` inside the display module.
        Detection could not see it, so detection stayed unguarded and the crash
        was unchanged -- while the code looked, on inspection, as though it had
        been fixed.
        """
        definitions = []
        for path in sorted(PKG.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and \
                        node.name.lstrip("_") == "edge_has_curve":
                    definitions.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(
            definitions, ["partestimator/smt/occ/topo.py"],
            f"the curve check must be defined once, in topo.py; found in "
            f"{definitions}",
        )

    def test_discretise_edge_returns_a_list_it_can_be_empty(self):
        """Callers must tolerate nothing coming back.

        Returning an empty run is always survivable: a missing segment is
        repaired geometrically, or shows up as an open loop and triggers the
        fallback. Crashing the process is not survivable.
        """
        source = (PKG / "occ" / "topo.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "discretise_edge"
        )
        returns_empty = any(
            isinstance(n, ast.Return) and isinstance(n.value, ast.List)
            and not n.value.elts
            for n in ast.walk(fn)
        )
        self.assertTrue(returns_empty, "discretise_edge cannot return empty")


class TestBreadcrumbs(unittest.TestCase):
    """A native crash must name what it died on.

    Nothing in Python can catch an access violation -- the process is gone. What
    can be done is to record what was about to be touched, so the last line of
    the log names it. That is the difference between guessing and knowing, and
    guessing has now cost two releases.
    """

    def test_face_reading_logs_before_each_face(self):
        source = (PKG / "occ" / "topo.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "describe_all"
        )
        logs = [
            n for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name) and n.func.value.id == "LOGGER"
        ]
        self.assertTrue(logs, "describe_all leaves no breadcrumb")

    def test_a_safe_outline_wrapper_exists_and_is_used(self):
        topo = (PKG / "occ" / "topo.py").read_text(encoding="utf-8")
        self.assertIn("def face_loops_safe(", topo)

        detect = (PKG / "occ" / "detect.py").read_text(encoding="utf-8")
        self.assertIn("face_loops_safe(", detect)


class TestCrashHandlerIsArmed(unittest.TestCase):
    def test_faulthandler_is_enabled_by_setup(self):
        from partestimator.smt.diag import setup_logging
        import faulthandler

        setup_logging(to_file=False)
        self.assertTrue(
            faulthandler.is_enabled(),
            "faulthandler must be armed, or a native crash says nothing at all",
        )


if __name__ == "__main__":
    unittest.main()


class TestNoUnresolvedNames(unittest.TestCase):
    """Every name used in the package resolves to something.

    Added after `thickness_from_face` shipped into `detect.py` as a call with no
    import. Nothing caught it: the module imports OCP, so the suite cannot
    import it to find out, and the existing guard only checks OCP and Qt
    symbols -- this was a name from the package's own kernel-free layer.

    It would have crashed the first time anyone used the manual sheet-face
    override, which is the feature it was written for.

    Resolving names statically covers every module, including the ones that can
    never be imported here, which is exactly where this class of mistake hides.
    """

    def test_every_loaded_name_is_defined_or_imported(self):
        import builtins

        problems = []
        for path in sorted(PKG.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            defined = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef,
                                     ast.AsyncFunctionDef)):
                    defined.add(node.name)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    defined.add(node.id)
                elif isinstance(node, ast.Import):
                    for a in node.names:
                        defined.add((a.asname or a.name).split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    for a in node.names:
                        defined.add(a.asname or a.name)
                elif isinstance(node, ast.arg):
                    defined.add(node.arg)
                elif isinstance(node, ast.ExceptHandler) and node.name:
                    defined.add(node.name)
                elif isinstance(node, ast.comprehension):
                    for t in ast.walk(node.target):
                        if isinstance(t, ast.Name):
                            defined.add(t.id)

            known = defined | set(dir(builtins)) | {
                "self", "cls", "annotations", "__file__", "__name__",
            }
            for node in ast.walk(tree):
                if (isinstance(node, ast.Name)
                        and isinstance(node.ctx, ast.Load)
                        and node.id not in known):
                    problems.append(
                        f"{path.relative_to(ROOT)}:{node.lineno} "
                        f"uses {node.id!r}, which is never defined or imported"
                    )

        self.assertEqual(problems, [], "\n" + "\n".join(sorted(set(problems))))
