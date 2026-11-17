"""Constructors must not use attributes they have not created yet.

WHY THIS EXISTS
---------------
SMTv10 shipped with this in `MainWindow.__init__`:

    self._build_ui()          # calls status(), which touches _stall
    self._set_busy(False)     # calls self._stall.stop()
    ...
    self._stall = QTimer(self)      # created here, three calls too late

It raised `AttributeError: 'MainWindow' object has no attribute '_stall'` on the
first line of the constructor that did any real work -- so the window never
opened at all.

Nothing existing could catch it. The name resolves fine, the import is fine, the
syntax is fine; it is purely a question of ORDER, and the only way to find it at
runtime is to construct the object -- which needs Qt, which is not installable
here. That is the same gap that has now produced several shipped bugs, so this
closes the part of it that can be closed statically.

WHAT IT CHECKS
--------------
Walk `__init__` top to bottom, tracking which `self.x` have been assigned so
far. Whenever it calls one of the class's own methods, check every attribute
that method reads. If the method reads an attribute that `__init__` assigns
LATER, that is the bug.

Deliberately narrow: it only flags attributes the constructor itself assigns, so
attributes set by a base class or created lazily elsewhere are not false
positives. It follows nested calls two levels deep, which is enough for the
helper-calls-helper pattern that caused this and shallow enough to stay fast and
predictable.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "partestimator" / "smt"

#: How far to follow method-calls-method before giving up.
MAX_DEPTH = 2


def self_attr_stores(node: ast.AST) -> list[str]:
    """`self.x = ...` targets inside a node, in source order."""
    out = []
    for sub in ast.walk(node):
        if not isinstance(sub, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            continue
        targets = sub.targets if isinstance(sub, ast.Assign) else [sub.target]
        for t in targets:
            if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                    and t.value.id == "self"):
                out.append(t.attr)
    return out


def self_attr_loads(node: ast.AST) -> set[str]:
    """`self.x` read anywhere inside a node."""
    out = set()
    for sub in ast.walk(node):
        if (isinstance(sub, ast.Attribute) and isinstance(sub.ctx, ast.Load)
                and isinstance(sub.value, ast.Name) and sub.value.id == "self"):
            out.add(sub.attr)
    return out


def self_method_calls(node: ast.AST) -> list[str]:
    """Names of `self.foo(...)` calls inside a node."""
    out = []
    for sub in ast.walk(node):
        if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                and isinstance(sub.func.value, ast.Name)
                and sub.func.value.id == "self"):
            out.append(sub.func.attr)
    return out


def reads_through(name: str, methods: dict, depth: int = 0,
                  seen: frozenset = frozenset()) -> set:
    """Every attribute a method reads, following its own calls `depth` deep."""
    fn = methods.get(name)
    if fn is None or name in seen or depth > MAX_DEPTH:
        return set()

    out = self_attr_loads(fn)
    seen = seen | {name}
    for callee in self_method_calls(fn):
        if callee in methods:
            out |= reads_through(callee, methods, depth + 1, seen)
    return out


def problems_in_class(cls: ast.ClassDef, where: str) -> list[str]:
    methods = {
        n.name: n for n in cls.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    init = methods.get("__init__")
    if init is None:
        return []

    created_by_init = set(self_attr_stores(init))
    assigned: set[str] = set()
    found: list[str] = []

    for stmt in init.body:
        # Calls first: within one statement, a call happens before the
        # assignment it feeds, and treating it the other way round would miss
        # `self.x = self.build()` where build() reads self.x.
        for callee in self_method_calls(stmt):
            if callee not in methods:
                continue
            for attr in reads_through(callee, methods):
                if attr in created_by_init and attr not in assigned:
                    found.append(
                        f"{where}:{stmt.lineno} {cls.name}.__init__ calls "
                        f"self.{callee}(), which reads self.{attr} -- but "
                        f"self.{attr} is not assigned until later"
                    )
        assigned.update(self_attr_stores(stmt))

    return found


class TestConstructorOrdering(unittest.TestCase):
    def test_no_constructor_uses_an_attribute_before_creating_it(self):
        found: list[str] = []
        for path in sorted(PKG.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            where = str(path.relative_to(ROOT))
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    found.extend(problems_in_class(node, where))

        self.assertEqual(
            found, [],
            "constructors use attributes before creating them:\n"
            + "\n".join(sorted(set(found))),
        )

    def test_the_check_actually_detects_the_v10_bug(self):
        """A self-test, because a checker that silently passes is worthless.

        This is the exact shape of the shipped bug: a helper called from
        `__init__` reads an attribute the constructor assigns further down.
        """
        source = '''
class Window:
    def __init__(self):
        self._build()
        self._stall = 1

    def _build(self):
        self._stall.stop()
'''
        cls = ast.parse(source).body[0]
        found = problems_in_class(cls, "example.py")
        self.assertEqual(len(found), 1, f"the check missed it: {found}")
        self.assertIn("_stall", found[0])

    def test_the_check_does_not_fire_on_correct_ordering(self):
        source = '''
class Window:
    def __init__(self):
        self._stall = 1
        self._build()

    def _build(self):
        self._stall.stop()
'''
        cls = ast.parse(source).body[0]
        self.assertEqual(problems_in_class(cls, "example.py"), [])

    def test_it_ignores_attributes_the_constructor_never_creates(self):
        """Base-class and lazily-created attributes are not this test's business.

        Narrow on purpose. A checker that reports things which are fine is one
        people stop reading, and then it stops catching the things that are not.
        """
        source = '''
class Window:
    def __init__(self):
        self._build()

    def _build(self):
        self.setWindowTitle("x")
        self.something_from_a_base_class.do()
'''
        cls = ast.parse(source).body[0]
        self.assertEqual(problems_in_class(cls, "example.py"), [])

    def test_it_follows_one_helper_calling_another(self):
        source = '''
class Window:
    def __init__(self):
        self._outer()
        self._timer = 1

    def _outer(self):
        self._inner()

    def _inner(self):
        self._timer.stop()
'''
        cls = ast.parse(source).body[0]
        found = problems_in_class(cls, "example.py")
        self.assertEqual(len(found), 1, f"nested call not followed: {found}")


if __name__ == "__main__":
    unittest.main()
