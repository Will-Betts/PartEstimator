"""A stand-in for PySide6, so the window can be built on a machine without Qt.

WHY
---
Three shipped bugs have now lived in `smt/gui/`, and every one of them was
invisible to the test suite for the same reason: the modules import PySide6, Qt
cannot be installed in the environment this is developed in, so nothing could
ever import them -- let alone construct anything.

  * v7  inverted the vertical drag and clamped rotation to 180 degrees
  * v10 called `thickness_from_face` with no import
  * v10 used `self._stall` three statements before creating it

The first two are now caught statically. The third was caught by a static
ordering check written afterwards, which works but only sees the shapes it was
taught to look for.

Actually CONSTRUCTING the window catches all of them and everything else of that
kind, without having to anticipate the shape of the mistake. That needs a Qt,
and a fake one is enough: none of these bugs is about Qt behaviour, they are
about Python -- names that do not resolve, attributes that do not exist yet,
signals connected to methods that were renamed.

WHAT THIS IS AND IS NOT
-----------------------
It is a permissive recorder. Every class accepts any constructor arguments,
every attribute access returns something callable, and every call returns
another stub. Building the window therefore exercises the real `__init__` of
every widget, the real layout code and the real signal wiring.

It is NOT a Qt emulator and it proves nothing about how anything LOOKS or
behaves. Passing here means the code runs; it does not mean the window is
usable. The rendering, the hit testing and the actual Qt API contracts remain
unverified, as SYSTEM_SPECS says.
"""

from __future__ import annotations

import sys
import types


class _StubMeta(type):
    """Class-level attribute access, for enums.

    Qt's enums are read off the CLASS -- `Qt.AlignRight`, `Qt.StrongFocus`,
    `QPainter.Antialiasing` -- and a `__getattr__` defined on the class only
    handles access through an INSTANCE. Without this, every enum lookup during
    construction raises, which is exactly what the first run of this stub did.
    """

    def __getattr__(cls, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        value = _Stub()
        setattr(cls, name, value)
        return value


class _Stub(metaclass=_StubMeta):
    """Accepts anything, returns more of itself.

    Supports the numeric and container protocols as well, because widget code
    reasonably does arithmetic on sizes and indexes into palettes, and a stub
    that explodes on `int()` would force the test to special-case exactly the
    code it is trying to exercise.
    """

    def __init__(self, *args, **kwargs):
        self._args = args

    def __call__(self, *args, **kwargs):
        return _Stub()

    def __getattr__(self, name):
        """Qt methods are invented. OUR attributes are not.

        THIS DISTINCTION IS THE WHOLE VALUE OF THE STUB, and the first version
        got it wrong. A blanket "return a stub for anything" meant that when
        `MainWindow.__init__` used `self._stall` before creating it, the stub
        base class cheerfully invented one -- so the smoke test passed on code
        that raised `AttributeError` the moment a real Qt was underneath it. The
        stub was hiding precisely the bug it was written to catch.

        The rule now: a leading underscore means the attribute belongs to this
        codebase, not to Qt. Qt's API is camelCase and public -- `setStyleSheet`,
        `menuBar`, `update`, `rect` -- and nothing here touches a private Qt
        member. So `_anything` that has not been assigned is a real mistake and
        must raise, exactly as it would in front of you.
        """
        if name.startswith("_"):
            raise AttributeError(
                f"{type(self).__name__!r} has no attribute {name!r} -- "
                f"names starting with an underscore belong to this codebase, "
                f"so the stub will not invent one"
            )
        return _Stub()

    # containers
    def __getitem__(self, item):
        return _Stub()

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0

    def __contains__(self, item):
        return False

    # numbers -- widget code does arithmetic on sizes and colour channels
    def __int__(self):
        return 0

    def __float__(self):
        return 0.0

    def __index__(self):
        return 0

    def __bool__(self):
        return True

    def __add__(self, other):
        return _Stub()

    __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __add__
    __truediv__ = __rtruediv__ = __add__

    def __eq__(self, other):
        return isinstance(other, _Stub)

    def __hash__(self):
        return id(self.__class__)

    def __repr__(self):
        return "<qtstub>"


class _Colour(_Stub):
    """QColor needs real numbers back: shading multiplies its channels."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nums = [a for a in args if isinstance(a, (int, float))]
        self._rgb = (nums + [128, 128, 128])[:3]

    def red(self):
        return int(self._rgb[0])

    def green(self):
        return int(self._rgb[1])

    def blue(self):
        return int(self._rgb[2])

    def name(self):
        return "#%02x%02x%02x" % tuple(int(c) & 0xFF for c in self._rgb)

    def rgb(self):
        return tuple(int(c) for c in self._rgb)


class _Signal(_Stub):
    """A class-level Signal that behaves like a bound one on instances."""

    def connect(self, *a, **k):
        return None

    def emit(self, *a, **k):
        return None

    def disconnect(self, *a, **k):
        return None

    def __get__(self, obj, owner=None):
        return self


class _Module(types.ModuleType):
    """Any attribute is a usable class or constant."""

    _OVERRIDES = {"QColor": _Colour, "Signal": _Signal, "pyqtSignal": _Signal}

    def __getattr__(self, name):
        if name in self._OVERRIDES:
            value = self._OVERRIDES[name]
        elif name[:1].isupper():
            # Widget and enum names: a real class, so `class X(QWidget)` works
            # and `super().__init__(...)` is a genuine call.
            value = _StubMeta(name, (_Stub,), {})
        else:
            value = _Stub()
        setattr(self, name, value)
        return value


#: The submodules the package imports from.
_NAMES = (
    "PySide6",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
)


def install() -> dict:
    """Put the stub into `sys.modules`. Returns what was displaced."""
    saved = {}
    for name in _NAMES:
        saved[name] = sys.modules.get(name)
        sys.modules[name] = _Module(name)

    # Drop any already-imported GUI modules so they re-import against the stub.
    for name in list(sys.modules):
        if name.startswith("partestimator.smt.gui"):
            saved.setdefault(name, sys.modules[name])
            del sys.modules[name]
    return saved


def remove(saved: dict) -> None:
    """Undo `install`, so the rest of the suite is unaffected."""
    for name in list(sys.modules):
        if name.startswith("partestimator.smt.gui") or name in _NAMES:
            del sys.modules[name]
    for name, module in saved.items():
        if module is not None:
            sys.modules[name] = module


def available() -> bool:
    """True when the REAL PySide6 is installed, in which case skip the stub."""
    try:
        import PySide6  # noqa: F401
    except ImportError:
        return False
    return True
