"""SMTv13 desktop interface (PySide6).

Imported lazily by `smt.__main__` so the CLI and the analysis library keep
working on a machine with no Qt installed.
"""

from __future__ import annotations

_HINT = """
The SMTv13 window needs PySide6.

Run `.\\pe.ps1 setup`, which installs the pinned dependencies into a .venv inside this
folder.

The command line tool (`.\\pe.ps1 sheet`) and the Python API work without Qt.
"""


def require_qt() -> None:
    try:
        import PySide6  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(_HINT) from exc
