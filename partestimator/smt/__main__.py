"""`python -m smt` opens the window.

Use `python -m smt.cli` for the command line tool.
"""

from __future__ import annotations

import sys


def main() -> int:
    from .config import UnfoldOptions
    from .diag import install_excepthook, setup_logging
    from .gui import require_qt

    setup_logging()
    install_excepthook()
    require_qt()

    from .gui.app import run

    path = sys.argv[1] if len(sys.argv) > 1 else None
    return run(UnfoldOptions(), path)


if __name__ == "__main__":
    raise SystemExit(main())
