"""`python -m partestimator [file.step]` opens the window.

    python -m partestimator --classify part.step   print sheet metal / machined
    python -m partestimator --sheet-cli part.step  sheet metal command line tool
                                                   (all the old SMTv13 options)
"""
from __future__ import annotations

import multiprocessing
import sys
import traceback
from pathlib import Path


def _tool_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _report_startup_failure(exc: BaseException) -> None:
    """The window runs without a console, so say what went wrong somewhere visible."""
    detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    log = _tool_dir() / "last_error.log"
    try:
        log.write_text(f"Python: {sys.version}\nExecutable: {sys.executable}\n\n{detail}",
                       encoding="utf-8")
        saved = f"\n\nFull details were saved to:\n{log}"
    except Exception:  # noqa: BLE001
        saved = ""
    hint = ""
    if isinstance(exc, ImportError):
        hint = ("\n\nThe environment is probably incomplete. Delete the .venv folder "
                "inside the Part Estimator folder and run the launcher again.")
    message = f"Part Estimator could not start.\n\n{type(exc).__name__}: {exc}{hint}{saved}"
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, message, "Part Estimator", 0x10)
            return
        except Exception:  # noqa: BLE001
            pass
    sys.stderr.write(message + "\n" + detail)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv[:1] == ["--classify"]:
        from .classify import classify_file
        from .settings import Settings

        rc = 0
        for p in argv[1:]:
            c = classify_file(p, Settings.load().sheet_cutoff_in)
            print(f"{Path(p).name}: {c.kind}  -- {c.reason}")
        return rc

    if argv[:1] == ["--sheet-cli"]:
        from .smt.cli import main as sheet_main

        return sheet_main(argv[1:])

    try:
        from .smt.diag import install_excepthook, setup_logging

        setup_logging()
        install_excepthook()
        from .app import run
    except BaseException as exc:  # noqa: BLE001
        _report_startup_failure(exc)
        return 1

    try:
        path = Path(argv[0]) if argv else None
        return run(path)
    except BaseException as exc:  # noqa: BLE001
        _report_startup_failure(exc)
        return 1


if __name__ == "__main__":
    # The classifier and the machining analysis run in child processes
    # ('spawn' on Windows re-imports this module there). freeze_support keeps
    # that working if this is ever bundled into an .exe.
    multiprocessing.freeze_support()
    raise SystemExit(main())
