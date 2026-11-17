"""Part Estimator command line: `python -m partestimator COMMAND ...`

Installed as the `part-estimator` console script too (`.\\pe.ps1 setup` does
an editable install into .venv), so with the venv active:

    part-estimator                          open the window
    part-estimator gui part.step            open the window on a file
    part-estimator classify a.step b.step   sheet metal or machined? (--json)
    part-estimator sheet part.step -o out   sheet metal flat pattern (all SMTv13 options)
    part-estimator calibrate times.csv      fit calibration factors to measured times
    part-estimator selftest                 full kernel + GUI check suite

Older forms still work: a bare file path opens the window on it, and
`--classify` / `--sheet-cli` behave as `classify` / `sheet`.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import traceback
from pathlib import Path

from . import __version__

PROG = "part-estimator"
COMMANDS = ("gui", "classify", "sheet", "calibrate", "selftest")
LEGACY_FLAGS = {"--classify": "classify", "--sheet-cli": "sheet"}

EPILOG = """\
examples:
  part-estimator gui bracket.step
  part-estimator gui --gl software        (force the software 3D renderer)
  part-estimator classify parts\\*.step --json
  part-estimator sheet bracket.step --material AL5052 -o out
  part-estimator calibrate training_table.csv

Run `part-estimator COMMAND --help` for a command's options.
"""


def _tool_dir() -> Path:
    return Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=PROG,
        description="Sheet metal flat patterns and machining cycle-time estimates "
                    "from STEP files. Runs offline.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    sub = p.add_subparsers(dest="command", metavar="COMMAND")

    g = sub.add_parser("gui", help="open the window (the default with no command)",
                       description="Open the Part Estimator window.")
    g.add_argument("step", nargs="?", type=Path, help="STEP file to open")
    g.add_argument("--gl", choices=("auto", "hardware", "software"), default="auto",
                   help="3D viewer OpenGL: auto (default) uses the graphics driver if "
                        "it offers 3.2+, else the software renderer bundled with "
                        "PySide6 (for VMs and remote desktops without a GPU)")

    c = sub.add_parser("classify", help="report whether parts are sheet metal or machined",
                       description="Classify STEP files as sheet metal or machined. "
                                   "Exit code 1 if any file could not be classified.")
    c.add_argument("steps", nargs="+", type=Path, metavar="STEP", help="STEP file(s)")
    c.add_argument("--json", action="store_true",
                   help="print one JSON array with every measurement, not text")
    c.add_argument("--cutoff", type=float, default=None, metavar="IN",
                   help="flat sheet thickness cutoff in inches "
                        "(default: settings.json, else 0.190)")

    # Dispatched before parsing so every option reaches the sheet metal
    # parser untouched; registered here only so it shows in --help.
    sub.add_parser("sheet", add_help=False,
                   help="develop a sheet metal flat pattern (DXF, STEP, report); "
                        "see `sheet --help`")

    k = sub.add_parser("calibrate", help="fit machining calibration factors to measured times",
                       description="Fit the machining cycle-time calibration to the "
                                   "measured times in an exported training CSV.")
    k.add_argument("csv", type=Path, help="exported CSV with cycle_time_min filled in")

    sub.add_parser("selftest", help="run every check (needs the CAD kernel and Qt)",
                   description="Run the machining self-test and the full unit-test suite.")
    return p


def _normalise(argv: list[str]) -> list[str]:
    """Map the pre-subcommand forms onto subcommands."""
    if not argv:
        return ["gui"]
    head = argv[0]
    if head in LEGACY_FLAGS:
        return [LEGACY_FLAGS[head], *argv[1:]]
    if head not in COMMANDS and not head.startswith("-"):
        # `python -m partestimator part.step`, or a file dropped on a shortcut.
        return ["gui", *argv]
    return argv


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def _expand(paths: list[Path]) -> list[Path]:
    """Expand wildcards ourselves: PowerShell and cmd pass `*.step` through as-is."""
    import glob

    out: list[Path] = []
    for p in paths:
        s = str(p)
        hits = sorted(glob.glob(s)) if any(ch in s for ch in "*?[") else []
        out.extend(Path(h) for h in hits) if hits else out.append(p)
    return out


def cmd_classify(args: argparse.Namespace) -> int:
    from .classify import classify_file

    cutoff = args.cutoff
    if cutoff is None:
        from .settings import Settings

        cutoff = Settings.load().sheet_cutoff_in
    rc = 0
    results = []
    for p in _expand(args.steps):
        if not p.is_file():
            print(f"error: {p}: no such file", file=sys.stderr)
            rc = 1
            continue
        try:
            c = classify_file(str(p), cutoff)
        except Exception as exc:  # noqa: BLE001 -- CLI boundary
            print(f"error: {p.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            rc = 1
            continue
        if args.json:
            results.append({"file": str(p), **c.to_dict()})
        else:
            print(f"{p.name}: {c.kind}  -- {c.reason}")
    if args.json:
        print(json.dumps(results, indent=2))
    return rc


def cmd_calibrate(args: argparse.Namespace) -> int:
    from .machining.calibrate import report

    return report(args.csv)


def cmd_selftest(args: argparse.Namespace) -> int:  # noqa: ARG001
    from .selftest import main as selftest_main

    return selftest_main()


def cmd_gui(args: argparse.Namespace) -> int:
    try:
        # Must run before anything imports Qt or VTK: see partestimator/opengl.py.
        from .opengl import choose

        gl = choose(getattr(args, "gl", "auto"))

        from .smt.diag import LOGGER, install_excepthook, setup_logging

        setup_logging()
        install_excepthook()
        LOGGER.info("3D viewer: %s", gl)
        from .app import run
    except BaseException as exc:  # noqa: BLE001
        _report_startup_failure(exc)
        return 1

    try:
        return run(args.step)
    except BaseException as exc:  # noqa: BLE001
        _report_startup_failure(exc)
        return 1


def _report_startup_failure(exc: BaseException) -> None:
    """Say what went wrong: on the console if there is one, else in a message box."""
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
        hint = ("\n\nThe environment is probably incomplete. Run `.\\pe.ps1 reset` "
                "and then `.\\pe.ps1 setup`.")
    message = f"Part Estimator could not start.\n\n{type(exc).__name__}: {exc}{hint}{saved}"

    # pythonw.exe has no console, and then sys.stderr is None.
    if sys.stderr is not None:
        sys.stderr.write(detail + "\n" + message + "\n")
        return
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, message, "Part Estimator", 0x10)
        except Exception:  # noqa: BLE001
            pass


HANDLERS = {
    "gui": cmd_gui,
    "classify": cmd_classify,
    "calibrate": cmd_calibrate,
    "selftest": cmd_selftest,
}


def main(argv: list[str] | None = None) -> int:
    argv = _normalise(list(sys.argv[1:] if argv is None else argv))

    if argv[0] == "sheet":
        from .smt.cli import main as sheet_main

        return sheet_main(argv[1:], prog=f"{PROG} sheet")

    args = build_parser().parse_args(argv)
    return HANDLERS[args.command](args)


if __name__ == "__main__":
    # The classifier and the machining analysis run in child processes
    # ('spawn' on Windows re-imports this module there). freeze_support keeps
    # that working if this is ever bundled into an .exe.
    multiprocessing.freeze_support()
    raise SystemExit(main())
