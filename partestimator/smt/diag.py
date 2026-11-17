"""Diagnostics: staged progress, timings, and logging that cannot leak.

An operation that can take a minute must narrate itself -- silence is
indistinguishable from a hang, and an early version of this tool could sit
apparently thinking forever with an empty console. Everything slow now reports
through a `Stage`, which logs to the console, to a file beside the application,
and to a callback the GUI wires into its status bar.

WHAT IS DELIBERATELY NOT LOGGED
-------------------------------
This tool runs on confidential geometry, so the log records WHAT HAPPENED and
HOW MUCH, never WHAT IT WAS. No file contents, no model text, and no full
paths -- a path is not the part, but on a real network share it names the
customer, the programme and the revision, and a log file that gets pasted into
an email should not carry that.

Two mechanisms, because one is not enough:

  * `name_only()` for every call site that deliberately mentions a file.
  * `RedactPaths`, a logging filter that strips directory components out of
    anything that still looks like a path by the time it reaches a handler.

The filter is the safety net. Call sites get missed -- one did, in the version
this was written for -- and a redaction that only works when the author
remembered it is not a control, it is a hope. It is deliberately conservative:
it needs a drive letter or two or more separators before it will touch a string,
so ordinary text like "area/thickness" or "and/or" is left alone.

SCRATCH FILES
-------------
Everything temporary goes in a subfolder of the tool directory and is removed on
exit. Nothing is written to %TEMP%, %APPDATA% or any user-profile cache, so
deleting the tool folder really is a complete uninstall.
"""

from __future__ import annotations

import faulthandler
import logging
import re
import shutil
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

APP_NAME = "SMTv13"

LOGGER = logging.getLogger("smt")

_CONFIGURED = False

#: Three shapes, because Windows paths contain spaces and POSIX-ish ones must
#: not swallow ordinary prose.
#:
#:   1. a drive letter          C:\Users\First Last\Jobs\rev C\part.step
#:   2. a UNC share             \\fileserver\eng\2291\part.step
#:   3. two or more separators  /mnt/share/project/part.step
#:
#: Forms 1 and 2 allow spaces, because a real Windows path routinely has them
#: and refusing to match one is the failure that matters. Form 3 does not,
#: because without a drive letter to anchor it a space-tolerant pattern turns
#: "ratio a/b and c/d" into "d". A single separator is never matched at all:
#: "and/or" and "mm/s" are prose, and a filter that mangles ordinary messages
#: is a filter somebody switches off.
_PATH_RE = re.compile(
    r"[A-Za-z]:[\\/][^\r\n\"'<>|]*"
    r"|\\\\[^\r\n\"'<>|]*"
    r"|(?:[\\/][^\\/\s\"'<>|]+){2,}"
)


def name_only(path) -> str:
    """The file name, with every directory component removed.

    Splits on BOTH separators rather than going through pathlib. `Path` follows
    the rules of the platform it is running on, so a Windows path handed to a
    POSIX interpreter comes back whole -- which is exactly the case that has to
    work when a log written on Windows is read anywhere else.

    Use this at every call site that mentions a file. `RedactPaths` will catch
    the ones that get missed, but a redaction that depends on the filter having
    caught it is a weaker guarantee than one made where the message is written.
    """
    try:
        text = str(path)
    except Exception:  # noqa: BLE001
        return "(unnamed)"
    return re.split(r"[\\/]", text)[-1].strip() or "(unnamed)"


def redact(text: str) -> str:
    """Strip directory components out of anything path-shaped."""

    def _one(m: re.Match) -> str:
        return name_only(m.group(0))

    return _PATH_RE.sub(_one, text)


class RedactPaths(logging.Filter):
    """Last line of defence: no full path reaches a handler.

    Applied to the formatted message rather than to the arguments, so it works
    regardless of how a call site built its string.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 -- never let logging break the run
            return True
        cleaned = redact(message)
        if cleaned != message:
            record.msg = cleaned
            record.args = ()
        return True


def tool_dir() -> Path:
    """The folder the tool was extracted into. Nothing is written outside it."""
    # partestimator/smt/diag.py -> the PartEstimator folder (two levels above
    # the package), so logs and scratch land beside the launcher.
    return Path(__file__).resolve().parents[2]


def log_path() -> Path:
    """Next to the application, not in a temp folder nobody can find."""
    return tool_dir() / "part_estimator.log"


def scratch_dir(create: bool = True) -> Path:
    """A working folder INSIDE the tool directory.

    Explicitly not %TEMP% and not %APPDATA%. Everything this tool writes lives
    under the folder you extracted, so deleting that folder is a complete
    uninstall with nothing left behind.
    """
    d = tool_dir() / "scratch"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def clean_scratch() -> None:
    """Remove the scratch folder. Safe to call when it was never created."""
    d = scratch_dir(create=False)
    try:
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            LOGGER.debug("scratch folder cleaned")
    except OSError:
        LOGGER.debug("scratch folder could not be removed")


def setup_logging(verbose: bool = True, to_file: bool = True) -> Path | None:
    """Console plus file logging. Safe to call more than once."""
    global _CONFIGURED
    if _CONFIGURED:
        return log_path() if to_file else None

    LOGGER.setLevel(logging.DEBUG if verbose else logging.INFO)
    LOGGER.propagate = False

    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", "%H:%M:%S")

    redactor = RedactPaths()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.addFilter(redactor)
    LOGGER.addHandler(console)

    path = None
    if to_file:
        try:
            path = log_path()
            fh = logging.FileHandler(path, mode="w", encoding="utf-8")
            fh.setFormatter(fmt)
            fh.setLevel(logging.DEBUG)
            fh.addFilter(redactor)
            LOGGER.addHandler(fh)
        except OSError:
            # A read-only install folder should not stop the program running.
            path = None

    _CONFIGURED = True
    install_crash_handler()
    _log_environment()
    return path


def _log_environment() -> None:
    """Record versions once. Most 'it works here' problems are version skew."""
    LOGGER.info("%s starting", APP_NAME)
    LOGGER.debug("python   %s", sys.version.split()[0])
    LOGGER.debug("platform %s", sys.platform)
    for name, mod in (("numpy", "numpy"), ("PySide6", "PySide6"),
                      ("ezdxf", "ezdxf")):
        try:
            m = __import__(mod)
            LOGGER.debug("%-8s %s", name, getattr(m, "__version__", "?"))
        except ImportError:
            LOGGER.debug("%-8s not installed", name)
    try:
        import OCP
        from OCP.Standard import Standard_Version

        LOGGER.debug("OCCT     %s", Standard_Version.Get_s())
    except Exception:  # noqa: BLE001
        try:
            import OCP  # noqa: F811

            LOGGER.debug("OCP      present, version unknown")
        except ImportError:
            LOGGER.debug("OCP      not installed")


#: Held open for the life of the process. `faulthandler` writes through this
#: handle from a signal context, so it must not be garbage collected.
_CRASH_FILE = None


def crash_path() -> Path:
    return tool_dir() / "part_estimator-crash.txt"


def install_crash_handler() -> Path | None:
    """Catch native crashes, which no Python exception handler can see.

    SMTv6 died with exit code -1073741819 -- 0xC0000005, an access violation
    inside native code. Nothing in Python runs after that: no `except`, no
    `finally`, no excepthook, no log line. The process is simply gone, and all
    that reaches you is a number.

    `faulthandler` installs an OS-level signal handler that dumps the Python
    stack of every thread at the moment of the fault, so a crash inside the CAD
    kernel names the exact Python frame that called into it. That is the
    difference between "it crashed" and "it crashed in `_mesh_shape`".

    Standard library, no dependency, no network, and the file lands in this
    tool's own folder alongside the log.
    """
    global _CRASH_FILE
    if _CRASH_FILE is not None:
        return crash_path()
    try:
        path = crash_path()
        _CRASH_FILE = open(path, "w", encoding="utf-8", buffering=1)
        faulthandler.enable(file=_CRASH_FILE, all_threads=True)
        LOGGER.debug("crash handler armed")
        return path
    except OSError:
        # A read-only folder must not stop the tool running.
        _CRASH_FILE = None
        return None


def install_excepthook() -> None:
    """Nothing should ever fail silently again -- including in Qt slots."""

    def hook(exc_type, exc, tb):
        LOGGER.error(
            "UNCAUGHT %s: %s\n%s",
            exc_type.__name__, exc,
            "".join(traceback.format_exception(exc_type, exc, tb)),
        )
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = hook

    # PySide6 swallows exceptions raised inside slots by default on some
    # versions; threading.excepthook covers the worker threads.
    try:
        import threading

        def thook(args):
            LOGGER.error(
                "UNCAUGHT in thread %s: %s\n%s",
                args.thread.name if args.thread else "?", args.exc_value,
                "".join(
                    traceback.format_exception(
                        args.exc_type, args.exc_value, args.exc_traceback
                    )
                ),
            )

        threading.excepthook = thook
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------


class Stage:
    """Reports what is happening, and how long it has taken.

    Pass one of these into anything slow. The GUI supplies a callback that
    forwards to the status bar and log pane; the CLI leaves it as plain logging.
    """

    def __init__(self, callback=None, cancel=None):
        self._cb = callback
        self._cancel = cancel
        self._t0 = time.perf_counter()
        self._stage_t0 = self._t0
        self.current = ""

    # -- reporting ------------------------------------------------------
    def __call__(self, message: str, detail: str = "") -> None:
        self.current = message
        self._stage_t0 = time.perf_counter()
        text = f"{message}{('  ' + detail) if detail else ''}"
        LOGGER.info("[%6.2fs] %s", self.elapsed, text)
        if self._cb:
            self._cb(text)

    def detail(self, message: str) -> None:
        """A sub-step. Logged, but does not reset the stage timer."""
        LOGGER.debug("[%6.2fs]     %s", self.elapsed, message)
        if self._cb:
            self._cb(f"{self.current} -- {message}")

    def done(self, message: str = "") -> None:
        LOGGER.info(
            "[%6.2fs] %s done in %.2fs%s",
            self.elapsed, self.current or "stage", self.stage_elapsed,
            f"  {message}" if message else "",
        )

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._t0

    @property
    def stage_elapsed(self) -> float:
        return time.perf_counter() - self._stage_t0

    # -- cancellation ---------------------------------------------------
    def check_cancelled(self) -> None:
        """Cooperative cancellation between stages.

        A kernel call already in flight cannot be interrupted, so this only
        takes effect at stage boundaries. That is still enough to stop a long
        pipeline promptly rather than making you kill the window.
        """
        if self._cancel is not None and self._cancel():
            raise Cancelled(f"cancelled during: {self.current}")

    @contextmanager
    def step(self, message: str, detail: str = ""):
        self.check_cancelled()
        self(message, detail)
        try:
            yield self
        finally:
            LOGGER.debug("[%6.2fs]   '%s' took %.2fs",
                         self.elapsed, message, self.stage_elapsed)


class Cancelled(Exception):
    """Raised when the user asks a long operation to stop."""


def null_stage() -> Stage:
    return Stage()
