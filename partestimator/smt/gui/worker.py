"""Background work, and the one bug that made it look like a hang.

THE GARBAGE COLLECTION BUG, RECORDED IN FULL
--------------------------------------------
The first version created the worker as a local variable and kept only the
`QThread`:

    thread = QThread()
    worker = Worker(path)          # <-- only reference, and it is local
    worker.moveToThread(thread)
    thread.start()
    return thread                  # `worker` goes out of scope here

Python then collected `worker` while it was mid-flight. The C++ object went with
it, so `finished` never arrived, `failed` never arrived, and no traceback was
printed anywhere. From the outside the program opened a file and thought
forever.

The fix is `_ACTIVE`: a module-level set holding a HARD reference to every
worker and thread until it reports back. It looks redundant and it is not. Any
future edit that drops it reintroduces a silent hang, which is why this comment
is longer than the code it explains.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, QThread, Signal, Slot

from ..config import UnfoldOptions
from ..diag import LOGGER, Cancelled, Stage, name_only

#: Hard references to everything in flight. See the module docstring.
_ACTIVE: set = set()


@dataclass
class JobHandle:
    """What the caller keeps so it can cancel and so nothing is collected."""

    thread: QThread
    worker: QObject
    cancelled: bool = False
    _flag: dict = field(default_factory=lambda: {"stop": False})

    def cancel(self) -> None:
        self.cancelled = True
        self._flag["stop"] = True

    def is_cancelled(self) -> bool:
        return bool(self._flag["stop"])


class _BaseWorker(QObject):
    progress = Signal(str)
    finished = Signal(object)
    failed = Signal(str, str)      # short message, full traceback

    def __init__(self, flag: dict):
        super().__init__()
        self._flag = flag

    def _stage(self) -> Stage:
        return Stage(
            callback=self.progress.emit,
            cancel=lambda: bool(self._flag["stop"]),
        )

    def run(self) -> None:
        try:
            result = self.work()
        except Cancelled as exc:
            LOGGER.info("cancelled: %s", exc)
            self.failed.emit("Cancelled.", "")
        except Exception as exc:  # noqa: BLE001 -- nothing may fail silently
            tb = traceback.format_exc()
            LOGGER.error("worker failed: %s\n%s", exc, tb)
            self.failed.emit(str(exc), tb)
        else:
            self.finished.emit(result)

    def work(self):
        raise NotImplementedError


@dataclass
class OpenResult:
    """Everything an opened file yields, in one pass.

    v6 through v9 split this in two: open the file, look at it, then press
    Flatten. The reasoning was that you should get to check the tool understood
    the part before committing to an unfold.

    In practice the blank IS the answer -- it is what the tool is for -- and
    making you ask for it twice was two clicks and a wait standing between a
    file and a number. So opening now does the whole job and lands on the flat.
    The folded part is still there behind the Form button, which is the right
    way round: the as-modelled shape is the reference, not the goal.
    """

    part: Any
    shape: Any
    stem: str = ""
    file_units: str | None = None

    #: The flat solid, and the folded part, as drawable scenes.
    flat_scene: Any = None
    flat_roles: dict = field(default_factory=dict)
    folded_scene: Any = None
    folded_roles: dict = field(default_factory=dict)

    flat2d: Any = None
    result: Any = None
    #: True when the flat is an extruded blank rather than the moved solid, so
    #: it carries no countersinks or chamfers.
    extruded: bool = False
    #: Set when the part has no bends: the blank IS the part.
    is_blank: bool = False


class OpenWorker(_BaseWorker):
    """Read a STEP file and produce the flat pattern. One pass, one wait.

    The folded scene is built LAST and deliberately: it is the least urgent
    thing on screen, and building it after the flat means the numbers and the
    blank appear as early as they can. If it fails, everything that matters is
    already in hand.
    """

    def __init__(self, path: Path, opts: UnfoldOptions, table, flag: dict):
        super().__init__(flag)
        self.path = Path(path)
        self.opts = opts
        self.table = table

    def work(self) -> OpenResult:
        from ..occ import require_kernel

        require_kernel()

        from ..model2d import build, oriented_flat_frame
        from ..occ.detect import classify_shape, detect
        from ..occ.load import load_step
        from ..occ.tess import build_scene
        from ..unfold import assign_allowances, solve
        from ..units import sniff_step_units

        stage = self._stage()
        opts = self.opts

        stage(f"Reading {name_only(self.path)}...")
        declared = sniff_step_units(self.path)
        shape = load_step(self.path)
        stage.done()

        stage.check_cancelled()
        part = detect(shape, opts, stage, file_units=declared)

        out = OpenResult(
            part=part, shape=shape, stem=self.path.stem, file_units=declared,
            is_blank=bool(part.is_blank),
        )

        stage.check_cancelled()
        stage("Unfolding...")
        assign_allowances(part, opts, self.table)
        result = solve(part, opts)
        part.base_flange = result.base
        part.transforms = result.transforms
        out.result = result
        out.flat2d = build(part, result, opts)
        stage.done(
            f"{out.flat2d.blank_length:.2f} x {out.flat2d.blank_width:.2f} mm"
        )

        # The same oriented frame the outline was built from, so the solid lands
        # in the identical place. Deriving it twice is how the DXF and the
        # preview came to disagree in an earlier version.
        M_frame, _angle, _saved = oriented_flat_frame(part, result, opts)

        stage.check_cancelled()
        solid, extruded = self._build_solid(part, result, M_frame, stage)
        part.flat_solid = solid
        part.flat_is_extruded = extruded
        out.extruded = extruded

        if solid is not None:
            stage.check_cancelled()
            stage("Drawing the blank...")
            try:
                roles, infos = classify_shape(solid, part.thickness, opts.tol)
                out.flat_scene = build_scene(solid, infos)
                out.flat_roles = roles
                stage.done(f"{len(out.flat_scene)} faces")
            except Exception as exc:  # noqa: BLE001 -- the numbers still stand
                LOGGER.warning("the flat view could not be drawn: %s", exc)

        # Last, because it is the least urgent thing on screen.
        stage.check_cancelled()
        stage("Drawing the folded part...")
        try:
            out.folded_scene = build_scene(
                shape, getattr(part, "face_infos", None),
                loop_cache=getattr(part, "loop_cache", None),
            )
            out.folded_roles = {i: f.role for i, f in part.features.items()}
            stage.done(f"{len(out.folded_scene)} faces")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("the folded view could not be drawn: %s", exc)

        return out

    def _build_solid(self, part, result, M_frame, stage):
        """The real unfolded solid, or an extruded blank. (solid, is_extruded)."""
        from ..occ.unfold3d import unfold_solid

        solid = unfold_solid(part, result, self.opts, M_frame, stage)
        if solid is not None:
            return solid, False

        stage("Building an extruded blank instead...")
        try:
            from ..occ.flatten import build_flat

            part.flat = build_flat(part, result, self.opts)
            stage.done()
            return part.flat.solid, True
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("extruded blank also failed: %s", exc)
            stage.detail("no 3D flat could be built")
            return None, True


class ExportWorker(_BaseWorker):
    """Write the requested files. Kernel work, so off the UI thread."""

    def __init__(self, part, opts, out_dir, stem, kinds: dict, flag: dict):
        super().__init__(flag)
        self.part = part
        self.opts = opts
        self.out_dir = Path(out_dir)
        self.stem = stem
        self.kinds = kinds

    def work(self) -> dict:
        from ..occ.flatten import build_flat
        from ..pipeline import write_outputs
        from ..unfold import solve

        stage = self._stage()

        if self.part.flat is None:
            stage("Building the flat solid...")
            result = solve(self.part, self.opts)
            self.part.base_flange = result.base
            self.part.transforms = result.transforms
            self.part.flat = build_flat(self.part, result, self.opts)
            stage.done()

        stage.check_cancelled()
        stage("Writing files...")
        written = write_outputs(
            self.part, self.opts, self.out_dir, self.stem,
            want_dxf=self.kinds.get("dxf", True),
            want_step=self.kinds.get("step", True),
            want_json=self.kinds.get("json", True),
        )
        stage.done(f"{len(written)} file(s)")
        return written


class _Relay(QObject):
    """Carries a worker's signals back to the UI thread.

    THE THREAD BUG (found when SMTv13 was merged into Part Estimator)
    ----------------------------------------------------------------
    v13 connected `finished` / `failed` / `progress` straight to plain Python
    closures. PySide decides queued-vs-direct delivery from the RECEIVER's
    thread, and a bare function has none -- so the connection was DIRECT and
    `_opened`, `log` and every widget update ran on the WORKER thread. Qt
    printed "Timers cannot be started from another thread" / "Cannot set
    parent, new parent is in a different thread" and, on Linux under a real
    display, the window segfaulted on the first file. On Windows it mostly got
    away with it, which is the worst kind of bug to have.

    This object is created on the UI thread, so its slots are reached by a
    QUEUED connection and every callback runs where widgets live.
    """

    def __init__(self, on_done, on_fail, on_progress, release):
        super().__init__()
        self._on_done = on_done
        self._on_fail = on_fail
        self._on_progress = on_progress
        self._release = release

    @Slot(object)
    def done(self, result) -> None:
        self._release()
        self._on_done(result)

    @Slot(str, str)
    def fail(self, message: str, tb: str) -> None:
        self._release()
        self._on_fail(message, tb)

    @Slot(str)
    def progress(self, text: str) -> None:
        if self._on_progress is not None:
            self._on_progress(text)


def run_in_thread(
    worker: _BaseWorker,
    on_done: Callable,
    on_fail: Callable,
    on_progress: Callable | None = None,
) -> JobHandle:
    """Start a worker on its own thread and keep it alive until it reports.

    Note the `_ACTIVE` bookkeeping. The thread, the worker AND the relay are
    held until `finished` or `failed` arrives; dropping any of them is what
    produced a silent hang with no traceback. See the module docstring, and
    `_Relay` for why callbacks go through it.
    """
    thread = QThread()
    worker.moveToThread(thread)
    handle = JobHandle(thread=thread, worker=worker, _flag=worker._flag)

    def _release() -> None:
        _ACTIVE.discard(worker)
        _ACTIVE.discard(thread)
        _ACTIVE.discard(relay)
        thread.quit()

    relay = _Relay(on_done, on_fail, on_progress, _release)

    _ACTIVE.add(worker)
    _ACTIVE.add(thread)
    _ACTIVE.add(relay)

    thread.started.connect(worker.run)
    worker.finished.connect(relay.done)
    worker.failed.connect(relay.fail)
    worker.progress.connect(relay.progress)

    thread.start()
    return handle
