"""The window. One file in, one blank out.

WHAT THIS VERSION IS TRYING TO BE
---------------------------------
Open a STEP file and the blank is on the screen. That is the whole design.

v6 through v9 grew the other way. Opening a file showed you the folded part and
what had been recognised; you then pressed Flatten, waited again, and switched
views to see the answer. The reasoning was sound -- check the tool understood
the geometry before committing to an unfold -- and in daily use it put two
clicks and a second wait between a file and a number you already knew you
wanted.

So the order is inverted. Opening does the whole job and lands on the FLAT
solid, because the blank is what the tool is for. The folded part is one button
away, which is the right way round: the as-modelled shape is the reference, not
the goal.

Alongside that, eleven always-visible controls became four. Nothing was removed
-- V-die, press capacity, cost, grain, method and blank placement all still work
and still reach the report. They are behind Advanced, closed, out of the way of
the common case.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..config import UnfoldOptions
from ..diag import APP_NAME, LOGGER, name_only
from ..features import Role
from .panels import (
    ActivityLog,
    AdvancedPanel,
    BlankPanel,
    DetailTable,
    SimpleControls,
)
from .view3d import ROLE_COLOURS
from .view3d_vtk import View3D
from .worker import ExportWorker, OpenWorker, run_in_thread



STALL_SECONDS = 30


class MainWindow(QMainWindow):
    def __init__(self, opts: UnfoldOptions | None = None):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} -- Sheet Metal Tool")
        self.resize(1280, 820)
        # No window-level style: the combined tool applies one light theme
        # to both tabs (partestimator/theme.py). SMTv13's dark sheet is gone.

        self.opts = opts or UnfoldOptions()
        self.part = None
        self.flat2d = None
        self.result = None
        self.scenes = {}          # "flat" / "folded" -> (scene, roles)
        self.stem = "part"
        self.current_path: Path | None = None
        self.source_dir: Path | None = None
        self._job = None
        self._table = None
        self._picking = False
        # Colouring is remembered PER VIEW. The formed part is a reference --
        # you want it to look like the model, plain and quick to read. The flat
        # is where the feature colours earn their keep, because that is where
        # you check a countersink came out on the side you meant.
        self._colour_pref = {"folded": False, "flat": True}
        self._current_view = "flat"

        # BEFORE anything that could touch it. `_set_busy` and `log` both start
        # and stop this timer, and `_build_ui` calls `status`, so creating it
        # after them is an AttributeError on the first line of the constructor
        # that does any real work. Every attribute the constructor's own helpers
        # rely on has to exist before those helpers are called.
        self._stall = QTimer(self)
        self._stall.setInterval(STALL_SECONDS * 1000)
        self._stall.setSingleShot(True)
        self._stall.timeout.connect(self._warn_stalled)

        self._build_ui()
        from partestimator.ui_common import guard_wheel

        guard_wheel(self)
        self._set_busy(False)
        self._enable(False)

    # -- construction ---------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)
        outer.addLayout(self._toolbar())

        split = QSplitter(Qt.Horizontal)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(5)

        self.view3d = View3D()
        self.view3d.picked.connect(self._face_picked)
        lv.addWidget(self.view3d, 1)
        lv.addLayout(self._view_controls())

        self.legend = _Legend()
        lv.addWidget(self.legend)

        # Panels on the LEFT, viewer on the right -- the same arrangement as
        # the machining tab, so both halves of the tool read the same way.
        from PySide6.QtWidgets import QScrollArea

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setMinimumWidth(420)
        right = QWidget()
        right_scroll.setWidget(right)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(6)

        # Inputs first, in one place, then the results -- the same order as
        # the machining tab.
        self.controls = SimpleControls()
        self.controls.changed.connect(self._recompute)
        self.controls.rescan.connect(self._reopen)
        self.controls.pick_sheet.connect(self._arm_picking)
        rv.addWidget(self.controls)

        self.advanced = AdvancedPanel()
        self.advanced.changed.connect(self._recompute)
        rv.addWidget(self.advanced)

        self.blank = BlankPanel()
        rv.addWidget(self.blank)

        self.tabs = QTabWidget()
        self.t_features = DetailTable()
        self.t_bends = DetailTable()
        self.t_notes = DetailTable()
        self.activity = ActivityLog()
        self.tabs.addTab(self.t_features, "Features")
        self.tabs.addTab(self.t_bends, "Bends")
        self.tabs.addTab(self.t_notes, "Notes")
        self.tabs.addTab(self.activity, "Log")
        self.tabs.setMinimumHeight(260)
        rv.addWidget(self.tabs, 1)

        split.addWidget(right_scroll)
        split.addWidget(left)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([440, 1000])
        outer.addWidget(split, 1)

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self.status("Open a STEP file.")
        self._menu()

    def _toolbar(self):
        row = QHBoxLayout()

        self.btn_open = QPushButton("Open STEP...")
        self.btn_open.setObjectName("primary")
        self.btn_open.clicked.connect(self.open_file)
        row.addWidget(self.btn_open)

        row.addSpacing(14)

        # Flat is the default and Form is the alternative, which is the whole
        # point of this version: the blank is the answer, the folded part is
        # the reference.
        self._views = {}
        group = QButtonGroup(self)
        group.setExclusive(True)
        for key, label, tip in (
            ("flat", "Flat", "The unfolded part, with its machined features."),
            ("folded", "Form", "The part as modelled."),
        ):
            b = QToolButton()
            b.setText(label)
            b.setCheckable(True)
            b.setToolTip(tip)
            b.clicked.connect(lambda _c=False, k=key: self.show_view(k))
            group.addButton(b)
            self._views[key] = b
            row.addWidget(b)
        self._views["flat"].setChecked(True)

        row.addStretch(1)

        self.btn_export = QPushButton("Export...")
        self.btn_export.clicked.connect(self.export)
        row.addWidget(self.btn_export)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self._cancel)
        row.addWidget(self.btn_cancel)
        return row

    def _view_controls(self):
        row = QHBoxLayout()
        for label, name in (
            ("Iso", "iso"), ("Top", "top"), ("Front", "front"), ("Right", "right"),
        ):
            b = QToolButton()
            b.setText(label)
            b.clicked.connect(lambda _c=False, n=name: self.view3d.set_view(n))
            row.addWidget(b)

        fit = QToolButton()
        fit.setText("Fit")
        fit.clicked.connect(self.view3d.fit)
        row.addWidget(fit)

        row.addSpacing(12)
        self.chk_colour = QCheckBox("Colour features")
        self.chk_colour.setChecked(True)
        self.chk_colour.setToolTip(
            "Colour every face by what the tool decided it is.\n\n"
            "Remembered separately for each view: off for the formed part, "
            "which reads better plain, and on for the flat, where it is how "
            "you check a countersink came out on the side you meant."
        )
        self.chk_colour.toggled.connect(self._colour_toggled)
        row.addWidget(self.chk_colour)

        self.chk_box = QCheckBox("Bounding box")
        self.chk_box.setChecked(True)
        self.chk_box.toggled.connect(self.view3d.set_show_box)
        row.addWidget(self.chk_box)

        row.addStretch(1)
        self.hint = QLabel("left-drag rotate   middle-drag pan   wheel zoom   click to pick")
        self.hint.setStyleSheet("color:#555; font-size:11px;")
        row.addWidget(self.hint)
        return row

    def _menu(self) -> None:
        m = self.menuBar().addMenu("&File")
        for text, shortcut, slot in (
            ("&Open STEP...", QKeySequence.Open, self.open_file),
            ("&Export...", "Ctrl+E", self.export),
        ):
            a = QAction(text, self)
            a.setShortcut(shortcut)
            a.triggered.connect(slot)
            m.addAction(a)
        m.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        m.addAction(quit_action)

        h = self.menuBar().addMenu("&Help")
        about = QAction("&About", self)
        about.triggered.connect(self._about)
        h.addAction(about)

    # -- state ----------------------------------------------------------

    def _enable(self, on: bool) -> None:
        self.btn_export.setEnabled(on)
        for b in self._views.values():
            b.setEnabled(on)

    def _set_busy(self, busy: bool) -> None:
        self.btn_open.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)
        self.controls.setEnabled(not busy)
        self.advanced.setEnabled(not busy)
        if busy:
            self._stall.start()
        else:
            self._stall.stop()

    def status(self, text: str) -> None:
        self.statusBar().showMessage(text)

    def log(self, text: str) -> None:
        self.activity.append(text)
        self.status(text)
        self._stall.start()

    def _warn_stalled(self) -> None:
        self.status(
            f"Still working -- the last step has been running over "
            f"{STALL_SECONDS}s. See the Log tab."
        )

    # -- opening --------------------------------------------------------

    def open_file(self) -> None:
        start = str(self.source_dir) if self.source_dir else ""
        path, _f = QFileDialog.getOpenFileName(
            self, "Open STEP file", start,
            "STEP files (*.step *.stp *.STEP *.STP);;All files (*)",
        )
        if not path:
            return
        # A new file means the old face index means nothing: indices are per
        # shape, so keeping it would apply one part's answer to another.
        self.opts.sheet_face = None
        self.view3d.set_highlight(None)
        self.load(Path(path))

    def load(self, path: Path) -> None:
        self.activity.clear()
        self.blank.clear()
        self.log(f"Opening {name_only(path)}")
        self.source_dir = path.parent
        self.current_path = path
        self.stem = path.stem
        self.scenes = {}
        self.view3d.clear()
        self._enable(False)
        self._set_busy(True)

        self._apply_options()
        worker = OpenWorker(path, self.opts, self._table, {"stop": False})
        self._job = run_in_thread(worker, self._opened, self._failed, self.log)

    def _reopen(self) -> None:
        """A control changed that detection itself depends on."""
        if self.current_path is not None and not self.btn_cancel.isEnabled():
            self.load(self.current_path)

    def _apply_options(self):
        self.controls.apply_to(self.opts)
        self.advanced.apply_to(self.opts)
        return self.opts

    def _opened(self, res) -> None:
        self._set_busy(False)
        self.part = res.part
        self.flat2d = res.flat2d
        self.result = res.result
        self.stem = res.stem

        if res.flat_scene is not None:
            self.scenes["flat"] = (res.flat_scene, res.flat_roles)
        if res.folded_scene is not None:
            self.scenes["folded"] = (res.folded_scene, res.folded_roles)

        from ..units import choose_display_unit

        unit = choose_display_unit(self.opts.display_units, res.file_units)
        self.controls.set_unit(unit)
        self.controls.show_detected_thickness(self.part.thickness)

        self._enable(True)

        # THE VIEW FIRST, AND SEPARATELY.
        #
        # v11 refreshed the panels and then showed the model. A single
        # exception anywhere in the results panels -- and there was one, a
        # string compared to an integer in the bend table -- aborted the rest
        # of this method, so the viewport stayed empty until you clicked a view
        # button and went down a different code path. The symptom looked like a
        # rendering problem and was nothing of the kind.
        #
        # Drawing the part is the one thing that must not depend on anything
        # else working.
        self.show_view("flat" if "flat" in self.scenes else "folded")

        try:
            self._refresh()
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("results panels failed to refresh")
            self.log(f"WARNING: the results panels failed to update: {exc}")

        if res.is_blank:
            self.log("Flat blank -- no bends. Measured directly.")
        else:
            self.log(f"{len(self.part.bends)} bend(s).")
        if res.extruded:
            self.log(
                "NOTE: the flat is an extruded blank, so countersinks and "
                "chamfers are not on it. Dimensions are unaffected -- see Notes."
            )

    # -- views ----------------------------------------------------------

    def _colour_toggled(self, on: bool) -> None:
        self._colour_pref[self._current_view] = bool(on)
        self.view3d.set_colour_by_feature(on)

    def show_view(self, key: str) -> None:
        if key not in self.scenes:
            key = next(iter(self.scenes), None)
            if key is None:
                return
        if key in self._views:
            self._views[key].setChecked(True)

        self._current_view = key
        want_colour = self._colour_pref.get(key, True)
        self.chk_colour.blockSignals(True)
        self.chk_colour.setChecked(want_colour)
        self.chk_colour.blockSignals(False)
        self.view3d.set_colour_by_feature(want_colour)

        scene, roles = self.scenes[key]
        # The camera is kept across the swap on purpose: watching the part fold
        # and unfold in place, from the angle you already chose, is most of the
        # value of having both.
        self.view3d.set_scene(scene, roles, keep_camera=True)
        self.legend.show_roles(set(roles.values()) if want_colour else set())
        if key == "flat":
            self._apply_box(scene)
        else:
            self.view3d.set_box(None)

    def _apply_box(self, scene) -> None:
        if self.part is None:
            return
        from ..report import resolve_unit

        lo, hi = scene.bounds()
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        size = hi - lo
        unit = resolve_unit(self.part, self.opts)
        labels = [unit.length(float(v)) for v in size]

        # Measured, not assumed. If the box is not snug on the blank, something
        # upstream is wrong and it should be visible rather than hidden by a
        # box drawn to the number it is supposed to be checking.
        measured, expected = float(size[2]), float(self.part.thickness)
        if expected > 0 and abs(measured - expected) > max(0.02 * expected, 1e-4):
            labels[2] += f"  (detected {unit.length(expected, False)})"
        self.view3d.set_box(lo, hi, tuple(labels))

    # -- results --------------------------------------------------------

    def _recompute(self) -> None:
        """The fast path: K, material and the advanced settings.

        Detection does not depend on any of them, so the cached boundary loops
        are simply re-transformed. That is what makes the K slider live.
        """
        if self.part is None or self.flat2d is None:
            return
        from ..model2d import recompute

        self._apply_options()
        try:
            self.flat2d, self.result = recompute(self.part, self.opts, self._table)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("recompute failed")
            self.status(f"Recompute failed: {exc}")
            return

        # The kernel products are now stale: K changed, so the bend zones are a
        # different width and the solid would be wrong.
        self.part.flat = None
        self.part.flat_solid = None
        try:
            self._refresh()
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("results panels failed to refresh")
            self.status(f"The results panels failed to update: {exc}")

    def _refresh(self) -> None:
        from ..materials import get_material
        from ..report import bend_table, manufacturability_warnings, resolve_unit

        part, flat = self.part, self.flat2d
        unit = resolve_unit(part, self.opts)
        self.controls.set_unit(unit)

        area = flat.polygon_area()
        weight = area * part.thickness * get_material(self.opts.material).density

        gauge = ""
        if part.nominal_thickness:
            gauge = f"  (stock {unit.length(part.nominal_thickness, False)})"
        self.blank.set_headline(
            f"{unit.length(flat.blank_length, False)} x "
            f"{unit.length(flat.blank_width)}",
            f"t {unit.length(part.thickness)}{gauge}",
        )
        self.blank.set("area", unit.area(area))
        self.blank.set("weight", f"{weight:.3f} kg")
        self.blank.set("bends", str(len(part.bends)) or "0")
        self.blank.set("features", str(len(getattr(part, "feature_groups", []))))
        self.blank.set(
            "cut",
            unit.length(part.flat.perimeter, False)
            if part.flat is not None else "on export",
        )
        self.blank.set(
            "cost",
            f"{weight * self.opts.cost_per_kg:,.2f}"
            if self.opts.cost_per_kg else "--",
        )

        # The box drawn round the flat carries its size as text, so it follows
        # the Units setting along with everything else on screen.
        if self._current_view == "flat" and "flat" in self.scenes:
            self._apply_box(self.scenes["flat"][0])

        rows = bend_table(part, self.opts)
        self.t_bends.show_bends(rows, unit)
        self.t_features.show_features(part, unit)

        warnings = part.warnings + manufacturability_warnings(part, self.opts)
        self.t_notes.show_notes(warnings, part.notes)
        self.tabs.setTabText(2, f"Notes ({len(warnings)})" if warnings else "Notes")

        if flat.missing_outlines:
            self.log(
                f"WARNING: {len(flat.missing_outlines)} flange(s) have no "
                f"outline and are missing from the flat pattern."
            )

    # -- sheet face picking ---------------------------------------------

    def _arm_picking(self) -> None:
        if self.part is None:
            return
        self._picking = True
        self.show_view("folded")
        self.status("Click the face that is the sheet surface.")

    def _face_picked(self, index: int) -> None:
        if not self._picking or self.current_path is None:
            return
        self._picking = False
        # Colouring is remembered PER VIEW. The formed part is a reference --
        # you want it to look like the model, plain and quick to read. The flat
        # is where the feature colours earn their keep, because that is where
        # you check a countersink came out on the side you meant.
        self._colour_pref = {"folded": False, "flat": True}
        self._current_view = "flat"
        self.view3d.set_highlight(index)
        self.opts.sheet_face = index
        self.log(f"Face {index} set as the sheet surface; re-reading the file.")
        # Re-read rather than patch: thickness is upstream of which cylinders
        # are bends and which planes are sheet, so a patched number would leave
        # a part whose bends were still chosen against the old one.
        self.load(self.current_path)

    # -- export ---------------------------------------------------------

    def export(self) -> None:
        if self.part is None:
            return
        start = str(self.source_dir) if self.source_dir else ""
        out = QFileDialog.getExistingDirectory(self, "Export into folder", start)
        if not out:
            return
        self._apply_options()
        self._set_busy(True)
        self.log("Exporting...")
        worker = ExportWorker(
            self.part, self.opts, Path(out), self.stem,
            {"dxf": True, "step": True, "json": True}, {"stop": False},
        )
        self._job = run_in_thread(worker, self._exported, self._failed, self.log)

    def _exported(self, written: dict) -> None:
        self._set_busy(False)
        names = ", ".join(sorted(name_only(p) for p in written.values()))
        self.log(f"Wrote {len(written)} file(s): {names}")

    # -- failure --------------------------------------------------------

    def _cancel(self) -> None:
        if self._job is not None:
            self._job.cancel()
            self.status("Cancelling at the next stage boundary...")

    def _failed(self, message: str, tb: str) -> None:
        self._set_busy(False)
        self.log(f"FAILED: {message}")
        if tb:
            self.activity.append(tb)
        if message == "Cancelled.":
            self.status("Cancelled.")
            return
        QMessageBox.critical(
            self, "Something went wrong",
            f"{message}\n\nThe full traceback is in the Log tab and in part_estimator.log "
            f"next to the application.",
        )

    def _about(self) -> None:
        QMessageBox.information(
            self, f"About {APP_NAME}",
            f"{APP_NAME} -- sheet metal flat patterns from STEP files.\n\n"
            "Runs entirely on this machine. It makes no network connections "
            "and writes nothing outside its own folder.\n\n"
            "Deleting the folder removes it completely.",
        )

    def closeEvent(self, event):  # noqa: N802
        from ..diag import clean_scratch

        if self._job is not None:
            self._job.cancel()
        try:
            self.view3d.close()
        except Exception:  # noqa: BLE001 -- VTK teardown must not block exit
            pass
        clean_scratch()
        super().closeEvent(event)


class _Legend(QWidget):
    """What the colours mean. The purple swatch is the one that matters."""

    LABELS = {
        Role.SHEET: "sheet", Role.BEND_INNER: "bend", Role.HOLE: "hole",
        Role.SLOT_END: "slot", Role.COUNTERSINK: "countersink",
        Role.COUNTERBORE: "counterbore", Role.CHAMFER: "chamfer",
        Role.CORNER_RADIUS: "corner radius", Role.FILLET: "fillet",
        Role.CUT_EDGE: "cut edge", Role.FORMED: "FORMED -- no exact flat",
        Role.UNKNOWN: "NOT RECOGNISED",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lay = QHBoxLayout(self)
        self._lay.setContentsMargins(2, 0, 2, 0)
        self._lay.setSpacing(9)
        self._lay.addStretch(1)

    def show_roles(self, roles: set) -> None:
        while self._lay.count():
            item = self._lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        # The two bend faces share a swatch: they are one bend, and listing
        # both would suggest they are different things.
        shown = {Role.BEND_INNER if r is Role.BEND_OUTER else r for r in roles}
        for role in [r for r in self.LABELS if r in shown]:
            colour = ROLE_COLOURS.get(role)
            swatch = QLabel("  ")
            swatch.setFixedSize(11, 11)
            swatch.setStyleSheet(
                f"background: {colour.name()}; border: 1px solid #555;"
            )
            self._lay.addWidget(swatch)
            text = QLabel(self.LABELS[role])
            loud = role in (Role.UNKNOWN, Role.FORMED)
            text.setStyleSheet(
                "color:#b45309; font-weight:600;" if loud else "color:#555;"
            )
            self._lay.addWidget(text)
        self._lay.addStretch(1)


def run(opts: UnfoldOptions | None = None, path: Path | None = None) -> int:
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    app.setApplicationName(APP_NAME)
    window = MainWindow(opts)
    window.show()
    if path is not None:
        QTimer.singleShot(80, lambda: window.load(Path(path)))
    return app.exec()
