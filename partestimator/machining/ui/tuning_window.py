"""The "Shop tuning" window: the library of measured parts, and the fit.

    Parts   one row per part: its inputs, the measured time, today's estimate
            and how far off it is. Every input, the stock included, can be
            edited in place.
    Tune    pick which settings to fit, fit, look at the result, save it.

The work is in machining/tuning.py; this is only the window. Capturing new
files and fitting both run off the GUI thread (analysis in child processes),
so the window stays usable. The STEP files are not kept: only what the
estimate reads from them.
"""
from __future__ import annotations

import math
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QBrush, QColor, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from partestimator.machining import tuning
from partestimator.machining.geometry.materials import MATERIALS

# Parts table columns.
(C_USE, C_PART, C_MAT, C_PROC, C_FINISH, C_OPS, C_TAPPED, C_STOCK, C_MEASURED,
 C_EST, C_ERR, C_STATUS) = range(12)
PART_HEADERS = ["Use", "Part", "Material", "Process", "Finish µin", "Ops", "Tapped",
                "Stock", "Measured min", "Estimate min", "Error", "Status"]
PART_TIPS = {
    C_USE: "Ticked parts are used in the fit.",
    C_FINISH: "Target surface finish, micro-inches Ra.",
    C_OPS: "Machine operations (setups) per part.",
    C_TAPPED: "Tapped holes. Blank = detect from tap-drill sizes.",
    C_STOCK: "Double-click to edit, e.g.\n"
             "  plate 1 x 4 x 6        (T x W x L)\n"
             "  round 2 x 6            (D x L)\n"
             "  extrusion AND10133-2004 @ 6.5   (profile @ length)\n"
             "Inches; fractions like 1-1/4 are fine.",
    C_MEASURED: "MEASURED door-to-door minutes per part: every operation, load/unload\n"
                "and deburr at the machine. Not the batch setup.",
    C_EST: "Today's estimate, with the tuning in use.",
    C_ERR: "Estimate against measured. + = the estimate is high.",
}

RED, AMBER, GREEN, GREY = "#b91c1c", "#b45309", "#15803d", "#6b7280"


def _item(text="", editable=False, align_right=False, tip=""):
    it = QTableWidgetItem(text)
    flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
    if editable:
        flags |= Qt.ItemIsEditable
    it.setFlags(flags)
    if align_right:
        it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
    if tip:
        it.setToolTip(tip)
    return it


def _err_colour(ratio: float) -> str:
    e = abs(ratio - 1.0)
    return GREEN if e <= 0.15 else AMBER if e <= 0.35 else RED


class _Worker(QObject):
    """Runs one function on a QThread and reports back."""
    progress = Signal(str)
    count = Signal(int, int)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn
        self.cancel = False

    def run(self):
        try:
            self.finished.emit(self._fn(self))
        except Exception as exc:  # noqa: BLE001 -- reported in the window
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class TuningWindow(QDialog):
    #: Show this entry in the Machining tab: (entry, library).
    openRequested = Signal(object, object)
    #: A fit was saved or cleared: reload the shop profile.
    tuningChanged = Signal()

    def __init__(self, table, defaults=None, folder: Path | None = None, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Shop tuning")
        self.resize(1250, 720)
        self.table = table                  # RateTable
        self._defaults = defaults or (lambda: {})
        self.library = tuning.Library.load(folder)
        self.profile, self.state = tuning.load_tuned_profile(self.library.folder)
        self._fit: tuning.Fit | None = None
        self._thread: QThread | None = None
        self._worker: _Worker | None = None
        self._filling = False
        # Read from disk once each: id(entry) -> snapshot / (inputs, Prepared).
        self._snapshots: dict[int, dict | None] = {}
        self._prepared: dict[int, tuple] = {}

        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.lbl_folder = QLabel()
        self.lbl_folder.setTextFormat(Qt.RichText)
        self.lbl_folder.setOpenExternalLinks(False)
        self.lbl_folder.linkActivated.connect(lambda _l: self.open_folder())
        top.addWidget(self.lbl_folder, 1)
        self.lbl_state = QLabel()
        self.lbl_state.setStyleSheet(f"color:{GREY};")
        top.addWidget(self.lbl_state)
        lay.addLayout(top)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_parts(), "Parts")
        self.tabs.addTab(self._build_tune(), "Tune")
        lay.addWidget(self.tabs, 1)

        row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setMaximumWidth(260)
        self.lbl_busy = QLabel()
        self.lbl_busy.setStyleSheet(f"color:{GREY};")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setVisible(False)
        self.btn_cancel.clicked.connect(self._cancel)
        row.addWidget(self.progress)
        row.addWidget(self.lbl_busy, 1)
        row.addWidget(self.btn_cancel)
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        row.addWidget(self.btn_close)
        lay.addLayout(row)

        self.refresh()

    # ------------------------------------------------------------ building
    def _build_parts(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.parts = QTableWidget(0, len(PART_HEADERS))
        self.parts.setHorizontalHeaderLabels(PART_HEADERS)
        for col, tip in PART_TIPS.items():
            self.parts.horizontalHeaderItem(col).setToolTip(tip)
        self.parts.verticalHeader().setVisible(False)
        self.parts.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.parts.setEditTriggers(QAbstractItemView.DoubleClicked
                                   | QAbstractItemView.EditKeyPressed
                                   | QAbstractItemView.AnyKeyPressed)
        self.parts.horizontalHeader().setSectionResizeMode(C_STATUS, QHeaderView.Stretch)
        self.parts.itemChanged.connect(self._on_item_changed)
        self.parts.itemSelectionChanged.connect(self._update_buttons)
        lay.addWidget(self.parts, 1)

        self.lbl_parts = QLabel()
        self.lbl_parts.setWordWrap(True)
        lay.addWidget(self.lbl_parts)

        row = QHBoxLayout()
        self.btn_add = QPushButton("Add STEP files…")
        self.btn_add.setToolTip("Analyse STEP files and keep what the estimate needs from\n"
                                "them (the files themselves are not copied). They start with\n"
                                "the material, finish and operations now set in the Machining\n"
                                "tab and the stock it would choose; fill in the measured time\n"
                                "and anything that differs.")
        self.btn_add.clicked.connect(self.add_files)
        self.btn_remove = QPushButton("Remove")
        self.btn_remove.clicked.connect(self.remove_selected)
        self.btn_open = QPushButton("Open in Machining tab")
        self.btn_open.setToolTip("Show the part with its saved inputs, from the STEP file\n"
                                 "it was added from (while that file is still there).")
        self.btn_open.clicked.connect(self.open_selected)
        self.btn_reload = QPushButton("Reload")
        self.btn_reload.setToolTip(f"Read {tuning.LIBRARY_FILE} again after editing it in Excel.")
        self.btn_reload.clicked.connect(self.reload)
        for b in (self.btn_add, self.btn_remove, self.btn_open):
            row.addWidget(b)
        row.addStretch(1)
        row.addWidget(self.btn_reload)
        lay.addLayout(row)
        return w

    def _build_tune(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        intro = QLabel("Ticked settings are fitted so the estimates match the measured "
                       "times. Each stays near its shipped value unless your parts give "
                       "a reason to move it.")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        self.knobs = QTableWidget(len(tuning.KNOBS), 6)
        self.knobs.setHorizontalHeaderLabels(["Setting", "Shipped", "In use", "Fitted",
                                              "±", "Note"])
        self.knobs.verticalHeader().setVisible(False)
        self.knobs.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.knobs.setSelectionMode(QAbstractItemView.NoSelection)
        self.knobs.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.knobs.setWordWrap(False)
        self.knobs.horizontalHeaderItem(4).setToolTip(
            "How tightly your parts pin the fitted value (one sigma).")
        for r, k in enumerate(tuning.KNOBS):
            it = QTableWidgetItem(k.label)
            it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if k.default else Qt.Unchecked)
            it.setToolTip(f"{k.tip}\n\nshop_profile key: {k.key}")
            self.knobs.setItem(r, 0, it)
            for c in range(1, 6):
                self.knobs.setItem(r, c, _item(align_right=c < 5))
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(self.knobs, 1)

        row = QHBoxLayout()
        self.btn_fit = QPushButton("Fit")
        self.btn_fit.setObjectName("primary")
        self.btn_fit.clicked.connect(self.run_fit)
        self.btn_save = QPushButton("Save tuning")
        self.btn_save.setToolTip(f"Write the fitted values to {tuning.TUNING_FILE}; every\n"
                                 "estimate uses them from now on.")
        self.btn_save.clicked.connect(self.save_fit)
        self.btn_clear = QPushButton("Back to shipped values")
        self.btn_clear.setToolTip("Stop using the fitted values. Hand-set rows in\n"
                                  f"{tuning.TUNING_FILE} are kept.")
        self.btn_clear.clicked.connect(self.clear_fit)
        row.addWidget(self.btn_fit)
        row.addWidget(self.btn_save)
        row.addStretch(1)
        row.addWidget(self.btn_clear)
        ll.addLayout(row)

        self.lbl_summary = QLabel()
        self.lbl_summary.setTextFormat(Qt.RichText)
        self.lbl_summary.setWordWrap(True)
        ll.addWidget(self.lbl_summary)

        self.fit_parts = QTableWidget(0, 6)
        self.fit_parts.setHorizontalHeaderLabels(["Part", "Measured", "Shipped est.",
                                                  "Tuned est.", "Tuned error",
                                                  "Unseen error"])
        self.fit_parts.horizontalHeaderItem(5).setToolTip(
            "The part estimated by a fit that did not include it: what to expect on a\n"
            "new part.")
        self.fit_parts.verticalHeader().setVisible(False)
        self.fit_parts.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.fit_parts.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(self.fit_parts)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        lay.addWidget(split, 1)
        return w

    # ------------------------------------------------------------- state
    def refresh(self) -> None:
        folder = self.library.folder
        self.lbl_folder.setText(f"Library: <a href='#'>{folder}</a>")
        self.lbl_folder.setToolTip("Open the folder. parts.csv can be edited in Excel; "
                                   "press Reload afterwards.")
        self.lbl_state.setText(self.state.describe() or "Using the shipped values.")
        self._fill_parts()
        self._fill_knobs()
        self._update_buttons()

    def reload(self) -> None:
        self.library = tuning.Library.load(self.library.folder)
        self.profile, self.state = tuning.load_tuned_profile(self.library.folder)
        self._forget()
        self.refresh()

    def _forget(self) -> None:
        self._snapshots.clear()
        self._prepared.clear()

    def _snapshot(self, e):
        if id(e) not in self._snapshots:
            self._snapshots[id(e)] = self.library.load_snapshot(e)
        return self._snapshots[id(e)]

    def _prepare(self, e):
        """The entry's Prepared, redone only when an input it bakes in
        (material, process, stock) changes."""
        key = (e.material, e.process, e.stock_type, e.stock_size, e.stock_profile)
        hit = self._prepared.get(id(e))
        if hit is None or hit[0] != key:
            snap = self._snapshot(e)
            if snap is None:
                raise ValueError("captured geometry missing")
            hit = (key, tuning.prepare(e, snap, self.profile))
            self._prepared[id(e)] = hit
        prep = hit[1]
        prep.finish_uin, prep.operations = float(e.finish_uin), int(e.operations)
        prep.tapped, prep.measured = e.tapped_holes, e.measured_min
        return prep

    def _status(self, e) -> tuple[str, str]:
        """(text, colour)."""
        if self._snapshot(e) is None:
            return f"captured geometry missing ({tuning.CAPTURED_DIR}\\{e.id}.json)", RED
        issues = tuning.problems(e, self.table)
        if issues:
            return ", ".join(issues), AMBER
        return ("ready" if e.include else "left out"), (GREEN if e.include else GREY)

    def _fill_parts(self) -> None:
        self._filling = True
        for r in range(self.parts.rowCount()):
            for c in (C_MAT, C_PROC):
                old = self.parts.cellWidget(r, c)
                if old is not None:
                    self.parts.removeCellWidget(r, c)
                    old.hide()
        entries = self.library.entries
        self.parts.setRowCount(len(entries))
        materials = [m for m in MATERIALS]
        for r, e in enumerate(entries):
            use = QTableWidgetItem()
            use.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
            use.setCheckState(Qt.Checked if e.include else Qt.Unchecked)
            self.parts.setItem(r, C_USE, use)
            self.parts.setItem(r, C_PART, _item(e.part, True, tip="\n".join(
                t for t in (e.notes, e.source_file and f"from {e.source_file}") if t)))

            mat = QComboBox()
            mat.addItems(materials if e.material in materials or not e.material
                         else materials + [e.material])
            mat.setCurrentText(e.material)
            if not e.material:
                mat.setCurrentIndex(-1)
            mat.currentTextChanged.connect(lambda t, row=r: self._set(row, "material", t))
            self.parts.setCellWidget(r, C_MAT, mat)

            proc = QComboBox()
            proc.addItems(list(tuning.PROCESSES))
            proc.setCurrentText(e.process)
            proc.currentTextChanged.connect(lambda t, row=r: self._set(row, "process", t))
            self.parts.setCellWidget(r, C_PROC, proc)

            self.parts.setItem(r, C_FINISH, _item(tuning._fmt(e.finish_uin), True, True))
            self.parts.setItem(r, C_OPS, _item(str(e.operations), True, True))
            self.parts.setItem(r, C_TAPPED, _item("" if e.tapped_holes is None
                                                  else str(e.tapped_holes), True, True))
            self.parts.setItem(r, C_STOCK, _item(tuning.stock_text(e), True,
                                                 tip=PART_TIPS[C_STOCK]))
            self.parts.setItem(r, C_MEASURED, _item(
                "" if e.measured_min is None else tuning._fmt(e.measured_min), True, True))
            for c in (C_EST, C_ERR):
                self.parts.setItem(r, c, _item(align_right=True))
            self.parts.setItem(r, C_STATUS, _item())
            self._fill_row_estimate(r)
        self._filling = False
        self.parts.resizeColumnsToContents()
        self.parts.setColumnWidth(C_MAT, max(self.parts.columnWidth(C_MAT), 170))
        for col, width in ((C_EST, 90), (C_ERR, 60), (C_STOCK, 150)):
            self.parts.setColumnWidth(col, max(self.parts.columnWidth(col), width))
        self._update_parts_summary()

    def _fill_row_estimate(self, r: int) -> None:
        e = self.library.entries[r]
        text, colour = self._status(e)
        est_text, err_text, err_colour = "", "", GREY
        if self._snapshot(e) is not None and e.material:
            try:
                prep = self._prepare(e)
                est = tuning.estimate(prep, self.table, self.profile).door_to_door_minutes
                if est > 0:
                    est_text = f"{est:,.1f}"
                    if e.measured_min:
                        ratio = est / e.measured_min
                        err_text, err_colour = f"{ratio - 1:+.0%}", _err_colour(ratio)
            except Exception as exc:  # noqa: BLE001 -- shown in the row
                text, colour = f"{type(exc).__name__}: {exc}", RED
        was = self._filling
        self._filling = True
        self.parts.item(r, C_EST).setText(est_text)
        err = self.parts.item(r, C_ERR)
        err.setText(err_text)
        err.setForeground(QBrush(QColor(err_colour)))
        status = self.parts.item(r, C_STATUS)
        status.setText(text)
        status.setForeground(QBrush(QColor(colour)))
        self._filling = was

    def _update_parts_summary(self) -> None:
        entries = self.library.entries
        ready = [e for e in entries if e.include and self._status(e)[0] == "ready"]
        ratios = []
        for r, e in enumerate(entries):
            if e in ready:
                t = self.parts.item(r, C_EST).text().replace(",", "")
                if t and e.measured_min:
                    ratios.append(float(t) / e.measured_min)
        bits = [f"{len(entries)} part{'s' if len(entries) != 1 else ''}, "
                f"{len(ready)} ready to fit"]
        if ratios:
            bits.append("today's estimates: " + tuning.Stats.of(ratios).text())
        errors = self.library.errors + self.state.errors
        text = ". ".join(bits) + "."
        if errors:
            text += "\n⚠ " + "\n⚠ ".join(errors[:5])
        self.lbl_parts.setText(text)

    def _fill_knobs(self) -> None:
        base = tuning.fit_baseline(self.library.folder)
        fitted = self._fit.values() if self._fit else {}
        results = {k.key: k for k in self._fit.knobs} if self._fit else {}
        for r, k in enumerate(tuning.KNOBS):
            shipped = base.get(k.key, 0.0)
            in_use = self.profile.get(k.key, 0.0)
            self.knobs.item(r, 1).setText(f"{shipped:.4g}")
            self.knobs.item(r, 2).setText(f"{in_use:.4g}")
            fit_item = self.knobs.item(r, 3)
            fit_item.setText(f"{fitted[k.key]:.4g}" if k.key in fitted else "")
            res = results.get(k.key)
            self.knobs.item(r, 4).setText(f"{(res.spread - 1) * 100:.0f}%" if res else "")
            note = ""
            if shipped <= 0:
                note = "off in the shop profile (0), so it cannot be fitted"
                self.knobs.item(r, 0).setFlags(Qt.ItemIsUserCheckable)
            elif res is not None and not res.pinned:
                note = ("not used by these parts" if res.used <= 0.01
                        else "loosely set: more parts would settle it")
            elif k.key in self.state.manual:
                note = f"set by hand in {tuning.TUNING_FILE}"
            self.knobs.item(r, 5).setText(note)
            colour = GREY if note and res is not None else "#111827"
            fit_item.setForeground(QBrush(QColor(colour)))
        self.knobs.resizeColumnsToContents()

    def _update_buttons(self) -> None:
        busy = self._thread is not None
        rows = self._selected_rows()
        self.btn_add.setEnabled(not busy)
        self.btn_remove.setEnabled(not busy and bool(rows))
        self.btn_open.setEnabled(not busy and len(rows) == 1)
        self.btn_reload.setEnabled(not busy)
        self.btn_fit.setEnabled(not busy)
        self.btn_save.setEnabled(not busy and self._fit is not None)
        self.btn_clear.setEnabled(not busy and bool(self.state.fitted))

    def _selected_rows(self) -> list[int]:
        return sorted({i.row() for i in self.parts.selectedIndexes()})

    # ------------------------------------------------------------- editing
    def _save_library(self) -> bool:
        try:
            self.library.save()
            return True
        except OSError as exc:
            QMessageBox.warning(self, "Shop tuning",
                                f"Could not save {tuning.LIBRARY_FILE}:\n{exc}\n\n"
                                "Is it open in Excel? Close it there and try again.")
            return False

    def _set(self, row: int, field: str, value) -> None:
        if self._filling or row >= len(self.library.entries):
            return
        setattr(self.library.entries[row], field, value)
        self._save_library()
        self._fill_row_estimate(row)
        self._update_parts_summary()

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._filling:
            return
        r, c = item.row(), item.column()
        e = self.library.entries[r]
        text = item.text().strip()
        try:
            if c == C_USE:
                e.include = item.checkState() == Qt.Checked
            elif c == C_PART:
                e.part = text or e.part
            elif c == C_FINISH:
                e.finish_uin = float(text)
            elif c == C_OPS:
                e.operations = max(int(float(text)), 1)
            elif c == C_TAPPED:
                e.tapped_holes = None if text.lower() in ("", "auto") else max(int(float(text)), 0)
            elif c == C_MEASURED:
                e.measured_min = None if not text else float(text)
                if e.measured_min is not None and e.measured_min <= 0:
                    e.measured_min = None
            elif c == C_STOCK:
                e.stock_type, e.stock_size, e.stock_profile = tuning.parse_stock_text(text)
            else:
                return
        except ValueError as exc:
            # Put the stored value back below, and say why.
            self.lbl_busy.setText(f"⚠ {exc}" if c == C_STOCK else "")
        self._filling = True
        if c == C_FINISH:
            item.setText(tuning._fmt(e.finish_uin))
        elif c == C_OPS:
            item.setText(str(e.operations))
        elif c == C_TAPPED:
            item.setText("" if e.tapped_holes is None else str(e.tapped_holes))
        elif c == C_MEASURED:
            item.setText("" if e.measured_min is None else tuning._fmt(e.measured_min))
        elif c == C_STOCK:
            item.setText(tuning.stock_text(e))
        self._filling = False
        self._save_library()
        self._fill_row_estimate(r)
        self._update_parts_summary()

    def add_files(self) -> None:
        paths, _f = QFileDialog.getOpenFileNames(
            self, "Add parts to the tuning library", "",
            "STEP files (*.step *.stp *.STEP *.STP)")
        if paths:
            self.add_paths([Path(p) for p in paths])

    def add_paths(self, paths) -> None:
        """Analyse and capture these STEP files in the background."""
        if self._thread is not None:
            return
        defaults = {k: v for k, v in self._defaults().items()
                    if k in ("material", "finish_uin", "operations")}
        lib, profile, paths = self.library, self.profile, [Path(p) for p in paths]

        def work(w: _Worker):
            def progress(done, total, message):
                w.count.emit(done, total)
                w.progress.emit(f"Capturing {done} / {total}: {message}")
            return tuning.add_files(lib, paths, profile=profile, progress=progress,
                                    cancelled=lambda: w.cancel, **defaults)

        self._start(work, self._on_added,
                    f"Capturing {len(paths)} part{'s' if len(paths) != 1 else ''}…")

    def _on_added(self, outcome) -> None:
        added, already, errors = outcome
        self._save_library()
        self._forget()
        self._fill_parts()
        msg = f"Added {len(added)} part{'s' if len(added) != 1 else ''}."
        if already:
            msg += f" Already in the library: {', '.join(already[:5])}" + \
                   ("…" if len(already) > 5 else "") + "."
        if errors:
            msg += f" {len(errors)} could not be read: " + "; ".join(
                f"{Path(p).name} ({err})" for p, err in list(errors.items())[:3])
        self.lbl_busy.setText(msg)

    def remove_selected(self) -> None:
        rows = self._selected_rows()
        if not rows:
            return
        names = ", ".join(self.library.entries[r].part for r in rows[:5])
        if QMessageBox.question(
                self, "Remove parts",
                f"Remove {len(rows)} part{'s' if len(rows) != 1 else ''} from the library?\n"
                f"{names}{'…' if len(rows) > 5 else ''}\n\n"
                "Their measured times and captured geometry are deleted.") != QMessageBox.Yes:
            return
        for r in reversed(rows):
            self.library.remove(self.library.entries[r])
        self._forget()
        self._save_library()
        self.refresh()

    def open_selected(self) -> None:
        rows = self._selected_rows()
        if len(rows) != 1:
            return
        e = self.library.entries[rows[0]]
        if not e.source_file or not Path(e.source_file).is_file():
            QMessageBox.information(
                self, "Shop tuning",
                f"The STEP file this part was added from is no longer there:\n"
                f"{e.source_file or '(not recorded)'}\n\n"
                "Its captured geometry is all the library needs, so it still fits; "
                "change its inputs here in the table.")
            return
        self.openRequested.emit(e, self.library)

    def open_folder(self) -> None:
        try:
            self.library.ensure()
        except OSError:
            pass
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.library.folder)))

    def entry_saved(self) -> None:
        """The Machining tab changed the library on disk."""
        if self._thread is None:
            self.reload()

    # ------------------------------------------------------------ background
    def _start(self, fn, on_done, label: str, count: bool = False) -> None:
        self._thread = QThread(self)
        self._worker = _Worker(fn)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.lbl_busy.setText)
        self._worker.count.connect(self._on_count)
        self._worker.finished.connect(on_done)
        self._worker.failed.connect(self._on_failed)
        for sig in (self._worker.finished, self._worker.failed):
            sig.connect(self._thread.quit)
        self._thread.finished.connect(self._on_thread_done)
        self.lbl_busy.setText(label)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.btn_cancel.setVisible(True)
        self._update_buttons()
        self._thread.start()

    def _on_count(self, done: int, total: int) -> None:
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(done)

    def _on_failed(self, message: str) -> None:
        self.lbl_busy.setText(f"⚠ {message}")

    def _on_thread_done(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._thread = self._worker = None
        self.progress.setVisible(False)
        self.btn_cancel.setVisible(False)
        self._update_buttons()

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel = True
            self.lbl_busy.setText("Stopping…")

    def is_busy(self) -> bool:
        return self._thread is not None

    def selected_keys(self) -> list[str]:
        return [k.key for r, k in enumerate(tuning.KNOBS)
                if self.knobs.item(r, 0).checkState() == Qt.Checked]

    def run_fit(self) -> None:
        if self._thread is not None:
            return
        keys = self.selected_keys()
        if not keys:
            self.lbl_summary.setText("Tick at least one setting to fit.")
            return
        lib, table = self.library, self.table
        base = tuning.fit_baseline(lib.folder)

        def work(w: _Worker):
            ready, skipped = tuning.prepare_library(lib, table, base)
            if not ready:
                raise ValueError("no part is ready to fit: each needs a material, a "
                                 "measured time and a valid stock (Parts tab)")
            return tuning.fit(ready, table, base, keys, progress=w.progress.emit,
                              cancelled=lambda: w.cancel), skipped

        self._start(work, self._on_fitted, "Fitting…")

    def _on_fitted(self, outcome) -> None:
        result, skipped = outcome
        self._fit = result
        self._fill_knobs()
        lines = [f"<b>{len(result.parts)} parts.</b>",
                 f"Shipped values: {result.before.text()}.",
                 f"<b>Tuned: {result.after.text()}.</b>"]
        if result.cross_checked:
            lines.append(f"On parts the fit did not see: {result.cross_checked.text()} "
                         f"&mdash; what to expect on a new part.")
        if skipped:
            lines.append(f"<span style='color:{GREY}'>{len(skipped)} part(s) not used: "
                         f"not ready (see the Parts tab).</span>")
        lines += [f"<span style='color:{AMBER}'>&#9888; {w}</span>" for w in result.warnings]
        self.lbl_summary.setText("<br>".join(lines))

        rows = sorted(result.parts, key=lambda p: -abs(math.log(p.ratio_after)))
        self.fit_parts.setRowCount(len(rows))
        for r, p in enumerate(rows):
            self.fit_parts.setItem(r, 0, _item(p.entry.part))
            self.fit_parts.setItem(r, 1, _item(f"{p.measured:,.1f}", align_right=True))
            self.fit_parts.setItem(r, 2, _item(f"{p.before:,.1f}", align_right=True))
            self.fit_parts.setItem(r, 3, _item(f"{p.after:,.1f}", align_right=True))
            err = _item(f"{p.ratio_after - 1:+.0%}", align_right=True)
            err.setForeground(QBrush(QColor(_err_colour(p.ratio_after))))
            self.fit_parts.setItem(r, 4, err)
            held = p.held_out / p.measured if p.held_out else None
            unseen = _item("" if held is None else f"{held - 1:+.0%}", align_right=True)
            if held is not None:
                unseen.setForeground(QBrush(QColor(_err_colour(held))))
            self.fit_parts.setItem(r, 5, unseen)
        self.fit_parts.resizeColumnsToContents()
        self.lbl_busy.setText("Fitted. Save tuning to use it.")
        self._update_buttons()

    def save_fit(self) -> None:
        if self._fit is None:
            return
        try:
            path = tuning.save_fit(self._fit, self.library.folder)
        except OSError as exc:
            QMessageBox.warning(self, "Shop tuning", f"Could not save the tuning:\n{exc}")
            return
        self.profile, self.state = tuning.load_tuned_profile(self.library.folder)
        self.lbl_busy.setText(f"Saved to {path}. Estimates use it now.")
        self.tuningChanged.emit()
        self.refresh()

    def clear_fit(self) -> None:
        if QMessageBox.question(self, "Shop tuning",
                                "Go back to the shipped values?\n\nThe fitted values are "
                                "removed; the library and hand-set rows are kept.") \
                != QMessageBox.Yes:
            return
        tuning.clear_fit(self.library.folder)
        self.profile, self.state = tuning.load_tuned_profile(self.library.folder)
        self.lbl_busy.setText("Back to the shipped values.")
        self.tuningChanged.emit()
        self.refresh()

    def closeEvent(self, event):  # noqa: N802 -- Qt override
        if self._worker is not None:
            self._worker.cancel = True
        super().closeEvent(event)


class SaveToLibraryDialog(QDialog):
    """Asks for the measured time of the part on screen."""

    def __init__(self, name: str, fields: dict, existing, estimate_min, folder, parent=None):
        from PySide6.QtWidgets import QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLineEdit

        super().__init__(parent)
        self.setWindowTitle("Save to tuning library")
        form = QFormLayout(self)
        intro = QLabel(
            ("This part is already in the library; its entry is updated with the "
             "inputs set now:" if existing is not None else
             "What the estimate needs from this part is kept in the library (not the "
             "STEP file), with the inputs set now:")
            + f"<br><span style='color:{GREY}'>{_inputs_text(fields)}</span>")
        intro.setTextFormat(Qt.RichText)
        intro.setWordWrap(True)
        form.addRow(intro)

        self.name = QLineEdit(name)
        form.addRow("Name:", self.name)
        self.measured = QDoubleSpinBox()
        self.measured.setDecimals(2)
        self.measured.setRange(0.0, 100000.0)
        self.measured.setSuffix(" min")
        self.measured.setSpecialValueText("not known yet")
        self.measured.setToolTip(PART_TIPS[C_MEASURED])
        if existing is not None and existing.measured_min:
            self.measured.setValue(existing.measured_min)
        form.addRow("Measured time per part:", self.measured)
        if estimate_min:
            est = QLabel(f"Today's estimate: {estimate_min:,.1f} min")
            est.setStyleSheet(f"color:{GREY};")
            form.addRow("", est)
        hint = QLabel("Door to door, every operation, load/unload and deburr at the "
                      "machine. Not the batch setup.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{GREY};")
        form.addRow("", hint)
        self.notes = QLineEdit(existing.notes if existing is not None else "")
        self.notes.setPlaceholderText("job number, machine, anything")
        form.addRow("Notes:", self.notes)
        where = QLabel(f"Library: {folder}")
        where.setStyleSheet(f"color:{GREY};")
        where.setWordWrap(True)
        form.addRow(where)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self.measured.setFocus()
        self.measured.selectAll()

    def values(self) -> dict:
        m = self.measured.value()
        out = {"measured_min": m if m > 0 else None, "notes": self.notes.text().strip()}
        if self.name.text().strip():
            out["part"] = self.name.text().strip()
        return out


def _inputs_text(f: dict) -> str:
    e = tuning.Entry(part="", id="", stock_type=f.get("stock_type", "plate"),
                     stock_size=f.get("stock_size", ""),
                     stock_profile=f.get("stock_profile", ""))
    tapped = f.get("tapped_holes")
    bits = [f.get("material") or "no material", f.get("process", "auto"),
            f"{f.get('finish_uin', 0):g} µin",
            f"{f.get('operations', 1)} op{'s' if f.get('operations', 1) != 1 else ''}",
            "tapped: auto" if tapped is None else f"{tapped} tapped",
            "stock: " + tuning.stock_text(e)]
    return ", ".join(bits)
