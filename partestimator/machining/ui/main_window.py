"""Main application window: file picker, material/finish inputs (placeholder
for a future time-estimate module), the 3D viewer, and all extracted
geometry / machining-indicator panels.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from partestimator.machining.geometry import cycle_time, derived, export, features, materials, rates, shop, step_io
from partestimator.machining import tuning
from partestimator.machining.geometry import orientation
from partestimator.machining.geometry import stock as stock_mod
from partestimator.machining.geometry.materials import MATERIALS
from partestimator.machining.ui.proc_workers import AnalysisRunner
from partestimator.machining.ui.stock_panel import StockControls
from partestimator.machining.ui.viewer import PartViewer, stock_box_extents
from partestimator.ui_common import guard_wheel, material_picker

# MATERIALS comes from geometry.materials so the dropdown and the density
# table can't drift apart -- a material with no density would otherwise
# silently produce a blank weight.

FINISH_UNITS = ["µin Ra", "µm Ra"]

REMOVAL_TIP = (
    "Mill: 40-taper machining center, 22 hp.\n"
    "Lathe: turn-mill with live tooling. Priced as turning -- passes at\n"
    "constant surface speed, facing, boring, Ra-driven finish feed --\n"
    "with flats, slots and cross holes done by the live tool.\n\n"
    "Set automatically when a part opens: round stock with most of its\n"
    "surface turned opens as Lathe. Change it freely afterwards.")


class MainWindow(QMainWindow):
    #: A file was opened from here rather than by the window around it (the
    #: tuning library's "Open in Machining tab"): its path.
    fileOpened = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("STEP Machining Geometry Estimator")

        self.display_unit = "IN"
        self.current_shape = None      # the loaded file's path (no C++ object crosses processes)
        self._current_path = None
        self.results: dict = {}
        # Live runners; must stay referenced while their process runs.
        self._runner = None
        self._thin_runner = None
        # Re-measures the turned profile when the bar is set another way round.
        self._turn_runner = None
        self._auto_turning = None

        # Loaded once. A missing or malformed rate file must not stop the
        # geometry side of the app from working, so this degrades to
        # "no estimate" rather than failing to start.
        self._rate_table = None
        self._shop_profile = None
        self._rate_error = ""
        try:
            self._rate_table = rates.load_rates()
        except Exception as exc:  # noqa: BLE001
            self._rate_error = f"{type(exc).__name__}: {exc}"
        # The shipped shop profile with this shop's own tuning laid over it
        # (machining/tuning.py). A broken tuning file falls back to shipped.
        self._tuning_state = None
        try:
            self._shop_profile, self._tuning_state = tuning.load_tuned_profile()
        except Exception as exc:  # noqa: BLE001
            try:
                self._shop_profile = shop.load_profile()
            except Exception as exc2:  # noqa: BLE001
                exc = exc2
            self._rate_error = (self._rate_error + "  ").strip() + \
                f"shop profile: {type(exc).__name__}: {exc}"
        self._tuning_window = None
        # A library part to show with its saved inputs once its analysis is in.
        self._pending_entry = None

        self._build_ui()
        guard_wheel(self)

        if self._rate_error:
            self.lbl_cycle_warnings.setText(
                f"⚠ Rate table not loaded — {self._rate_error}")

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(420)
        left_panel = QWidget()
        left_scroll.setWidget(left_panel)
        self.left_layout = QVBoxLayout(left_panel)

        # The viewer is constructed before the control panels because the
        # display checkboxes emit toggled() when their initial state is set,
        # and those handlers talk to the viewer.
        self.viewer = PartViewer(self)

        # One work area for everything the user enters about the part, then
        # the results below it, top to bottom.
        self._build_file_section()
        self._build_input_section()
        self._build_basic_properties_section()
        self._build_bounding_box_section()
        self._build_stock_section()
        self._build_cycle_time_section()
        self._build_hole_section()
        self._build_features_section()
        self._build_timing_section()
        self.left_layout.addStretch(1)

        splitter.addWidget(left_scroll)
        # Viewer with the same view buttons and hint row as the sheet metal
        # tab, so both halves of the tool handle the same way.
        view_box = QWidget()
        vb = QVBoxLayout(view_box)
        vb.setContentsMargins(0, 0, 0, 0)
        vb.setSpacing(5)
        vb.addWidget(self.viewer.widget, 1)
        row = QHBoxLayout()
        for label, method in (("Iso", "view_isometric"), ("Top", "view_xy"),
                              ("Front", "view_xz"), ("Right", "view_yz")):
            b = QPushButton(label)
            b.setMaximumWidth(70)
            b.clicked.connect(lambda _c=False, m=method: self._set_view(m))
            row.addWidget(b)
        fit = QPushButton("Fit")
        fit.setMaximumWidth(70)
        fit.clicked.connect(lambda: self._set_view(None))
        row.addWidget(fit)
        row.addStretch(1)
        hint = QLabel("left-drag rotate   middle-drag pan   wheel zoom")
        hint.setStyleSheet("color:#555; font-size:11px;")
        row.addWidget(hint)
        vb.addLayout(row)
        # What to draw over the part: a viewing choice, not a part input, so
        # it lives with the view buttons rather than in the input panel.
        vb.addLayout(self._build_display_row())
        splitter.addWidget(view_box)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.statusBar().showMessage("Open a STEP file to begin.")

    def _build_file_section(self):
        box = QGroupBox("STEP File")
        layout = QVBoxLayout(box)
        row = QHBoxLayout()
        self.open_button = QPushButton("Open STEP File…")
        self.open_button.clicked.connect(self._on_open_clicked)
        row.addWidget(self.open_button)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._on_cancel_clicked)
        self.cancel_button.setEnabled(False)
        row.addWidget(self.cancel_button)
        self.file_label = QLabel("(none loaded)")
        self.file_label.setWordWrap(True)
        row.addWidget(self.file_label, 1)
        layout.addLayout(row)

        unit_row = QHBoxLayout()
        unit_row.addWidget(QLabel("Display units:"))
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["mm", "in"])
        self.unit_combo.setCurrentText("in")
        self.unit_combo.currentTextChanged.connect(self._on_unit_changed)
        unit_row.addWidget(self.unit_combo)
        unit_row.addStretch(1)
        layout.addLayout(unit_row)

        self.left_layout.addWidget(box)

    # Analysis settings are fixed -- these were the chosen defaults, and the
    # panel that exposed them has been removed. Edit here to change them.
    # Shared with the tuning library, whose cached analyses must match.
    ANALYSIS_OPTIONS = dict(tuning.ANALYSIS_OPTIONS)

    def _analysis_options(self) -> dict:
        # The worker process starts on the shipped stock allowances: hand it
        # the ones in use here (Settings...).
        return {**self.ANALYSIS_OPTIONS, "stock_allowances": stock_mod.allowances()}

    def refresh_stock_allowances(self) -> None:
        """The stock allowances changed (Settings...): resize the open part's
        stock if it was sized by the rules."""
        self.stock_controls.refresh_allowances()

    def _build_display_row(self) -> QHBoxLayout:
        """Only the overlays worth toggling. Model edges are always on and
        the tessellation wireframe is a diagnostic that stays off -- neither
        needed a permanent control."""
        layout = QHBoxLayout()
        layout.addWidget(QLabel("Show:"))

        self.chk_show_box = QCheckBox("Bounding box")
        self.chk_show_box.setChecked(False)
        self.chk_show_box.setToolTip("Grey dashed outline of the selected bounding box.")
        self.chk_show_box.toggled.connect(self._on_overlay_toggled)
        layout.addWidget(self.chk_show_box)

        self.chk_show_stock = QCheckBox("Stock size")
        self.chk_show_stock.setChecked(True)
        self.chk_show_stock.setToolTip(
            "Translucent light-blue stock envelope with a blue dashed outline, "
            "drawn concentric with the bounding box."
        )
        self.chk_show_stock.toggled.connect(self._on_overlay_toggled)
        layout.addWidget(self.chk_show_stock)

        self.chk_show_captured = QCheckBox("Captured hole axes")
        self.chk_show_captured.setChecked(True)
        self.chk_show_captured.setToolTip(
            "Red axis through any hole whose every open mouth is blocked along "
            "its own axis, so no straight tool reaches it from either side. "
            "Draws nothing when a part has none."
        )
        self.chk_show_captured.toggled.connect(self._on_overlay_toggled)
        layout.addWidget(self.chk_show_captured)

        self.chk_show_com = QCheckBox("Center of mass")
        self.chk_show_com.setChecked(False)
        self.chk_show_com.setToolTip(
            "Amber sphere with crosshair lines at the part's center of mass. "
            "The crosshairs extend past the part because the centroid is "
            "usually buried inside solid material."
        )
        self.chk_show_com.toggled.connect(self._on_overlay_toggled)
        layout.addWidget(self.chk_show_com)
        layout.addStretch(1)
        return layout

    def _on_overlay_toggled(self, _checked: bool):
        if self.results:
            self._draw_overlays(fit=False)

    def _build_input_section(self):
        """Every input about the part, in one place: material and process,
        the quality targets, setups, and the stock it is cut from."""
        box = QGroupBox("Part Inputs")
        box.setStyleSheet("QGroupBox { font-weight: bold; }"
                          "QGroupBox * { font-weight: normal; }")
        layout = QFormLayout(box)
        self.material_combo = material_picker(
            [m for m in MATERIALS if m != "Other"], ["Other"])
        self.material_combo.currentTextChanged.connect(self._on_material_changed)
        layout.addRow("Material:", self.material_combo)

        # Third axis of the cycle-time pivot. Material and finish below are
        # the other two.
        self.removal_combo = QComboBox()
        self.removal_combo.addItems(["Mill", "Lathe"])
        self.removal_combo.setToolTip(REMOVAL_TIP)
        self.removal_combo.currentTextChanged.connect(lambda _t: self._refresh_cycle_time())
        layout.addRow("Removal type:", self.removal_combo)

        finish_row = QHBoxLayout()
        self.finish_value = QDoubleSpinBox()
        self.finish_value.setRange(0.0, 1000.0)
        self.finish_value.setDecimals(1)
        self.finish_value.setValue(63.0)
        finish_row.addWidget(self.finish_value)
        self.finish_unit_combo = QComboBox()
        self.finish_unit_combo.addItems(FINISH_UNITS)
        finish_row.addWidget(self.finish_unit_combo)
        layout.addRow("Target surface finish:", finish_row)

        # STEP files rarely carry PMI, so the tightest tolerance has to be
        # entered. It is among the largest non-geometric drivers of cycle
        # time: a 0.0005" bore needs boring and reaming where a 0.010" one
        # needs only a drill.
        self.tolerance_value = QDoubleSpinBox()
        self.tolerance_value.setDecimals(4)
        self.tolerance_value.setRange(0.0, 1.0)
        self.tolerance_value.setSingleStep(0.0005)
        self.tolerance_value.setValue(0.0050)
        self.tolerance_value.setSuffix(" in")
        self.tolerance_value.setToolTip(
            "Tightest dimensional tolerance on the part. Exported with the feature "
            "row; set to 0 if unknown."
        )
        self.tolerance_value.valueChanged.connect(lambda _v: self._refresh_feature_table())
        layout.addRow("Tightest tolerance:", self.tolerance_value)

        self.finish_value.valueChanged.connect(lambda _v: self._refresh_cycle_time())
        self.finish_unit_combo.currentTextChanged.connect(lambda _t: self._refresh_cycle_time())

        # Setups cannot be inferred from the solid, and getting it wrong
        # scales the whole estimate -- a two-op part estimated as one op
        # is half the real door-to-door before any other error.
        self.operations_spin = QSpinBox()
        self.operations_spin.setRange(1, 12)
        self.operations_spin.setValue(1)
        self.operations_spin.setToolTip(
            "Machine operations (setups) per part.\n"
            "Tool changes and load/unload repeat per setup.")
        self.operations_spin.valueChanged.connect(lambda _v: self._refresh_cycle_time())
        layout.addRow("Operations:", self.operations_spin)

        # STEP carries no thread data, so tapped holes are inferred from
        # tap-drill diameters. -1 means "trust the detection".
        self.tapped_spin = QSpinBox()
        self.tapped_spin.setRange(-1, 999)
        self.tapped_spin.setValue(-1)
        self.tapped_spin.setSpecialValueText("auto (detect)")
        self.tapped_spin.setToolTip(
            "Number of tapped holes.\n"
            "'auto' infers them from tap-drill diameters, which is a\n"
            "heuristic — override it if the detected count is wrong.\n"
            "Goes back to 'auto' when the next part opens.")
        self.tapped_spin.valueChanged.connect(lambda _v: self._refresh_cycle_time())
        layout.addRow("Tapped holes:", self.tapped_spin)

        stock_title = QLabel("Stock")
        stock_title.setStyleSheet("font-weight: bold; margin-top: 6px;")
        layout.addRow(stock_title)
        self.stock_controls = StockControls()
        self.stock_controls.stockChanged.connect(self._on_stock_changed)
        self.stock_controls.pickRequested.connect(self._on_pick_requested)
        self.stock_controls.orientationChanged.connect(
            lambda: self.viewer.clear_highlights("stock"))
        self.stock_controls.btn_auto_orient.clicked.connect(
            lambda: self.viewer.clear_highlights())
        self.stock_controls.orient_dialog.startOver.connect(
            lambda: self.viewer.clear_highlights())
        self.stock_controls.orient_dialog.closed.connect(
            lambda: self.viewer.clear_highlights())
        layout.addRow(self.stock_controls)

        self.left_layout.addWidget(box)

    def _build_cycle_time_section(self):
        """Bottom-up cycle estimate from published cutting data.

        Deliberately separate from the feature vector: this is a
        handbook estimate, not a fitted model, and the export carries it
        alongside the blank measured column so the two can be compared.
        """
        box = QGroupBox("Cycle Time")
        box.setToolTip(
            "Door-to-door estimate from published manufacturer rates at 80%.\n"
            "A bottom-up estimate, not a fitted model: compare it with your\n"
            "measured times and tune data/machining_rates.csv.")
        layout = QVBoxLayout(box)

        self.lbl_cycle_total = QLabel("—")
        self.lbl_cycle_total.setStyleSheet(
            "font-size: 17px; font-weight: bold; color: #1f4e79;")
        layout.addWidget(self.lbl_cycle_total)

        self.cycle_table = QTableWidget(0, 3)
        self.cycle_table.setHorizontalHeaderLabels(["Term", "Time", "Basis"])
        self.cycle_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.cycle_table.horizontalHeader().setStretchLastSection(True)
        self.cycle_table.verticalHeader().setVisible(False)
        layout.addWidget(self.cycle_table)

        self.lbl_cycle_warnings = QLabel("")
        self.lbl_cycle_warnings.setWordWrap(True)
        self.lbl_cycle_warnings.setStyleSheet("color: #92400e;")
        layout.addWidget(self.lbl_cycle_warnings)

        row = QHBoxLayout()
        self.btn_copy_cycle = QPushButton("Copy breakdown")
        self.btn_copy_cycle.setToolTip(
            "Copy the term breakdown as tab-separated text for Excel.")
        self.btn_copy_cycle.clicked.connect(self._copy_cycle)
        self.btn_show_rates = QPushButton("Show rates and sources…")
        self.btn_show_rates.setToolTip(
            "Every published figure this estimate rests on, with its citation.")
        self.btn_show_rates.clicked.connect(self._show_rate_sources)
        row.addWidget(self.btn_copy_cycle)
        row.addWidget(self.btn_show_rates)
        row.addStretch(1)
        # Fitting the model to the shop's own measured parts.
        self.btn_tuning = QPushButton("Shop tuning")
        self.btn_tuning.setToolTip(
            "Fit the estimate to parts your shop has run and timed.\n"
            "The adjustments stay on this computer (or your shop's shared folder).")
        tune_menu = QMenu(self.btn_tuning)
        self.act_save_to_library = tune_menu.addAction(
            "Save this part to the tuning library…", self.save_to_library)
        self.act_open_library = tune_menu.addAction(
            "Open the tuning library…", self.open_tuning)
        self.btn_tuning.setMenu(tune_menu)
        row.addWidget(self.btn_tuning)
        layout.addLayout(row)

        self.lbl_tuning = QLabel("")
        self.lbl_tuning.setWordWrap(True)
        self.lbl_tuning.setStyleSheet("color: #555;")
        layout.addWidget(self.lbl_tuning)
        self._refresh_tuning_label()

        self.left_layout.addWidget(box)

    def _set_view(self, method) -> None:
        p = self.viewer.plotter
        if method:
            getattr(p, method)()
        p.reset_camera()
        p.render()

    def _auto_removal_type(self, results: dict) -> None:
        """Round stock with most of its surface turned opens as Lathe."""
        removal, why = cycle_time.auto_removal_type(results, self._shop_profile)
        self.removal_combo.blockSignals(True)
        self.removal_combo.setCurrentText("Lathe" if removal == "lathe" else "Mill")
        self.removal_combo.blockSignals(False)
        self.removal_combo.setToolTip(f"{why}\n\n{REMOVAL_TIP}")

    def _current_removal_type(self) -> str:
        return self.removal_combo.currentText().strip().lower()

    def _finish_uin(self) -> float:
        """The finish target in micro-inches, whichever unit it was typed in."""
        value = self.finish_value.value()
        if self.finish_unit_combo.currentText().startswith("µm"):
            return value / 0.0254          # 1 µin = 0.0254 µm
        return value

    def _current_estimate(self):
        if not self.results or self._rate_table is None:
            return None
        # The pipeline stores the feature vector under "derived"; the
        # material-dependent columns (density) are refreshed in-place by
        # _refresh_feature_table when the material combo changes, so this
        # is the same dict the export will use.
        feats = self.results.get("derived")
        if not feats:
            feats = derived.compute_derived_features(
                self.results, material=self.material_combo.currentText())
        try:
            tapped = self.tapped_spin.value()
            return cycle_time.estimate_cycle_time(
                feats, self.results, self._rate_table,
                material=self.material_combo.currentText(),
                finish_uin=self._finish_uin(),
                removal_type=self._current_removal_type(),
                profile=self._shop_profile,
                operations=self.operations_spin.value(),
                tapped_override=None if tapped < 0 else tapped,
            )
        except Exception as exc:  # noqa: BLE001 - never let the estimate break the UI
            self.statusBar().showMessage(f"Cycle estimate failed: {exc}", 6000)
            return None

    def _refresh_cycle_time(self):
        if not self.results:
            return
        est = self._current_estimate()
        if est is None:
            self.lbl_cycle_total.setText("—")
            self.cycle_table.setRowCount(0)
            return

        self.lbl_cycle_total.setText(
            f"{est.door_to_door_minutes:,.1f} min  "
            f"({est.door_to_door_minutes / 60.0:,.2f} hr)")

        rows = est.as_rows()
        self.cycle_table.setRowCount(len(rows))
        for r, (label, value, note) in enumerate(rows):
            name = QTableWidgetItem(label)
            if label in ("DOOR TO DOOR", "Machine cycle", "Cut subtotal"):
                font = name.font()
                font.setBold(True)
                name.setFont(font)
            self.cycle_table.setItem(r, 0, name)
            val = QTableWidgetItem(value)
            val.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if label in ("DOOR TO DOOR", "Machine cycle", "Cut subtotal"):
                font = val.font()
                font.setBold(True)
                val.setFont(font)
            self.cycle_table.setItem(r, 1, val)
            self.cycle_table.setItem(r, 2, QTableWidgetItem(note))
        self.cycle_table.resizeColumnsToContents()

        warnings = list(est.warnings)
        stock = self.results.get("stock")
        if self._current_removal_type() == "lathe" and stock is not None and not stock.is_round:
            warnings.append("Lathe pricing turns a round bar sized from the part; the "
                            "chosen stock is not used. Pick Round bar to set the bar.")
        self.lbl_cycle_warnings.setText(
            "  ".join("⚠ " + w for w in warnings) if warnings else "")

    def _copy_cycle(self):
        est = self._current_estimate()
        if est is None:
            return
        lines = ["Term\tMinutes\tBasis"]
        for label, value, note in est.as_rows():
            lines.append(f"{label}\t{value.replace(' min', '')}\t{note}")
        QApplication.clipboard().setText("\n".join(lines))
        self.statusBar().showMessage("Cycle breakdown copied", 4000)

    def _show_rate_sources(self):
        if self._rate_table is None:
            return
        material = self.material_combo.currentText()
        rows = self._rate_table.sources_for(material, self._current_removal_type())
        if not rows:
            QMessageBox.information(
                self, "Rates",
                f"No published rates on file for {material}.\n\n"
                f"Add rows to data/machining_rates.csv.")
            return
        parts = [f"<b>{material}</b> — ISO group "
                 f"{self._rate_table.iso_group(material)}<br><br>"]
        for r in rows:
            flag = "  <i>(overridden)</i>" if r.is_overridden else ""
            parts.append(
                f"<b>{r.operation}</b>: {r.effective_value:g} {r.units}"
                f" &nbsp;=&nbsp; {r.published_value:g} published × {r.derate:g}{flag}"
                f"<br><span style='color:#555'>{r.notes}</span>"
                f"<br><a href='{r.source_url}'>{r.source}</a>"
                f" <span style='color:#777'>(retrieved {r.retrieved})</span><br><br>")
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Rates and sources")
        dlg.setTextFormat(Qt.RichText)
        dlg.setText("".join(parts))
        dlg.exec()

    def _build_basic_properties_section(self):
        box = QGroupBox("Basic Properties")
        layout = QFormLayout(box)
        self.lbl_volume = QLabel("—")
        self.lbl_area = QLabel("—")
        self.lbl_solidity = QLabel("—")
        self.lbl_weight = QLabel("—")
        self.lbl_solidity.setToolTip(
            "Part volume divided by its convex hull volume. Near 1.0 is a simple "
            "blocky part; low values mean deep pockets, webs and open structure."
        )
        self.lbl_weight.setToolTip("Part volume x material density.")
        layout.addRow("Volume:", self.lbl_volume)
        layout.addRow("Surface area:", self.lbl_area)
        layout.addRow("Solidity:", self.lbl_solidity)
        layout.addRow("Est. weight:", self.lbl_weight)
        self.left_layout.addWidget(box)

    def _build_bounding_box_section(self):
        """Only the selected box is shown -- whichever of the rectilinear or
        cylindrical fit has the smaller volume."""
        box = QGroupBox("Bounding Box")
        layout = QFormLayout(box)
        self.lbl_bbox_kind = QLabel("—")
        self.lbl_bbox_kind.setStyleSheet("font-weight: bold;")
        self.lbl_bbox_dims = QLabel("—")
        self.lbl_bbox_volume = QLabel("—")
        layout.addRow("Selected:", self.lbl_bbox_kind)
        layout.addRow("Dimensions:", self.lbl_bbox_dims)
        layout.addRow("Volume:", self.lbl_bbox_volume)
        self.left_layout.addWidget(box)

    def _build_stock_section(self):
        box = QGroupBox("Stock Size")
        layout = QVBoxLayout(box)

        # A real table rather than a formatted string, so the three values
        # can be selected and pasted into Excel as three cells. Qt puts
        # tab-separated text on the clipboard for a multi-cell selection,
        # which is exactly what Excel splits on.
        self.stock_table = QTableWidget(1, 3)
        self.stock_table.setHorizontalHeaderLabels(["OD/ST", "W", "L"])
        self.stock_table.verticalHeader().setVisible(False)
        self.stock_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.stock_table.setSelectionMode(QAbstractItemView.ContiguousSelection)
        self.stock_table.horizontalHeader().setStretchLastSection(True)
        # Two header rows' worth of chrome plus one data row; without this
        # the table claims far more vertical space than it needs.
        self.stock_table.setFixedHeight(
            self.stock_table.horizontalHeader().height()
            + self.stock_table.rowHeight(0) + 4
        )
        layout.addWidget(self.stock_table)

        row = QHBoxLayout()
        self.btn_copy_stock = QPushButton("Copy row")
        self.btn_copy_stock.setToolTip(
            "Copy OD/ST, W, L to the clipboard as tab-separated text.\n"
            "Pastes into Excel as three cells.")
        self.btn_copy_stock.clicked.connect(lambda: self._copy_stock(with_header=False))
        self.btn_copy_stock_hdr = QPushButton("Copy with header")
        self.btn_copy_stock_hdr.setToolTip(
            "Same, with the OD/ST, W, L header line above it.")
        self.btn_copy_stock_hdr.clicked.connect(lambda: self._copy_stock(with_header=True))
        row.addWidget(self.btn_copy_stock)
        row.addWidget(self.btn_copy_stock_hdr)
        row.addStretch(1)
        layout.addLayout(row)

        form = QFormLayout()
        self.lbl_stock_volume = QLabel("—")
        self.lbl_stock_weight = QLabel("—")
        form.addRow("Stock volume:", self.lbl_stock_volume)
        form.addRow("Stock weight:", self.lbl_stock_weight)
        layout.addLayout(form)

        self.left_layout.addWidget(box)

    def _copy_stock(self, with_header: bool = False):
        stock = self.results.get("stock") if self.results else None
        if stock is None:
            return
        QApplication.clipboard().setText(stock.tsv(include_header=with_header))
        self.statusBar().showMessage(
            "Stock copied — paste into Excel", 4000)

    def _build_hole_section(self):
        box = QGroupBox("Holes")
        box.setToolTip("Holes grouped by diameter: concave cylindrical faces only.")
        layout = QVBoxLayout(box)
        self.hole_table = QTableWidget(0, 4)
        self.hole_table.setHorizontalHeaderLabels(["Nominal ⌀", "Holes", "Max depth", "Avg depth"])
        self.hole_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.hole_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.hole_table)
        self.left_layout.addWidget(box)

    def _build_features_section(self):
        """The ML feature vector: everything scale-free and
        authoring-independent that the tool can derive."""
        box = QGroupBox("Derived Features")
        box.setToolTip(
            "For training a cycle-time model: scale-free ratios plus size terms,\n"
            "excluding anything positional or exporter-dependent. Export appends\n"
            "one row per part; fill in cycle_time_min from your job records.")
        layout = QVBoxLayout(box)

        self.feature_table = QTableWidget(0, 2)
        self.feature_table.setHorizontalHeaderLabels(["Feature", "Value"])
        self.feature_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.feature_table.horizontalHeader().setStretchLastSection(True)
        self.feature_table.verticalHeader().setVisible(False)
        self.feature_table.setMinimumHeight(260)
        layout.addWidget(self.feature_table)

        # Thin-wall scanning now exists purely to enrich the feature vector
        # (min wall thickness), so its control lives here rather than in a
        # panel of its own.
        thin_row = QHBoxLayout()
        thin_row.addWidget(QLabel("Thin-wall threshold:"))
        self.thin_threshold_spin = QDoubleSpinBox()
        self.thin_threshold_spin.setDecimals(3)
        thin_row.addWidget(self.thin_threshold_spin)
        self._sync_thin_threshold_spin_to_unit(reset_to_default=True)
        self.thin_wall_button = QPushButton("Run thin-wall scan")
        self.thin_wall_button.setToolTip(
            "Optional: adds the minimum wall thickness to the features.\n"
            "Slow on complex parts, so it is not run when a part opens.")
        self.thin_wall_button.clicked.connect(self._on_run_thin_wall_scan)
        self.thin_wall_button.setEnabled(False)
        thin_row.addWidget(self.thin_wall_button)
        layout.addLayout(thin_row)

        self.lbl_thin_wall = QLabel("")
        self.lbl_thin_wall.setWordWrap(True)
        self.lbl_thin_wall.setStyleSheet("color: #555;")
        layout.addWidget(self.lbl_thin_wall)

        row = QHBoxLayout()
        self.export_csv_button = QPushButton("Append to CSV…")
        self.export_csv_button.clicked.connect(self._on_export_csv)
        self.export_csv_button.setEnabled(False)
        row.addWidget(self.export_csv_button)
        self.export_json_button = QPushButton("Save JSON…")
        self.export_json_button.clicked.connect(self._on_export_json)
        self.export_json_button.setEnabled(False)
        row.addWidget(self.export_json_button)
        layout.addLayout(row)

        self.left_layout.addWidget(box)

    def _refresh_feature_table(self):
        # Recompute so weight-dependent features track the material dropdown
        # without needing a reanalysis.
        if self.results:
            try:
                self.results["derived"] = derived.compute_derived_features(
                    self.results, material=self.material_combo.currentText()
                )
            except Exception:  # noqa: BLE001 - a display refresh must not kill the load
                pass
        feats = self.results.get("derived") or {}
        rows = sorted(feats.items())
        self.feature_table.setRowCount(len(rows))
        for r, (key, value) in enumerate(rows):
            self.feature_table.setItem(r, 0, QTableWidgetItem(key))
            if isinstance(value, float):
                text = f"{value:,.6g}"
            else:
                text = f"{value:,}" if isinstance(value, int) else str(value)
            self.feature_table.setItem(r, 1, QTableWidgetItem(text))
        self.feature_table.resizeColumnsToContents()
        enabled = bool(feats)
        self.export_csv_button.setEnabled(enabled)
        self.export_json_button.setEnabled(enabled)

    def _feature_row(self) -> dict:
        return export.build_row(
            self.results,
            self.results.get("derived") or {},
            material=self.material_combo.currentText(),
            finish_value=self.finish_value.value(),
            finish_units=self.finish_unit_combo.currentText(),
            tolerance_in=self.tolerance_value.value(),
            quality=derived.quality_flags(self.results),
            removal_type=self._current_removal_type(),
            estimate=self._current_estimate(),
        )

    def _on_export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Append features to CSV", "cycle_time_training_set.csv", "CSV files (*.csv)"
        )
        if not path:
            return
        try:
            export.append_csv(path, self._feature_row())
            self.statusBar().showMessage(f"Appended features for this part to {Path(path).name}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export failed", f"{type(exc).__name__}: {exc}")

    def _on_export_json(self):
        default = f"{Path(self.results.get('path', 'part')).stem}_features.json"
        path, _ = QFileDialog.getSaveFileName(self, "Save features as JSON", default, "JSON files (*.json)")
        if not path:
            return
        try:
            export.write_json(path, self._feature_row())
            self.statusBar().showMessage(f"Saved features to {Path(path).name}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export failed", f"{type(exc).__name__}: {exc}")

    def _build_timing_section(self):
        """Shows where load time actually went, per stage. The costs shift a
        lot between parts (a big freeform part is dominated by tessellation,
        a part with thousands of small faces by the face scan), so measuring
        beats guessing."""
        box = QGroupBox("Load Time Breakdown")
        layout = QVBoxLayout(box)
        self.timing_table = QTableWidget(0, 3)
        self.timing_table.setHorizontalHeaderLabels(["Stage", "Time", "%"])
        self.timing_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.timing_table.horizontalHeader().setStretchLastSection(True)
        self.timing_table.verticalHeader().setVisible(False)
        layout.addWidget(self.timing_table)
        self.lbl_mesh_stats = QLabel("—")
        self.lbl_mesh_stats.setWordWrap(True)
        layout.addWidget(self.lbl_mesh_stats)
        self.left_layout.addWidget(box)

    def _refresh_timing_table(self):
        timings = self.results.get("timings") or {}
        total = sum(timings.values()) or 1.0
        rows = sorted(timings.items(), key=lambda kv: -kv[1])
        self.timing_table.setRowCount(len(rows) + 1)
        for r, (label, seconds) in enumerate(rows):
            self.timing_table.setItem(r, 0, QTableWidgetItem(label))
            self.timing_table.setItem(r, 1, QTableWidgetItem(f"{seconds:.2f} s"))
            self.timing_table.setItem(r, 2, QTableWidgetItem(f"{100 * seconds / total:.0f}%"))
        last = len(rows)
        total_item = QTableWidgetItem("TOTAL")
        font = total_item.font()
        font.setBold(True)
        total_item.setFont(font)
        self.timing_table.setItem(last, 0, total_item)
        self.timing_table.setItem(last, 1, QTableWidgetItem(f"{sum(timings.values()):.2f} s"))
        self.timing_table.setItem(last, 2, QTableWidgetItem(""))
        self.timing_table.resizeColumnsToContents()

        n_tri = len(self.results.get("render_triangles", []))
        n_edges = len(self.results.get("edge_polylines", []))
        defl = self.results.get("deflection_mm", 0.0)
        self.lbl_mesh_stats.setText(
            f"{n_tri:,} triangles at {defl:.3g} mm deflection · {n_edges:,} model edges. "
            f"If tessellation dominates, raise DEFAULT_RELATIVE_DEFLECTION in geometry/mesh_utils.py."
        )

    # ------------------------------------------------------------- actions
    def _on_open_clicked(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open STEP File", "", "STEP files (*.step *.stp)")
        if not path:
            return
        self._load_and_process(path)

    def _on_cancel_clicked(self):
        """Terminates the worker process outright. Because the work is in a
        separate process this interrupts even an OCCT call that would
        otherwise ignore Python entirely."""
        for runner in (self._runner, self._thin_runner, self._turn_runner):
            if runner is not None:
                runner.cancel()
        self.statusBar().showMessage("Cancelled.")

    def _on_unit_changed(self, text: str):
        old_unit = self.display_unit
        self.display_unit = "IN" if text == "in" else "MM"
        self._sync_thin_threshold_spin_to_unit(reset_to_default=False, previous_unit=old_unit)
        if self.results:
            self._refresh_display()

    def _sync_thin_threshold_spin_to_unit(self, reset_to_default: bool, previous_unit: str | None = None):
        """Keeps the thin-wall threshold spin box's numeric value and range
        consistent with the current display unit. On a genuine unit switch
        (previous_unit given), the underlying mm value is preserved and just
        re-expressed in the new unit rather than silently reinterpreted."""
        unit = self.display_unit
        if reset_to_default:
            value_mm = features.THIN_WALL_DEFAULT_THRESHOLD_MM
        elif previous_unit is not None:
            value_mm = step_io.display_to_mm(self.thin_threshold_spin.value(), previous_unit)
        else:
            value_mm = step_io.display_to_mm(self.thin_threshold_spin.value(), unit)

        if unit == "IN":
            self.thin_threshold_spin.setRange(0.002, 2.0)
            self.thin_threshold_spin.setSuffix(" in")
        else:
            self.thin_threshold_spin.setRange(0.05, 50.0)
            self.thin_threshold_spin.setSuffix(" mm")
        self.thin_threshold_spin.setValue(step_io.mm_to_display(value_mm, unit))

    def _set_busy(self, busy: bool):
        """Disables the actions that would conflict with a running job. The
        window itself stays responsive (pan/zoom/resize still work) because
        the work is on a worker thread -- no processEvents() pumping needed."""
        self.open_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.thin_wall_button.setEnabled(not busy and self.current_shape is not None)
        if busy:
            QApplication.setOverrideCursor(Qt.WaitCursor)
        else:
            QApplication.restoreOverrideCursor()

    def _load_and_process(self, path: str):
        """Runs the analysis in a separate PROCESS.

        A thread isn't enough: OCP doesn't release the GIL around OCCT
        calls, so one long tessellation would freeze the whole UI with no
        progress shown. See ui/proc_workers.py for the full reasoning.
        """
        self._set_busy(True)
        self._current_path = path
        self.statusBar().showMessage(f"Loading {Path(path).name}…")

        runner = AnalysisRunner(self)
        runner.progress.connect(self.statusBar().showMessage)
        runner.finished.connect(self._on_analysis_finished)
        runner.failed.connect(self._on_worker_failed)
        runner.done.connect(self._on_analysis_done)
        self._runner = runner
        runner.start_analysis(path, self._analysis_options())

    def _on_analysis_done(self):
        self._runner = None
        self._set_busy(False)

    def _on_analysis_finished(self, results: dict):
        # No TopoDS_Shape comes back from the child process (C++ objects
        # can't be pickled); the file path is what later stages re-use.
        self.current_shape = results.get("path")
        self.results = results
        if self._turn_runner is not None:
            self._turn_runner.cancel()
            self._turn_runner = None
        self._auto_turning = results.get("turning")

        self.file_label.setText(Path(results["path"]).name)
        self._auto_removal_type(results)
        # Part-specific inputs start fresh for each part; material, finish,
        # tolerance and setups carry over, since a batch usually shares them.
        self.tapped_spin.blockSignals(True)
        self.tapped_spin.setValue(-1)
        self.tapped_spin.blockSignals(False)
        sp = results.get("shape_props")
        stock = self.stock_controls.set_part(
            results.get("rect_box"), results.get("cylinder"), results.get("primary"),
            emit=False,
            part_volume_in3=(sp.volume_mm3 / 25.4 ** 3) if sp is not None else None,
            points_mm=results.get("render_points"),
            triangles=results.get("render_triangles"),
            extrusion=results.get("extrusion"))
        if stock is not None:
            results["stock"] = stock
        self.lbl_thin_wall.setText("")

        self.viewer.display_part(
            results["render_points"], results["render_triangles"], results.get("edge_polylines"),
        )
        self._draw_overlays()
        # Features first: the cycle estimate inside _refresh_display reads
        # results["derived"], and the material-dependent columns (density,
        # which drives the load/unload term) are only filled in here.
        self._refresh_feature_table()
        self._refresh_display()
        self._refresh_timing_table()

        timings = results.get("timings") or {}
        total = sum(timings.values())
        slowest = max(timings.items(), key=lambda kv: kv[1])[0] if timings else "—"
        n_tri = len(results["render_triangles"])
        self.statusBar().showMessage(
            f"Loaded {Path(results['path']).name} in {total:.1f}s "
            f"(slowest stage: {slowest}) — {n_tri:,} triangles "
            f"at {results['deflection_mm']:.3g} mm deflection"
        )
        pending, self._pending_entry = self._pending_entry, None
        if pending is not None and self._current_path and \
                Path(pending[2]) == Path(self._current_path):
            self._apply_entry_inputs(pending[0])

    def _on_worker_failed(self, title: str, message: str):
        QMessageBox.critical(self, title, message)
        self.statusBar().showMessage(title)

    def _on_run_thin_wall_scan(self):
        """Thin-wall scanning is opt-in (not run on load) because it's the
        slowest step by far. Like the main analysis it runs out-of-process,
        re-loading the file there since a shape can't cross a process
        boundary."""
        if not self._current_path:
            return
        self._set_busy(True)
        rect_box = self.results["rect_box"]
        max_dist = float(np.linalg.norm(rect_box.extents_mm)) * 1.5
        threshold_mm = step_io.display_to_mm(self.thin_threshold_spin.value(), self.display_unit)

        runner = AnalysisRunner(self)
        runner.progress.connect(self.statusBar().showMessage)
        runner.finished.connect(self._on_thin_wall_finished)
        runner.failed.connect(self._on_worker_failed)
        runner.done.connect(self._on_thin_done)
        self._thin_runner = runner
        runner.start_thin_wall(self._current_path, threshold_mm, max_dist)

    def _on_thin_done(self):
        self._thin_runner = None
        self._set_busy(False)

    def _on_thin_wall_finished(self, result):
        self.results["thin_wall"] = result
        self._refresh_thin_wall_label()
        self._draw_overlays()
        self.statusBar().showMessage("Thin-wall scan complete.")

    def closeEvent(self, event):
        """Shut down cleanly: let any running worker thread finish and close
        the VTK render window, otherwise PyVista/VTK can emit errors or hang
        on exit."""
        for runner in (self._runner, self._thin_runner, self._turn_runner):
            if runner is not None:
                try:
                    runner.cancel()
                except Exception:  # noqa: BLE001
                    pass
        try:
            self.viewer.close()
        except Exception:  # noqa: BLE001
            pass
        super().closeEvent(event)

    # ------------------------------------------------------------ tuning
    def _refresh_tuning_label(self) -> None:
        state = self._tuning_state
        text = state.describe() if state is not None else ""
        if state is not None and state.errors:
            text = (text + "  " if text else "") + "⚠ " + "; ".join(state.errors[:2])
        self.lbl_tuning.setText(text)
        self.lbl_tuning.setVisible(bool(text))

    def reload_shop_profile(self) -> None:
        """The tuning changed on disk: estimates follow at once."""
        try:
            self._shop_profile, self._tuning_state = tuning.load_tuned_profile()
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"Shop tuning not loaded: {exc}", 8000)
            return
        self._refresh_tuning_label()
        self._refresh_cycle_time()

    def library_defaults(self) -> dict:
        """What a part added to the library starts with: today's inputs."""
        return {"material": self.material_combo.currentText(),
                "finish_uin": round(self._finish_uin(), 2),
                "operations": self.operations_spin.value()}

    def open_tuning(self):
        from partestimator.machining.ui.tuning_window import TuningWindow

        if self._rate_table is None:
            QMessageBox.warning(self, "Shop tuning",
                                f"The rate table is not loaded: {self._rate_error}")
            return None
        if self._tuning_window is None:
            w = TuningWindow(self._rate_table, defaults=self.library_defaults, parent=self)
            w.openRequested.connect(self.open_library_entry)
            w.tuningChanged.connect(self.reload_shop_profile)
            self._tuning_window = w
        self._tuning_window.show()
        self._tuning_window.raise_()
        self._tuning_window.activateWindow()
        return self._tuning_window

    def _entry_fields(self) -> dict:
        """The inputs on screen, as tuning-library fields."""
        tapped = self.tapped_spin.value()
        tol = self.tolerance_value.value()
        stock_type, stock_size, stock_profile = tuning.stock_fields(self.stock_controls.stock())
        return {
            "material": self.material_combo.currentText(),
            "process": self._current_removal_type(),
            "finish_uin": round(self._finish_uin(), 2),
            "tolerance_in": tol if tol > 0 else None,
            "operations": self.operations_spin.value(),
            "tapped_holes": None if tapped < 0 else tapped,
            "stock_type": stock_type,
            "stock_size": stock_size,
            "stock_profile": stock_profile,
        }

    def save_to_library(self) -> None:
        """Record this part, as set up on screen, with its measured time.

        What the estimate reads from the geometry is captured now; the STEP
        file is not copied, only its path noted."""
        from partestimator.machining.ui.tuning_window import SaveToLibraryDialog

        if not self.results or not self._current_path:
            QMessageBox.information(self, "Shop tuning", "Open a part first.")
            return
        path = Path(self._current_path)
        try:
            lib = tuning.Library.load()
            part_id = tuning.file_id(path)
        except OSError as exc:
            QMessageBox.warning(self, "Shop tuning", f"Could not read the library:\n{exc}")
            return
        existing = lib.find(part_id)
        fields = self._entry_fields()
        est = self._current_estimate()
        dlg = SaveToLibraryDialog(path.stem if existing is None else existing.part, fields,
                                  existing, est.door_to_door_minutes if est else None,
                                  lib.folder, self)
        if not dlg.exec():                     # cancelled
            return
        fields.update(dlg.values())
        try:
            snap = tuning.capture(self.results, self.stock_controls.stock(),
                                  self.material_combo.currentText())
            entry = existing or tuning.Entry(part=path.stem, id=part_id)
            for k, v in fields.items():
                setattr(entry, k, v)
            entry.source_file = str(path.resolve())
            lib.add(entry, snap)
            lib.save()
        except OSError as exc:
            QMessageBox.warning(self, "Shop tuning",
                                f"Could not save to the library:\n{exc}\n\n"
                                f"Is {tuning.LIBRARY_FILE} open in Excel?")
            return
        self.statusBar().showMessage(
            f"{'Updated' if existing else 'Saved'} {entry.part} in the tuning library "
            f"({lib.csv_path}).", 8000)
        if self._tuning_window is not None:
            self._tuning_window.entry_saved()

    def open_library_entry(self, entry, library) -> None:
        """Show a library part, from the file it was added from, with the
        inputs saved for it."""
        path = entry.source_file
        if not path or not Path(path).is_file():
            QMessageBox.information(self, "Shop tuning",
                                    f"The STEP file is no longer there:\n{path}")
            return
        self._pending_entry = (entry, library, path)
        self.fileOpened.emit(path)
        self._load_and_process(path)            # inputs applied when it is in
        self.window().raise_()
        self.window().activateWindow()

    def _apply_entry_inputs(self, entry) -> None:
        if entry.material:
            self.material_combo.setCurrentText(entry.material)
        if entry.process in ("mill", "lathe"):
            self.removal_combo.blockSignals(True)
            self.removal_combo.setCurrentText(entry.process.capitalize())
            self.removal_combo.blockSignals(False)
        self.finish_unit_combo.setCurrentText(FINISH_UNITS[0])
        self.finish_value.setValue(entry.finish_uin)
        if entry.tolerance_in is not None:
            self.tolerance_value.setValue(entry.tolerance_in)
        self.operations_spin.setValue(entry.operations)
        self.tapped_spin.setValue(-1 if entry.tapped_holes is None else entry.tapped_holes)
        try:
            stock = tuning.build_stock(entry)
            if stock.is_extrusion:
                # Placed on this part the way the stock panel would place it.
                stock = self.stock_controls.extrusion_for(stock.profile, stock.length_in)
        except ValueError as exc:
            self.statusBar().showMessage(f"Saved stock not used: {exc}", 10000)
            stock = None
        if stock is not None:
            self.stock_controls.set_stock(stock)       # refreshes the estimate
        else:
            self._refresh_feature_table()
            self._refresh_display()
        self.statusBar().showMessage(
            f"{entry.part}: opened from the tuning library with its saved inputs"
            + (f"; measured {entry.measured_min:g} min." if entry.measured_min else "."),
            10000)

    # ------------------------------------------------------------- display
    def _draw_overlays(self, fit: bool = True):
        self.viewer.clear_overlays()
        rect_box = self.results["rect_box"]
        cyl = self.results["cylinder"]
        primary = self.results["primary"]
        if self.chk_show_box.isChecked():
            # `cyl` is None when the cylinder fit was switched off; in that
            # case `primary` is already forced to "rectilinear".
            self.viewer.show_primary_bounding_box(primary, rect_box, cyl)

        if self.chk_show_stock.isChecked():
            self.viewer.show_stock(primary, self.results.get("stock"), rect_box, cyl)

        if self.chk_show_captured.isChecked():
            self.viewer.show_captured_hole_axes(self.results.get("holes") or [])

        if self.chk_show_com.isChecked():
            self.viewer.show_center_of_mass(self.results["shape_props"].centroid_mm)

        thin_wall = self.results.get("thin_wall")
        if thin_wall is not None and thin_wall.worst_points_mm:
            points = np.array([p for p, _dist in thin_wall.worst_points_mm])
            self.viewer.show_thin_wall_points(points)

        if fit:
            self.viewer.fit_all()
        else:
            self.viewer.plotter.render()

    def _refresh_display(self):
        unit = self.display_unit
        len_suffix = "mm" if unit == "MM" else "in"
        area_suffix = "mm²" if unit == "MM" else "in²"
        vol_suffix = "mm³" if unit == "MM" else "in³"

        sp = self.results["shape_props"]
        self.lbl_volume.setText(f"{step_io.mm3_to_display(sp.volume_mm3, unit):.4g} {vol_suffix}")
        self.lbl_area.setText(f"{step_io.mm2_to_display(sp.surface_area_mm2, unit):.4g} {area_suffix}")

        solidity = (self.results.get("derived") or {}).get("solidity")
        self.lbl_solidity.setText(f"{solidity:.4f}" if solidity is not None else "—")
        self._refresh_weight_labels()

        self._refresh_bounding_box_labels(unit, len_suffix, vol_suffix)
        self._refresh_stock_labels()
        self._refresh_cycle_time()
        self._refresh_hole_table(unit, len_suffix)
        self._refresh_thin_wall_label()

    def _refresh_weight_labels(self):
        """Weight depends on the material dropdown, so it refreshes on its
        own whenever the selection changes -- no reanalysis needed."""
        sp = self.results.get("shape_props")
        material = self.material_combo.currentText()
        if sp is None:
            self.lbl_weight.setText("—")
        else:
            w = materials.weight_lb(sp.volume_mm3, material)
            self.lbl_weight.setText(f"{w:,.3f} lb" if w is not None
                                     else "— (no density for this material)")

        stock = self.results.get("stock")
        density = materials.density_lb_per_in3(material)
        if stock is not None and density:
            self.lbl_stock_weight.setText(f"{stock.volume_in3 * density:,.3f} lb")
        elif stock is not None:
            self.lbl_stock_weight.setText("— (no density for this material)")
        else:
            self.lbl_stock_weight.setText("—")

    def _on_stock_changed(self, stock):
        """The user picked a different stock: everything that depends on it --
        features, weight, cycle time and the overlay -- follows at once."""
        if not self.results:
            return
        self.results["stock"] = stock
        # A bar set another way round: the cylinder everything reads follows,
        # and the turned profile is measured again about its axis.
        cyl = self.stock_controls.cylinder()
        if cyl is not None and cyl is not self.results.get("cylinder"):
            self.results["cylinder"] = cyl
            self._remeasure_turning(cyl)
        # The part may have been turned in its stock: the box everything
        # else reads (labels, removal ratio, the 3D outline) turns with it.
        rect = self.stock_controls.rect_box()
        if rect is not None and rect is not self.results.get("rect_box"):
            self.results["rect_box"] = rect
            unit = self.display_unit
            self._refresh_bounding_box_labels(
                unit, "mm" if unit == "MM" else "in", "mm³" if unit == "MM" else "in³")
        self._refresh_feature_table()
        self._refresh_weight_labels()
        self._refresh_stock_labels()
        self._refresh_cycle_time()
        self._draw_overlays(fit=False)

    def _remeasure_turning(self, cyl) -> None:
        """The lathe estimate is built from the profile turned about the
        bar's axis. The automatic bar's profile came with the analysis; for
        any other axis it is measured again, in a child process (it needs
        the shape), and the estimate follows when it arrives."""
        if self._turn_runner is not None:
            self._turn_runner.cancel()
            self._turn_runner = None
        if cyl is self.stock_controls._auto_cyl:
            self.results["turning"] = self._auto_turning
            return
        if not self._current_path:
            return
        runner = AnalysisRunner(self)
        runner.progress.connect(self.statusBar().showMessage)
        runner.finished.connect(lambda tg, c=cyl: self._on_turning_measured(c, tg))
        runner.failed.connect(lambda title, msg: self.statusBar().showMessage(
            f"Turned profile not re-measured: {title}", 8000))
        runner.done.connect(lambda r=runner: self._on_turning_done(r))
        self._turn_runner = runner
        runner.start_turning(self._current_path, cyl, self.results.get("holes"))

    def _on_turning_measured(self, cyl, turning_geo) -> None:
        if self.results.get("cylinder") is not cyl:
            return                          # the bar has moved again since
        self.results["turning"] = turning_geo
        self._refresh_feature_table()
        self._refresh_cycle_time()
        self.statusBar().showMessage("Turned profile measured about the new bar axis.", 6000)

    def _on_turning_done(self, runner) -> None:
        if self._turn_runner is runner:
            self._turn_runner = None

    def turning_busy(self) -> bool:
        return self._turn_runner is not None

    # ------------------------------------------------------- orientation
    def _on_pick_requested(self, mode: str):
        """The stock panel wants a face clicked in the 3D view."""
        self._pick_mode = mode
        if not mode or not self.results:
            self.viewer.set_click_handler(None)
            self.viewer.set_picking_cursor(False)
            if not mode:
                self.statusBar().clearMessage()
            return
        if mode == "stock" and not (self.chk_show_box.isChecked()
                                    or self.chk_show_stock.isChecked()):
            self.chk_show_stock.setChecked(True)       # something to click on
        self.viewer.set_click_handler(self._on_view_click)
        self.viewer.set_picking_cursor(True)

    def _part_face_labels(self):
        """Which CAD face each triangle belongs to, worked out on the first
        pick and kept for the part."""
        tris = self.results["render_triangles"]
        cached = getattr(self, "_face_labels", None)
        if cached is None or cached[0] is not tris:
            cached = (tris, orientation.face_labels(tris, len(self.results["render_points"])))
            self._face_labels = cached
        return cached[1]

    def _on_view_click(self, x, y):
        mode = getattr(self, "_pick_mode", "")
        if not self.results or not mode:
            return
        if mode == "part":
            tri = self.viewer.pick_part_triangle(x, y)
            if tri is None:
                self.stock_controls.pick_problem("That missed the part.")
                return
            labels = self._part_face_labels()
            face = orientation.planar_face(self.results["render_points"],
                                           self.results["render_triangles"], labels, tri)
            if face is None:
                self.stock_controls.pick_problem("That face is curved.")
                return
            self.viewer.highlight_triangles(np.flatnonzero(labels == labels[tri]), tag="part")
            self.stock_controls.set_part_face(face[0])
            return

        rect = self.stock_controls.rect_box()
        if rect is None:
            return
        origin, direction = self.viewer.pick_ray(x, y)
        stock = self.results.get("stock")
        extents = np.asarray(rect.extents_mm, dtype=float)
        if (self.chk_show_stock.isChecked() and stock is not None
                and stock.kind == "rectilinear"):
            extents = stock_box_extents(stock, rect)
        hit = orientation.ray_box_face(origin, direction, rect, extents)
        if hit is None:
            self.stock_controls.pick_problem("That missed the stock.")
            return
        k, sign = hit
        self.viewer.highlight_quad(
            orientation.box_face_corners(rect.center_mm, rect.axes, extents, k, sign),
            tag="stock")
        self.stock_controls.set_stock_face(k)

    def _on_material_changed(self, _text: str):
        if self.results:
            self._refresh_weight_labels()
            self._refresh_feature_table()
            self._refresh_cycle_time()

    def _refresh_bounding_box_labels(self, unit, len_suffix, vol_suffix):
        """Shows only the selected box -- whichever fit has the smaller
        volume."""
        rect_box = self.results["rect_box"]
        cyl = self.results["cylinder"]
        primary = self.results["primary"]

        if primary == "cylindrical" and cyl is not None:
            self.lbl_bbox_kind.setText("Cylindrical (round stock)")
            diam = step_io.mm_to_display(2 * cyl.radius_mm, unit)
            length = step_io.mm_to_display(cyl.height_mm, unit)
            self.lbl_bbox_dims.setText(f"⌀{diam:.4g} x {length:.4g} {len_suffix}")
            self.lbl_bbox_volume.setText(
                f"{step_io.mm3_to_display(cyl.volume_mm3, unit):.4g} {vol_suffix}")
        else:
            suffix = "" if cyl is not None else "  (cylinder fit disabled)"
            self.lbl_bbox_kind.setText(f"Rectilinear (plate/block){suffix}")
            dims = sorted(rect_box.extents_mm, reverse=True)
            d = [step_io.mm_to_display(x, unit) for x in dims]
            self.lbl_bbox_dims.setText(f"{d[0]:.4g} x {d[1]:.4g} x {d[2]:.4g} {len_suffix}")
            self.lbl_bbox_volume.setText(
                f"{step_io.mm3_to_display(rect_box.volume_mm3, unit):.4g} {vol_suffix}")

    def _refresh_stock_labels(self):
        """Stock is always reported in inches -- the allowance rules
        (+2.00", +0.25" rounded up to the next 1/4") are imperial by
        definition, so showing them in mm would obscure them."""
        stock = self.results.get("stock")
        if stock is None:
            for col in range(3):
                self.stock_table.setItem(0, col, QTableWidgetItem("—"))
            self.lbl_stock_volume.setText("—")
            self.stock_table.setToolTip("—")
            return

        for col, (_label, value) in enumerate(stock.table_rows()):
            item = QTableWidgetItem(value)
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.stock_table.setItem(0, col, item)

        if stock.is_extrusion:
            p = stock.profile
            self.stock_table.setToolTip(
                f'Extrusion: {p.describe()}, section {p.area_in2:.3f} in² '
                f'({"as listed in the standard" if p.listed_area_in2 else "sharp corners"}). '
                f'OD/ST and W are the profile\'s outer envelope. '
                f'L: minimum {stock.min_length_in:.3f}", recommended '
                f'{stock.recommended_length_in:.3f}" (part + 1.00", rounded up to the '
                f'next 1/4").')
        elif stock.basis == stock_mod.MINIMUM:
            self.stock_table.setToolTip(
                "Minimum: the part's own bounding box, with no cleanup or "
                "work-holding allowance. Chosen in Part Inputs.")
        elif stock.basis == stock_mod.CUSTOM:
            self.stock_table.setToolTip("Custom size, entered in Part Inputs.")
        elif stock.is_round:
            self.stock_table.setToolTip(
                'Barstock. OD: part diameter + 0.125", rounded up to the next 1/4". '
                'L: part length + 1.00", rounded up to the next 1/4". '
                'W is blank — a round bar has no second cross dimension.')
        elif stock.is_sheet:
            self.stock_table.setToolTip(
                f'Sheet (part is ≤ {stock_mod.SHEET_THICKNESS_MAX_IN:.3f}" thick). '
                'ST is the part thickness as-is: sheet is bought to gauge and the '
                'faces are not machined. W and L: part + 1.00", rounded up to the '
                'next 1/4".')
        else:
            self.stock_table.setToolTip(
                'Plate. ST: part thickness + 0.125" for cleanup, rounded up to the '
                'next 1/4". W and L: part + 1.00" for work-holding, rounded up to '
                'the next 1/4".')
        self.lbl_stock_volume.setText(f"{stock.volume_in3:,.3f} in³")

    def _refresh_hole_table(self, unit: str, len_suffix: str):
        groups = self.results["hole_groups"]
        self.hole_table.setRowCount(len(groups))
        for row, g in enumerate(groups):
            values = [
                f"{step_io.mm_to_display(g['nominal_diameter_mm'], unit):.4g} {len_suffix}",
                str(g["hole_count"]),
                f"{step_io.mm_to_display(g['max_depth_mm'], unit):.4g} {len_suffix}",
                f"{step_io.mm_to_display(g.get('avg_depth_mm', 0.0), unit):.4g} {len_suffix}",
            ]
            for col, text in enumerate(values):
                self.hole_table.setItem(row, col, QTableWidgetItem(text))
        self.hole_table.resizeColumnsToContents()

    def _refresh_thin_wall_label(self):
        result = self.results.get("thin_wall")
        if result is None:
            return
        unit = self.display_unit
        len_suffix = "mm" if unit == "MM" else "in"
        if result.min_thickness_mm is None:
            self.lbl_thin_wall.setText("Scan ran, but found no opposing-surface hits (check the part / threshold).")
            return
        # p10 leads, because the minimum is an extreme-order statistic that
        # one bad ray can set. Showing both, in this order, makes it obvious
        # when the two disagree -- which is the signal to open the model and
        # look rather than to trust either number.
        p10_disp = step_io.mm_to_display(result.p10_thickness_mm, unit)
        med_disp = step_io.mm_to_display(result.median_thickness_mm, unit)
        min_disp = step_io.mm_to_display(result.min_thickness_mm, unit)
        thresh_disp = step_io.mm_to_display(result.threshold_mm, unit)
        rejected = ""
        if result.samples_rejected:
            rejected = (f"  ({result.samples_rejected} corner/grazing hit(s) "
                        f"discarded)")
        self.lbl_thin_wall.setText(
            f"{result.samples_checked} usable samples.{rejected}\n"
            f"P10 thickness {p10_disp:.3g} {len_suffix}, "
            f"median {med_disp:.3g} {len_suffix}, "
            f"thinnest single sample {min_disp:.3g} {len_suffix}.\n"
            f"{result.thin_area_fraction * 100:.1f}% of measured area is below "
            f"the {thresh_disp:.3g} {len_suffix} threshold."
        )
