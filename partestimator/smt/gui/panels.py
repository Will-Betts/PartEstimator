"""The controls, and the numbers you came for.

DESIGN NOTE: WHAT IS ON SCREEN, AND WHAT IS NOT
------------------------------------------------
v9 put eleven controls in front of you at all times -- units, blank placement,
material, forming method, V-die, K-factor, thickness, grain, scrap, cost per kg,
press capacity -- plus five tabs. Every one of them is a real thing somebody
wants, and together they made the window unreadable and the job unclear.

Four are visible now, because these are the four that change the answer often
enough to earn permanent space:

    units          you work in inches and the file might say millimetres
    material       drives the K curve, the weight and the minimum bend radius
    K-factor       the single number that decides blank length
    thickness      the one that goes wrong, and the manual override for it

Everything else moved behind Advanced, closed by default. Nothing was removed --
V-die, press capacity, cost, grain, method and blank placement all still work
and still affect the report exactly as before. They are simply not in the way of
the common case, which is: open a file, read the blank size.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..materials import DEFAULT_MATERIAL, MATERIALS, STANDARD_V_DIES

K_MIN, K_MAX = 0.20, 0.50
K_STEPS = 300


class SimpleControls(QWidget):
    """The four that matter, and a button for the fifth thing that goes wrong."""

    changed = Signal()          # fast path: recompute from cached detection
    rescan = Signal()           # slow path: needs the file read again
    pick_sheet = Signal()       # arm face picking in the 3D view

    def __init__(self, parent=None):
        super().__init__(parent)
        self._unit = None

        box = QGroupBox("Settings")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        self.units = QComboBox()
        self.units.addItems(["auto (from file)", "inch", "mm"])
        self.units.setToolTip(
            "What you read on screen and in the report.\n\n"
            "The DXF is ALWAYS written in inches, whatever this says.\n\n"
            "Geometry is always computed in millimetres internally; this "
            "changes nothing about the answer, only how it is written down.\n\n"
            "'auto' follows whatever the STEP file declares."
        )
        self.units.currentTextChanged.connect(lambda _: self.changed.emit())
        form.addRow("Units", self.units)

        self.material = QComboBox()
        from partestimator.ui_common import fill_material_combo

        fill_material_combo(self.material, list(MATERIALS))
        self.material.setCurrentText(DEFAULT_MATERIAL)
        self.material.setToolTip(
            "Sets the K-factor curve, the density used for weight, the minimum "
            "bend radius check and the tonnage calculation.\n\n"
            "If yours is not listed, pick the closest and check the bend radius "
            "warning yourself."
        )
        self.material.currentTextChanged.connect(lambda _: self.changed.emit())
        form.addRow("Material", self.material)

        # --- K factor -------------------------------------------------
        krow = QHBoxLayout()
        self.k_auto = QCheckBox("auto")
        self.k_auto.setChecked(True)
        self.k_auto.setToolTip(
            "Interpolate K from the material curve as K = f(r/t), which is what "
            "a published table does and is right within a few percent for "
            "normal work.\n\n"
            "K is not a constant: it rises from about 0.33 at a tight radius "
            "toward 0.5 as the radius grows. Untick only if you have measured "
            "your own."
        )
        self.k_auto.toggled.connect(self._k_auto_toggled)
        krow.addWidget(self.k_auto)

        self.k_slider = QSlider(Qt.Horizontal)
        self.k_slider.setRange(0, K_STEPS)
        self.k_slider.setValue(int((0.44 - K_MIN) / (K_MAX - K_MIN) * K_STEPS))
        self.k_slider.setEnabled(False)
        self.k_slider.valueChanged.connect(self._k_slider_moved)
        krow.addWidget(self.k_slider, 1)

        self.k_label = QLabel("0.440")
        self.k_label.setMinimumWidth(40)
        krow.addWidget(self.k_label)
        form.addRow("K-factor", krow)

        # --- thickness ------------------------------------------------
        trow = QHBoxLayout()
        self.thickness_auto = QCheckBox("auto")
        self.thickness_auto.setChecked(True)
        self.thickness_auto.toggled.connect(self._thickness_auto_toggled)
        trow.addWidget(self.thickness_auto)

        self.thickness = QDoubleSpinBox()
        self.thickness.setDecimals(4)
        self.thickness.setRange(0.004, 0.5)
        self.thickness.setSingleStep(0.005)
        self.thickness.setEnabled(False)
        self.thickness.valueChanged.connect(lambda _: self.rescan.emit())
        trow.addWidget(self.thickness, 1)
        form.addRow("Thickness", trow)

        self.btn_pick = QPushButton("Pick the sheet face...")
        self.btn_pick.setToolTip(
            "Click the face in the 3D view that really is the sheet surface.\n\n"
            "Thickness is then measured from that face to whatever lies one "
            "material behind it, and everything else is re-derived from it.\n\n"
            "Use this when the tool has measured the wrong pair of faces: a "
            "boss, a weld prep or a mounting pad can leave more than one "
            "defensible answer, and you can see the model."
        )
        self.btn_pick.clicked.connect(lambda: self.pick_sheet.emit())
        form.addRow("", self.btn_pick)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(box)

    # -- internals -------------------------------------------------------

    def _k_auto_toggled(self, on: bool) -> None:
        self.k_slider.setEnabled(not on)
        self.k_label.setText("auto" if on else f"{self.k_value():.3f}")
        self.changed.emit()

    def _k_slider_moved(self, _v: int) -> None:
        self.k_label.setText(f"{self.k_value():.3f}")
        self.changed.emit()

    def _thickness_auto_toggled(self, on: bool) -> None:
        self.thickness.setEnabled(not on)
        self.rescan.emit()

    def k_value(self) -> float:
        return K_MIN + (K_MAX - K_MIN) * self.k_slider.value() / K_STEPS

    # -- read / write ----------------------------------------------------

    def apply_to(self, opts):
        u = self.units.currentText()
        opts.display_units = (
            "auto" if u.startswith("auto") else ("in" if u == "inch" else "mm")
        )
        opts.material = self.material.currentText()
        opts.k_factor = None if self.k_auto.isChecked() else self.k_value()
        opts.thickness = (
            None if self.thickness_auto.isChecked() else self.thickness_mm()
        )
        return opts

    def set_unit(self, unit) -> None:
        """Retune the thickness box for whichever unit is showing."""
        self._unit = unit
        inch = unit.name == "in"
        self.thickness.blockSignals(True)
        self.thickness.setDecimals(4 if inch else 3)
        self.thickness.setSingleStep(0.005 if inch else 0.1)
        # The upper bound is the same half inch the detector enforces, so the
        # control cannot ask for something the analysis will refuse.
        self.thickness.setRange(0.004 if inch else 0.1, 0.5 if inch else 12.7)
        self.thickness.setSuffix(f"  {unit.label}")
        self.thickness.blockSignals(False)

    def show_detected_thickness(self, t_mm: float) -> None:
        if self.thickness_auto.isChecked():
            self.thickness.blockSignals(True)
            self.thickness.setValue(self._to_display(t_mm))
            self.thickness.blockSignals(False)

    def _to_display(self, mm: float) -> float:
        return mm * self._unit.per_mm if self._unit else mm

    def thickness_mm(self) -> float:
        v = self.thickness.value()
        return v / self._unit.per_mm if self._unit else v


class AdvancedPanel(QGroupBox):
    """Everything that used to be in the way. Nothing was removed."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__("Advanced", parent)
        self.setCheckable(True)
        self.setChecked(False)
        form = QFormLayout(self)

        self.method = QComboBox()
        self.method.addItems(["air", "bottoming", "coining"])
        self.method.setToolTip(
            "Air bending springs back most and has the highest K; coining "
            "crushes the material into the die and has the lowest.\n\n"
            "If you are bending on a brake with a standard V-die and not "
            "hitting bottom, it is air bending."
        )
        form.addRow("Method", self.method)

        self.orient = QComboBox()
        self.orient.addItems(["minimum area", "as modelled"])
        self.orient.setToolTip(
            "Minimum area turns the blank until its tightest rectangle is "
            "square to the axes, which is what a material estimate is built "
            "on. A grain direction always overrides it."
        )
        form.addRow("Blank placement", self.orient)

        self.v_die = QComboBox()
        self.v_die.addItem("auto")
        for v in STANDARD_V_DIES:
            self.v_die.addItem(f"{v:g}")
        form.addRow("V-die", self.v_die)

        self.grain = QComboBox()
        self.grain.addItems(["unconstrained", "0", "90"])
        self.grain.setToolTip(
            "Sheet bends more readily across the rolling direction than along "
            "it. Setting a grain direction overrides minimum-area placement, "
            "because that constraint comes from the material."
        )
        form.addRow("Grain", self.grain)

        self.press_tons = QDoubleSpinBox()
        self.press_tons.setRange(0.0, 2000.0)
        self.press_tons.setSuffix("  t")
        self.press_tons.setSpecialValueText("none")
        form.addRow("Press capacity", self.press_tons)

        self.cost_per_kg = QDoubleSpinBox()
        self.cost_per_kg.setRange(0.0, 1000.0)
        self.cost_per_kg.setDecimals(2)
        self.cost_per_kg.setSpecialValueText("none")
        form.addRow("Cost per kg", self.cost_per_kg)

        for w in (self.method, self.orient, self.v_die, self.grain):
            w.currentTextChanged.connect(lambda _: self.changed.emit())
        for w in (self.press_tons, self.cost_per_kg):
            w.valueChanged.connect(lambda _: self.changed.emit())
        self.toggled.connect(lambda _: self.changed.emit())

    def apply_to(self, opts):
        opts.method = self.method.currentText()
        opts.orient = (
            "min_area" if self.orient.currentText().startswith("minimum")
            else "as_modelled"
        )
        v = self.v_die.currentText()
        opts.v_die = None if v == "auto" else float(v)
        g = self.grain.currentText()
        opts.grain_deg = None if g.startswith("un") else int(g)
        opts.press_capacity_t = (
            self.press_tons.value() if self.press_tons.value() > 0 else None
        )
        opts.cost_per_kg = (
            self.cost_per_kg.value() if self.cost_per_kg.value() > 0 else None
        )
        return opts


class BlankPanel(QWidget):
    """The answer, in the largest type on the screen.

    This is what the tool is for. v9 put the blank size in a grid of eight
    equally weighted rows, where it read as one fact among many rather than as
    the reason you opened the file.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 2, 6, 6)
        lay.setSpacing(2)

        self.headline = QLabel("--")
        f = QFont()
        f.setPointSize(19)
        f.setBold(True)
        self.headline.setFont(f)
        self.headline.setStyleSheet("color:#1f2933;")
        lay.addWidget(self.headline)

        self.subhead = QLabel("")
        self.subhead.setStyleSheet("color:#555;")
        lay.addWidget(self.subhead)
        lay.addSpacing(6)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(3)
        self._fields = {}
        for row, (key, label) in enumerate((
            ("area", "Profile area"),
            ("weight", "Weight"),
            ("bends", "Bends"),
            ("features", "Features"),
            ("cut", "Cut length"),
            ("cost", "Material cost"),
        )):
            cap = QLabel(label)
            cap.setStyleSheet("color:#555;")
            val = QLabel("--")
            grid.addWidget(cap, row, 0)
            grid.addWidget(val, row, 1)
            self._fields[key] = val
        grid.setColumnStretch(1, 1)
        lay.addLayout(grid)
        lay.addStretch(1)

    def set_headline(self, blank: str, detail: str) -> None:
        self.headline.setText(blank)
        self.subhead.setText(detail)

    def set(self, key: str, text: str) -> None:
        if key in self._fields:
            self._fields[key].setText(text)

    def clear(self) -> None:
        self.headline.setText("--")
        self.subhead.setText("")
        for v in self._fields.values():
            v.setText("--")


class DetailTable(QTableWidget):
    """One table, three modes. v9 had three tables in three tabs."""

    def __init__(self, parent=None):
        super().__init__(0, 0, parent)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setAlternatingRowColors(True)
        self.horizontalHeader().setStretchLastSection(True)

    def _load(self, columns, rows) -> None:
        self.clear()
        self.setColumnCount(len(columns))
        self.setHorizontalHeaderLabels(columns)
        self.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, cell in enumerate(row):
                self.setItem(r, c, QTableWidgetItem(str(cell)))
        self.resizeColumnsToContents()
        self.horizontalHeader().setSectionResizeMode(
            max(len(columns) - 1, 0), QHeaderView.Stretch
        )

    def show_bends(self, rows, unit) -> None:
        # `BendRow.direction` is ALREADY a string -- "UP" or "DOWN" -- decided
        # in `report.bend_table`. v11 compared it to an integer here, which
        # raised TypeError, and because this runs from `_refresh` during open,
        # it aborted the whole open before the 3D view was ever shown. One
        # wrong comparison, and the window came up blank.
        conv, dp = unit.from_mm, unit.decimals
        self._load(
            ("#", "Angle", "Radius", "Dir", "Length", "K", "Allowance"),
            [
                (b.n, f"{b.angle_deg:.1f} deg", f"{conv(b.r_inner):.{dp}f}",
                 str(b.direction).lower(),
                 f"{conv(b.length):.{dp}f}", f"{b.k_factor:.3f}",
                 f"{conv(b.bend_allowance):.{dp}f}")
                for b in rows
            ],
        )

    def show_features(self, part, unit) -> None:
        from ..report import _group_label

        groups = list(getattr(part, "feature_groups", []))
        if not groups:
            self._load(("Feature",), [("no secondary features found",)])
            return
        self._load(
            ("Qty", "Feature"),
            [(g.count, _group_label(g, unit)) for g in groups],
        )

    def show_notes(self, warnings, notes) -> None:
        rows = [("warning", w) for w in warnings] + [("note", n) for n in notes]
        if not rows:
            rows = [("", "nothing to report")]
        self._load(("", "Message"), rows)


class ActivityLog(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        f = QFont("Consolas")
        f.setPointSize(9)
        self.setFont(f)
