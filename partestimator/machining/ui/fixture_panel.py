"""Part Inputs: tabbed fixturing for one part.

The shop standard (Settings...) decides the tab size, channel widths, frame
and vise allowances; here the part can be taken off tabs, its sides swapped,
the vise put on another edge, and the channel width or tab count set by
hand. Shows what the plan came to, and exports the STEP file.
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from partestimator.machining.geometry import fixture


class FixtureControls(QWidget):
    changed = Signal()          # anything that changes the layout
    exportRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self.chk = QCheckBox("Tabbed fixturing")
        self.chk.setChecked(True)
        self.chk.setToolTip(
            "Cut the part out of its plate with a channel round its outline, held to\n"
            "the frame of left-over stock by tabs: the shop's standard procedure.\n"
            "Vertical in a 5-axis vise in one operation, or flat in two for thin or\n"
            "tall plate. Milled plate parts only; the standard is in Settings...")
        self.chk.toggled.connect(self._on_input)
        lay.addWidget(self.chk)

        # Everything below the checkbox: hidden while it is unchecked.
        self.details = QWidget()
        outer, lay = lay, QVBoxLayout(self.details)
        lay.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.details)

        self.lbl_plan = QLabel("")
        self.lbl_plan.setWordWrap(True)
        self.lbl_plan.setStyleSheet("color:#555;")
        lay.addWidget(self.lbl_plan)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.btn_flip = QPushButton("Flip sides")
        self.btn_flip.setCheckable(True)
        self.btn_flip.setToolTip("Swap side 1 and side 2 (flat: top and bottom). Side 1 opens\n"
                                 "as the side with more flat area facing it.")
        self.btn_flip.toggled.connect(self._on_input)
        row.addWidget(self.btn_flip)
        self.clamp = QComboBox()
        self.clamp.addItems([e.capitalize() for e in fixture.CLAMP_EDGES])
        self.clamp.setToolTip("Which edge of the blank goes in the vise (vertical procedure).\n"
                              "A long edge keeps the stick-out above the jaws lowest.")
        self.clamp.currentIndexChanged.connect(self._on_input)
        row.addWidget(self.clamp, 1)
        form.addRow("Setup:", row)

        row = QHBoxLayout()
        self.channel = QDoubleSpinBox()
        self.channel.setDecimals(3)
        self.channel.setRange(0.0, 2.0)
        self.channel.setSingleStep(0.125)
        self.channel.setSuffix(" in")
        self.channel.setSpecialValueText("auto")
        self.channel.setToolTip("Channel width (the profiling cutter). 'auto' takes it from the\n"
                                "table in Settings by the depth cut from one side.")
        self.channel.valueChanged.connect(self._on_input)
        row.addWidget(QLabel("Channel"))
        row.addWidget(self.channel)
        self.tabs = QSpinBox()
        self.tabs.setRange(0, 50)
        self.tabs.setSpecialValueText("auto")
        self.tabs.setToolTip("Number of tabs. 'auto': one per tab spacing of outline (Settings),\n"
                             "at least the minimum.")
        self.tabs.valueChanged.connect(self._on_input)
        row.addWidget(QLabel("Tabs"))
        row.addWidget(self.tabs)
        form.addRow(row)
        lay.addLayout(form)

        self.lbl_warn = QLabel("")
        self.lbl_warn.setWordWrap(True)
        self.lbl_warn.setStyleSheet("color:#b35a00;")
        self.lbl_warn.setVisible(False)
        lay.addWidget(self.lbl_warn)

        self.btn_export = QPushButton("Export fixture STEP…")
        self.btn_export.setToolTip(
            "One STEP file with the stock, the stock after side 1 (or op 1), the part\n"
            "held in its frame by the tabs, and the part, overlaid in the stock's frame:\n"
            "origin at the stock corner on the clamped edge, Z out of side 1.")
        self.btn_export.clicked.connect(self.exportRequested)
        lay.addWidget(self.btn_export)
        self._updating = False
        self.show_plan(None, "")

    # -- state ---------------------------------------------------------------

    def wanted(self) -> bool:
        return self.chk.isChecked()

    def options(self) -> dict:
        """plan() keyword options from the controls."""
        return {"flipped": self.btn_flip.isChecked(),
                "clamp_edge": fixture.CLAMP_EDGES[max(self.clamp.currentIndex(), 0)],
                "channel_width_in": self.channel.value() or None,
                "tab_count": self.tabs.value() or None}

    def reset(self) -> None:
        """A new part: the standard's defaults, overrides cleared."""
        self._updating = True
        self.chk.setChecked(fixture.standard().enabled)
        self.btn_flip.setChecked(False)
        self.clamp.setCurrentIndex(0)
        self.channel.setValue(0.0)
        self.tabs.setValue(0)
        self._updating = False

    def _on_input(self, *_args) -> None:
        if not self._updating:
            self.changed.emit()

    # -- showing the plan ----------------------------------------------------

    def show_plan(self, plan, why_not: str = "") -> None:
        on = plan is not None
        self.details.setVisible(self.chk.isChecked())
        # The clamping edge stays editable when flat: a short edge's stick-out
        # can be what laid it flat, and the long edge puts it back on edge.
        for w in (self.btn_flip, self.clamp, self.channel, self.tabs, self.btn_export):
            w.setEnabled(on)
        if not on:
            self.lbl_plan.setText(why_not)
            self.lbl_plan.setVisible(bool(why_not))
            self.lbl_warn.setVisible(False)
            return
        how = ("Vertical in the vise, one operation" if plan.procedure == fixture.VERTICAL
               else "Flat, two operations (op 2 faces the floor off)" if plan.operations > 1
               else "Flat, one operation")
        self.lbl_plan.setText(
            f"{how} — {plan.why}. {plan.tab_count} tabs, "
            f"{plan.channel_width_in:.3f}\" channel.")
        self.lbl_plan.setVisible(True)
        self.lbl_warn.setText("\n".join("⚠ " + w for w in plan.warnings))
        self.lbl_warn.setVisible(bool(plan.warnings))
