"""The "Change part orientation" window.

A small tool window that stays open beside the 3D view, so faces can be
clicked in the view while it is up. Two things, in one place:

  faces    click a flat face of the part, then the face of the stock it
           should lie against; the two are made parallel
  rotate   turn the part about the stock's X, Y or Z axis, in degrees

On round bar there is one thing to set: which way the bar runs. Click a flat
face of the part and the bar's axis is set square to it, so that face lies
parallel to the bar's ends. Turning about the axes does not apply.

It holds only widgets. StockControls owns the orientation and drives it.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

AXES = (("X", "length"), ("Y", "width"), ("Z", "thickness"))


def _angle_spin(axis: str, along: str) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setDecimals(1)
    s.setRange(-180.0, 180.0)
    s.setSingleStep(1.0)
    s.setWrapping(True)
    s.setSuffix("°")
    s.setKeyboardTracking(False)
    s.setToolTip(f"Turn the part about the stock's {axis} axis (along its {along}).\n"
                 "Right-hand rule. The stock is refitted round the part.")
    return s


class OrientationDialog(QDialog):
    startOver = Signal()
    automatic = Signal()
    closed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Tool)
        self.setWindowTitle("Part orientation")
        lay = QVBoxLayout(self)

        faces = self.faces = QGroupBox("Make two faces parallel")
        fl = QVBoxLayout(faces)
        self.lbl_step = QLabel()
        self.lbl_step.setWordWrap(True)
        self.lbl_step.setTextFormat(Qt.RichText)
        self.lbl_step.setMinimumWidth(300)
        fl.addWidget(self.lbl_step)
        row = QHBoxLayout()
        self.btn_start_over = QPushButton("Start over")
        self.btn_start_over.setToolTip("Forget the faces clicked so far.")
        row.addWidget(self.btn_start_over)
        row.addStretch(1)
        fl.addLayout(row)
        lay.addWidget(faces)

        turn = self.turn = QGroupBox("Rotate part")
        grid = QGridLayout(turn)
        self.spins = []
        for col, (axis, along) in enumerate(AXES):
            lbl = QLabel(f"{axis}  ({along})")
            lbl.setAlignment(Qt.AlignCenter)
            spin = _angle_spin(axis, along)
            grid.addWidget(lbl, 0, col)
            grid.addWidget(spin, 1, col)
            self.spins.append(spin)
        self.spin_rx, self.spin_ry, self.spin_rz = self.spins
        lay.addWidget(turn)

        row = QHBoxLayout()
        self.btn_automatic = QPushButton("Automatic")
        self.btn_automatic.setToolTip("Back to the smallest box round the part.")
        self.btn_close = QPushButton("Done")
        self.btn_close.setDefault(True)
        row.addWidget(self.btn_automatic)
        row.addStretch(1)
        row.addWidget(self.btn_close)
        lay.addLayout(row)

        self.btn_start_over.clicked.connect(self.startOver.emit)
        self.btn_automatic.clicked.connect(self.automatic.emit)
        self.btn_close.clicked.connect(self.close)

    def set_round(self, round_bar: bool) -> None:
        """Round bar: only the face square to the bar's axis is picked."""
        self.faces.setTitle("Set a face square to the bar" if round_bar
                            else "Make two faces parallel")
        self.turn.setVisible(not round_bar)
        self.btn_automatic.setToolTip(
            "Back to the smallest bar round the part." if round_bar
            else "Back to the smallest box round the part.")
        self.adjustSize()

    def set_step(self, html: str) -> None:
        self.lbl_step.setText(html)

    def closeEvent(self, event):  # noqa: N802 -- Qt override
        self.closed.emit()
        super().closeEvent(event)
