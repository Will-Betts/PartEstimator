"""The stock the part is cut from: recommended, minimum, typed in, or an
extrusion profile cut to length (issues #2 and #4).

A self-contained widget: give it the part's fits with `set_part()`, and it
emits `stockChanged(StockSize)` whenever the choice changes. The main window
puts that StockSize in results["stock"], which is what the features, the
cycle-time estimate, the stock weight and the 3D overlay all read.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from partestimator.machining.geometry import extrusions as ex
from partestimator.machining.geometry import stock as st

PLATE, ROUND, EXTRUSION = "Plate / block", "Round bar", "Extrusion profile"
KIND_OF = {PLATE: "rectilinear", ROUND: "cylindrical"}

BASIS_TEXT = {
    st.RECOMMENDED: "Recommended size (the stock rules).",
    st.MINIMUM: "Minimum size: the part's own envelope, no allowance.",
    st.CUSTOM: "Custom size.",
}


def profile_picker():
    from partestimator.materials_db import token_match
    from partestimator.ui_common import HierarchicalPicker

    profiles = ex.load_profiles()
    groups = []
    for shape, label in ex.SHAPES.items():
        items = [(f"{p.shape}|{p.name}", p.name) for p in profiles if p.shape == shape]
        groups.append((label, items))

    def text(value: str) -> str:
        shape, name = value.split("|", 1)
        return f"{ex.SHAPES.get(shape, shape)} {shape} {name}"

    picker = HierarchicalPicker(groups, token_match, text,
                                placeholder="Search profiles, e.g. angle 2 x 2 or tube 1",
                                item_tooltips=False)
    picker.group_combo.setToolTip("Profile shape")
    picker.item_combo.setToolTip("Size (inches)")
    return picker


def _spin(tip: str) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setDecimals(3)
    s.setRange(0.0, 1000.0)
    s.setSingleStep(0.125)
    s.setSuffix(" in")
    # Recompute when the number is committed (Enter, Tab, arrows), not on
    # every keystroke while it is being typed.
    s.setKeyboardTracking(False)
    s.setToolTip(tip)
    return s


class StockControls(QWidget):
    stockChanged = Signal(object)      # StockSize

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rect = None
        self._cyl = None
        self._part_volume = None
        self._points = None
        self._stock: st.StockSize | None = None
        self._updating = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        row = QHBoxLayout()
        row.addWidget(QLabel("Stock type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems([PLATE, ROUND, EXTRUSION])
        self.type_combo.setToolTip(
            "What the part is cut from. Opens as recommended by the stock rules;\n"
            "change it to whatever the NC programmer will actually use.")
        row.addWidget(self.type_combo, 1)
        lay.addLayout(row)

        self.profile = profile_picker()
        lay.addWidget(self.profile)

        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        self.lbl_a, self.lbl_b, self.lbl_l = QLabel("OD/ST"), QLabel("W"), QLabel("L")
        self.spin_a = _spin("Thickness (plate) or diameter (bar).")
        self.spin_b = _spin("Width.")
        self.spin_l = _spin("Length.")
        for col, (lbl, spin) in enumerate(((self.lbl_a, self.spin_a), (self.lbl_b, self.spin_b),
                                           (self.lbl_l, self.spin_l))):
            lbl.setAlignment(Qt.AlignCenter)
            grid.addWidget(lbl, 0, col)
            grid.addWidget(spin, 1, col)
        lay.addLayout(grid)

        row = QHBoxLayout()
        self.btn_recommended = QPushButton("Snap to recommended")
        self.btn_recommended.setToolTip(
            "Plate and bar: the stock rules (cleanup and work-holding allowance,\n"
            "rounded up to the next 1/4\").\n"
            "Extrusion: the recommended length.")
        self.btn_minimum = QPushButton("Snap to minimum")
        self.btn_minimum.setToolTip(
            "Plate and bar: the part's own bounding box, no allowance.\n"
            "Extrusion: the part's length along the profile.")
        row.addWidget(self.btn_recommended)
        row.addWidget(self.btn_minimum)
        row.addStretch(1)
        lay.addLayout(row)

        self.lbl_status = QLabel("Open a part to size its stock.")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setTextFormat(Qt.RichText)
        lay.addWidget(self.lbl_status)

        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        self.profile.currentTextChanged.connect(lambda _v: self._on_profile_changed())
        for s in (self.spin_a, self.spin_b, self.spin_l):
            s.valueChanged.connect(lambda _v: self._on_dims_edited())
        self.btn_recommended.clicked.connect(lambda: self.snap(st.RECOMMENDED))
        self.btn_minimum.clicked.connect(lambda: self.snap(st.MINIMUM))

        self._show_for_type()
        self.setEnabled(False)

    # -- public ------------------------------------------------------------

    def stock(self) -> st.StockSize | None:
        return self._stock

    def set_part(self, rect_box, cylinder, primary: str, emit: bool = True,
                 part_volume_in3: float | None = None, points_mm=None, triangles=None):
        """A new part: back to the recommended stock of the best-fitting kind.
        `points_mm`, the part's mesh vertices, lets an extrusion profile be
        checked against the part's real section and turned to match it."""
        self._rect, self._cyl = rect_box, cylinder
        self._part_volume = part_volume_in3
        self._points = ex._sample(ex.surface_samples(points_mm, triangles))
        self.setEnabled(rect_box is not None or cylinder is not None)
        round_ok = cylinder is not None
        self.type_combo.model().item(1).setEnabled(round_ok)
        kind = ROUND if (primary == "cylindrical" and round_ok) else PLATE
        self._updating = True
        self.type_combo.setCurrentText(kind)
        self._updating = False
        self._show_for_type()
        stock = st.snap(KIND_OF[kind], st.RECOMMENDED, rect_box, cylinder)
        self._apply(stock, emit=emit)
        return stock

    def clear(self) -> None:
        self._rect = self._cyl = self._stock = self._part_volume = self._points = None
        self.setEnabled(False)
        self.lbl_status.setText("Open a part to size its stock.")

    def snap(self, basis: str) -> None:
        kind = self.type_combo.currentText()
        if kind == EXTRUSION:
            if self._stock is None or not self._stock.is_extrusion:
                return
            length = (self._stock.recommended_length_in if basis == st.RECOMMENDED
                      else self._stock.min_length_in)
            self._apply(self._extrusion(length))
            return
        self._apply(st.snap(KIND_OF[kind], basis, self._rect, self._cyl))

    # -- reacting to the user ----------------------------------------------

    def _on_type_changed(self, kind: str) -> None:
        if self._updating:
            return
        self._show_for_type()
        if kind == EXTRUSION:
            best = self._best_profile()
            if best is not None:
                self._updating = True
                self.profile.setCurrentText(f"{best.shape}|{best.name}")
                self._updating = False
            self._apply(self._extrusion(None))
        else:
            self._apply(st.snap(KIND_OF[kind], st.RECOMMENDED, self._rect, self._cyl))

    def _on_profile_changed(self) -> None:
        if self._updating or self.type_combo.currentText() != EXTRUSION:
            return
        self._apply(self._extrusion(None))

    def _on_dims_edited(self) -> None:
        if self._updating or self._stock is None:
            return
        kind = self.type_combo.currentText()
        a, b, length = self.spin_a.value(), self.spin_b.value(), self.spin_l.value()
        if kind == EXTRUSION:
            self._apply(self._extrusion(length), reflect=False)
            return
        if kind == ROUND:
            stock = st.custom_cylindrical(a, length)
        else:
            stock = st.custom_rectilinear(a, b, length)
        self._apply(st.classify_basis(stock, self._rect, self._cyl), reflect=False)

    # -- internals ---------------------------------------------------------

    def _current_profile(self):
        value = self.profile.currentText()
        if "|" not in value:
            return None
        return ex.find(*value.split("|", 1))

    def _extrusion(self, length_in):
        prof = self._current_profile()
        if prof is None:
            return None
        return ex.extrusion_stock(prof, self._rect, self._cyl, length_in=length_in,
                                  part_volume_in3=self._part_volume, points_mm=self._points)

    def _best_profile(self):
        """The profile that fits this part with the least material."""
        best, best_vol = None, None
        for prof in ex.load_profiles():
            s = ex.extrusion_stock(prof, self._rect, self._cyl,
                                   part_volume_in3=self._part_volume, points_mm=self._points)
            if (s is not None and s.fits and s.enough_material
                    and (best_vol is None or s.volume_in3 < best_vol)):
                best, best_vol = prof, s.volume_in3
        return best

    def _show_for_type(self) -> None:
        kind = self.type_combo.currentText()
        self.profile.setVisible(kind == EXTRUSION)
        if kind == ROUND:
            self.lbl_a.setText("Diameter")
            self.lbl_b.setText("")
            self.lbl_l.setText("Length")
        elif kind == EXTRUSION:
            self.lbl_a.setText("Profile A")
            self.lbl_b.setText("Profile B")
            self.lbl_l.setText("Length")
        else:
            self.lbl_a.setText("Thickness")
            self.lbl_b.setText("Width")
            self.lbl_l.setText("Length")
        self.spin_b.setVisible(kind != ROUND)
        # A profile's section is fixed by the catalog; only its length is chosen.
        self.spin_a.setEnabled(kind != EXTRUSION)
        self.spin_b.setEnabled(kind != EXTRUSION)

    def _apply(self, stock, emit: bool = True, reflect: bool = True) -> None:
        if stock is None:
            return
        self._stock = stock
        if reflect:
            self._updating = True
            self.spin_a.setValue(stock.thickness_or_diameter_in)
            self.spin_b.setValue(stock.width_in or 0.0)
            self.spin_l.setValue(stock.length_in)
            self._updating = False
        self._update_status()
        if emit:
            self.stockChanged.emit(stock)

    def _update_status(self) -> None:
        s = self._stock
        if s is None:
            return
        lines = []
        if s.is_extrusion:
            p = s.profile
            lines.append(f"<b>{p.describe()}</b>, section {p.area_in2:.3f} in&sup2;.")
            lines.append(f'Length: minimum {s.min_length_in:.3f}", '
                         f'recommended {s.recommended_length_in:.3f}".')
            if s.section_inside is not None and s.enough_material:
                lines.append("<span style='color:#555'>The part lies inside the profile's "
                             "section.</span>")
        else:
            lines.append(BASIS_TEXT.get(s.basis, ""))
        for w in st.fit_warnings(s, self._rect, self._cyl):
            lines.append(f"<span style='color:#b91c1c'>&#9888; {w}</span>")
        self.lbl_status.setText("<br>".join(lines))
