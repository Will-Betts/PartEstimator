"""The stock the part is cut from: recommended, minimum, typed in, or an
extrusion profile cut to length (issues #2 and #4).

A self-contained widget: give it the part's fits with `set_part()`, and it
emits `stockChanged(StockSize)` whenever the choice changes. The main window
puts that StockSize in results["stock"], which is what the features, the
cycle-time estimate, the stock weight and the 3D overlay all read.
"""
from __future__ import annotations

import numpy as np
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

from partestimator.machining.geometry import bounding_box as bb
from partestimator.machining.geometry import extrusions as ex
from partestimator.machining.geometry import orientation as orient
from partestimator.machining.geometry import stock as st
from partestimator.machining.ui.orientation_dialog import OrientationDialog

PLATE, ROUND, EXTRUSION = "Plate / block", "Round bar", "Extrusion profile"
KIND_OF = {PLATE: "rectilinear", ROUND: "cylindrical"}

STEP_PART = "Click a <b>flat face of the part</b> in the 3D view."
STEP_STOCK = "Now click the <b>face of the stock</b> it should lie against."
STEP_BAR = ("Click a <b>flat face of the part</b> to set square to the bar: it ends up "
            "parallel to the bar's ends, and the bar runs along its normal.")

CUSTOM_GROUP = "Custom library"


def _value(p) -> str:
    return f"{p.shape}|{p.name}"


def profile_groups(custom=()):
    """The picker's groups: the user's own library first, then the catalog
    by shape, then one group per AND / AMS standard."""
    catalog, standard = ex.load_catalog(), ex.load_standard()
    groups = [(CUSTOM_GROUP, [(_value(p), f"{p.name}   {p.shape_label} {p.size_text}")
                              for p in custom])]
    for shape, label in ex.SHAPES.items():
        groups.append((label, [(_value(p), p.name) for p in catalog if p.shape == shape]))
    for spec, title in ex.STANDARDS.items():
        items = [(_value(p), f"{p.name}   {p.size_text}") for p in standard if p.spec == spec]
        groups.append((f"{spec} / {ex.ams_name(spec)}  {title}", items))
    return groups


def _search_text(p) -> str:
    if p.custom:
        return " ".join(("custom", p.name, p.shape_label, p.shape, p.size_text, p.notes))
    if p.is_standard:
        return " ".join((p.name, p.alias, ex.STANDARDS.get(p.spec, ""), p.shape_label,
                         p.size_text))
    return f"{p.shape_label} {p.shape} {p.name}"


def _display_text(p) -> str:
    """How a profile reads in the box: the shape and size, the part number
    first for a standard die."""
    if p.custom:
        return f"{p.name}  (custom {p.shape_label}, {p.size_text})"
    if p.is_standard:
        return f"{p.name}  {p.shape_label} {p.size_text}"
    return f"{p.shape_label} {p.name}"


def profile_picker(custom=()):
    from partestimator.materials_db import token_match
    from partestimator.ui_common import NestedPicker

    # Shared with refresh_picker(), which swaps the custom profiles in place.
    by_value = {_value(p): p for p in ex.load_catalog() + ex.load_standard() + tuple(custom)}

    def text(value: str) -> str:
        p = by_value.get(value)
        return value if p is None else _search_text(p)

    def shown(value: str) -> str:
        p = by_value.get(value)
        return value if p is None else _display_text(p)

    picker = NestedPicker(profile_groups(custom), token_match, text,
                          placeholder="Type a profile, e.g. angle 2 x 2, tube 1 "
                                      "or AND10134-1205",
                          display=shown)
    picker.setToolTip("Type to search by shape, size or part number, or open the list\n"
                      "for a shape or AND / AMS standard, then a size (inches).\n"
                      "(AMS101xx replaced AND101xx with the same dash numbers.)")
    picker.profiles_by_value = by_value
    return picker


def refresh_picker(picker, custom) -> None:
    """Offer a reloaded custom library, keeping the pick where it still exists."""
    by_value = picker.profiles_by_value
    for key in [k for k, p in by_value.items() if p.custom]:
        del by_value[key]
    by_value.update({_value(p): p for p in custom})
    picker.set_groups(profile_groups(custom))


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


def _rank(box) -> list[int]:
    """For each row of `box.axes`: 0 if it is the length, 1 the width, 2 the
    thickness -- the stock rules name a box's sides by size."""
    order = np.argsort(-np.asarray(box.extents_mm, dtype=float), kind="stable")
    rank = [0, 0, 0]
    for r, row in enumerate(order):
        rank[int(row)] = r
    return rank


class StockControls(QWidget):
    stockChanged = Signal(object)      # StockSize
    #: "part" or "stock": click that face in the 3D view; "" stops picking.
    pickRequested = Signal(str)
    #: The part was turned in its stock (a face pair applied, or reset).
    orientationChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # Set by the window: rect box -> the recommended PLATE for it, or
        # None for the plain rule. Tabbed fixturing grows the plate so a
        # frame stands outside the channel.
        self.recommend_plate = None
        self._rect = None
        self._auto_rect = None
        self._cyl = None
        self._part_volume = None
        self._points = None
        self._all_points = None
        self._stock: st.StockSize | None = None
        self._updating = False
        # Orientation: the frame the angles turn from, and the face pair.
        self._base_axes = None
        self._oriented = False
        self._how = ""
        self._part_normal = None
        self._stock_axis = None
        self._pick_step = ""           # "part", "stock" or "" (not picking)
        # Round bar keeps its own orientation: which way the bar's axis runs.
        self._auto_cyl = None
        self._cyl_oriented = False
        self._cyl_how = ""

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

        library = ex.custom_library()
        self.profile = profile_picker(library.profiles)
        lay.addWidget(self.profile)

        # The user's own profiles, kept in custom_extrusions.csv beside the
        # launcher: edit it in any spreadsheet or text editor, then reload.
        self.custom_row = QWidget()
        row = QHBoxLayout(self.custom_row)
        row.setContentsMargins(0, 0, 0, 0)
        self.btn_edit_custom = QPushButton("Edit custom library")
        self.btn_edit_custom.setToolTip(
            f"Open {ex.CUSTOM_PATH.name} (beside the launcher) to add your own dies\n"
            "and supplier sizes. It is created with instructions the first time.")
        self.btn_reload_custom = QPushButton("Reload")
        self.btn_reload_custom.setToolTip(f"Read {ex.CUSTOM_PATH.name} again after editing it.")
        row.addWidget(self.btn_edit_custom)
        row.addWidget(self.btn_reload_custom)
        row.addStretch(1)
        lay.addWidget(self.custom_row)
        self.lbl_custom = QLabel()
        self.lbl_custom.setWordWrap(True)
        self.lbl_custom.setTextFormat(Qt.RichText)
        lay.addWidget(self.lbl_custom)
        self._show_custom_status(library)

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

        # Problems only (stock too small, profile cannot hold the part).
        self.lbl_status = QLabel()
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setTextFormat(Qt.RichText)
        self.lbl_status.setVisible(False)
        lay.addWidget(self.lbl_status)

        self._build_orientation(lay)

        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        self.profile.currentTextChanged.connect(lambda _v: self._on_profile_changed())
        for s in (self.spin_a, self.spin_b, self.spin_l):
            s.valueChanged.connect(lambda _v: self._on_dims_edited())
        self.btn_recommended.clicked.connect(lambda: self.snap(st.RECOMMENDED))
        self.btn_minimum.clicked.connect(lambda: self.snap(st.MINIMUM))
        self.btn_edit_custom.clicked.connect(self.edit_custom_library)
        self.btn_reload_custom.clicked.connect(self.reload_custom_library)

        self._show_for_type()
        self.setEnabled(False)

    def _build_orientation(self, lay) -> None:
        """One button; the controls live in a tool window beside the view."""
        self.orient_row = QWidget()
        row = QHBoxLayout(self.orient_row)
        row.setContentsMargins(0, 0, 0, 0)
        self.btn_orient = QPushButton("Change part orientation…")
        self.btn_orient.setToolTip(
            "Make a face of the part parallel to a face of the stock, and/or turn\n"
            "the part about the stock's X, Y or Z axis.")
        row.addWidget(self.btn_orient)
        self.lbl_oriented = QLabel("set by hand")
        self.lbl_oriented.setStyleSheet("color: #555;")
        row.addWidget(self.lbl_oriented)
        row.addStretch(1)
        lay.addWidget(self.orient_row)

        self.orient_dialog = OrientationDialog(self)
        d = self.orient_dialog
        self.spin_rx, self.spin_ry, self.spin_rz = d.spin_rx, d.spin_ry, d.spin_rz
        self.btn_auto_orient = d.btn_automatic
        for spin in d.spins:
            spin.valueChanged.connect(lambda _v: self._on_angles_changed())
        d.startOver.connect(self._start_picking)
        d.automatic.connect(self.reset_orientation)
        d.closed.connect(self.cancel_picking)
        self.btn_orient.clicked.connect(self.open_orientation)
        self._show_orientation()

    # -- public ------------------------------------------------------------

    def stock(self) -> st.StockSize | None:
        return self._stock

    def rect_box(self):
        """The part's box in its current orientation (what the stock is
        sized from), or None."""
        return self._rect

    def cylinder(self):
        """The bar round the part in its current orientation (what round
        stock is sized from), or None."""
        return self._cyl

    def is_reoriented(self) -> bool:
        return self._cyl_oriented if self._is_round() else self._oriented

    def _is_round(self) -> bool:
        return self.type_combo.currentText() == ROUND

    def angles(self) -> tuple[float, float, float]:
        return (self.spin_rx.value(), self.spin_ry.value(), self.spin_rz.value())

    def open_orientation(self) -> None:
        """Show the orientation window and start picking faces."""
        d = self.orient_dialog
        if not d.isVisible():
            top = self.window()
            d.move(top.geometry().topLeft() + top.rect().center() / 2)
        d.set_round(self._is_round())
        d.show()
        d.raise_()
        self._start_picking()

    def picking(self) -> str:
        """What a click in the 3D view should pick: "part", "stock" or ""."""
        return self._pick_step

    def set_part_face(self, normal) -> None:
        """A flat face of the part was picked: its outward normal."""
        if self._is_round():
            self.square_to_bar(normal)
            return
        self._part_normal = orient._unit(normal)
        self._stock_axis = None
        self._set_step("stock")

    def set_stock_face(self, axis_index: int) -> None:
        """A stock face was picked: the row of `rect_box().axes` it faces.
        With a part face already picked, the two are made parallel."""
        if self._part_normal is None:
            return
        normal, self._part_normal = self._part_normal, None
        self.align(normal, int(axis_index))
        self._set_step("part")

    def cancel_picking(self) -> None:
        self._part_normal = self._stock_axis = None
        self._set_step("")

    def pick_problem(self, message: str) -> None:
        """A click that picked nothing usable: say so, and ask again."""
        step = (STEP_STOCK if self._pick_step == "stock"
                else STEP_BAR if self._is_round() else STEP_PART)
        self.orient_dialog.set_step(f"<span style='color:#b45309'>{message}</span> {step}")

    def align(self, part_normal, stock_axis: int) -> None:
        """Make the stock face along row `stock_axis` of the current box
        parallel to a part face with this normal."""
        if self._rect is None or self._all_points is None:
            return
        picked = _rank(self._rect)[stock_axis]
        frame = orient.align_axis_to_face(self._rect.axes, stock_axis, part_normal,
                                          self._all_points)
        self._base_axes = orient.sorted_frame(orient.box_in_frame(self._all_points, frame))
        # The stock's faces are named by size, so after the refit the part
        # face may lie against a different-sized stock face. Say so.
        landed = int(np.argmax(np.abs(self._base_axes @ orient._unit(part_normal))))
        self._how = (f"Part face parallel to the stock's "
                     f"{orient.STOCK_FACE_NAMES[landed]}")
        if landed != picked:
            self._how += (f" (you picked its {orient.STOCK_FACE_NAMES[picked]}, but the "
                          f"part's size that way makes it the "
                          f"{['length', 'width', 'thickness'][landed]})")
        self._how += "."
        self._oriented = True
        self.orient_dialog.set_step(self._how + "<br><br>" + STEP_PART)
        self._set_angles((0.0, 0.0, 0.0))
        self._reorient()

    def square_to_bar(self, normal) -> None:
        """Run the bar along this face normal: the face lies parallel to the
        bar's ends, and the bar is the tightest one round the part that way."""
        if self._all_points is None:
            return
        self._cyl = bb.cylinder_about(self._all_points, orient._unit(normal))
        self._cyl_oriented = True
        self._cyl_how = "Bar axis set square to the picked face (parallel to the bar's ends)."
        self.orient_dialog.set_step(self._cyl_how + "<br><br>" + STEP_BAR)
        self._show_orientation()
        self.orientationChanged.emit()
        self._restock()

    def reset_orientation(self) -> None:
        """Back to the automatic, smallest-volume fit."""
        self._part_normal = self._stock_axis = None
        if self._pick_step:
            self._start_picking()
        if self._is_round():
            self._cyl = self._auto_cyl
            self._cyl_oriented = False
            self._cyl_how = ""
            self._show_orientation()
            self.orientationChanged.emit()
            self._restock()
            return
        if self._auto_rect is None:
            return
        self._base_axes = orient.sorted_frame(self._auto_rect)
        self._oriented = False
        self._how = ""
        self._set_angles((0.0, 0.0, 0.0))
        self._rect = self._auto_rect
        self._show_orientation()
        self.orientationChanged.emit()
        self._restock()

    def set_part(self, rect_box, cylinder, primary: str, emit: bool = True,
                 part_volume_in3: float | None = None, points_mm=None, triangles=None,
                 extrusion=None):
        """A new part: back to the recommended stock of the best-fitting kind.
        `points_mm`, the part's mesh vertices, lets an extrusion profile be
        checked against the part's real section and turned to match it.
        `extrusion`, a (shape, name) the analysis found holds the part with
        less material than plate, opens it on that profile instead."""
        self._rect, self._cyl = rect_box, cylinder
        self._auto_rect, self._auto_cyl = rect_box, cylinder
        self._cyl_oriented = False
        self._cyl_how = ""
        self._part_volume = part_volume_in3
        self._points = ex._sample(ex.surface_samples(points_mm, triangles))
        # Every vertex, not a sample: a re-oriented box must still enclose
        # the whole part. Without them only the automatic box is on offer.
        self._all_points = (np.asarray(points_mm, dtype=float).reshape(-1, 3)
                            if points_mm is not None and len(points_mm) else None)
        self._base_axes = orient.sorted_frame(rect_box) if rect_box is not None else None
        self._oriented = False
        self._how = ""
        self._part_normal = self._stock_axis = None
        self.orient_dialog.close()
        self.cancel_picking()
        self._set_angles((0.0, 0.0, 0.0))
        self.btn_orient.setEnabled(self._all_points is not None
                                   and (rect_box is not None or cylinder is not None))
        self.setEnabled(rect_box is not None or cylinder is not None)
        round_ok = cylinder is not None
        self.type_combo.model().item(1).setEnabled(round_ok)
        kind = ROUND if (primary == "cylindrical" and round_ok) else PLATE
        profile = ex.find(*extrusion) if extrusion and kind == PLATE else None
        self._updating = True
        self.type_combo.setCurrentText(EXTRUSION if profile is not None else kind)
        if profile is not None:
            self.profile.setCurrentText(_value(profile))
        self._updating = False
        self._show_for_type()
        stock = (self._extrusion(None) if profile is not None
                 else self._snap(KIND_OF[kind], st.RECOMMENDED))
        self._apply(stock, emit=emit)
        return stock

    def set_stock(self, stock, emit: bool = True) -> None:
        """Show this stock (e.g. one saved in the tuning library) as if it had
        been picked here."""
        if stock is None:
            return
        kind = EXTRUSION if stock.is_extrusion else ROUND if stock.is_round else PLATE
        self._updating = True
        self.type_combo.setCurrentText(kind)
        if stock.is_extrusion and stock.profile is not None:
            self.profile.setCurrentText(_value(stock.profile))
        self._updating = False
        self._show_for_type()
        self._apply(stock, emit=emit)

    def refresh_allowances(self) -> None:
        """The stock allowances changed: a recommended stock is resized to
        the new rules, an extrusion at its recommended length is recut. A
        size typed by hand is kept."""
        s = self._stock
        if s is None:
            return
        if s.is_extrusion:
            keep = abs(s.length_in - s.recommended_length_in) > st.SAME_SIZE_TOL_IN
            self._apply(self._extrusion(s.length_in if keep else None))
        else:
            self._restock()

    def clear(self) -> None:
        self._rect = self._cyl = self._stock = self._part_volume = self._points = None
        self._auto_rect = self._all_points = self._base_axes = self._auto_cyl = None
        self._cyl_oriented = False
        self._part_normal = self._stock_axis = None
        self._oriented = False
        self.orient_dialog.close()
        self.cancel_picking()
        self.setEnabled(False)
        self.lbl_status.setText("")
        self.lbl_status.setVisible(False)

    def snap(self, basis: str) -> None:
        kind = self.type_combo.currentText()
        if kind == EXTRUSION:
            if self._stock is None or not self._stock.is_extrusion:
                return
            length = (self._stock.recommended_length_in if basis == st.RECOMMENDED
                      else self._stock.min_length_in)
            self._apply(self._extrusion(length))
            return
        self._apply(self._snap(KIND_OF[kind], basis))

    def edit_custom_library(self) -> None:
        """Open the custom library in the system's editor for .csv files,
        creating it from the template first if need be."""
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        try:
            path = ex.ensure_custom_file()
        except OSError as exc:
            self.lbl_custom.setText(f"<span style='color:#b91c1c'>&#9888; Cannot create "
                                    f"{ex.CUSTOM_PATH.name}: {exc}</span>")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        self.lbl_custom.setText(f"Editing {path.name}. Save it, then press Reload.")
        self.lbl_custom.setVisible(True)

    def reload_custom_library(self) -> None:
        library = ex.custom_library(reload=True)
        self._updating = True
        refresh_picker(self.profile, library.profiles)
        self._updating = False
        self._show_custom_status(library)
        if self.type_combo.currentText() == EXTRUSION and self._stock is not None:
            # The profile in use may have changed or gone.
            self._apply(self._extrusion(None))

    def _show_custom_status(self, library) -> None:
        if not library.exists:
            self.lbl_custom.setText("")
            self.lbl_custom.setVisible(False)
            return
        lines = []
        for err in library.errors[:5]:
            lines.append(f"<span style='color:#b45309'>&#9888; Skipped {err}</span>")
        if len(library.errors) > 5:
            lines.append(f"<span style='color:#b45309'>... and {len(library.errors) - 5} "
                         f"more rows skipped.</span>")
        self.lbl_custom.setText("<br>".join(lines))
        self.lbl_custom.setVisible(bool(lines))
        n = len(library.profiles)
        self.btn_reload_custom.setToolTip(
            f"Read {ex.CUSTOM_PATH.name} again after editing it.\n"
            f"It holds {n} profile{'s' if n != 1 else ''}.")

    # -- orientation -------------------------------------------------------

    def _start_picking(self) -> None:
        self._part_normal = self._stock_axis = None
        self.orient_dialog.set_step(STEP_BAR if self._is_round() else STEP_PART)
        self._set_step("part")

    def _set_step(self, step: str) -> None:
        """Ask for the next click in the 3D view (or stop asking)."""
        if step == "stock":
            self.orient_dialog.set_step(STEP_STOCK)
        changed = step != self._pick_step
        self._pick_step = step
        if changed:
            self.pickRequested.emit(step)

    def _set_angles(self, angles) -> None:
        for spin, a in zip((self.spin_rx, self.spin_ry, self.spin_rz), angles):
            spin.blockSignals(True)
            spin.setValue(a)
            spin.blockSignals(False)

    def _on_angles_changed(self) -> None:
        if self._base_axes is None:
            return
        self._part_normal = self._stock_axis = None
        self._reorient()

    def _reorient(self) -> None:
        """Refit the box in the frame the face pair and angles give, then
        resize the stock to it."""
        if self._base_axes is None or self._all_points is None:
            return
        angles = self.angles()
        if any(abs(a) > 1e-9 for a in angles):
            self._oriented = True
        frame = orient.rotated(self._base_axes, angles)
        self._rect = orient.box_in_frame(self._all_points, frame)
        self._show_orientation()
        self.orientationChanged.emit()
        self._restock()

    def _restock(self) -> None:
        """The same kind and basis of stock, sized for the box as it now is."""
        kind = self.type_combo.currentText()
        if kind == EXTRUSION:
            self._apply(self._extrusion(None))
        else:
            # Same kind and basis; a size typed in by hand is kept as typed.
            want_round = kind == ROUND
            basis = self._stock.basis if self._stock is not None else st.RECOMMENDED
            if (basis == st.CUSTOM and self._stock is not None
                    and self._stock.is_round == want_round):
                self._apply(self._classify(self._stock))
            else:
                self._apply(self._snap(KIND_OF[kind], basis if basis in
                                       (st.RECOMMENDED, st.MINIMUM) else st.RECOMMENDED))

    def _show_orientation(self) -> None:
        if not hasattr(self, "orient_row"):
            return
        self.orient_row.setVisible(True)
        if self._is_round():
            self.lbl_oriented.setVisible(self._cyl_oriented)
            self.lbl_oriented.setToolTip(self._cyl_how)
            return
        self.lbl_oriented.setVisible(self._oriented)
        if self._oriented:
            turns = [f"{n} {a:+g}°" for n, a in zip("XYZ", self.angles()) if abs(a) > 1e-9]
            tip = " ".join(([self._how] if self._how else []) +
                           (["Turned " + ", ".join(turns) + "."] if turns else []))
            self.lbl_oriented.setToolTip(tip)

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
            self._apply(self._snap(KIND_OF[kind], st.RECOMMENDED))

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
        self._apply(self._classify(stock), reflect=False)

    # -- internals ---------------------------------------------------------

    def _snap(self, kind: str, basis: str):
        """st.snap, with the window's recommended plate when it has one."""
        if (basis == st.RECOMMENDED and kind == "rectilinear" and self._rect is not None
                and self.recommend_plate is not None):
            s = self.recommend_plate(self._rect)
            if s is not None:
                return s
        return st.snap(kind, basis, self._rect, self._cyl)

    def _classify(self, stock):
        """st.classify_basis, knowing the window's recommended plate."""
        hooked = (stock is not None and stock.kind == "rectilinear" and self._rect is not None
                  and self.recommend_plate is not None)
        ref = self.recommend_plate(self._rect) if hooked else None
        if ref is None:
            return st.classify_basis(stock, self._rect, self._cyl)
        from dataclasses import replace

        if st.same_size(stock, ref):
            return replace(stock, basis=st.RECOMMENDED, is_sheet=ref.is_sheet)
        if st.same_size(stock, st.snap("rectilinear", st.MINIMUM, self._rect, None)):
            return replace(stock, basis=st.MINIMUM)
        return replace(stock, basis=st.CUSTOM)

    def _current_profile(self):
        value = self.profile.currentText()
        if "|" not in value:
            return None
        return ex.find(*value.split("|", 1))

    def _extrusion(self, length_in):
        prof = self._current_profile()
        if prof is None:
            return None
        return self.extrusion_for(prof, length_in)

    def extrusion_for(self, profile, length_in):
        """`profile` cut to `length_in` and placed on the current part."""
        return ex.extrusion_stock(profile, self._rect, self._cyl, length_in=length_in,
                                  part_volume_in3=self._part_volume, points_mm=self._points)

    def _best_profile(self):
        """The profile that fits this part with the least material, from the
        catalog and the AND / AMS standards alike. One the part reaches
        across if there is one; asked for an extrusion by hand, any that
        holds it will do."""
        for fill in (True, False):
            best = ex.best_profile(ex.load_profiles(), self._rect, self._cyl,
                                   part_volume_in3=self._part_volume, points_mm=self._points,
                                   require_fill=fill)
            if best is not None:
                return best
        return None

    def _show_for_type(self) -> None:
        kind = self.type_combo.currentText()
        self.profile.setVisible(kind == EXTRUSION)
        self.custom_row.setVisible(kind == EXTRUSION)
        self.lbl_custom.setVisible(kind == EXTRUSION and bool(self.lbl_custom.text()))
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
        if self.orient_dialog.isVisible():
            # Plate and bar are set differently; start over in the right mode.
            self.orient_dialog.set_round(kind == ROUND)
            self._start_picking()
        self._show_orientation()

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
        if s.is_extrusion:
            p = s.profile
            basis = "per the standard" if p.listed_area_in2 else "sharp corners"
            self.profile.setToolTip(f"{p.describe()}\nSection {p.area_in2:.3f} in² ({basis}).")
            self.spin_l.setToolTip(f'Length. Minimum {s.min_length_in:.3f}", '
                                   f'recommended {s.recommended_length_in:.3f}".')
        else:
            self.spin_l.setToolTip("Length.")
        warnings = st.fit_warnings(s, self._rect, self._cyl)
        self.lbl_status.setText("<br>".join(
            f"<span style='color:#b91c1c'>&#9888; {w}</span>" for w in warnings))
        self.lbl_status.setVisible(bool(warnings))
