"""Small Qt helpers shared by both tools (imported lazily; Qt required).

  HierarchicalPicker  family -> item selection with a keyword search box,
                      used for materials and for extrusion profiles
  material_picker()   a HierarchicalPicker filled from the material library
  guard_wheel()       stop the mouse wheel changing inputs as you scroll past
"""
from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QEvent, QObject, QStringListModel, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QCompleter,
    QHBoxLayout,
    QLineEdit,
    QSlider,
    QVBoxLayout,
    QWidget,
)

#: (value, label shown in the item list)
Item = tuple[str, str]


class HierarchicalPicker(QWidget):
    """Pick one value in two short steps -- a group, then an item in it --
    or type a few words to search everything at once.

    Behaves like the QComboBox it replaces where callers need it to:
    `currentText()`, `setCurrentText()` and `currentTextChanged(str)` all
    deal in the full value (e.g. "Aluminum 6061-T6"), whatever short label
    the item list shows.
    """

    currentTextChanged = Signal(str)

    def __init__(self, groups: list[tuple[str, list[Item]]],
                 matcher: Callable[[str, str], bool],
                 search_text: Callable[[str], str] | None = None,
                 placeholder: str = "Search...", item_tooltips: bool = True,
                 parent=None):
        super().__init__(parent)
        self._item_tooltips = item_tooltips
        self._groups = [(g, list(items)) for g, items in groups if items]
        self._matcher = matcher
        self._search_text = search_text or (lambda v: v)
        self._value = ""
        self._group_of = {v: g for g, items in self._groups for v, _ in items}

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)

        self.search = QLineEdit()
        self.search.setPlaceholderText(placeholder)
        self.search.setClearButtonEnabled(True)
        self._matches = QStringListModel(self)
        self._completer = QCompleter(self._matches, self)
        self._completer.setCompletionMode(QCompleter.UnfilteredPopupCompletion)
        self._completer.setCaseSensitivity(Qt.CaseInsensitive)
        self._completer.setMaxVisibleItems(15)
        self.search.setCompleter(self._completer)
        self.search.textEdited.connect(self._on_search_edited)
        self.search.returnPressed.connect(self._on_search_return)
        self._completer.activated[str].connect(self._on_completion)
        lay.addWidget(self.search)

        row = QHBoxLayout()
        row.setSpacing(4)
        self.group_combo = QComboBox()
        self.group_combo.addItems([g for g, _ in self._groups])
        self.item_combo = QComboBox()
        self.item_combo.setMaxVisibleItems(25)
        row.addWidget(self.group_combo)
        row.addWidget(self.item_combo, 1)
        lay.addLayout(row)

        self.group_combo.currentIndexChanged.connect(self._on_group_changed)
        self.item_combo.currentIndexChanged.connect(self._on_item_changed)
        self._fill_items(0)
        self._on_item_changed(self.item_combo.currentIndex())

    # -- combo-like API --------------------------------------------------

    def currentText(self) -> str:  # noqa: N802 -- Qt naming, drop-in for QComboBox
        return self._value

    def setCurrentText(self, value: str) -> None:  # noqa: N802
        group = self._group_of.get(value)
        if group is None:
            return
        gi = [g for g, _ in self._groups].index(group)
        if self.group_combo.currentIndex() != gi:
            self.group_combo.blockSignals(True)
            self.group_combo.setCurrentIndex(gi)
            self.group_combo.blockSignals(False)
            self._fill_items(gi)
        ii = self.item_combo.findData(value)
        if ii >= 0:
            self.item_combo.setCurrentIndex(ii)
            self._on_item_changed(ii)

    def values(self) -> list[str]:
        return [v for _, items in self._groups for v, _ in items]

    def matches(self, query: str) -> list[str]:
        return [v for v in self.values() if self._matcher(query, self._search_text(v))]

    # -- internals -------------------------------------------------------

    def _fill_items(self, group_index: int) -> None:
        self.item_combo.blockSignals(True)
        self.item_combo.clear()
        if 0 <= group_index < len(self._groups):
            for value, label in self._groups[group_index][1]:
                self.item_combo.addItem(label, value)
                if self._item_tooltips and label != value:
                    self.item_combo.setItemData(self.item_combo.count() - 1, value,
                                                Qt.ToolTipRole)
        self.item_combo.blockSignals(False)

    def _on_group_changed(self, index: int) -> None:
        self._fill_items(index)
        self.item_combo.setCurrentIndex(0)
        self._on_item_changed(0)

    def _on_item_changed(self, index: int) -> None:
        value = self.item_combo.itemData(index) if index >= 0 else None
        if value and value != self._value:
            self._value = value
            self.currentTextChanged.emit(value)

    def _on_search_edited(self, text: str) -> None:
        self._matches.setStringList(self.matches(text) if text.strip() else [])
        if text.strip():
            self._completer.complete()

    def _on_search_return(self) -> None:
        found = self.matches(self.search.text())
        if found:
            self._on_completion(found[0])

    def _on_completion(self, value: str) -> None:
        if value in self._group_of:
            self.setCurrentText(value)
            # Leave the box empty for the next search; the combos now show
            # what was picked.
            self.search.clear()
            self._matches.setStringList([])


def material_picker(names: list[str], extra: list[str] | None = None) -> HierarchicalPicker:
    """Family, then alloy, with keyword search: "7075", "stainless 304",
    "inconel", or an old short key such as "CRS"."""
    from . import materials_db as db

    order = db.families()
    by_family: dict[str, list[Item]] = {f: [] for f in order}
    for n in names:
        rec = db.get(n)
        fam = rec.family if rec else "Other"
        label = n
        if label.lower().startswith(fam.lower() + " "):
            label = label[len(fam) + 1:]        # "Aluminum 6061-T6" -> "6061-T6"
        by_family.setdefault(fam, []).append((n, label))
    for n in extra or []:
        by_family.setdefault("Other", []).append((n, n))
    groups = [(f, by_family[f]) for f in by_family if by_family[f]]
    picker = HierarchicalPicker(groups, db.token_match, db.search_text,
                                placeholder="Search materials, e.g. 7075 or stainless 304")
    picker.group_combo.setToolTip("Material family")
    picker.item_combo.setToolTip("Alloy / grade within the family")
    return picker


# ---------------------------------------------------------------------------
# mouse wheel
# ---------------------------------------------------------------------------


class _WheelGuard(QObject):
    """Turns a wheel event over an input into a scroll of the panel it sits in.

    Without this, scrolling down a long settings panel changes whatever
    dropdown or number happens to pass under the pointer -- a quiet way to
    enter a wrong value. Inputs still change by clicking, typing and the
    arrow keys.
    """

    def eventFilter(self, obj, event):  # noqa: N802 -- Qt override
        if event.type() != QEvent.Wheel:
            return False
        w = obj.parentWidget() if isinstance(obj, QWidget) else None
        while w is not None and not isinstance(w, QAbstractScrollArea):
            w = w.parentWidget()
        if w is not None:
            QApplication.sendEvent(w.verticalScrollBar(), event)
        return True


GUARDED = (QComboBox, QAbstractSpinBox, QSlider)


def guard_wheel(root: QWidget) -> int:
    """Stop the wheel changing any dropdown, number box or slider under
    `root`. Call after the widgets exist. Returns how many were guarded."""
    guard = root.findChild(_WheelGuard)
    if guard is None:
        guard = _WheelGuard(root)
    n = 0
    for w in root.findChildren(QWidget):
        if isinstance(w, GUARDED):
            w.installEventFilter(guard)
            # Wheel no longer gives focus either; a click or Tab does.
            w.setFocusPolicy(Qt.StrongFocus)
            n += 1
    return n
