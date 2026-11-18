"""Small Qt helpers shared by both tools (imported lazily; Qt required).

  NestedPicker        one dropdown you can type in; its arrow opens a menu of
                      groups, each with its items in a submenu (materials,
                      extrusion profiles)
  material_picker()   a NestedPicker filled from the material library
  guard_wheel()       stop the mouse wheel changing inputs as you scroll past
"""
from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QEvent, QObject, QPoint, QStringListModel, Qt, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QCompleter,
    QHBoxLayout,
    QMenu,
    QSlider,
    QWidget,
)
from shiboken6 import isValid

#: (value, label shown in the item list)
Item = tuple[str, str]


class _MenuCombo(QComboBox):
    """An editable combo whose arrow opens the owner's nested menu in place
    of the usual flat list."""

    def __init__(self, owner: NestedPicker):
        super().__init__(owner)
        self._owner = owner

    def showPopup(self) -> None:  # noqa: N802 -- Qt override
        self._owner.show_menu()


class NestedPicker(QWidget):
    """One dropdown for a two-level list, such as material family -> alloy.

    Type in it to search everything ("7075", "stainless 304"): matches drop
    down below as you type, and Enter takes the first. Click the arrow for a
    menu of groups, each opening a submenu of its items. Typing while that
    menu is open goes straight to the search.

    Offers the same `currentText()`, `setCurrentText()` and
    `currentTextChanged(str)` as a QComboBox, in full values (e.g.
    "Aluminum 6061-T6"). The signal fires only when a value is picked, never
    on a half-typed search.

    `display` turns a value into the words shown for it in the box and the
    search results, for values that are keys rather than names; each must
    give a different text.
    """

    currentTextChanged = Signal(str)

    MAX_MATCHES = 40

    def __init__(self, groups: list[tuple[str, list[Item]]],
                 matcher: Callable[[str, str], bool],
                 search_text: Callable[[str], str] | None = None,
                 placeholder: str = "Type to search, or pick from the list",
                 display: Callable[[str], str] | None = None,
                 parent=None):
        super().__init__(parent)
        self._matcher = matcher
        self._search_text = search_text or (lambda v: v)
        self._display = display or (lambda v: v)
        self._value_of: dict[str, str] = {}     # shown text -> value
        self._groups: list[tuple[str, list[Item]]] = []
        self._group_of: dict[str, str] = {}
        self._value = ""
        self.menu: QMenu | None = None

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self.combo = _MenuCombo(self)
        self.combo.setEditable(True)
        self.combo.setInsertPolicy(QComboBox.NoInsert)
        self.combo.setMinimumContentsLength(18)
        self.combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        lay.addWidget(self.combo)

        self.line = self.combo.lineEdit()
        self.line.setPlaceholderText(placeholder)
        self._matches = QStringListModel(self)
        self._completer = QCompleter(self._matches, self)
        self._completer.setCompletionMode(QCompleter.UnfilteredPopupCompletion)
        self._completer.setCaseSensitivity(Qt.CaseInsensitive)
        self._completer.setMaxVisibleItems(15)
        self._completer.popup().setObjectName("completerPopup")
        # On the line edit itself: QComboBox.setCompleter would also tie the
        # completer to the combo's own (empty) item list.
        self.line.setCompleter(self._completer)
        self.line.textEdited.connect(self._on_search_edited)
        self.line.returnPressed.connect(self._on_return)
        self.line.editingFinished.connect(self._restore_text)
        self._completer.activated[str].connect(self._on_completion)
        self.line.installEventFilter(self)

        self.set_groups(groups)

    # -- combo-like API --------------------------------------------------

    def currentText(self) -> str:  # noqa: N802 -- Qt naming, drop-in for QComboBox
        return self._value

    def setCurrentText(self, value: str) -> None:  # noqa: N802
        if value in self._group_of:
            self.commit(value)

    def setToolTip(self, text: str) -> None:  # noqa: N802
        super().setToolTip(text)
        self.combo.setToolTip(text)

    def set_groups(self, groups: list[tuple[str, list[Item]]]) -> None:
        """Replace everything on offer, keeping the current value when it is
        still there. When it is gone, the first item is taken, and
        `currentTextChanged` says so."""
        self._groups = [(g, list(items)) for g, items in groups if items]
        self._group_of = {v: g for g, items in self._groups for v, _ in items}
        self._value_of = {self._display(v): v for v in self._group_of}
        values = self.values()
        if self._value not in self._group_of and values:
            self.commit(values[0])
        else:
            self._restore_text()

    def values(self) -> list[str]:
        return [v for _, items in self._groups for v, _ in items]

    def matches(self, query: str) -> list[str]:
        return [v for v in self.values() if self._matcher(query, self._search_text(v))]

    def group_of(self, value: str) -> str | None:
        return self._group_of.get(value)

    def display_text(self, value: str) -> str:
        return self._display(value) if value else ""

    def commit(self, value: str) -> None:
        """Make `value` the current one, if it is on offer."""
        if value not in self._group_of:
            self._restore_text()
            return
        changed = value != self._value
        self._value = value
        self._matches.setStringList([])
        self._completer.popup().hide()
        self._restore_text()
        if changed:
            self.currentTextChanged.emit(value)

    # -- the nested menu -------------------------------------------------

    def build_menu(self) -> QMenu:
        """Groups, each a submenu of its items; the current item is ticked
        and its group shown in bold."""
        menu = QMenu(self)
        menu.setMinimumWidth(self.combo.width())
        current_group = self._group_of.get(self._value)
        for group, items in self._groups:
            if len(items) == 1 and items[0][1] == group:
                parent = menu                   # e.g. "Other": no submenu of one
            else:
                parent = menu.addMenu(group)
                parent.setToolTipsVisible(True)
                if group == current_group:
                    f = QFont(parent.menuAction().font())
                    f.setBold(True)
                    parent.menuAction().setFont(f)
            for value, label in items:
                act = parent.addAction(label)
                act.setCheckable(True)
                act.setChecked(value == self._value)
                if label != self._display(value):
                    act.setToolTip(self._display(value))
                act.triggered.connect(lambda _=False, v=value: self.commit(v))
        menu.installEventFilter(self)
        return menu

    def show_menu(self) -> None:
        if self.menu is not None:
            self.menu.close()
        self.menu = self.build_menu()
        self.menu.aboutToHide.connect(self._menu_closed)
        self.menu.popup(self.combo.mapToGlobal(QPoint(0, self.combo.height())))

    def _menu_closed(self) -> None:
        menu, self.menu = self.menu, None
        if menu is not None:
            menu.deleteLater()

    # -- typing ----------------------------------------------------------

    def eventFilter(self, obj, event):  # noqa: N802 -- Qt override
        et = event.type()
        if isinstance(obj, QMenu) and et == QEvent.KeyPress:
            # A letter or digit typed into the open menu starts a search.
            text = event.text()
            if text and text.isprintable() and not text.isspace():
                obj.close()
                self.line.setFocus(Qt.PopupFocusReason)
                self.line.setText(text)
                self._on_search_edited(text)
                return True
        elif obj is self.line and et == QEvent.FocusIn:
            # Typing replaces what is there rather than adding to it.
            QTimer.singleShot(0, self.line.selectAll)
        elif obj is self.line and et == QEvent.KeyPress and event.key() == Qt.Key_Escape:
            self._matches.setStringList([])
            self._completer.popup().hide()
            self._restore_text()
            self.line.selectAll()
            return True
        return super().eventFilter(obj, event)

    def _on_search_edited(self, text: str) -> None:
        query = text.strip()
        found = self.matches(query)[: self.MAX_MATCHES] if query else []
        self._matches.setStringList([self._display(v) for v in found])
        if found:
            self._completer.setCompletionPrefix("")
            self._completer.complete()
        else:
            self._completer.popup().hide()

    def _on_completion(self, shown: str) -> None:
        self.commit(self._value_of.get(shown, shown))

    def _on_return(self) -> None:
        text = self.line.text().strip()
        if text in self._value_of:
            self.commit(self._value_of[text])
            return
        found = self.matches(text) if text else []
        self.commit(found[0] if found else self._value)

    def _restore_text(self) -> None:
        """Show the current value again, dropping any unfinished search."""
        # editingFinished can arrive while the window is being torn down.
        if not (isValid(self.line) and isValid(self._completer)):
            return
        if self._completer.popup().isVisible():
            return
        shown = self.display_text(self._value)
        if self.line.text() != shown:
            self.line.setText(shown)
        self.line.setCursorPosition(0)


def material_picker(names: list[str], extra: list[str] | None = None) -> NestedPicker:
    """One dropdown: type to search ("7075", "stainless 304", "inconel", or
    an old short key such as "CRS"), or open it for family -> alloy menus."""
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
    return NestedPicker(groups, db.token_match, db.search_text,
                        placeholder="Type a material, e.g. 7075 or stainless 304")


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
