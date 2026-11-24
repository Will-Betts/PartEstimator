"""Face colours for the sheet metal views: the 3D view and the legend."""
from __future__ import annotations

from PySide6.QtGui import QColor

from ..features import Role

#: Base colour per role. Steel for the sheet and a distinct hue only where
#: something is worth looking at -- a view where everything is a different
#: colour tells you nothing.
ROLE_COLOURS: dict[Role, QColor] = {
    Role.SHEET: QColor(158, 166, 176),
    Role.CUT_EDGE: QColor(132, 140, 150),
    Role.BEND_INNER: QColor(226, 154, 64),
    Role.BEND_OUTER: QColor(238, 174, 86),
    Role.HOLE: QColor(112, 120, 130),
    Role.SLOT_END: QColor(112, 120, 130),
    Role.CORNER_RADIUS: QColor(142, 178, 144),
    Role.COUNTERSINK: QColor(98, 188, 198),
    Role.COUNTERBORE: QColor(80, 158, 168),
    Role.CHAMFER: QColor(142, 178, 144),
    Role.FILLET: QColor(136, 164, 142),
    Role.FORMED: QColor(216, 98, 98),
    Role.UNKNOWN: QColor(198, 124, 202),
}
