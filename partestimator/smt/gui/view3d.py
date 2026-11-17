"""A CAD-style shaded view, drawn on the CPU, one path per face.

WHY NOT OPENGL
--------------
This has to run on a locked-down workstation, often over a remote desktop
session, with no admin rights and no guarantee about the graphics stack -- and a
remote session is exactly where OpenGL is most likely to be missing, emulated to
uselessness, or present but broken. The failure mode there is a black viewport
with nothing to explain it, which is worse than being a little slower.

WHY v6 LOOKED LIKE A MESH, AND WHY THIS DOES NOT
------------------------------------------------
v6 drew triangles. Thousands of them, individually, at a chord tolerance of
about a millimetre and twenty degrees around a curve. That is slow because the
cost of software rendering is Qt call count, not arithmetic, and it looks
faceted because it *is* faceted.

The fix is to stop working in triangles, because the part is not made of
triangles. A sheet metal part is made of planes and cylinders:

  * a PLANAR face is drawn as ONE filled polygon built from its real boundary
    curves, with its holes as sub-paths. No triangles are involved at all, so
    the outline is as smooth as the geometry and the fill is exactly the right
    shade everywhere. This is most of the part.
  * a CURVED face is drawn as ONE path of its front-facing triangles, filled
    with a LINEAR GRADIENT fitted to how the shading actually varies across it.
    A cylinder reads as genuinely round, because the shading is continuous
    rather than constant per facet, however coarse the mesh underneath is.

A typical bracket goes from ~4000 draw calls to ~60, and no facet is visible
anywhere.

THE GRADIENT FIT
----------------
Rather than deriving a gradient analytically per surface type -- which needs a
separate case for cylinders, cones and everything else, and gets the cone wrong
-- the shading is fitted from the data that is already there.

For each front-facing triangle of a face we have a screen position and a
Lambert intensity. Fit intensity as a plane over screen space by least squares:

    shade ~ a + b . (x, y)

`b` is then the direction in which shading varies, and the gradient runs along
it. Sampling the real intensities into stops along that direction reproduces
the true variation, including the non-linear falloff around a cylinder, without
knowing what kind of surface it was. Degenerate fits fall back to a flat fill,
which is the right answer for a face whose shading does not vary.

HIDDEN LINES
------------
There is no true hidden-line removal -- doing it properly is a harder problem
than the shading. Instead every edge knows which faces it borders, and an edge
whose owning faces are all turned away from the camera is not drawn. That
removes the great majority of the wrong lines for almost no work, and it is the
difference between reading as a solid model and reading as a wireframe.
"""

from __future__ import annotations


import numpy as np

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import QWidget

from .. import camera
from ..features import Role

# --- appearance ------------------------------------------------------------

BACKGROUND_TOP = QColor(44, 49, 56)
BACKGROUND_BOTTOM = QColor(26, 29, 34)
EDGE_COLOUR = QColor(16, 18, 21)
BOX_COLOUR = QColor(120, 176, 208, 190)
BOX_TEXT = QColor(168, 206, 228)
HIGHLIGHT = QColor(96, 210, 130)

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
PLAIN_STEEL = QColor(158, 166, 176)

#: Direction the key light comes from, in camera space. Slightly up and to the
#: left of the viewer rather than straight down the view axis: a pure headlight
#: gives every surface the same brightness and the part reads as a silhouette.
LIGHT = np.array([-0.32, 0.44, 0.84])
LIGHT = LIGHT / np.linalg.norm(LIGHT)

#: Number of stops in a fitted gradient. Eight is past the point where any
#: banding is visible.
GRADIENT_STOPS = 8


def _shade(cos_theta: np.ndarray | float, facing: np.ndarray | float = 0.0):
    """Lambert plus an ambient floor and a little rim from the view direction.

    The floor matters: a surface turned away from the key light with no ambient
    goes pure black, and the inside of a countersink is exactly such a surface.
    """
    lam = np.clip(cos_theta, 0.0, 1.0)
    return np.clip(0.30 + 0.54 * lam + 0.16 * np.clip(facing, 0.0, 1.0), 0.0, 1.0)


def _tint(base: QColor, level: float) -> QColor:
    f = 0.30 + 0.78 * float(level)
    return QColor(
        min(255, int(base.red() * f)),
        min(255, int(base.green() * f)),
        min(255, int(base.blue() * f)),
    )


class View3D(QWidget):
    """Orbit, pan and zoom around a `smt.occ.tess.Scene`.

    The widget knows nothing about sheet metal -- it is handed faces, a role per
    face and a camera. That is what lets the same widget show the folded part
    and the unfolded solid with no special casing at all.
    """

    view_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 260)
        self.setFocusPolicy(Qt.StrongFocus)
        # Right-drag is a pan modifier, so the context menu has to go or it
        # fires on every pan release.
        self.setContextMenuPolicy(Qt.NoContextMenu)

        self._scene = None
        self._roles: dict[int, Role] = {}
        self._tri_normals = None
        self._edges_by_face = {}

        self.colour_by_feature = True
        self.show_edges = True

        # Orthographic, because a machinist reading a length off the screen
        # should not have it foreshortened by perspective.
        #
        # The orientation is a MATRIX, not a pair of angles. See `smt.camera`:
        # angles gave a 180 degree wall at the poles and an inverted vertical
        # drag, and both problems disappear once there are no Euler angles left
        # to lock up.
        self.R = camera.named("iso")
        self.zoom = 1.0
        self.pan = np.zeros(2)

        # Optional wireframe box drawn round the model, with its dimensions.
        self._highlight = None    # face index to pick out in colour
        self._box = None          # (lo, hi) in model coordinates
        self._box_labels = ()
        self.show_box = True

        self._last_pos = None
        self._mode = None          # "orbit" | "pan"
        self._interacting = False

    # -- content --------------------------------------------------------

    def set_scene(self, scene, roles: dict[int, Role] | None = None,
                  keep_camera: bool = False) -> None:
        self._scene = scene
        self._roles = roles or {}
        self._tri_normals = (
            scene.triangle_normals() if scene is not None else None
        )
        self._edges_by_face = _group_edges(scene)
        if not keep_camera:
            self.fit()
        self.update()

    def clear(self) -> None:
        self._scene = None
        self._tri_normals = None
        self._edges_by_face = {}
        self.update()

    def has_scene(self) -> bool:
        return self._scene is not None and not self._scene.is_empty

    def set_colour_by_feature(self, on: bool) -> None:
        self.colour_by_feature = bool(on)
        self.update()

    def set_show_edges(self, on: bool) -> None:
        self.show_edges = bool(on)
        self.update()

    def set_box(self, lo=None, hi=None, labels=()) -> None:
        """A wireframe bounding box round the model, with its dimensions.

        Shown on the flattened part, where the box IS the answer: it is the
        stock you have to buy, and after the minimum-area orientation pass it is
        also the tightest one that exists. Seeing it drawn round the blank is
        the difference between a number in a table and something you can check
        against the sheet in the rack.
        """
        if lo is None or hi is None:
            self._box = None
        else:
            self._box = (np.asarray(lo, float), np.asarray(hi, float))
        self._box_labels = tuple(labels)
        self.update()

    def set_show_box(self, on: bool) -> None:
        self.show_box = bool(on)
        self.update()

    # -- camera ---------------------------------------------------------

    def fit(self) -> None:
        self.pan = np.zeros(2)
        self.zoom = 1.0
        self.update()
        self.view_changed.emit()

    def set_view(self, name: str) -> None:
        self.R = camera.named(name)
        self.update()
        self.view_changed.emit()

    def _rotation(self) -> np.ndarray:
        return self.R

    def _scale(self) -> float:
        scene = self._scene
        diag = scene.diagonal() if scene is not None else 1.0
        if self._box is not None:
            lo, hi = self._box
            diag = max(diag, float(np.linalg.norm(hi - lo)))
        base = 0.82 * min(self.width(), self.height()) / max(diag, 1e-9)
        return base * self.zoom

    def _offset(self) -> np.ndarray:
        return np.array([self.width() / 2.0, self.height() / 2.0]) + self.pan

    def project_all(self):
        """Every vertex in the scene, projected once.

        THIS IS THE PERFORMANCE BUG THAT MADE v9 FEEL SLOW.

        `_prepare_curved` called `project(scene.vertices)` -- the WHOLE vertex
        array -- and it is called once per face. On a part with sixty curved
        faces and twenty thousand vertices that is 1.2 million point transforms
        per frame to do twenty thousand points' worth of work. Sixty times the
        arithmetic, every frame, for nothing.

        It is an easy mistake to make and an invisible one to read: the call
        looks local, and the cost only shows up multiplied by the loop it sits
        in. Now it happens once per pass and the result is handed down.
        """
        return self.project(self._scene.vertices)

    def project(self, points: np.ndarray):
        """World points to (screen xy, camera depth). Depth grows toward you."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        if len(pts) == 0:
            return np.zeros((0, 2)), np.zeros(0)
        centre = self._scene.centre() if self._scene is not None else np.zeros(3)
        R, scale, offset = self._rotation(), self._scale(), self._offset()
        cam = (pts - centre) @ R.T
        xy = np.empty((len(cam), 2))
        xy[:, 0] = cam[:, 0] * scale + offset[0]
        xy[:, 1] = -cam[:, 1] * scale + offset[1]      # screen Y runs downward
        return xy, cam[:, 2]

    # -- painting -------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 -- Qt naming
        painter = QPainter(self)
        self._paint_background(painter)

        if not self.has_scene():
            painter.setPen(QColor(126, 134, 144))
            painter.drawText(
                self.rect(), Qt.AlignCenter, "Open a STEP file to see the part"
            )
            painter.end()
            return

        # Antialiasing is the single most expensive rendering hint, and while the
        # part is moving nobody can see the difference. It comes back the
        # instant the mouse is released.
        painter.setRenderHint(QPainter.Antialiasing, not self._interacting)
        painter.setPen(Qt.NoPen)

        R = self._rotation()
        projected = self.project_all()
        drawables = []
        for face in self._scene.faces:
            item = self._prepare_face(face, R, projected)
            if item is not None:
                drawables.append((item[0], item[1], item[2], face.index))

        # Painter's algorithm: furthest first. Exact for convex bodies, and
        # after backface culling the residual error on a thin folded sheet is
        # rare and local.
        drawables.sort(key=lambda d: d[0])

        pen = self._edge_pen() if self.show_edges else None

        # EACH FACE, THEN ITS OWN EDGES, INTERLEAVED.
        #
        # This is what removes hidden lines, and it costs nothing. v11 drew
        # every visible face and then drew every edge on top, so an edge on the
        # far side of the part was painted over the near side and you were
        # looking at a wireframe pretending to be a solid.
        #
        # Drawing a face's boundary immediately after the face puts the edge at
        # that face's depth, so any nearer face painted later covers it -- the
        # same painter's algorithm already used for the surfaces, extended to
        # the lines. An edge shared by two faces is drawn with each of them,
        # and the copy belonging to the nearer face is the one that survives,
        # which is exactly right.
        for _depth, path, brush, index in drawables:
            painter.fillPath(path, brush)
            if pen is not None:
                self._paint_face_edges(painter, pen, index)

        if self.show_box and self._box is not None:
            self._paint_box(painter)

        painter.end()

    def _paint_box(self, painter: QPainter) -> None:
        """The bounding box: 12 dashed edges, and L / W / T labels on 3 of them.

        v13 called this from paintEvent but never defined it, so every paint of
        the flat view raised AttributeError part-way through -- the box was
        never drawn, and the painter was left open (Qt then warns about
        destroying a paint device that is being painted).
        """
        lo, hi = self._box
        corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                            for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        xy, _depth = self.project(corners)
        # Corner index bits: x -> 4, y -> 2, z -> 1. An edge joins corners that
        # differ in exactly one bit.
        edges = [(a, b) for a in range(8) for b in range(a + 1, 8)
                 if bin(a ^ b).count("1") == 1]
        pen = QPen(BOX_COLOUR)
        pen.setWidthF(1.2)
        pen.setStyle(Qt.DashLine)
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        for a, b in edges:
            painter.drawLine(QPointF(*xy[a]), QPointF(*xy[b]))
        labels = list(self._box_labels) + ["", "", ""]
        painter.setPen(BOX_TEXT)
        for text, (a, b) in zip(labels[:3], ((0, 4), (0, 2), (0, 1))):
            if text:
                mid = (xy[a] + xy[b]) / 2.0
                painter.drawText(QPointF(mid[0] + 6, mid[1] + 14), text)
        painter.restore()

    def _paint_background(self, painter: QPainter) -> None:
        g = QLinearGradient(0.0, 0.0, 0.0, float(self.height()))
        g.setColorAt(0.0, BACKGROUND_TOP)
        g.setColorAt(1.0, BACKGROUND_BOTTOM)
        painter.fillRect(self.rect(), g)

    def _colour_for(self, face) -> QColor:
        if self._highlight is not None and face.index == self._highlight:
            return HIGHLIGHT
        if not self.colour_by_feature:
            return PLAIN_STEEL
        role = self._roles.get(face.index, Role.SHEET)
        return ROLE_COLOURS.get(role, PLAIN_STEEL)

    def _prepare_face(self, face, R, projected):
        """(depth, path, brush) for one face, or None if it is not visible."""
        if face.drawable_as_polygon:
            return self._prepare_planar(face, R)
        return self._prepare_curved(face, R, projected)

    def _prepare_planar(self, face, R):
        """A plane: one polygon from its real boundary, one exact shade."""
        n_cam = R @ face.normal
        if n_cam[2] <= 1e-6:
            return None                     # facing away

        path = QPainterPath()
        # Odd-even so interior loops punch holes regardless of which way round
        # the kernel happened to orient them.
        path.setFillRule(Qt.OddEvenFill)

        depths = []
        for loop in face.loops:
            xy, d = self.project(loop)
            if len(xy) < 3:
                continue
            path.addPolygon(QPolygonF([QPointF(p[0], p[1]) for p in xy]))
            path.closeSubpath()
            depths.append(d.mean())

        if not depths:
            return None

        level = float(_shade(float(np.dot(n_cam, LIGHT)), float(n_cam[2])))
        return (float(np.mean(depths)), path, _tint(self._colour_for(face), level))

    def _prepare_curved(self, face, R, projected):
        """A curved face: one path of front-facing triangles, gradient filled."""
        if face.n_triangles == 0 or self._tri_normals is None:
            return None

        scene = self._scene
        rows = slice(face.tri_lo, face.tri_hi)
        normals = self._tri_normals[rows] @ R.T
        front = normals[:, 2] > 1e-6
        if not front.any():
            return None

        tris = scene.triangles[rows][front]
        normals = normals[front]

        xy, depth = projected
        pts = xy[tris]                                   # (T, 3, 2)

        path = QPainterPath()
        path.setFillRule(Qt.WindingFill)
        for tri in pts:
            path.addPolygon(QPolygonF([
                QPointF(tri[0, 0], tri[0, 1]),
                QPointF(tri[1, 0], tri[1, 1]),
                QPointF(tri[2, 0], tri[2, 1]),
            ]))

        levels = _shade(normals @ LIGHT, normals[:, 2])
        # While dragging, take the mean instead of fitting a gradient. The fit
        # is a least-squares solve per curved face and the difference is not
        # visible on a moving part.
        if self._interacting:
            brush = _tint(self._colour_for(face), float(levels.mean()))
        else:
            centres = pts.mean(axis=1)
            brush = self._fit_gradient(self._colour_for(face), centres, levels)
        mean_depth = float(depth[tris].mean())
        return (mean_depth, path, brush)

    def _fit_gradient(self, colour: QColor, centres: np.ndarray,
                      levels: np.ndarray):
        """A linear gradient matching how the shading really varies.

        Least squares fit of `shade ~ a + b.(x, y)` over the face's triangles.
        `b` is the direction shading changes fastest; the gradient runs along
        it, and its stops are sampled from the actual intensities rather than
        from the fitted plane, so a cylinder's non-linear falloff is reproduced
        rather than straightened out.
        """
        if len(centres) < 3:
            return _tint(colour, float(np.mean(levels)))

        c0 = centres.mean(axis=0)
        rel = centres - c0
        try:
            coeffs, *_ = np.linalg.lstsq(
                np.column_stack([rel, np.ones(len(rel))]), levels, rcond=None
            )
        except np.linalg.LinAlgError:
            return _tint(colour, float(np.mean(levels)))

        b = coeffs[:2]
        mag = float(np.linalg.norm(b))
        if not np.isfinite(mag) or mag < 1e-9:
            return _tint(colour, float(np.mean(levels)))

        direction = b / mag
        t = rel @ direction
        t_lo, t_hi = float(t.min()), float(t.max())
        if t_hi - t_lo < 1e-6:
            return _tint(colour, float(np.mean(levels)))

        start = c0 + direction * t_lo
        end = c0 + direction * t_hi
        grad = QLinearGradient(
            float(start[0]), float(start[1]), float(end[0]), float(end[1])
        )

        # Bin the real intensities along the gradient axis. Empty bins are
        # skipped rather than interpolated, so a face that only occupies part of
        # its own span does not get invented colour at the ends.
        edges = np.linspace(t_lo, t_hi, GRADIENT_STOPS + 1)
        which = np.clip(np.digitize(t, edges) - 1, 0, GRADIENT_STOPS - 1)
        for k in range(GRADIENT_STOPS):
            sel = which == k
            if not sel.any():
                continue
            pos = (k + 0.5) / GRADIENT_STOPS
            grad.setColorAt(pos, _tint(colour, float(levels[sel].mean())))

        return grad

    def _edge_pen(self) -> QPen:
        pen = QPen(EDGE_COLOUR)
        pen.setWidthF(1.1)
        pen.setCosmetic(True)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        return pen

    def _paint_face_edges(self, painter, pen, index: int) -> None:
        """The boundary of one face, drawn at that face's depth."""
        polys = self._edges_by_face.get(index)
        if not polys:
            return
        path = QPainterPath()
        for poly in polys:
            xy, _d = self.project(poly)
            if len(xy) < 2:
                continue
            path.moveTo(QPointF(xy[0, 0], xy[0, 1]))
            for p in xy[1:]:
                path.lineTo(QPointF(p[0], p[1]))
        painter.strokePath(path, pen)

    def _curved_face_faces_us(self, face, R) -> bool:
        if face.n_triangles == 0 or self._tri_normals is None:
            return True
        normals = self._tri_normals[face.tri_lo:face.tri_hi] @ R.T
        return bool((normals[:, 2] > 1e-6).any())

    # -- interaction ----------------------------------------------------
    #
    # MCAD conventions, because that is what the muscle memory is:
    #
    #     middle drag                 orbit
    #     ctrl or shift + middle      pan
    #     right drag                  pan
    #     wheel                       zoom, toward the cursor
    #     left                        left free
    #
    # This matches SolidWorks, Creo, NX and Inventor. v6 used a game-style
    # left-drag orbit, which is why it felt wrong.

    def _button_mode(self, event) -> str | None:
        buttons = event.buttons()
        mods = event.modifiers()
        if buttons & Qt.MiddleButton:
            return "pan" if mods & (Qt.ControlModifier | Qt.ShiftModifier) else "orbit"
        if buttons & Qt.RightButton:
            return "pan"
        return None

    picked = Signal(int)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            index = self.pick(event.position().x(), event.position().y())
            if index is not None:
                self.picked.emit(index)
            return

        self._mode = self._button_mode(event)
        if self._mode is None:
            super().mousePressEvent(event)
            return
        self._last_pos = event.position()
        self._interacting = True
        self.setCursor(
            Qt.SizeAllCursor if self._mode == "pan" else Qt.OpenHandCursor
        )

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._mode is None or self._last_pos is None:
            return
        pos = event.position()
        delta = pos - self._last_pos
        self._last_pos = pos

        # Re-read the modifiers every move rather than trusting the ones from
        # the press, so holding ctrl part-way through a drag switches to pan.
        mode = self._mode
        if mode == "orbit" and event.modifiers() & (
            Qt.ControlModifier | Qt.ShiftModifier
        ):
            mode = "pan"

        if mode == "pan":
            self.pan += np.array([delta.x(), delta.y()])
        else:
            # Free tumble, unlimited in every direction, with the signs derived
            # rather than guessed. See `smt.camera` -- the rule is that the
            # surface under the cursor follows the cursor.
            self.R = camera.tumble(self.R, delta.x(), delta.y())
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._mode = None
        self._last_pos = None
        self._interacting = False
        self.unsetCursor()
        self.update()
        self.view_changed.emit()

    def wheelEvent(self, event) -> None:  # noqa: N802
        """Zoom toward the cursor, not toward the middle of the window.

        Zooming about the centre means every zoom has to be followed by a pan to
        put back what you were looking at. Keeping the model point under the
        pointer fixed is what makes a wheel feel like it is doing what you meant.

        In an orthographic view that is exact: the screen position of a model
        point is `p * scale + offset`, so holding it still while the scale
        changes needs `offset' = cursor - (cursor - offset) * scale'/scale`.
        """
        steps = event.angleDelta().y() / 120.0
        if not steps:
            return

        old_zoom = self.zoom
        new_zoom = float(np.clip(old_zoom * (1.18 ** steps), 0.02, 200.0))
        if new_zoom == old_zoom:
            return

        cursor = np.array([event.position().x(), event.position().y()])
        centre = np.array([self.width() / 2.0, self.height() / 2.0])
        ratio = new_zoom / old_zoom

        # offset = centre + pan, so solve for the new pan directly.
        offset = centre + self.pan
        self.pan = (cursor - (cursor - offset) * ratio) - centre
        self.zoom = new_zoom
        self.update()
        self.view_changed.emit()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.fit()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        keys = {
            Qt.Key_1: "front", Qt.Key_2: "back", Qt.Key_3: "left",
            Qt.Key_4: "right", Qt.Key_5: "top", Qt.Key_6: "bottom",
            Qt.Key_0: "iso",
        }
        name = keys.get(event.key())
        if name:
            self.set_view(name)
        elif event.key() in (Qt.Key_F, Qt.Key_Home):
            self.fit()
        else:
            super().keyPressEvent(event)

    # -- picking --------------------------------------------------------

    def pick(self, x: float, y: float) -> int | None:
        """The B-rep face index under a screen point, or None.

        Screen-space hit testing rather than a 3D ray cast, because everything
        needed is already projected for drawing: test containment in what is on
        screen, and keep the candidate nearest the viewer.

        Planar faces are tested against their real boundary -- inside the outer
        loop and outside every hole -- so clicking through a hole selects
        whatever is behind it rather than the face the hole is cut in. Curved
        faces are tested against their front-facing triangles.
        """
        if not self.has_scene():
            return None

        R = self._rotation()
        projected = self.project_all()
        point = np.array([float(x), float(y)])
        best_index, best_depth = None, -np.inf

        for face in self._scene.faces:
            hit, depth = self._hit_face(face, R, point, projected)
            if hit and depth > best_depth:
                best_index, best_depth = face.index, depth

        return best_index

    def _hit_face(self, face, R, point, projected):
        if face.drawable_as_polygon:
            n_cam = R @ face.normal
            if n_cam[2] <= 1e-6:
                return False, 0.0

            inside = False
            depth = 0.0
            for i, loop in enumerate(face.loops):
                xy, d = self.project(loop)
                if len(xy) < 3:
                    continue
                if _point_in_polygon(point, xy):
                    if i == 0:
                        inside, depth = True, float(d.mean())
                    else:
                        return False, 0.0        # in a hole: not this face
            return inside, depth

        if face.n_triangles == 0 or self._tri_normals is None:
            return False, 0.0

        rows = slice(face.tri_lo, face.tri_hi)
        normals = self._tri_normals[rows] @ R.T
        front = normals[:, 2] > 1e-6
        if not front.any():
            return False, 0.0

        tris = self._scene.triangles[rows][front]
        xy, depth = projected
        pts = xy[tris]

        inside = _point_in_triangles(point, pts)
        if not inside.any():
            return False, 0.0
        return True, float(depth[tris][inside].mean())

    def set_highlight(self, index: int | None) -> None:
        self._highlight = index
        self.update()


def _point_in_polygon(point: np.ndarray, poly: np.ndarray) -> bool:
    """Ray casting, vectorised. Odd number of crossings means inside."""
    x, y = float(point[0]), float(point[1])
    xs, ys = poly[:, 0], poly[:, 1]
    xs2, ys2 = np.roll(xs, -1), np.roll(ys, -1)

    straddles = (ys > y) != (ys2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        cross_x = xs + (y - ys) * (xs2 - xs) / (ys2 - ys)
    hits = straddles & np.isfinite(cross_x) & (cross_x > x)
    return bool(hits.sum() % 2 == 1)


def _point_in_triangles(point: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Barycentric containment for many triangles at once."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    v0, v1, v2 = c - a, b - a, point - a

    d00 = (v0 * v0).sum(axis=1)
    d01 = (v0 * v1).sum(axis=1)
    d11 = (v1 * v1).sum(axis=1)
    d20 = (v2 * v0).sum(axis=1)
    d21 = (v2 * v1).sum(axis=1)

    denom = d00 * d11 - d01 * d01
    ok = np.abs(denom) > 1e-12
    u = np.zeros(len(tris))
    v = np.zeros(len(tris))
    np.divide(d11 * d20 - d01 * d21, denom, out=u, where=ok)
    np.divide(d00 * d21 - d01 * d20, denom, out=v, where=ok)
    return ok & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)


def _group_edges(scene) -> dict:
    """Edge polylines, indexed by each face that borders them.

    An edge with two neighbouring faces appears under both. That duplication is
    deliberate: it is what lets the edge be drawn at whichever face's depth is
    being painted, so the nearer one wins and the farther one is covered.
    """
    out: dict[int, list] = {}
    if scene is None:
        return out
    for poly, owners in getattr(scene, "edges", ()):
        for owner in owners or ():
            out.setdefault(int(owner), []).append(poly)
    return out
