"""The sheet metal 3D view, on the same VTK viewer the machining tab uses.

SMTv13 drew its own view in software (QPainter, painter's algorithm). In the
combined tool both tabs now share one viewer: the machining tab's PyVista /
VTK widget, with its white background, grey part colour, crisp near-black
B-rep edges, parallel projection and VTK's standard mouse controls
(left-drag rotate, middle-drag or shift-drag pan, wheel zoom).

The public API is the one SMTv13's window already calls, so the window
itself did not have to change: set_scene / clear / set_colour_by_feature /
set_show_edges / set_box / set_show_box / set_view / fit / set_highlight, and
the `picked(face_index)` signal used by "Pick the sheet face...". A pick is a
left click WITHOUT a drag, so rotating never picks a face by accident.
"""
from __future__ import annotations

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from partestimator.machining.ui.viewer import (
    EDGE_COLOR,
    EDGE_LINE_WIDTH,
    PART_COLOR,
    SHADING_FEATURE_ANGLE,
    _dash_segments,
    _dashed_polydata,
    _line_polydata,
    _style_line_actor,
)

from .view3d import ROLE_COLOURS

HIGHLIGHT = (0.20, 0.72, 0.35)
BOX_COLOR = (0.12, 0.40, 0.90)          # same blue as the machining stock outline
LABEL_COLOR = (0.10, 0.25, 0.55)
CLICK_TOLERANCE_PX = 4

_VIEWS = {
    "iso": "view_isometric", "top": "view_xy", "bottom": "view_yx",
    "front": "view_xz", "back": "view_zx", "right": "view_yz", "left": "view_zy",
}


def _rgb(qcolor) -> tuple[float, float, float]:
    return (qcolor.redF(), qcolor.greenF(), qcolor.blueF())


class View3D(QWidget):
    picked = Signal(int)
    view_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self)
        self.plotter.set_background("white")
        self.plotter.enable_parallel_projection()
        lay.addWidget(self.plotter)

        self.colour_by_feature = True
        self.show_edges = True
        self.show_box = True
        self._scene = None
        self._roles: dict = {}
        self._mesh = None
        self._face_of_cell = None
        self._surface = None
        self._edges_actor = None
        self._box_actors: list = []
        self._box = None
        self._box_labels = ()
        self._highlight = None
        self._press = None

        try:
            self.plotter.iren.add_observer("LeftButtonPressEvent", self._on_press)
            self.plotter.iren.add_observer("LeftButtonReleaseEvent", self._on_release)
        except Exception:  # noqa: BLE001 -- picking is a convenience
            pass

    # -- content ----------------------------------------------------------

    def set_scene(self, scene, roles=None, keep_camera: bool = False) -> None:
        cam = self._save_camera() if keep_camera and self._mesh is not None else None
        self._scene = scene
        self._roles = roles or {}
        self.plotter.clear()
        self._surface = self._edges_actor = None
        self._box_actors = []
        self._mesh = None
        if scene is None or scene.is_empty:
            self.plotter.render()
            return
        tris = np.asarray(scene.triangles, dtype=np.int64)
        faces = np.hstack([np.full((len(tris), 1), 3, dtype=np.int64), tris]).ravel()
        self._mesh = pv.PolyData(np.asarray(scene.vertices, float), faces)
        face_of_cell = np.full(len(tris), -1, dtype=np.int64)
        for f in scene.faces:
            face_of_cell[f.tri_lo:f.tri_hi] = f.index
        self._face_of_cell = face_of_cell
        self._mesh.cell_data["face_id"] = face_of_cell
        self._mesh.cell_data["rgb"] = self._cell_colours()
        try:
            self._surface = self.plotter.add_mesh(
                self._mesh, scalars="rgb", rgb=True, show_edges=False,
                smooth_shading=True, split_sharp_edges=True,
                feature_angle=SHADING_FEATURE_ANGLE, specular=0.25, specular_power=12,
                show_scalar_bar=False)
        except TypeError:
            self._surface = self.plotter.add_mesh(
                self._mesh, scalars="rgb", rgb=True, show_edges=False,
                show_scalar_bar=False)
        self._draw_edges()
        self._draw_box()
        if cam is not None:
            self._restore_camera(cam)
        else:
            self.plotter.view_isometric()
            self.plotter.reset_camera()
        self.plotter.render()

    def clear(self) -> None:
        self.set_scene(None)

    def has_scene(self) -> bool:
        return self._scene is not None and not self._scene.is_empty

    def set_colour_by_feature(self, on: bool) -> None:
        self.colour_by_feature = bool(on)
        self._recolour()

    def set_show_edges(self, on: bool) -> None:
        self.show_edges = bool(on)
        self._draw_edges()
        self.plotter.render()

    def set_box(self, lo=None, hi=None, labels=()) -> None:
        self._box = None if lo is None or hi is None else (np.asarray(lo, float),
                                                          np.asarray(hi, float))
        self._box_labels = tuple(labels)
        self._draw_box()
        self.plotter.render()

    def set_show_box(self, on: bool) -> None:
        self.show_box = bool(on)
        self._draw_box()
        self.plotter.render()

    def set_highlight(self, index) -> None:
        self._highlight = index
        self._recolour()

    # -- camera -------------------------------------------------------------

    def fit(self) -> None:
        self.plotter.reset_camera()
        self.plotter.render()
        self.view_changed.emit()

    def set_view(self, name: str) -> None:
        getattr(self.plotter, _VIEWS.get(name, "view_isometric"))()
        self.plotter.reset_camera()
        self.plotter.render()
        self.view_changed.emit()

    def _save_camera(self):
        c = self.plotter.camera
        d = np.array(c.position) - np.array(c.focal_point)
        n = np.linalg.norm(d)
        return (d / n if n > 0 else np.array([1.0, 1.0, 1.0]), np.array(c.up))

    def _restore_camera(self, cam) -> None:
        direction, up = cam
        centre = np.array(self._mesh.center)
        c = self.plotter.camera
        c.focal_point = tuple(centre)
        c.position = tuple(centre + direction * 100.0)
        c.up = tuple(up)
        self.plotter.reset_camera()

    # -- drawing --------------------------------------------------------------

    def _cell_colours(self) -> np.ndarray:
        n = len(self._face_of_cell)
        cols = np.tile(np.array(PART_COLOR), (n, 1))
        if self.colour_by_feature and self._roles:
            for idx, role in self._roles.items():
                q = ROLE_COLOURS.get(role)
                if q is not None:
                    cols[self._face_of_cell == idx] = _rgb(q)
        if self._highlight is not None:
            cols[self._face_of_cell == self._highlight] = HIGHLIGHT
        return (np.clip(cols, 0, 1) * 255).astype(np.uint8)

    def _recolour(self) -> None:
        if self._mesh is None:
            return
        self._mesh.cell_data["rgb"] = self._cell_colours()
        self.plotter.render()

    def _draw_edges(self) -> None:
        if self._edges_actor is not None:
            self.plotter.remove_actor(self._edges_actor)
            self._edges_actor = None
        if not (self.show_edges and self._scene is not None and self._scene.edges):
            return
        pts, cells, base = [], [], 0
        for poly, _owners in self._scene.edges:
            poly = np.asarray(poly, float)
            if len(poly) < 2:
                continue
            pts.append(poly)
            cells.append(np.concatenate([[len(poly)], np.arange(base, base + len(poly))]))
            base += len(poly)
        if not pts:
            return
        poly = _line_polydata(np.vstack(pts), np.concatenate(cells).astype(np.int64))
        self._edges_actor = _style_line_actor(
            self.plotter.add_mesh(poly, color=EDGE_COLOR, line_width=EDGE_LINE_WIDTH,
                                  lighting=False),
            depth_offset=True)

    def _draw_box(self) -> None:
        for a in self._box_actors:
            self.plotter.remove_actor(a)
        self._box_actors = []
        if not (self.show_box and self._box is not None and self._mesh is not None):
            return
        lo, hi = self._box
        corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                            for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        diag = float(np.linalg.norm(hi - lo)) or 1.0
        segs = []
        for a in range(8):
            for b in range(a + 1, 8):
                if bin(a ^ b).count("1") == 1:
                    segs += _dash_segments(corners[a], corners[b], diag * 0.02)
        if segs:
            self._box_actors.append(_style_line_actor(self.plotter.add_mesh(
                _dashed_polydata(segs), color=BOX_COLOR, line_width=1.5, lighting=False)))
        labels = list(self._box_labels)
        pts, texts = [], []
        for text, (a, b) in zip(labels, ((0, 4), (0, 2), (0, 1))):
            if text:
                pts.append((corners[a] + corners[b]) / 2.0)
                texts.append(text)
        if pts:
            self._box_actors.append(self.plotter.add_point_labels(
                np.array(pts), texts, font_size=12, text_color=LABEL_COLOR,
                shape=None, show_points=False, always_visible=True))

    # -- picking ----------------------------------------------------------------

    def _on_press(self, *_args) -> None:
        try:
            self._press = tuple(self.plotter.iren.get_event_position())
        except Exception:  # noqa: BLE001
            self._press = None

    def _on_release(self, *_args) -> None:
        if self._press is None or self._mesh is None:
            return
        try:
            x, y = self.plotter.iren.get_event_position()
        except Exception:  # noqa: BLE001
            return
        if abs(x - self._press[0]) > CLICK_TOLERANCE_PX or abs(y - self._press[1]) > CLICK_TOLERANCE_PX:
            return
        index = self.pick(x, y)
        if index is not None:
            self.picked.emit(index)

    def pick(self, x, y):
        """Face index under display coordinates (x, y), or None."""
        import vtk

        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.0005)
        if self._surface is not None:
            # Only the part surface: an edge line drawn over it must not win.
            picker.PickFromListOn()
            picker.AddPickList(self._surface)
        if not picker.Pick(x, y, 0, self.plotter.renderer):
            return None
        if picker.GetDataSet() is None:
            return None
        cid = picker.GetCellId()
        if cid < 0 or self._face_of_cell is None or cid >= len(self._face_of_cell):
            return None
        idx = int(self._face_of_cell[cid])
        return idx if idx >= 0 else None

    def close(self) -> bool:  # noqa: D401 -- Qt override
        try:
            self.plotter.close()
        except Exception:  # noqa: BLE001
            pass
        return super().close()
