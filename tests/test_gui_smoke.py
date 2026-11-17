"""Build the whole window, for real, against a stub Qt.

This is the test that would have caught every GUI bug that has shipped:

    v10  `AttributeError: 'MainWindow' object has no attribute '_stall'`
    v10  `thickness_from_face` called with no import
    v7   a method renamed without updating the code that called it

None of them is about Qt behaviour. They are Python mistakes -- a name that does
not resolve, an attribute used before it exists, a signal connected to something
that is no longer there -- and every one of them shows up the instant the object
is actually constructed.

It could not be constructed here because the modules import PySide6 and Qt
cannot be installed in this environment. `tests/qtstub.py` supplies a permissive
fake, which is enough: the real `__init__` of every widget runs, the real layout
code runs, and the real signal wiring runs.

WHAT PASSING HERE DOES NOT MEAN
-------------------------------
It does not mean the window looks right, renders correctly, or that any Qt API
is being used the way Qt intends. A stub accepts everything, including calls Qt
would reject. This raises the floor from "the source parses" to "the code runs";
it does not reach "the tool works", and SYSTEM_SPECS says so plainly.
"""

from __future__ import annotations

import unittest

from tests import qtstub


class _StubbedQt(unittest.TestCase):
    """Installs the stub for the duration of the test and takes it out after.

    Removing it matters: leaving a fake PySide6 in `sys.modules` would make
    every later test in the run see it, and a suite that quietly reconfigures
    the interpreter under itself is worse than no suite.
    """

    @classmethod
    def setUpClass(cls):
        if qtstub.available():
            raise unittest.SkipTest("real PySide6 present; the stub is not needed")
        cls._saved = qtstub.install()

    @classmethod
    def tearDownClass(cls):
        qtstub.remove(cls._saved)


class TestModulesImport(_StubbedQt):
    def test_every_gui_module_imports(self):
        import importlib

        for name in ("partestimator.smt.gui.view3d", "partestimator.smt.gui.panels", "partestimator.smt.gui.worker",
                     "partestimator.smt.gui.app"):
            with self.subTest(module=name):
                importlib.import_module(name)


class TestWidgetsConstruct(_StubbedQt):
    """Each panel on its own, so a failure says which one."""

    def test_simple_controls(self):
        from partestimator.smt.gui.panels import SimpleControls

        SimpleControls()

    def test_advanced_panel(self):
        from partestimator.smt.gui.panels import AdvancedPanel

        AdvancedPanel()

    def test_blank_panel(self):
        from partestimator.smt.gui.panels import BlankPanel

        BlankPanel()

    def test_detail_table(self):
        from partestimator.smt.gui.panels import DetailTable

        DetailTable()

    def test_view3d(self):
        from partestimator.smt.gui.view3d import View3D

        View3D()

    def test_legend(self):
        from partestimator.smt.gui.app import _Legend

        _Legend()


class TestMainWindowConstructs(_StubbedQt):
    """The one that matters. This is where `_stall` blew up."""

    def test_the_window_can_be_built(self):
        from partestimator.smt.config import UnfoldOptions
        from partestimator.smt.gui.app import MainWindow

        MainWindow(UnfoldOptions())

    def test_it_builds_with_no_options_passed(self):
        from partestimator.smt.gui.app import MainWindow

        MainWindow()

    def test_constructing_it_twice_is_fine(self):
        """Catches state accidentally kept on the class rather than instances."""
        from partestimator.smt.gui.app import MainWindow

        MainWindow()
        MainWindow()


class TestCallbacksResolve(_StubbedQt):
    """Every method the window wires up must actually exist.

    A signal connected to a renamed method fails at click time, not at build
    time, so constructing the window alone does not catch it. Checking the
    attributes exist does.
    """

    HANDLERS = (
        "open_file", "load", "export", "show_view", "status", "log",
        "_recompute", "_reopen", "_refresh", "_opened", "_failed",
        "_exported", "_cancel", "_about", "_apply_box", "_apply_options",
        "_arm_picking", "_face_picked", "_warn_stalled", "_set_busy",
        "_enable", "_build_ui", "_toolbar", "_view_controls", "_menu",
    )

    def test_every_handler_exists(self):
        from partestimator.smt.gui.app import MainWindow

        window = MainWindow()
        missing = [h for h in self.HANDLERS if not hasattr(window, h)]
        self.assertEqual(missing, [], f"the window has no {missing}")

    def test_the_view_switch_knows_both_views(self):
        from partestimator.smt.gui.app import MainWindow

        window = MainWindow()
        self.assertEqual(set(window._views), {"flat", "folded"})

    def test_show_view_survives_having_no_scenes(self):
        """Called before a file is open, it must do nothing, not raise."""
        from partestimator.smt.gui.app import MainWindow

        window = MainWindow()
        window.show_view("flat")
        window.show_view("folded")

    def test_recompute_does_nothing_without_a_part(self):
        from partestimator.smt.gui.app import MainWindow

        MainWindow()._recompute()

    def test_picking_a_face_before_opening_anything_is_harmless(self):
        from partestimator.smt.gui.app import MainWindow

        window = MainWindow()
        window._face_picked(3)


class TestViewerHandlesAnEmptyScene(_StubbedQt):
    """The viewer's own guards, exercised rather than assumed."""

    def test_picking_an_empty_view_returns_nothing(self):
        from partestimator.smt.gui.view3d import View3D

        self.assertIsNone(View3D().pick(10.0, 10.0))

    def test_clearing_an_empty_view_is_fine(self):
        from partestimator.smt.gui.view3d import View3D

        view = View3D()
        view.clear()
        self.assertFalse(view.has_scene())

    def test_named_views_and_fit_work_with_nothing_loaded(self):
        from partestimator.smt.gui.view3d import View3D

        view = View3D()
        for name in ("iso", "top", "front", "right", "bottom", "left", "back"):
            view.set_view(name)
        view.fit()

    def test_the_box_can_be_set_and_cleared(self):
        from partestimator.smt.gui.view3d import View3D

        view = View3D()
        view.set_box([0, 0, 0], [10, 5, 1], ("10", "5", "1"))
        view.set_box(None)


if __name__ == "__main__":
    unittest.main()
