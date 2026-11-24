"""The command line's start-up failures and the searchable picker's menu and
typing paths.

The start-up tests are plain Python; the picker needs Qt.
"""
from __future__ import annotations

import argparse
import io
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

if os.name != "nt" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QFocusEvent, QKeyEvent
    from PySide6.QtWidgets import QApplication, QMenu

    HAVE_QT = True
except Exception:  # noqa: BLE001
    HAVE_QT = False


class TestStartUpFailures(unittest.TestCase):
    def setUp(self):
        from partestimator import __main__ as cli

        self.cli = cli
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        p = mock.patch.object(cli, "_tool_dir", return_value=self.tmp)
        p.start()
        self.addCleanup(p.stop)

    def test_gui_fails_before_the_window(self):
        args = argparse.Namespace(gl="auto", step=None)
        err = io.StringIO()
        with mock.patch("partestimator.opengl.choose", side_effect=ImportError("no Qt")), \
                mock.patch.object(sys, "stderr", err):
            self.assertEqual(self.cli.cmd_gui(args), 1)
        self.assertIn("pe.ps1 reset", err.getvalue())
        self.assertIn("no Qt", (self.tmp / "last_error.log").read_text(encoding="utf-8"))

    @unittest.skipUnless(HAVE_QT, "needs Qt")
    def test_gui_fails_in_the_window(self):
        args = argparse.Namespace(gl="auto", step=None)
        with mock.patch("partestimator.opengl.choose", return_value="software"), \
                mock.patch("partestimator.smt.diag.setup_logging"), \
                mock.patch("partestimator.smt.diag.install_excepthook"), \
                mock.patch("partestimator.app.run", side_effect=RuntimeError("window")), \
                mock.patch.object(sys, "stderr", io.StringIO()) as err:
            self.assertEqual(self.cli.cmd_gui(args), 1)
        self.assertIn("window", err.getvalue())

    def test_no_console_shows_a_message_box(self):
        boxes = []
        fake = mock.MagicMock()
        fake.windll.user32.MessageBoxW.side_effect = lambda *a: boxes.append(a)
        with mock.patch.object(sys, "stderr", None), \
                mock.patch.object(sys, "platform", "win32"), \
                mock.patch.dict(sys.modules, {"ctypes": fake}), \
                mock.patch.object(Path, "write_text", side_effect=OSError("read-only")):
            self.cli._report_startup_failure(ValueError("boom"))
        self.assertIn("boom", boxes[0][1])
        self.assertNotIn("saved to", boxes[0][1])
        broken = mock.MagicMock()
        broken.windll.user32.MessageBoxW.side_effect = OSError("no desktop")
        with mock.patch.object(sys, "stderr", None), \
                mock.patch.object(sys, "platform", "win32"), \
                mock.patch.dict(sys.modules, {"ctypes": broken}):
            self.cli._report_startup_failure(ValueError("boom"))      # never raises

    def test_classify_uses_the_saved_cutoff(self):
        (self.tmp / "a.step").write_text("x", encoding="utf-8")
        out = io.StringIO()
        with mock.patch("partestimator.classify.classify_file",
                        side_effect=OSError("unreadable")) as cf, \
                mock.patch.object(sys, "stdout", out), \
                mock.patch.object(sys, "stderr", io.StringIO()):
            rc = self.cli.main(["classify", str(self.tmp / "a.step")])
        self.assertNotEqual(rc, 0)
        self.assertTrue(cf.called)


@unittest.skipUnless(HAVE_QT, "needs Qt")
class TestPickerInteraction(unittest.TestCase):
    def setUp(self):
        from partestimator.ui_common import material_picker

        QApplication.instance() or QApplication([])
        self.p = material_picker(["Aluminum 6061-T6", "Aluminum 7075-T6", "Steel 1018"],
                                 ["Other"])
        self.got = []
        self.p.currentTextChanged.connect(self.got.append)

    def test_menu_opens_once_and_typing_in_it_searches(self):
        p = self.p
        p.combo.showPopup()
        first = p.menu
        p.show_menu()                                      # a second open replaces it
        self.assertIsNot(p.menu, first)
        menu = p.menu
        handled = p.eventFilter(menu, QKeyEvent(QEvent.KeyPress, Qt.Key_S, Qt.NoModifier, "s"))
        self.assertTrue(handled)
        self.assertEqual(p.line.text(), "s")
        p.eventFilter(menu, QKeyEvent(QEvent.KeyPress, Qt.Key_Space, Qt.NoModifier, " "))
        p._menu_closed()
        self.assertIsNone(p.menu)
        p._menu_closed()                                   # already closed

    def test_typing_in_the_box(self):
        p = self.p
        p.eventFilter(p.line, QFocusEvent(QEvent.FocusIn))
        p._on_search_edited("zzz")                         # nothing matches
        p._on_search_edited("")
        p._on_search_edited("7075")
        p.line.setText("Steel 1018")
        p._on_return()
        self.assertEqual(p.currentText(), "Steel 1018")
        p.line.setText("6061")
        p._on_return()
        self.assertEqual(p.currentText(), "Aluminum 6061-T6")
        p.line.setText("")
        p._on_return()
        self.assertEqual(p.currentText(), "Aluminum 6061-T6")
        p._on_completion(p.display_text("Aluminum 7075-T6"))
        self.assertEqual(p.currentText(), "Aluminum 7075-T6")
        p.line.setText("half typed")
        handled = p.eventFilter(p.line, QKeyEvent(QEvent.KeyPress, Qt.Key_Escape,
                                                  Qt.NoModifier))
        self.assertTrue(handled)
        self.assertEqual(p.line.text(), p.display_text("Aluminum 7075-T6"))
        p.commit("Unobtainium")                            # not on offer
        self.assertEqual(p.currentText(), "Aluminum 7075-T6")


if __name__ == "__main__":
    unittest.main()
