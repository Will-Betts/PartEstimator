"""The real window, offscreen: open parts, check the right tool comes up.

Drives `UnifiedWindow` exactly as a click would -- classification in its
process pool, the sheet metal tool's worker thread, the machining child
process -- and checks where each part lands and that overriding works.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

# Use the real display when there is one (Windows, or Linux with $DISPLAY);
# otherwise Qt's offscreen platform. VTK cannot render there -- it aborts the
# process -- so the tests that load the machining viewer need a real display.
if os.name != "nt" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HEADLESS = os.environ.get("QT_QPA_PLATFORM") == "offscreen"

try:
    import OCP  # noqa: F401
    from PySide6.QtWidgets import QApplication

    HAVE = True
except Exception:  # noqa: BLE001
    HAVE = False


@unittest.skipUnless(HAVE, "needs OCP and PySide6")
class TestUnifiedWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests import occparts

        cls.app = QApplication.instance() or QApplication([])
        cls.tmp = Path(tempfile.mkdtemp(prefix="unified_"))
        devnull = os.open(os.devnull, os.O_WRONLY)
        saved = os.dup(1)
        os.dup2(devnull, 1)
        try:
            cls.sheet_part = occparts.l_bracket(cls.tmp)
            cls.mach_part = occparts.machined_plate(cls.tmp)
        finally:
            os.dup2(saved, 1)
            os.close(devnull)
            os.close(saved)

        from partestimator.app import UnifiedWindow

        cls.w = UnifiedWindow()
        # Not shown: nothing here needs pixels, and VTK cannot render on
        # Qt's offscreen platform (no OpenGL context there).

    @classmethod
    def tearDownClass(cls):
        cls.w.close()
        cls._pump(0.3)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def _pump(cls, seconds):
        end = time.time() + seconds
        while time.time() < end:
            cls.app.processEvents()
            time.sleep(0.02)

    def _wait(self, timeout=120):
        w = self.w
        end = time.time() + timeout
        self._pump(0.3)
        while time.time() < end:
            busy = (w._future is not None or w.sheet.btn_cancel.isEnabled()
                    or w.mach.cancel_button.isEnabled())
            if not busy:
                break
            self._pump(0.1)
        self._pump(0.5)

    def test_1_sheet_part_opens_in_sheet_tool(self):
        from partestimator.app import PAGE_SHEET
        from partestimator.classify import SHEET

        self.w.open_path(self.sheet_part)
        self._wait()
        self.assertEqual(self.w.detected.kind, SHEET)
        self.assertEqual(self.w.stack.currentIndex(), PAGE_SHEET)
        self.assertIsNotNone(self.w.sheet.part, "sheet tool did not finish opening")
        self.assertEqual(len(self.w.sheet.part.bends), 1)
        self.assertTrue(self.w.sheet.btn_export.isEnabled())

    @unittest.skipIf(HEADLESS, "machining viewer (VTK) needs a real display")
    def test_2_machined_part_opens_in_machining_tool(self):
        from partestimator.app import PAGE_MACHINED
        from partestimator.classify import MACHINED

        self.w.open_path(self.mach_part)
        self._wait()
        self.assertEqual(self.w.detected.kind, MACHINED)
        self.assertEqual(self.w.stack.currentIndex(), PAGE_MACHINED)
        self.assertTrue(self.w.mach.results, "machining analysis did not finish")
        self.assertIn("min", self.w.mach.lbl_cycle_total.text())

    @unittest.skipIf(HEADLESS, "machining viewer (VTK) needs a real display")
    def test_3_override_loads_the_other_tool(self):
        from partestimator.app import PAGE_SHEET
        from partestimator.classify import SHEET

        self.w.open_path(self.mach_part)
        self._wait()
        self.w.set_mode(SHEET, user=True)
        self._wait()
        self.assertEqual(self.w.stack.currentIndex(), PAGE_SHEET)
        self.assertIn("by hand", self.w.banner.text())


if __name__ == "__main__":
    unittest.main()
