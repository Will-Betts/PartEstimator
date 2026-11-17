"""Drive the window's panels with REAL analysis output.

The bug this exists for:

    TypeError: '>' not supported between instances of 'str' and 'int'

`report.BendRow.direction` is a string -- "UP" or "DOWN" -- and the bend table
compared it to an integer. The mismatch is invisible to every static check,
because both sides are perfectly valid Python; it only appears when a real row
meets the code that formats it.

Worse, it surfaced during `_refresh`, which runs while a file is opening, so the
exception aborted the open before the 3D view was ever shown. The reported
symptom was "nothing appears until I click a view button", which sounds like a
rendering fault and is nothing of the kind.

So these tests feed genuine `report` output through the genuine panels, using
the synthetic parts the transform solver is checked against. No CAD kernel is
needed for any of it: the parts, the unfold, the report and the panels are all
in the kernel-free layer, and only Qt had to be faked.
"""

from __future__ import annotations

import unittest

from tests import qtstub
from tests.test_gui_smoke import _StubbedQt


def _analysed(part_factory):
    """A part carried all the way to a flat pattern, with no kernel involved."""
    from partestimator.smt.config import UnfoldOptions
    from partestimator.smt.model2d import recompute

    part = part_factory()
    opts = UnfoldOptions()
    flat, _result = recompute(part, opts)
    return part, flat, opts


class TestBendTableAcceptsRealRows(_StubbedQt):
    def test_real_bend_rows_render(self):
        from partestimator.smt.gui.panels import DetailTable
        from partestimator.smt.report import bend_table, resolve_unit
        from tests import synthetic

        part, _flat, opts = _analysed(synthetic.u_channel)
        rows = bend_table(part, opts)
        self.assertTrue(rows, "the fixture produced no bends to render")

        DetailTable().show_bends(rows, resolve_unit(part, opts))

    def test_direction_really_is_a_string(self):
        """Pins the contract the panel got wrong, on the report's side."""
        from partestimator.smt.report import bend_table
        from tests import synthetic

        part, _flat, opts = _analysed(synthetic.u_channel)
        for row in bend_table(part, opts):
            with self.subTest(bend=row.n):
                self.assertIsInstance(row.direction, str)
                self.assertIn(row.direction.upper(), ("UP", "DOWN"))

    def test_it_renders_for_several_different_parts(self):
        from partestimator.smt.gui.panels import DetailTable
        from partestimator.smt.report import bend_table, resolve_unit
        from tests import synthetic

        # `closed_box` is deliberately excluded: it is the fixture that proves a
        # closed section CANNOT be developed, so it raises by design and never
        # reaches a panel.
        for name in ("u_channel", "z_section"):
            with self.subTest(part=name):
                part, _flat, opts = _analysed(getattr(synthetic, name))
                DetailTable().show_bends(
                    bend_table(part, opts), resolve_unit(part, opts)
                )


class TestTheWholeRefreshRuns(_StubbedQt):
    """Every results panel, against a real part, in one go.

    This is the test that would have caught the shipped bug: `_refresh` touches
    the blank panel, the bend table, the feature table, the notes table and the
    unit formatting, and any type mismatch in any of them shows up here.
    """

    def _window_with(self, factory):
        from partestimator.smt.gui.app import MainWindow

        part, flat, opts = _analysed(factory)
        window = MainWindow(opts)
        window.part = part
        window.flat2d = flat
        return window

    def test_refresh_runs_on_a_bent_part(self):
        from tests import synthetic

        self._window_with(synthetic.u_channel)._refresh()

    def test_refresh_runs_on_every_fixture(self):
        from tests import synthetic

        for name in ("u_channel", "z_section"):
            with self.subTest(part=name):
                self._window_with(getattr(synthetic, name))._refresh()

    def test_refresh_runs_in_inches_too(self):
        """Unit conversion runs through every formatted field."""
        from tests import synthetic

        window = self._window_with(synthetic.u_channel)
        window.opts.display_units = "in"
        window._refresh()


class TestUnflattenablePartsStillReportCleanly(_StubbedQt):
    """A closed section cannot be developed, and must say so, not crash."""

    def test_a_closed_box_raises_the_right_error(self):
        from partestimator.smt.graph import UnflattenableError
        from tests import synthetic

        with self.assertRaises(UnflattenableError):
            _analysed(synthetic.closed_box)


class TestOpeningSurvivesABrokenPanel(_StubbedQt):
    """Drawing the part must not depend on the results panels working.

    v11 refreshed the panels and then showed the model, so one exception in a
    table left the viewport empty. The 3D view is the one thing that has to
    appear regardless -- if the numbers cannot be formatted, you should still be
    able to look at the geometry and see why.
    """

    def test_the_view_is_shown_even_if_refresh_raises(self):
        from partestimator.smt.gui.app import MainWindow
        from partestimator.smt.gui.worker import OpenResult
        from tests import synthetic

        part, flat, opts = _analysed(synthetic.u_channel)

        window = MainWindow(opts)
        shown = []
        window.show_view = lambda key: shown.append(key)

        def explode():
            raise TypeError("exactly the kind of thing that shipped")

        window._refresh = explode

        window._opened(OpenResult(
            part=part, shape=None, stem="x", flat2d=flat, result=None,
        ))

        self.assertTrue(
            shown, "the 3D view was never shown when the panels failed"
        )


if __name__ == "__main__":
    unittest.main()
