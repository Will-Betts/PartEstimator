"""SMTv13 -- Sheet Metal Tool.

STEP in, flat pattern out, for air-bent press brake parts. Detects brake bends
by their geometric signature, develops the blank at an adjustable K-factor, and
emits a DXF, an unfolded STEP solid, a bend table and a material summary.

Three ways in::

    python -m smt                      the window
    smt-unfold part.step --k 0.44      the command line
    from smt import unfold_step        the library

Units: geometry is ALWAYS millimetres internally. Inches exist only at the two
edges -- what is displayed and what is written into a DXF. See `smt.units`.

Layering, which matters if you extend this: `smt.occ` is the ONLY package that
imports a CAD kernel. The transform solver, the allowance maths, the 2D flat
geometry and the reporting all run on numpy alone. That is what lets the
K-factor slider be live, and what lets the test suite check the risky maths on a
machine with nothing but numpy installed.
"""

from .config import Tolerance, UnfoldOptions
from .model import Bend, BendKind, FaceKind, FlatPattern, Flange, Hole, Part

__version__ = "5.0.1"
__appname__ = "SMTv13"

__all__ = [
    "UnfoldOptions",
    "Tolerance",
    "Part",
    "Flange",
    "Bend",
    "BendKind",
    "FaceKind",
    "Hole",
    "FlatPattern",
    "unfold_step",
    "__version__",
    "__appname__",
]


def unfold_step(path, options=None):
    """Load a STEP file and develop its flat pattern.

    Imported lazily so that `import smt` works without a CAD kernel present --
    useful for the maths tests, and for using the reporting layer on
    already-extracted data.
    """
    from .pipeline import unfold_step as _impl

    return _impl(path, options)
