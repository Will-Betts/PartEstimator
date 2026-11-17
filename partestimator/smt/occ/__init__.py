"""The only part of the package that touches a CAD kernel.

Everything else runs on numpy alone, which is what lets the transform solver,
the allowance maths and the whole feature classifier be tested without a 500 MB
OpenCASCADE install.

Kernel access is via OCP -- the OpenCASCADE bindings that CadQuery and build123d
both sit on -- called DIRECTLY rather than through a modelling API. Feature
recognition needs surface parameters, axis data, face orientation flags and
topology adjacency, none of which a fluent modelling API is designed to expose,
and depending on one for STEP reading alone would add a package to the supply
chain for no benefit. `cadquery-ocp` is therefore the only CAD dependency.

THE COST OF THIS BOUNDARY, STATED HONESTLY
------------------------------------------
Quarantining the kernel here is what makes the risky mathematics testable. It
also means every bug that lives on THIS side of the line is invisible to the
test suite until the code reaches a machine with OCP installed. Several have.
`tests/test_imports.py` closes part of that gap by statically resolving every
OCP symbol against its module's imports, which catches the commonest kind --
a name used but never imported -- without needing the kernel present.
"""

from __future__ import annotations

_IMPORT_HINT = """
smt.occ requires the OpenCASCADE Python bindings (cadquery-ocp-novtk).

Run `.\\pe.ps1 setup`, which installs the exact pinned versions into a .venv inside
this folder.

The rest of the package -- the unfold solver, allowance maths, feature
classification and reporting -- imports and runs without a kernel.
"""


def require_kernel() -> None:
    try:
        import OCP  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(_IMPORT_HINT) from exc
