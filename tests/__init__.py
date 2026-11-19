"""Test package.

A package rather than a loose folder so `from tests import synthetic` resolves
the same way whether the suite is run by `main.py --test`, by `python -m
unittest discover`, or by pytest -- three entry points that otherwise disagree
about what the top level directory is.

It also points the extrusion custom library at a file that never exists, so a
developer's own custom_extrusions.csv beside the launcher cannot change what
the tests see. Tests of the custom library pass their own files.

Likewise the shop tuning folder (machining/tuning.py) is pointed at an empty
temporary folder, so a developer's own tuning cannot change any estimate the
tests check. Tests of the tuning library pass their own folders.
"""
import os as _os
import tempfile as _tempfile
from pathlib import Path as _Path

from partestimator.machining.geometry import extrusions as _extrusions

_extrusions.CUSTOM_PATH = _Path(__file__).resolve().parent / "no_custom_extrusions.csv"
_os.environ["PE_TUNING_DIR"] = str(_Path(_tempfile.gettempdir()) / "partestimator-tests-no-tuning")
