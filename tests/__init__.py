"""Test package.

A package rather than a loose folder so `from tests import synthetic` resolves
the same way whether the suite is run by `main.py --test`, by `python -m
unittest discover`, or by pytest -- three entry points that otherwise disagree
about what the top level directory is.
"""
