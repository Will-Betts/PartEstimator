"""Unit tests for the machining timing model (cycle_time, turn_time, calibrate).

Everything here is pure Python plus numpy: no CAD kernel, no Qt. That is what
lets these run on a bare CI runner. Run them with:

    python -m pytest tests/timing
"""
