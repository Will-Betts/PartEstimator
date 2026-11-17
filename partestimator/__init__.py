"""Part Estimator -- one window for sheet metal and machined parts.

Open a STEP file; the part is classified (see `partestimator.classify`) and
opened in either:

  * the sheet metal tool (`partestimator.smt`, formerly SMTv13) -- flat
    pattern, bend table, DXF/STEP/report export; or
  * the machining cycle-time estimator (`partestimator.machining`, formerly
    STEP Estimator v2.5) -- geometry features, stock, cycle-time estimate,
    CSV/JSON training rows.

Kept import-light on purpose: nothing here pulls in the CAD kernel or Qt.
"""

__version__ = "1.1.0"
__appname__ = "Part Estimator"
