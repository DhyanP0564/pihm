r"""The analysis notebooks' shared figure style and figures; nothing in the package imports it.

Every figure reads only the results tables and arrays (``summary.csv`` and the records' NPZ files,
through :mod:`pihm.pipeline.summary`) and never recomputes.
matplotlib is imported only when a figure is drawn: it belongs to the optional ``notebooks``
extra, so the package itself imports without it.

The style follows a reference data-visualisation palette. The encoding is fixed:

* **Colour follows the entity.** The corrected variant is always categorical slot 1 (blue), the
  faithful variant slot 2 (orange) and a third series slot 3 (aqua); a figure that sets the two
  forms side by side draws the standard form in slot 1 and the descriptor in slot 3. These three
  slots are validated for colour-vision deficiency across all pairs. Exact solutions are drawn in
  ink, and context (bounds) in muted grey.
* **Colour is never alone.** Every variant also has its own marker shape, and a legend is present
  wherever two or more series are.
* **Marks are thin**, and grids and axes are solid hairlines. There is one y-axis per chart:
  quantities on different scales get separate panels.
* **An untrusted ground state** is a hollow marker.

The figures are grouped by what they show:

* :mod:`~pihm.plotting.reproduction`: the paper's panels;
* :mod:`~pihm.plotting.scaling`: both forms' quantities against the resolution;
* :mod:`~pihm.plotting.stateprep`: the filter map;
* :mod:`~pihm.plotting.comparison`: the forms and their encodings;
* :mod:`~pihm.plotting.checks`: measured quantities against their bounds;
* :mod:`~pihm.plotting.lift`: the Carleman lift against its order, resolution and strength;
* :mod:`~pihm.plotting.solutions` and :mod:`~pihm.plotting.viewer`: a case's decoded fields, the
  spy plots of its matrices and its metrics, and the solution viewer that shows them. Signed
  fields there take the diverging blue–red map about a grey zero.

:mod:`~pihm.plotting.style` holds the palette and the helpers. Every name below is available as
``plotting.<name>``.
"""

from pihm.plotting.checks import Check, checks
from pihm.plotting.comparison import (
    by_resolution,
    cost_factors,
    cost_ratios,
    cost_walls,
    exponent_dots,
    panel_dots,
)
from pihm.plotting.lift import layout_cost, lift_convergence, order_sweep
from pihm.plotting.reproduction import (
    convergence,
    error_maps,
    gap_collapse,
    maclaurin_order,
    scale_factor_rows,
    solution_panels,
)
from pihm.plotting.scaling import both_forms
from pihm.plotting.solutions import (
    case_markdown,
    metrics_markdown,
    snapshot,
    solution_figures,
    spy,
)
from pihm.plotting.stateprep import EDGE, filter_map
from pihm.plotting.style import (
    DIVERGING,
    ENCODING_STYLE,
    INK,
    REGIME_STYLE,
    SERIES,
    STYLES,
    SURFACE,
    VARIANT_STYLE,
    apply_style,
    markdown_table,
    select,
)
from pihm.plotting.viewer import show_case, viewer

__all__ = [
    "DIVERGING",
    "EDGE",
    "ENCODING_STYLE",
    "INK",
    "REGIME_STYLE",
    "SERIES",
    "STYLES",
    "SURFACE",
    "VARIANT_STYLE",
    "Check",
    "apply_style",
    "both_forms",
    "by_resolution",
    "case_markdown",
    "checks",
    "convergence",
    "cost_factors",
    "cost_ratios",
    "cost_walls",
    "error_maps",
    "exponent_dots",
    "filter_map",
    "gap_collapse",
    "layout_cost",
    "lift_convergence",
    "maclaurin_order",
    "markdown_table",
    "metrics_markdown",
    "order_sweep",
    "panel_dots",
    "scale_factor_rows",
    "select",
    "show_case",
    "snapshot",
    "solution_figures",
    "solution_panels",
    "spy",
    "viewer",
]
