r"""A quantity of both forms against the resolution, a panel per problem and quantity."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pihm.plotting.style import REGIME_STYLE, _finite, _grid, _legend_above, select

__all__ = ["both_forms"]


def both_forms(
    rows: Sequence[Mapping[str, Any]],
    problems: Sequence[str],
    *,
    metrics: Sequence[tuple[str, str]] = (("looseness", "alpha_R/||R||"), ("cx", "cx per V")),
    columns: int = 3,
    linear: Sequence[str] = (),
) -> Any:
    """Draw a quantity of both forms against n: a panel per problem and metric.

    A value either form lacks is left out of its line.

    Parameters
    ----------
    rows
        One row per case with each metric of both forms, as
        :func:`pihm.pipeline.summary.encoding_comparison` and
        :func:`~pihm.pipeline.summary.side_by_side` give them.
    problems
        Problem names, one panel each per metric.
    metrics
        ``(column, label)`` pairs: the column stem, suffixed ``_standard`` and ``_descriptor``.
    columns
        Panels per row.
    linear
        The metrics drawn on a linear axis, such as a count; the rest are logarithmic.

    Returns
    -------
    object
        The matplotlib figure.
    """
    from matplotlib.ticker import MaxNLocator

    figure, axes = _grid(len(problems) * len(metrics), columns, (3.0, 2.4))
    panels = [(column, label, name) for column, label in metrics for name in problems]
    for axis, (column, label, name) in zip(axes, panels, strict=True):
        points = select(rows, problem=name)
        for form, (colour, marker) in REGIME_STYLE.items():
            kept = [p for p in points if _finite(p.get(f"{column}_{form}"))]
            axis.plot(
                [int(p["n"]) for p in kept],
                [float(p[f"{column}_{form}"]) for p in kept],
                color=colour,
                marker=marker,
                linewidth=1.4,
                label=f"{form} form",
            )
        if column in linear:
            axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            axis.set_yscale("log")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_title(f"{name}: {label}", loc="left")
        axis.set_xlabel("n (qubits per axis)")
    _legend_above(figure, axes)
    return figure
