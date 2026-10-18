r"""Measured quantities against their bounds or references: the acceptance checks, in both forms.

:func:`checks` draws one panel per :class:`Check`, a point per record: the measured quantity
against its bound or reference, each series in its style (:data:`~pihm.plotting.style.STYLES`),
with the diagonal where the two are equal and, optionally, a bar. A record whose reference is not
trusted is hollow.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pihm.plotting.style import MUTED, STYLES, SURFACE, _grid, _legend_above, _titles

__all__ = ["Check", "checks"]


@dataclass(frozen=True)
class Check:
    """One panel of :func:`checks`.

    Attributes
    ----------
    series
        Each series' records, by its name in :data:`~pihm.plotting.style.STYLES`.
    x, y
        The columns drawn against each other.
    xlabel, ylabel, title
        The axes' labels and the panel's title.
    diagonal
        Whether to draw the line where ``y`` equals ``x``.
    bar
        A horizontal line at this value (an acceptance bar), labelled ``bar_label``.
    bar_label
        The bar's legend entry.
    logx, logy
        Whether each axis is logarithmic.
    """

    series: Mapping[str, Sequence[Mapping[str, Any]]]
    x: str
    y: str
    xlabel: str
    ylabel: str
    title: str
    diagonal: bool = True
    bar: float | None = None
    bar_label: str = ""
    logx: bool = True
    logy: bool = True


def _drawable(value: Any, log: bool) -> bool:
    """Whether a cell holds a finite number, positive on a logarithmic axis."""
    finite = isinstance(value, (int, float)) and math.isfinite(value)
    return finite and (value > 0 or not log)


def checks(panels: Sequence[Check], *, columns: int = 2) -> Any:
    """Draw each check's records, a panel per :class:`Check`.

    Parameters
    ----------
    panels
        The checks, in order.
    columns
        Panels per row.

    Returns
    -------
    object
        The matplotlib figure.
    """
    wide = len(panels) == 1
    figure, axes = _grid(len(panels), min(columns, len(panels)), (4.4 if wide else 3.6, 3.4))
    for axis, check in zip(axes, panels, strict=True):
        low, high = math.inf, -math.inf
        for name, rows in check.series.items():
            colour, marker = STYLES[name]
            points = [
                r
                for r in rows
                if _drawable(r.get(check.x), check.logx) and _drawable(r.get(check.y), check.logy)
            ]
            for trusted in (True, False):
                chosen = [r for r in points if (r.get("trusted", True) is True) == trusted]
                axis.scatter(
                    [float(r[check.x]) for r in chosen],
                    [float(r[check.y]) for r in chosen],
                    marker=marker,
                    edgecolors=colour,
                    facecolors=colour if trusted else SURFACE,
                    label=name if trusted else None,
                    zorder=3,
                )
            values = [float(r[k]) for r in points for k in (check.x, check.y)]
            low, high = min([low, *values]), max([high, *values])
        if check.diagonal and low < high:
            axis.plot([low, high], [low, high], color=MUTED, linewidth=0.9, zorder=1)
        if check.bar is not None:
            axis.axhline(
                check.bar, color=MUTED, linewidth=0.9, linestyle=(0, (4, 2)), label=check.bar_label
            )
        axis.set_xscale("log" if check.logx else "linear")
        axis.set_yscale("log" if check.logy else "linear")
        axis.set_xlabel(check.xlabel)
        axis.set_ylabel(check.ylabel)
        _titles(axis, check.title, "")
    untrusted = any(
        r.get("trusted", True) is not True
        for check in panels
        for rows in check.series.values()
        for r in rows
    )
    _legend_above(figure, axes, untrusted=untrusted, columns=3 if wide else None)
    return figure
