"""State preparation: the filter study's map."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

from pihm.plotting.style import (
    MUTED,
    REGIME_STYLE,
    SURFACE,
    _finite,
    _grid,
    _legend_above,
)

__all__ = ["EDGE", "filter_map"]


#: The ground state's height above the edge, in gaps, below which the map's scale turns linear.
EDGE: Final = 1e-4


def filter_map(forms: Mapping[str, Sequence[Mapping[str, Any]]], *, epsilon: float = 1e-6) -> Any:
    r"""Draw both forms' records on the filter study's map, and QITE's degree over the minimax's.

    Left, each ``"filters"`` record where the map places it: the gap on :math:`H'`'s scale,
    :math:`x_1-x_0`, against the ground state's height above :math:`-1` in gaps,
    :math:`\lambda_0/\Delta`. Right, QITE's measured degree over the minimax filter's at
    ``epsilon``, against :math:`\lambda_0/\Delta`. A record whose reference is not trusted is
    hollow. :math:`\lambda_0/\Delta` is drawn on a scale linear below :data:`EDGE`, where every
    value places the ground state on the map's edge, and logarithmic above it.

    Parameters
    ----------
    forms
        Each form's summary rows of its filters campaign, keyed by the forms of
        :data:`~pihm.plotting.style.REGIME_STYLE`.
    epsilon
        The infidelity whose measured degrees the right panel compares.

    Returns
    -------
    object
        The matplotlib figure.
    """
    figure, axes = _grid(2, 2, (3.8, 3.0))
    prefix = f"e{round(-math.log10(epsilon))}_"
    qite, minimax = prefix + "qite_degree", prefix + "minimax_degree"
    for form, rows in forms.items():
        colour, marker = REGIME_STYLE[form]
        placed = [r for r in rows if r.get("status") == "ok" and _finite(r.get("offset"))]
        for trusted in (True, False):
            chosen = [r for r in placed if (r.get("trusted") is True) == trusted]
            style = {
                "marker": marker,
                "edgecolors": colour,
                "facecolors": colour if trusted else SURFACE,
                "zorder": 3,
            }
            label = f"{form} form" if trusted else None
            offsets = [float(r["offset"]) for r in chosen]
            axes[0].scatter([float(r["delta"]) for r in chosen], offsets, label=label, **style)
            compared = [r for r in chosen if _finite(r.get(qite)) and _finite(r.get(minimax))]
            axes[1].scatter(
                [float(r["offset"]) for r in compared],
                [float(r[qite]) / float(r[minimax]) for r in compared],
                **style,
            )
    axes[0].set_xscale("log")
    axes[0].set_yscale("symlog", linthresh=EDGE)
    axes[0].set_ylim(bottom=-EDGE / 2.0)
    axes[0].set_xlabel(r"$x_1-x_0$, the gap on the scale of $H'$")
    axes[0].set_ylabel(r"$\lambda_0/\Delta$")
    axes[0].set_title("the map", loc="left")
    axes[1].set_xscale("symlog", linthresh=EDGE)
    axes[1].set_xlim(left=-EDGE / 2.0)
    axes[1].axhline(1.0, color=MUTED, linewidth=0.8, zorder=1)
    axes[1].set_xlabel(r"$\lambda_0/\Delta$")
    axes[1].set_ylabel("QITE degree / minimax degree")
    axes[1].set_title(f"measured, at infidelity {epsilon:g}", loc="left")
    _legend_above(figure, axes, untrusted=True)
    return figure
