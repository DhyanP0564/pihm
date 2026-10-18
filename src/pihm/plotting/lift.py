r"""The Carleman lift's figures: its error against the order, the resolution and the strength.

Each line is one Carleman order, or one strength :math:`a/\nu` of Burgers, in the sequential ramp
from light (the lowest) to dark, with a marker of its own; a panel per form, or per number of
modes. An untrusted ground state is a hollow marker, as everywhere.
:func:`layout_cost` draws the lift's coupling on a circuit in its two layouts against the number
of modes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pihm.plotting.style import LAYOUT_STYLE, SEQUENTIAL, SURFACE, _finite, _grid, _legend_above

__all__ = ["layout_cost", "lift_convergence", "order_sweep"]

#: A marker per line, in order.
_MARKERS = ("o", "s", "D", "^", "v", "P", "X")


def _ramp(count: int) -> list[str]:
    """Return ``count`` colours from the sequential ramp, light to dark, skipping its palest."""
    usable = SEQUENTIAL[2:]
    if count <= 1:
        return [usable[-1]]
    return [usable[round(i * (len(usable) - 1) / (count - 1))] for i in range(count)]


def _line(axis: Any, points: Sequence[tuple[float, float, bool]], colour: str, marker: str) -> None:
    """Draw one line with its markers, hollow where untrusted."""
    axis.plot([p[0] for p in points], [p[1] for p in points], color=colour, linewidth=1.3)
    for x, value, trusted in points:
        axis.plot(
            x,
            value,
            marker=marker,
            color=colour,
            markerfacecolor=colour if trusted else SURFACE,
            markeredgecolor=colour if not trusted else SURFACE,
            markeredgewidth=1.2,
            zorder=3,
        )


def order_sweep(
    rows: Sequence[Mapping[str, Any]],
    column: str,
    label: str,
    *,
    forms: Sequence[str] = ("standard", "descriptor"),
) -> Any:
    """Draw ``column`` against n, a line per Carleman order, a panel per form, on shared axes.

    Parameters
    ----------
    rows
        One problem's summary rows, each with ``carleman``, ``regime`` and ``n``, and
        ``trusted`` where the row has a reference (a missing one draws as trusted).
    column
        The column drawn, on a logarithmic axis.
    label
        The y-axis label.
    forms
        The forms, one panel each.
    """
    from matplotlib.ticker import MaxNLocator

    orders = sorted({int(r["carleman"]) for r in rows})
    colours = _ramp(len(orders))
    figure, axes = _grid(len(forms), len(forms), (3.2, 2.6))
    for axis, form in zip(axes, forms, strict=True):
        for order, colour, marker in zip(orders, colours, _MARKERS, strict=False):
            chosen = sorted(
                (
                    r
                    for r in rows
                    if r["regime"] == form and int(r["carleman"]) == order and _finite(r[column])
                ),
                key=lambda r: int(r["n"]),
            )
            points = [
                (int(r["n"]), float(r[column]), r.get("trusted", True) is True) for r in chosen
            ]
            _line(axis, [p for p in points if p[1] > 0], colour, marker)
            axis.plot([], [], color=colour, marker=marker, label=f"K = {order}")
        axis.set_yscale("log")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_title(form, loc="left")
        axis.set_xlabel("n (qubits on the time axis)")
    axes[0].set_ylabel(label)
    for axis in axes[1:]:
        axis.sharey(axes[0])
    untrusted = any(r.get("trusted", True) is not True for r in rows)
    _legend_above(figure, axes, untrusted=untrusted)
    return figure


def lift_convergence(
    series: Mapping[str, Mapping[int, Sequence[tuple[int, float, bool]]]], label: str
) -> Any:
    """Draw the lift's error against its order K, a line per strength, a panel per mode count.

    Parameters
    ----------
    series
        By strength's label (``"a/ν = 1"``, weakest first), by number of modes, the points
        ``(K, error, trusted)``.
    label
        The y-axis label.
    """
    from matplotlib.ticker import MaxNLocator

    modes = sorted({m for per in series.values() for m in per})
    colours = _ramp(len(series))
    figure, axes = _grid(len(modes), len(modes), (3.0, 2.6))
    for axis, count in zip(axes, modes, strict=True):
        for (name, per), colour, marker in zip(series.items(), colours, _MARKERS, strict=False):
            points = sorted(p for p in per.get(count, ()) if _finite(p[1]) and p[1] > 0)
            _line(axis, points, colour, marker)
            axis.plot([], [], color=colour, marker=marker, label=name)
        axis.set_yscale("log")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_title(f"M = {count} modes", loc="left")
        axis.set_xlabel("Carleman order K")
    axes[0].set_ylabel(label)
    for axis in axes[1:]:
        axis.sharey(axes[0])
    untrusted = any(not p[2] for per in series.values() for points in per.values() for p in points)
    _legend_above(figure, axes, untrusted=untrusted)
    return figure


def layout_cost(rows: Sequence[Mapping[str, Any]], column: str, label: str) -> Any:
    """Draw a lift's ``column`` against its number of modes: a line per layout and order.

    A panel per rung, on logarithmic axes; the layout's colour (the tensor layout in its own, the
    symmetric one in secondary ink, dashed), a marker per order.

    Parameters
    ----------
    rows
        ``"lift"`` summary rows, one per problem, order and layout, each with ``rung``,
        ``modes``, ``carleman`` and ``lift``.
    column
        The column drawn, on a logarithmic axis, such as ``cx_count``.
    label
        The y-axis label.
    """
    from matplotlib.ticker import NullLocator

    rungs = sorted({str(r["rung"]) for r in rows}, key=lambda rung: int(rung[1:]))
    orders = sorted({int(r["carleman"]) for r in rows})
    figure, axes = _grid(len(rungs), len(rungs), (3.2, 2.6))
    for axis, rung in zip(axes, rungs, strict=True):
        for layout, (colour, _marker) in LAYOUT_STYLE.items():
            for order, marker in zip(orders, _MARKERS, strict=False):
                points = sorted(
                    (float(r["modes"]), float(r[column]))
                    for r in rows
                    if r["rung"] == rung
                    and r["lift"] == layout
                    and int(r["carleman"]) == order
                    and _finite(r[column])
                )
                if not points:
                    continue
                axis.plot(
                    [p[0] for p in points],
                    [p[1] for p in points],
                    color=colour,
                    marker=marker,
                    linestyle="-" if layout == "tensor" else "--",
                    linewidth=1.3,
                    label=f"{layout}, K = {order}",
                )
        axis.set_xscale("log")
        axis.set_yscale("log")
        counts = sorted({int(r["modes"]) for r in rows if r["rung"] == rung})
        axis.set_xticks(counts, [str(count) for count in counts])
        axis.xaxis.set_minor_locator(NullLocator())
        axis.set_title(rung, loc="left")
        axis.set_xlabel("modes M")
    axes[0].set_ylabel(label)
    _legend_above(figure, axes, columns=len(orders))
    return figure
