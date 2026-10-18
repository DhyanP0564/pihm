r"""The forms and their encodings compared: where the cost lies, and case by case.

:func:`cost_walls` and :func:`exponent_dots` read :func:`pihm.pipeline.results.encoding_scaling`
and :func:`~pihm.pipeline.results.degree_exponents`: each encoding's degree against the
resolution, with the two ratios that set it. :func:`cost_ratios` and :func:`cost_factors` read
:func:`~pihm.pipeline.results.cost_comparison`'s table, whose ratios are the standard form's over
the descriptor's: above the break-even line at one, the descriptor costs less. A case whose
reference is not trusted is hollow.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pihm.plotting.style import (
    ENCODING_STYLE,
    INK,
    INK_SECONDARY,
    MUTED,
    STYLES,
    SURFACE,
    _finite,
    _legend_above,
    _titles,
)

__all__ = [
    "FACTORS",
    "by_resolution",
    "cost_factors",
    "cost_ratios",
    "cost_walls",
    "exponent_dots",
    "panel_dots",
]

#: The factors of a preparation's total cx, as :func:`cost_factors` draws them.
FACTORS = (
    ("total", "total cx"),
    ("preparations", "preparations"),
    ("degree", "filter degree"),
    ("step", "cx per walk step"),
)


def _mark(axis: Any, x: Sequence[float], y: Sequence[float], trusted: Sequence[bool]) -> None:
    """Draw points in ink, hollow where untrusted, with a surface ring between overlaps."""
    for u, v, sure in zip(x, y, trusted, strict=True):
        axis.plot(
            u,
            v,
            marker="o",
            color=INK,
            markerfacecolor=INK if sure else SURFACE,
            markeredgecolor=INK,
            markeredgewidth=1.2,
            zorder=3,
        )


def _key(figure: Any) -> None:
    """Add the break-even line and the hollow marker to the legend above the panels."""
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=MUTED, linewidth=0.9),
        Line2D([], [], marker="o", linestyle="none", color=INK, markerfacecolor=SURFACE),
    ]
    labels = ["break-even", "a reference not trusted, or not built"]
    figure.legend(handles, labels, loc="outside upper center", ncols=2)


def cost_ratios(table: Sequence[Mapping[str, Any]], *, columns: int = 5) -> Any:
    """Draw the standard form's total cx over the descriptor's against n, a panel per problem.

    The panels share one logarithmic y-axis, so their ratios compare across problems.

    Parameters
    ----------
    table
        :func:`pihm.pipeline.results.cost_comparison`'s rows.
    columns
        Panels per row.

    Returns
    -------
    object
        The matplotlib figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    problems = sorted({str(row["problem"]) for row in table})
    rows = -(-len(problems) // columns)
    figure, grid = plt.subplots(
        rows,
        columns,
        figsize=(2.3 * columns, 1.9 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    axes = list(grid.ravel())
    for spare in axes[len(problems) :]:
        spare.remove()
    for index, (axis, problem) in enumerate(zip(axes, problems, strict=False)):
        points = sorted((r for r in table if r["problem"] == problem), key=lambda r: int(r["n"]))
        ns = [int(r["n"]) for r in points]
        ratios = [float(r["total"]) for r in points]
        axis.axhline(1.0, color=MUTED, linewidth=0.9, zorder=1)
        axis.plot(ns, ratios, color=INK, linewidth=1.2, zorder=2)
        _mark(axis, ns, ratios, [bool(r["trusted"]) for r in points])
        axis.set_yscale("log")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_title(problem, loc="left")
        if index % columns == 0:
            axis.set_ylabel("standard ÷ descriptor")
        if index >= len(problems) - columns:
            axis.set_xlabel("n (qubits per axis)")
    _key(figure)
    return figure


def cost_factors(table: Sequence[Mapping[str, Any]]) -> Any:
    """Draw, at each problem's largest n both forms estimated, the cost ratio and its factors.

    One column per factor of :data:`FACTORS`, sharing a logarithmic x-axis; the total is the
    product of the other three.

    Parameters
    ----------
    table
        :func:`pihm.pipeline.results.cost_comparison`'s rows.

    Returns
    -------
    object
        The matplotlib figure.
    """
    import matplotlib.pyplot as plt

    last: dict[str, Mapping[str, Any]] = {}
    for row in table:
        problem = str(row["problem"])
        if problem not in last or int(row["n"]) > int(last[problem]["n"]):
            last[problem] = row
    problems = sorted(last, reverse=True)
    figure, axes = plt.subplots(
        1,
        len(FACTORS),
        figsize=(2.3 * len(FACTORS), 0.26 * len(problems) + 1.3),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    positions = list(range(len(problems)))
    for axis, (column, label) in zip(axes, FACTORS, strict=True):
        axis.axvline(1.0, color=MUTED, linewidth=0.9, zorder=1)
        values = [float(last[p][column]) for p in problems]
        _mark(axis, values, positions, [bool(last[p]["trusted"]) for p in problems])
        axis.set_xscale("log")
        axis.grid(axis="y", visible=False)
        _titles(axis, label, "standard ÷ descriptor")
    axes[0].set_yticks(positions, [f"{p}, n = {last[p]['n']}" for p in problems])
    _key(figure)
    return figure


def _series(axis: Any, rows: Sequence[Mapping[str, Any]], column: str, encoding: str) -> None:
    """Draw one encoding's ``column`` against :math:`N`: a line, and markers hollow if untrusted."""
    colour, marker = ENCODING_STYLE[encoding]
    Ns = [2 ** int(row["n"]) for row in rows]
    values = [float(row[column]) for row in rows]
    axis.plot(Ns, values, color=colour, linewidth=1.4, label=encoding, zorder=2)
    for x, y, row in zip(Ns, values, rows, strict=True):
        trusted = bool(row["trusted"])
        axis.plot(
            x,
            y,
            marker=marker,
            color=colour,
            markerfacecolor=colour if trusted else SURFACE,
            markeredgecolor=colour if not trusted else SURFACE,
            markeredgewidth=1.2,
            zorder=3,
        )


def cost_walls(
    table: Sequence[Mapping[str, Any]], exponents: Sequence[Mapping[str, Any]] = ()
) -> Any:
    r"""Draw where one problem's cost lies: the encoding, the Hamiltonian, and the filter's degree.

    Three panels against :math:`N=2^n`, log-log, one line per encoding: the looseness
    :math:`\alpha_H/\|H\|`, the relative gap :math:`\Delta/\|H\|` and the degree, which grows
    as the square root of their ratio. An encoding whose relative gaps are an earlier one's
    encodes the same Hamiltonian, and its line is left out of the middle panel. The degree is
    T3's, with T2's measured degree marked where it ran; each ``exponents`` entry is written
    beside its line's last point.

    Parameters
    ----------
    table
        :func:`pihm.pipeline.results.encoding_scaling`'s rows, of one problem.
    exponents
        :func:`pihm.pipeline.results.degree_exponents`'s rows of the same problem.

    Returns
    -------
    object
        The matplotlib figure.
    """
    import matplotlib.pyplot as plt

    encodings = list(dict.fromkeys(str(row["encoding"]) for row in table))
    series = {
        name: sorted((r for r in table if r["encoding"] == name), key=lambda r: int(r["n"]))
        for name in encodings
    }
    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), constrained_layout=True)
    drawn: list[list[float]] = []
    for name, rows in series.items():
        _series(axes[0], rows, "looseness", name)
        gaps = [float(row["relative_gap"]) for row in rows]
        if gaps not in drawn:
            drawn.append(gaps)
            _series(axes[1], rows, "relative_gap", name)
        _series(axes[2], rows, "degree", name)
        measured = [row for row in rows if row["measured_degree"] is not None]
        if measured:
            axes[2].plot(
                [2 ** int(row["n"]) for row in measured],
                [float(row["measured_degree"]) for row in measured],
                linestyle="none",
                marker="x",
                markersize=8,
                markeredgewidth=1.4,
                color=INK,
                label="T2, measured",
                zorder=4,
            )
    for fit in exponents:
        last = series[str(fit["encoding"])][-1]
        axes[2].annotate(
            f"N^{float(fit['exponent']):.1f}",
            xy=(2 ** int(last["n"]), float(last["degree"])),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK_SECONDARY,
        )
    titles = (
        ("the encoding", r"looseness $\alpha_H/\|H\|$"),
        ("the Hamiltonian", r"relative gap $\Delta/\|H\|$"),
        ("the filter's degree", "T3, with T2 where it ran"),
    )
    for axis, (title, subtitle) in zip(axes, titles, strict=True):
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlabel("N = 2^n")
        axis.margins(x=0.15)
        _titles(axis, title, subtitle)
    _legend_above(figure, axes, untrusted=True)
    return figure


def exponent_dots(exponents: Sequence[Mapping[str, Any]]) -> Any:
    """Draw each problem's fitted degree exponent in :math:`N`, one marker per encoding.

    Parameters
    ----------
    exponents
        :func:`pihm.pipeline.results.degree_exponents`'s rows.

    Returns
    -------
    object
        The matplotlib figure.
    """
    import matplotlib.pyplot as plt

    problems = sorted({str(row["problem"]) for row in exponents}, reverse=True)
    figure, axis = plt.subplots(figsize=(5.6, 0.28 * len(problems) + 1.2), constrained_layout=True)
    for encoding in dict.fromkeys(str(row["encoding"]) for row in exponents):
        colour, marker = ENCODING_STYLE[encoding]
        rows = [row for row in exponents if row["encoding"] == encoding]
        for index, row in enumerate(rows):
            trusted = bool(row["trusted"])
            axis.plot(
                float(row["exponent"]),
                problems.index(str(row["problem"])),
                marker=marker,
                linestyle="none",
                color=colour,
                markerfacecolor=colour if trusted else SURFACE,
                markeredgecolor=colour if not trusted else SURFACE,
                markeredgewidth=1.2,
                label=encoding if index == 0 else None,
                zorder=3,
            )
    axis.set_yticks(range(len(problems)), problems)
    axis.grid(axis="y", visible=False)
    axis.set_xlabel("fitted exponent x of the degree, degree ∝ N^x")
    fitted = sorted({n for row in exponents for n in row["ns"]})
    _titles(axis, "the degree's growth", f"fitted over n = {fitted[0]} to {fitted[-1]}")
    _legend_above(figure, [axis], untrusted=not all(row["trusted"] for row in exponents))
    return figure


def panel_dots(
    table: Sequence[Mapping[str, Any]],
    columns: Sequence[tuple[str, str]],
    *,
    key: str = "form",
    title: str = "",
    linear: Sequence[str] = (),
) -> Any:
    """Draw each problem's values, one marker per series and one panel per column.

    The panels share the problems' axis, and each series takes its style from
    :data:`~pihm.plotting.style.STYLES`; a row whose reference is not trusted is hollow.

    Parameters
    ----------
    table
        One row per problem and series: ``problem``, the series' name under ``key``, each
        column, and optionally ``trusted``.
    columns
        ``(column, label)`` pairs, one panel each.
    key
        The column naming each row's series.
    title
        A subtitle for the first panel, such as the resolution the values are at.
    linear
        The columns drawn on a linear axis, such as a count; the rest are logarithmic.

    Returns
    -------
    object
        The matplotlib figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    problems = sorted({str(row["problem"]) for row in table}, reverse=True)
    figure, axes = plt.subplots(
        1,
        len(columns),
        figsize=(2.6 * len(columns) + 0.6, 0.26 * len(problems) + 1.4),
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    panels = list(axes[0])
    for index, (axis, (column, label)) in enumerate(zip(panels, columns, strict=True)):
        for name in dict.fromkeys(str(row[key]) for row in table):
            colour, marker = STYLES[name]
            rows = [r for r in table if r[key] == name and isinstance(r.get(column), (int, float))]
            for trusted in (True, False):
                chosen = [r for r in rows if (r.get("trusted", True) is True) == trusted]
                axis.scatter(
                    [float(r[column]) for r in chosen],
                    [problems.index(str(r["problem"])) for r in chosen],
                    marker=marker,
                    edgecolors=colour,
                    facecolors=colour if trusted else SURFACE,
                    label=name if trusted and index == 0 else None,
                    zorder=3,
                )
        if column in linear:
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            axis.set_xscale("log")
        axis.grid(axis="y", visible=False)
        _titles(axis, label, title if index == 0 else "")
    panels[0].set_yticks(range(len(problems)), problems)
    untrusted = any(row.get("trusted", True) is not True for row in table)
    _legend_above(figure, panels, untrusted=untrusted)
    return figure


def by_resolution(
    series: Mapping[str, Sequence[Mapping[str, Any]]],
    column: str,
    label: str,
    *,
    problems: Sequence[str] | None = None,
    columns: int = 5,
) -> Any:
    """Draw ``column`` against n, a panel per problem and a line per series, on shared axes.

    Each series takes its style from :data:`~pihm.plotting.style.STYLES`; a row whose reference
    is not trusted is hollow. The panels share one logarithmic y-axis, so their values compare.

    Parameters
    ----------
    series
        Each series' rows, by its name, one row per problem and resolution.
    column
        The column drawn.
    label
        The y-axis label.
    problems
        The problems, one panel each; by default every problem any series has.
    columns
        Panels per row.

    Returns
    -------
    object
        The matplotlib figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    names = problems or sorted({str(r["problem"]) for rows in series.values() for r in rows})
    count = min(columns, len(names))
    rows_needed = -(-len(names) // count)
    figure, grid = plt.subplots(
        rows_needed,
        count,
        figsize=(2.5 * count, 2.1 * rows_needed),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    axes = list(grid.ravel())
    for spare in axes[len(names) :]:
        spare.remove()
    for index, (axis, problem) in enumerate(zip(axes, names, strict=False)):
        for name, rows in series.items():
            colour, marker = STYLES[name]
            points = sorted(
                (r for r in rows if r["problem"] == problem and _finite(r.get(column))),
                key=lambda r: int(r["n"]),
            )
            points = [r for r in points if r[column] > 0]
            ns = [int(r["n"]) for r in points]
            values = [float(r[column]) for r in points]
            axis.plot(ns, values, color=colour, linewidth=1.3, label=name)
            for x, value, row in zip(ns, values, points, strict=True):
                trusted = row.get("trusted", True) is True
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
        axis.set_yscale("log")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_title(problem, loc="left")
        if index % count == 0:
            axis.set_ylabel(label)
        if index >= len(names) - count:
            axis.set_xlabel("n (qubits per axis)")
    untrusted = any(r.get("trusted", True) is not True for rows in series.values() for r in rows)
    _legend_above(figure, axes[: len(names)], untrusted=untrusted)
    return figure
