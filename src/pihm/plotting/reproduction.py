r"""The paper's reproduction: its panels rebuilt, their convergence with n, and the triage figures.

Each function reads the build records of the reproduction campaign, ``p2_reproduction``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pihm.pipeline.summary import read_arrays
from pihm.plotting.style import (
    GRID,
    INK,
    INK_SECONDARY,
    MUTED,
    SEQUENTIAL,
    SERIES,
    SURFACE,
    VARIANT_STYLE,
    _axis_names,
    _grid,
    _legend_above,
    _title,
    _titles,
    select,
)

__all__ = [
    "convergence",
    "error_maps",
    "gap_collapse",
    "maclaurin_order",
    "scale_factor_rows",
    "solution_panels",
]


def solution_panels(
    rows: Sequence[Mapping[str, Any]],
    root: Path,
    problems: Sequence[str],
    *,
    n: int | None = None,
    columns: int = 4,
) -> Any:
    """Draw the paper's one-axis panels: the exact solution and both variants' readouts.

    Parameters
    ----------
    rows
        Summary rows (:func:`pihm.pipeline.summary.read_summary`).
    root
        The results directory the rows were collected from.
    problems
        One-axis problem names, such as ``["R2a", "R2b", "R2c"]``.
    n
        The resolution; by default each panel's printed :math:`n`.
    columns
        Panels per row.

    Returns
    -------
    object
        The matplotlib figure.
    """
    figure, axes = _grid(len(problems), min(columns, len(problems)), (3.1, 2.4))
    for index, (axis, name) in enumerate(zip(axes, problems, strict=True)):
        drawn = False
        title = name
        errors: list[str] = []
        for variant in ("faithful", "corrected"):
            candidates = select(rows, problem=name, variant=variant, label="")
            if not candidates:
                continue
            size = n or int(candidates[0].get("printed_n") or candidates[0]["n"])
            matches = [row for row in candidates if int(row["n"]) == size]
            if not matches:
                continue
            row = matches[0]
            arrays = read_arrays(root, row)
            x = arrays["grid_0"]
            if not drawn:
                axis.plot(x, arrays["reference"], color=INK, linewidth=1.2, label="exact")
                title = f"{_title(row)}, n = {size}"
                drawn = True
            colour, _ = VARIANT_STYLE[variant]
            errors.append(f"{variant} {float(row['field_err_linf']):.1e}")
            axis.plot(
                x,
                arrays["field"],
                color=colour,
                linewidth=1.6 if variant == "corrected" else 1.8,
                linestyle="-" if variant == "faithful" else (0, (4, 2)),
                label=variant,
            )
        _titles(axis, title, "L∞: " + " · ".join(errors))
        axis.set_xlabel("x")
        if index % columns == 0:
            axis.set_ylabel("f(x)")
    _legend_above(figure, axes)
    return figure


def error_maps(
    rows: Sequence[Mapping[str, Any]],
    root: Path,
    problems: Sequence[str],
    *,
    n: int | None = None,
) -> Any:
    r"""Draw :math:`|f_Q - f|` for the two-axis panels, faithful above corrected, one log scale.

    Returns
    -------
    object
        The matplotlib figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    cmap = LinearSegmentedColormap.from_list("pihm-blue", [SURFACE, *SEQUENTIAL])
    figure, axes = plt.subplots(
        2,
        len(problems),
        figsize=(3.0 * len(problems), 5.2),
        squeeze=False,
        constrained_layout=True,
    )
    fields: dict[tuple[int, int], tuple[Any, Any, Any, str]] = {}
    names: dict[int, list[str]] = {}
    for column, name in enumerate(problems):
        for line, variant in enumerate(("faithful", "corrected")):
            candidates = select(rows, problem=name, variant=variant, label="")
            size = n or int(candidates[0].get("printed_n") or candidates[0]["n"])
            row = next(r for r in candidates if int(r["n"]) == size)
            arrays = read_arrays(root, row)
            error = np.abs(arrays["field"] - arrays["reference"])
            fields[(line, column)] = (arrays["grid_0"], arrays["grid_1"], error, _title(row))
            names[column] = _axis_names(root, row)
    high = max(float(np.max(e)) for _, _, e, _ in fields.values())
    floor = high * 1e-9
    norm = LogNorm(vmin=floor, vmax=high)
    images = []
    for (line, column), (first, second, error, title) in fields.items():
        axis = axes[line][column]
        images.append(
            axis.pcolormesh(
                second, first, np.maximum(error, floor), cmap=cmap, norm=norm, shading="auto"
            )
        )
        variant = ("faithful", "corrected")[line]
        _titles(axis, title, f"{variant}, L∞ {float(np.max(error)):.1e}")
        axis.set_xlabel(names[column][1])
        axis.set_ylabel(names[column][0])
        axis.grid(False)
    colourbar = figure.colorbar(images[-1], ax=axes, shrink=0.8)
    colourbar.set_label("|f_Q - f|")
    return figure


def convergence(
    rows: Sequence[Mapping[str, Any]],
    problems: Sequence[str],
    *,
    extra: tuple[str, str] | None = None,
    columns: int = 4,
    metric: str = "field_err_linf",
) -> Any:
    """Draw an error against n per problem: faithful, corrected and any labelled series.

    The muted line is Wedin's bound on the SVD reference's own error (the double-precision
    floor of the standard form); hollow markers are untrusted ground states.

    Parameters
    ----------
    rows
        Summary rows.
    problems
        Problem names, one panel each.
    extra
        One further series as ``(label, variant)``, drawn in slot 3 (one only, so that no two
        series share a colour).
    columns
        Panels per row.
    metric
        The summary column to plot on a log scale.

    Returns
    -------
    object
        The matplotlib figure.
    """
    figure, axes = _grid(len(problems), min(columns, len(problems)), (3.0, 2.4))
    series: list[tuple[str, str, str, str]] = [
        ("faithful", "", VARIANT_STYLE["faithful"][0], "s"),
        ("corrected", "", VARIANT_STYLE["corrected"][0], "o"),
    ]
    if extra is not None:
        series.append((extra[1], extra[0], SERIES[2], "D"))
    for index, (axis, name) in enumerate(zip(axes, problems, strict=True)):
        title = ""
        for variant, label, colour, marker in series:
            points = select(rows, problem=name, variant=variant, label=label)
            if not points:
                continue
            title = title or _title(points[0])
            ns = [int(p["n"]) for p in points]
            values = [max(float(p[metric]), 1e-17) for p in points]
            axis.plot(ns, values, color=colour, label=label or variant, zorder=2)
            for x, y, point in zip(ns, values, points, strict=True):
                trusted = bool(point["trusted"])
                axis.plot(
                    x,
                    y,
                    marker=marker,
                    color=colour,
                    markerfacecolor=colour if trusted else SURFACE,
                    markeredgecolor=colour if not trusted else SURFACE,
                    markeredgewidth=1.4,
                    zorder=3,
                )
            if variant == "corrected" and not label and metric == "field_err_linf":
                bound = [float(p["vector_error_bound"]) for p in points]
                axis.plot(ns, bound, color=MUTED, linewidth=1.0, label="SVD bound", zorder=1)
        axis.set_yscale("log")
        axis.set_title(title, loc="left")
        axis.set_xlabel("n (qubits per axis)")
        if index % columns == 0:
            axis.set_ylabel("L∞ error" if metric == "field_err_linf" else metric)
    _legend_above(figure, axes, untrusted=True)
    return figure


def maclaurin_order(rows: Sequence[Mapping[str, Any]]) -> Any:
    """Draw Fig. 5d's error at the printed n = 4 for each source and constraint choice (R-D3).

    Returns
    -------
    object
        The matplotlib figure.
    """
    import matplotlib.pyplot as plt

    cases = [
        ("printed root, Maclaurin 7 (faithful)", "faithful", ""),
        ("exact root, Maclaurin 7", "faithful", "exact root, Maclaurin source"),
        ("printed root, Chebyshev source", "faithful", "printed root, Chebyshev source"),
        ("exact root, Maclaurin 11", "corrected", "exact root, Maclaurin p = 11"),
        ("exact root, Maclaurin 15", "corrected", "exact root, Maclaurin p = 15"),
        ("exact root, Chebyshev (corrected)", "corrected", ""),
    ]
    names, values = [], []
    for name, variant, label in cases:
        found = [
            row
            for row in select(rows, problem="R4d", variant=variant, label=label)
            if int(row["n"]) == 4
        ]
        if found:
            names.append(name)
            values.append(float(found[0]["field_err_linf"]))
    figure, axis = plt.subplots(figsize=(6.4, 2.6), constrained_layout=True)
    positions = np.arange(len(names))
    axis.hlines(positions, 1e-8, values, color=GRID, linewidth=1.0, zorder=1)
    axis.plot(values, positions, "o", color=SERIES[0], markeredgecolor=SURFACE, zorder=2)
    for position, value in zip(positions, values, strict=True):
        axis.annotate(
            f"{value:.1e}",
            (value, float(position)),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=INK_SECONDARY,
            fontsize=8,
        )
    axis.set_yticks(positions, names)
    axis.invert_yaxis()
    axis.set_xscale("log")
    axis.set_xlim(1e-8, 1.0)
    axis.set_xlabel("L∞ error of the decoded field at n = 4")
    axis.set_title("Fig. 5d (R4d): which printed choice causes the error", loc="left")
    axis.grid(axis="y", visible=False)
    return figure


def gap_collapse(rows: Sequence[Mapping[str, Any]], problems: Sequence[str]) -> Any:
    """Draw the absolute gap Δ = λ₁ − λ₀ against n for the PDE panels.

    Returns
    -------
    object
        The matplotlib figure.
    """
    figure = convergence(rows, problems, metric="gap", columns=len(problems))
    figure.axes[0].set_ylabel("gap Δ = λ₁ - λ₀")
    return figure


def scale_factor_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    r"""Return, per paper panel, the printed, ideal and ground-state :math:`\eta_e` at printed n.

    Each entry holds ``figure``, ``problem``, ``n``, ``printed``, ``ideal``, ``faithful``,
    ``corrected`` (the ground states read out at the regular point) and ``trusted`` (the
    faithful ground state's), for a table in the notebook or the report.
    """
    table: list[dict[str, Any]] = []
    panels = {str(row["problem"]) for row in rows if row.get("figure")}
    for name in sorted(panels):
        faithful = [
            row
            for row in select(rows, problem=name, variant="faithful", label="")
            if row["n"] == row["printed_n"]
        ]
        corrected = [
            row
            for row in select(rows, problem=name, variant="corrected", label="")
            if row["n"] == row["printed_n"]
        ]
        if not faithful:
            continue
        head = faithful[0]
        table.append(
            {
                "figure": head["figure"],
                "problem": name,
                "n": head["n"],
                "printed": head["eta_printed"],
                "ideal": head["eta_ideal"],
                "faithful": head["eta"],
                "corrected": corrected[0]["eta"] if corrected else float("nan"),
                "trusted": head["trusted"],
            }
        )
    return table
