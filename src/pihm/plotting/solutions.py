r"""A case's solution: its decoded fields in both forms, the spy plots of its matrices, its metrics.

The solution viewer's figures; :mod:`pihm.pipeline.solutions` gathers what they
draw. :func:`solution_figures` draws a case's fields by its family:

* ``curve`` and ``fields``: each field against the reference, and beneath it each layer's error on
  a log scale, with each form's field bound;
* ``surface``: maps of the reference, each layer and its error; on a time axis, slices in time;
* ``burgers``: :math:`u(x, s)` on the periodic interval, profiles against the Galerkin reference
  and the Cole–Hopf solution, the energy and the spectrum;
* ``vorticity``: :math:`\omega(x, y)` with the stream function's contours and each layer's
  departure from the reference, the energy, the enstrophy and the spectrum.

A form keeps its colour (:data:`~pihm.plotting.style.REGIME_STYLE`) and a state its line: the
ground state solid, T2 dashed and T1 dotted. Where the fields agree to the eye, as they mostly do,
the standard form's wider line shows around the descriptor's, and the reference, thin and in ink,
runs on top of both. Signed fields take the diverging map
(:data:`~pihm.plotting.style.DIVERGING`) about zero, magnitudes the sequential blue ramp.
:func:`snapshot` draws one moment of a time-dependent case, for the viewer's slider, :func:`spy`
draws :math:`R` and :math:`H` in each form with their blocks marked, and
:func:`metrics_markdown` and :func:`case_markdown` write a case's metrics and records.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from pihm.pipeline.solutions import (
    Band,
    CaseKey,
    CaseRecords,
    FieldSource,
    Layer,
    MetricsTable,
    Solution,
    Structure,
)
from pihm.plotting.style import (
    DIVERGING,
    INK,
    INK_SECONDARY,
    MUTED,
    REGIME_STYLE,
    SEQUENTIAL,
    SURFACE,
    _legend_above,
    _titles,
)

__all__ = [
    "SOURCE_LINE",
    "case_markdown",
    "metrics_markdown",
    "snapshot",
    "solution_figures",
    "spy",
]

#: The line of each state a field is decoded from.
SOURCE_LINE: Final[Mapping[FieldSource, Any]] = {
    "ground": "-",
    "T2": (0, (4, 2)),
    "T1": (0, (1, 1.5)),
}

#: How a legend names each state.
_SOURCE_NAME: Final[Mapping[FieldSource, str]] = {
    "ground": "ground state",
    "T2": "prepared, T2",
    "T1": "prepared, T1",
}

#: Each form's line width: the standard form's wider, so that coincident lines all show.
_WIDTH: Final[Mapping[str, float]] = {"standard": 2.6, "descriptor": 1.4}

#: The reference's line: thin, in ink, above every layer.
_REFERENCE: Final[Mapping[str, Any]] = {"color": INK, "linewidth": 0.9, "zorder": 5}

#: The smallest error a log scale shows; an exact field sits on it.
_FLOOR: Final = 1e-17

#: The fractions of the time window at which a time-dependent case is drawn.
_MOMENTS: Final = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)

#: Points on the periodic interval (Burgers) and per side of the periodic square (vorticity).
_PERIODIC_POINTS: Final = {1: 129, 2: 97}

#: The share of a spy plot's side under which a band is narrow: it shares a tick (:func:`_ticks`).
_NARROW: Final = 0.04

#: The most bins per side of a spy plot. Each bin covers a whole number of entries, the same on
#: both sides, so that a uniform pattern draws uniformly; a small matrix gets one bin per entry.
_SPY_BINS: Final = 360


def _pyplot() -> Any:
    """Return matplotlib's pyplot, imported only when a figure is drawn."""
    import matplotlib.pyplot as plt

    return plt


def _maps() -> tuple[Any, Any, Any]:
    """Return the diverging, sequential and density colour maps."""
    from matplotlib.colors import LinearSegmentedColormap

    return (
        LinearSegmentedColormap.from_list("pihm-diverging", DIVERGING),
        LinearSegmentedColormap.from_list("pihm-blue", [SURFACE, *SEQUENTIAL]),
        LinearSegmentedColormap.from_list("pihm-density", SEQUENTIAL[3:]),
    )


def _label(layer: Layer) -> str:
    """Return a layer's legend entry: its form and its state."""
    return f"{layer.regime}, {_SOURCE_NAME[layer.source]}"


def _style(layer: Layer) -> dict[str, Any]:
    """Return a layer's line style: its form's colour and its state's line."""
    return {
        "color": REGIME_STYLE[layer.regime][0],
        "linestyle": SOURCE_LINE[layer.source],
        "linewidth": _WIDTH[layer.regime],
        "label": _label(layer),
    }


def _reference_name(solution: Solution) -> str:
    """Return how the reference is named: exact for a closed form, else by its solver."""
    kind = solution.problem.reference.kind
    return "exact" if kind == "expression" else f"reference ({kind.upper()})"


def _indices(size: int, at: float | None) -> list[int]:
    """Return the grid indices of the moments drawn: ``at`` (a fraction of the window) or four."""
    fractions = _MOMENTS if at is None else (at,)
    return [round(fraction * (size - 1)) for fraction in fractions]


def _bounds(solution: Solution) -> dict[str, float]:
    """Return each form's field bound from its T2 record, where it applies."""
    bounds = {}
    for regime, found in solution.records.t2.items():
        bound = found.row.get("field_bound")
        if isinstance(bound, float) and np.isfinite(bound):
            bounds[regime] = bound
    return bounds


def _heading(axis: Any, solution: Solution) -> None:
    """Title a figure's first panel with the case, the problem's description beneath it."""
    description = solution.problem.description
    _titles(axis, str(solution.key), description if len(description) <= 110 else "")


def _symmetric(values: Sequence[NDArray[Any]]) -> Any:
    """Return a colour norm symmetric about zero over every array of ``values``."""
    from matplotlib.colors import CenteredNorm

    high = max(float(np.max(np.abs(v))) for v in values) or 1.0
    return CenteredNorm(vcenter=0.0, halfrange=high)


def _log_norm(values: Sequence[NDArray[Any]]) -> Any:
    """Return a log norm over nine decades below the largest of ``values``."""
    from matplotlib.colors import LogNorm

    high = max(float(np.max(v)) for v in values)
    high = high if high > 0 else 1.0
    return LogNorm(vmin=high * 1e-9, vmax=high)


# ---------------------------------------------------------------------------- one axis


def _curves(solution: Solution) -> Any:
    """Draw each field against the reference, and each layer's error beneath it."""
    plt = _pyplot()
    fields = solution.problem.fields
    count = len(fields)
    x = solution.axes[0]
    axis_name = solution.axis_names[0]
    reference = solution.reference.reshape(count, -1)
    figure, axes = plt.subplots(
        2,
        count,
        figsize=(max(4.6 * count, 7.6), 5.2),
        squeeze=False,
        sharex="col",
        constrained_layout=True,
        height_ratios=(3, 2),
    )
    bounds = _bounds(solution)
    readout = {regime: found.arrays() for regime, found in solution.records.readout.items()}
    for index, field in enumerate(fields):
        top, bottom = axes[0][index], axes[1][index]
        top.plot(x, reference[index], label=_reference_name(solution), **_REFERENCE)
        for layer in solution.layers:
            values = layer.field.reshape(count, -1)[index]
            top.plot(x, values, **_style(layer))
            error = np.maximum(np.abs(values - reference[index]), _FLOOR)
            bottom.semilogy(x, error, **_style(layer))
        for regime, bound in bounds.items():
            bottom.axhline(
                bound,
                color=REGIME_STYLE[regime][0],
                linewidth=0.9,
                linestyle=(0, (6, 2, 1, 2)),
                alpha=0.7,
                label=f"field bound, {regime}",
            )
        if count == 1:
            for regime, arrays in readout.items():
                if "points" in arrays:
                    top.plot(
                        arrays["points"],
                        np.real(arrays["field"]),
                        linestyle="none",
                        marker="s",
                        markerfacecolor=SURFACE,
                        color=REGIME_STYLE[regime][0],
                        label=f"readout, {regime}",
                    )
        top.set_ylabel(f"{field.name}({axis_name})")
        bottom.set_ylabel("|error|")
        bottom.set_xlabel(axis_name)
    _heading(axes[0][0], solution)
    _legend_above(figure, [*axes[0], *axes[1]], columns=3)
    return figure


# ---------------------------------------------------------------------------- two axes


def _map_row(
    figure: Any,
    axes: Sequence[Any],
    horizontal: NDArray[Any],
    vertical: NDArray[Any],
    maps: Sequence[tuple[str, NDArray[Any]]],
    norm: Any,
    cmap: Any,
    label: str,
) -> None:
    """Draw one row of maps on a shared colour scale, with its colour bar."""
    image = None
    for axis, (title, values) in zip(axes, maps, strict=True):
        image = axis.pcolormesh(horizontal, vertical, values, cmap=cmap, norm=norm, shading="auto")
        axis.set_title(title, loc="left", fontsize=8.5)
        axis.grid(False)
    if image is not None:
        figure.colorbar(image, ax=list(axes), shrink=0.85).set_label(label)


def _map_grid(
    solution: Solution,
    horizontal: NDArray[Any],
    vertical: NDArray[Any],
    reference: NDArray[Any],
    layers: Sequence[tuple[Layer, NDArray[Any]]],
    names: tuple[str, str],
) -> Any:
    """Draw the reference and each layer's field above each layer's error, as maps."""
    plt = _pyplot()
    diverging, sequential, _ = _maps()
    columns = 1 + len(layers)
    figure, axes = plt.subplots(
        2, columns, figsize=(3.0 * columns + 1.0, 5.6), squeeze=False, constrained_layout=True
    )
    fields = [reference, *(values for _, values in layers)]
    signed = any(float(np.min(v)) < 0 for v in fields)
    _map_row(
        figure,
        axes[0],
        horizontal,
        vertical,
        [(_reference_name(solution), reference)]
        + [(_label(layer), values) for layer, values in layers],
        _symmetric(fields) if signed else None,
        diverging if signed else sequential,
        "value",
    )
    errors = [np.maximum(np.abs(values - reference), _FLOOR) for _, values in layers]
    axes[1][0].remove()
    if errors:
        _map_row(
            figure,
            axes[1][1:],
            horizontal,
            vertical,
            [
                (f"|error|, {_label(layer)}", e)
                for (layer, _), e in zip(layers, errors, strict=True)
            ],
            _log_norm(errors),
            sequential,
            "|error|",
        )
    for axis in [*axes[0], *axes[1][1:]]:
        axis.set_xlabel(names[1])
        axis.set_ylabel(names[0])
    _heading(axes[0][0], solution)
    return figure


def _surface(solution: Solution) -> Any:
    """Draw a field on two axes: the reference and each layer, then each layer's error."""
    first, second = solution.axes
    names = (solution.axis_names[0], solution.axis_names[1])
    layers = [(layer, layer.field) for layer in solution.layers]
    return _map_grid(solution, second, first, solution.reference, layers, names)


def _slices(solution: Solution, at: float | None) -> Any:
    """Draw a field on a time axis and another at moments in time, against the reference."""
    plt = _pyplot()
    time = solution.time_axis
    assert time is not None
    other = 1 - time
    coordinates = solution.axes[other]
    moments = _indices(solution.axes[time].size, at)
    figure, axes = plt.subplots(
        1, len(moments), figsize=(3.4 * len(moments), 3.0), squeeze=False, constrained_layout=True
    )

    def moment(values: NDArray[Any], index: int) -> NDArray[Any]:
        return np.asarray(np.take(values, index, axis=time))

    for axis, index in zip(axes[0], moments, strict=True):
        reference = moment(solution.reference, index)
        axis.plot(coordinates, reference, label=_reference_name(solution), **_REFERENCE)
        for layer in solution.layers:
            axis.plot(coordinates, moment(layer.field, index), **_style(layer))
        t = float(solution.axes[time][index])
        axis.set_title(f"{solution.axis_names[time]} = {t:.3g}", loc="left", fontsize=8.5)
        axis.set_xlabel(solution.axis_names[other])
    axes[0][0].set_ylabel(solution.problem.fields[0].name)
    _legend_above(figure, list(axes[0]), columns=3)
    return figure


# ---------------------------------------------------------------------------- Burgers


def _burgers_maps(solution: Solution) -> Any:
    """Draw :math:`u(x, s)` of the reference and each layer, then each layer's error."""
    space = solution.space
    assert space is not None
    x = np.linspace(0.0, 2.0 * np.pi, _PERIODIC_POINTS[1])
    s = solution.seconds
    reference = space.field(solution.reference, x).T
    layers = [(layer, space.field(layer.field, x).T) for layer in solution.layers]
    return _map_grid(solution, x, s, reference, layers, ("s", "x"))


def _burgers_profiles(solution: Solution, at: float | None) -> Any:
    """Draw :math:`u(x)` at moments against the reference and Cole–Hopf; the energy; the modes."""
    plt = _pyplot()
    space = solution.space
    assert space is not None
    x = np.linspace(0.0, 2.0 * np.pi, _PERIODIC_POINTS[1])
    s = solution.seconds
    moments = _indices(s.size, at)
    panels = len(moments) + 2
    figure, axes = plt.subplots(
        1, panels, figsize=(3.2 * panels, 3.1), squeeze=False, constrained_layout=True
    )
    axes = axes[0]
    for axis, index in zip(axes[: len(moments)], moments, strict=True):
        axis.plot(
            x,
            space.exact(x, s[index]),
            color=MUTED,
            linewidth=2.6,
            alpha=0.6,
            label="Cole–Hopf (the PDE)",
        )
        axis.plot(
            x,
            space.field(solution.reference[:, index], x),
            label=_reference_name(solution),
            **_REFERENCE,
        )
        for layer in solution.layers:
            axis.plot(x, space.field(layer.field[:, index], x), **_style(layer))
        axis.set_title(f"s = {s[index]:.3g}", loc="left", fontsize=8.5)
        axis.set_xlabel("x")
    axes[0].set_ylabel("u")
    energy, spectrum = axes[-2], axes[-1]
    energy.plot(s, space.energy(solution.reference), **_REFERENCE)
    last = moments[-1]
    k = np.array([kx for _, kx, _ in space.modes])
    spectrum.semilogy(k, np.abs(solution.reference[:, last]) + _FLOOR, color=INK, marker="o")
    for layer in solution.layers:
        energy.plot(s, space.energy(layer.field), **_style(layer))
        spectrum.semilogy(k, np.abs(layer.field[:, last]) + _FLOOR, marker="o", **_style(layer))
    energy.set_title("energy E(s)", loc="left", fontsize=8.5)
    energy.set_xlabel("s")
    spectrum.set_title(f"|b_k| at s = {s[last]:.3g}", loc="left", fontsize=8.5)
    spectrum.set_xlabel("k")
    _legend_above(figure, list(axes), columns=4)
    return figure


# ---------------------------------------------------------------------------- the vorticity


def _vorticity_maps(solution: Solution, at: float | None) -> Any:
    r"""Draw :math:`\omega` with the streamlines at moments, and each layer's departure from it."""
    plt = _pyplot()
    diverging, _, _ = _maps()
    space = solution.space
    assert space is not None
    side = np.linspace(0.0, 2.0 * np.pi, _PERIODIC_POINTS[2])
    X, Y = np.meshgrid(side, side)
    s = solution.seconds
    moments = _indices(s.size, at)
    rows = 1 + len(solution.layers)
    figure, axes = plt.subplots(
        rows,
        len(moments),
        figsize=(2.9 * len(moments) + 1.2, 2.7 * rows + 0.4),
        squeeze=False,
        constrained_layout=True,
    )
    reference = [space.field(solution.reference[:, i], X, Y) for i in moments]
    departures = [
        [
            space.field(layer.field[:, i], X, Y) - omega
            for i, omega in zip(moments, reference, strict=True)
        ]
        for layer in solution.layers
    ]
    _map_row(
        figure,
        axes[0],
        side,
        side,
        [(f"s = {s[i]:.3g}", omega) for i, omega in zip(moments, reference, strict=True)],
        _symmetric(reference),
        diverging,
        f"ω, {_reference_name(solution)}",
    )
    for axis, index in zip(axes[0], moments, strict=True):
        psi = space.stream(solution.reference[:, index], X, Y)
        axis.contour(side, side, psi, levels=8, colors=INK_SECONDARY, linewidths=0.6)
    if departures:
        norm = _symmetric([d for row in departures for d in row])
        for row, (layer, maps) in enumerate(zip(solution.layers, departures, strict=True), 1):
            _map_row(
                figure,
                axes[row],
                side,
                side,
                [("", d) for d in maps],
                norm,
                diverging,
                "ω − reference" if row == len(departures) else "",
            )
            axes[row][0].set_ylabel(f"{_label(layer)}\ny")
    axes[0][0].set_ylabel(f"{_reference_name(solution)}\ny")
    for axis in axes.ravel():
        axis.set_aspect("equal")
    for axis in axes[-1]:
        axis.set_xlabel("x")
    return figure


def _vorticity_dynamics(solution: Solution) -> Any:
    """Draw the energy and the enstrophy against time, and the spectrum at the end."""
    plt = _pyplot()
    space = solution.space
    assert space is not None
    s = solution.seconds
    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.1), constrained_layout=True)
    energy, enstrophy, spectrum = axes
    wavenumbers = space.wavenumbers
    energy.plot(s, space.energy(solution.reference), label=_reference_name(solution), **_REFERENCE)
    enstrophy.plot(s, space.enstrophy(solution.reference), **_REFERENCE)
    spectrum.semilogy(
        wavenumbers,
        np.abs(solution.reference[:, -1]) + _FLOOR,
        color=INK,
        marker="o",
        linestyle="none",
    )
    for layer in solution.layers:
        energy.plot(s, space.energy(layer.field), **_style(layer))
        enstrophy.plot(s, space.enstrophy(layer.field), **_style(layer))
        spectrum.semilogy(
            wavenumbers,
            np.abs(layer.field[:, -1]) + _FLOOR,
            marker=REGIME_STYLE[layer.regime][1],
            markerfacecolor="none",
            linestyle="none",
            color=REGIME_STYLE[layer.regime][0],
            label=_label(layer),
        )
    energy.set_title("energy E(s)", loc="left", fontsize=8.5)
    enstrophy.set_title("enstrophy Z(s)", loc="left", fontsize=8.5)
    spectrum.set_title(f"|w_m| at s = {s[-1]:.3g}", loc="left", fontsize=8.5)
    energy.set_xlabel("s")
    enstrophy.set_xlabel("s")
    spectrum.set_xlabel("|k|²")
    _legend_above(figure, list(axes), columns=3)
    return figure


# ---------------------------------------------------------------------------- the entry points


def solution_figures(solution: Solution) -> list[Any]:
    """Return a case's figures by its family (module documentation), time at four moments."""
    if solution.family in ("curve", "fields"):
        return [_curves(solution)]
    if solution.family == "surface":
        slices = [_slices(solution, None)] if solution.time_axis is not None else []
        return [_surface(solution), *slices]
    if solution.family == "burgers":
        return [_burgers_maps(solution), _burgers_profiles(solution, None)]
    return [_vorticity_maps(solution, None), _vorticity_dynamics(solution)]


def snapshot(solution: Solution, at: float) -> Any | None:
    """Return a time-dependent case at the fraction ``at`` of its window, or ``None``.

    A field on a time axis as slices, Burgers' profile with its energy and spectrum, or the
    vorticity with its departures; a stationary case (``None``) has no moment to draw.
    """
    if solution.family == "burgers":
        return _burgers_profiles(solution, at)
    if solution.family == "vorticity":
        return _vorticity_maps(solution, at)
    if solution.family == "surface" and solution.time_axis is not None:
        return _slices(solution, at)
    return None


# ---------------------------------------------------------------------------- the matrices


def _ticks(bands: Sequence[Band], extent: int) -> tuple[list[float], list[str]]:
    """Return a tick per band at its centre, a run of narrow bands sharing one tick.

    A band under :data:`_NARROW` of the axis would have its label collide with its neighbour's;
    consecutive narrow bands are labelled together, at their run's centre.
    """
    groups: list[list[Band]] = []
    for band in bands:
        narrow = band.stop - band.start < _NARROW * extent
        previous = groups[-1] if groups else None
        if narrow and previous and previous[-1].stop - previous[-1].start < _NARROW * extent:
            previous.append(band)
        else:
            groups.append([band])
    positions = [(group[0].start + group[-1].stop) / 2 for group in groups]
    return positions, [", ".join(band.label for band in group) for group in groups]


def _spy_panel(
    axis: Any, matrix: Any, rows: Sequence[Band], columns: Sequence[Band], title: str
) -> None:
    """Draw a sparse matrix's nonzeros binned (one bin per entry when small), blocks marked."""
    from matplotlib.colors import LogNorm

    _, _, density = _maps()
    height, width = matrix.shape
    entries = matrix.tocoo()
    step = -(-max(height, width) // _SPY_BINS)
    bins = (-(-height // step), -(-width // step))
    counts, _, _ = np.histogram2d(
        entries.row, entries.col, bins=bins, range=((0, bins[0] * step), (0, bins[1] * step))
    )
    most = max(float(np.max(counts)), 2.0)
    axis.imshow(
        np.ma.masked_equal(counts, 0),
        extent=(0, bins[1] * step, bins[0] * step, 0),
        cmap=density,
        norm=LogNorm(vmin=1, vmax=most),
        interpolation="nearest",
        aspect="equal",
    )
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.set_facecolor(SURFACE)
    axis.grid(False)
    for band in rows[1:]:
        axis.axhline(band.start, color=MUTED, linewidth=0.6)
    for band in columns[1:]:
        axis.axvline(band.start, color=MUTED, linewidth=0.6)
    axis.set_yticks(*_ticks(rows, height), fontsize=7.5)
    axis.xaxis.tick_top()
    positions, labels = _ticks(columns, width)
    crowded = len(positions) > 3
    axis.set_xticks(
        positions,
        labels,
        fontsize=7.5,
        rotation=30 if crowded else 0,
        ha="left" if crowded else "center",
    )
    axis.tick_params(length=0)
    density_ = entries.nnz / (height * width)
    axis.set_xlabel(
        f"{title}\n{height:,} × {width:,}, {entries.nnz:,} nonzeros ({density_:.2%})",
        color=INK,
        fontsize=8.5,
    )


def spy(structures: Mapping[str, Structure]) -> Any:
    """Draw :math:`R` and :math:`H` of each form, row blocks and the state's blocks marked.

    Each panel bins the nonzeros, at most 360 bins a side, coloured by how many fall in a bin
    (a small matrix has one bin per entry). :math:`H` is left out past
    :data:`~pihm.pipeline.solutions.HAMILTONIAN_LIMIT`.
    """
    plt = _pyplot()
    figure, axes = plt.subplots(
        len(structures),
        2,
        figsize=(9.6, 4.8 * len(structures)),
        squeeze=False,
        constrained_layout=True,
    )
    for (regime, structure), (left, right) in zip(structures.items(), axes, strict=True):
        _spy_panel(left, structure.residual, structure.rows, structure.columns, f"R, {regime} form")
        if structure.hamiltonian is None:
            right.axis("off")
            right.text(
                0.5,
                0.5,
                "H is not formed: R's nonzeros pass the limit",
                ha="center",
                color=INK_SECONDARY,
                transform=right.transAxes,
            )
        else:
            _spy_panel(
                right,
                structure.hamiltonian,
                structure.columns,
                structure.columns,
                f"H = R†R, {regime} form",
            )
    return figure


# ---------------------------------------------------------------------------- the tables


def _format(value: Any) -> str:
    """Write one metric for a table: counts with separators, other numbers to four figures."""
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            return f"{int(value):,}"
        return f"{value:.4g}" if 1e-3 <= abs(value) < 1e5 else f"{value:.3g}"
    text = str(value).replace("|", "\\|")
    return text if len(text) <= 90 else text[:87] + "…"


def metrics_markdown(table: MetricsTable) -> str:
    """Return a case's metrics as a Markdown table, a bold row opening each section."""
    lines = [
        "| | " + " | ".join(table.columns) + " |",
        "|---|" + "---|" * len(table.columns),
    ]
    section = None
    for title, name, values in table.rows:
        if title != section:
            lines.append(f"| **{title}** |" + " |" * len(table.columns))
            section = title
        lines.append(f"| {name} | " + " | ".join(_format(v) for v in values) + " |")
    return "\n".join(lines)


def case_markdown(
    key: CaseKey,
    records: CaseRecords,
    labels: Mapping[str, str],
    solution: Solution | None = None,
) -> str:
    """Return a case's heading, its problem, what is missing, and its records' run keys."""
    lines = [f"### {key}"]
    if solution is not None and solution.problem.description:
        lines.append(solution.problem.description)
    if solution is None:
        lines.append("No record of this case holds a field; its metrics follow.")
    else:
        lines.extend(f"* {note}" for note in solution.notes)
    keys = records.run_keys(labels)
    if keys:
        lines += ["", "| stage | form | run key |", "|---|---|---|"]
        lines += [f"| {stage} | {name} | `{run}` |" for stage, name, run in keys]
    return "\n".join(lines)
