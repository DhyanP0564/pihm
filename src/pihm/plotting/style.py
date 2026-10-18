r"""The shared figure style: the palette, matplotlib's defaults and the helpers every figure uses.

The encoding is described in :mod:`pihm.plotting`. The helpers with a leading underscore are the
package's own: small multiples, titles, and one legend above the panels.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from pihm.pipeline.records import loads_strict

__all__ = [
    "AXIS",
    "DIVERGING",
    "ENCODING_STYLE",
    "GRID",
    "INK",
    "INK_SECONDARY",
    "LAYOUT_STYLE",
    "MUTED",
    "REGIME_STYLE",
    "SEQUENTIAL",
    "SERIES",
    "STYLES",
    "SURFACE",
    "VARIANT_STYLE",
    "apply_style",
    "markdown_table",
    "select",
]

#: The chart surface, primary and secondary ink, muted text, grid and axis colours.
SURFACE: Final = "#fcfcfb"
INK: Final = "#0b0b0b"
INK_SECONDARY: Final = "#52514e"
MUTED: Final = "#898781"
GRID: Final = "#e1e0d9"
AXIS: Final = "#c3c2b7"

#: Categorical slots 1 to 3 of the reference palette (blue, orange, aqua), in fixed order.
SERIES: Final = ("#2a78d6", "#eb6834", "#1baf7a")

#: The sequential blue ramp, light to dark, for magnitudes such as error maps.
SEQUENTIAL: Final = (
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
)

#: The diverging map for signed fields, from the sequential blue ramp's steps 700, 450 and 200
#: (negative) through a neutral grey zero to red (positive), stepped to the same OKLab lightness
#: (0.34, 0.58 and 0.81) so that neither sign reads as the stronger.
DIVERGING: Final = (
    "#0d366b",
    "#2a78d6",
    "#9ec5f4",
    "#f0efec",
    "#f2adad",
    "#cc4241",
    "#5f1e1e",
)

#: Colour and marker of each variant; the marker is the non-colour channel.
VARIANT_STYLE: Final[Mapping[str, tuple[str, str]]] = {
    "corrected": (SERIES[0], "o"),
    "faithful": (SERIES[1], "s"),
}

#: Colour and marker of each form, where a figure compares them.
REGIME_STYLE: Final[Mapping[str, tuple[str, str]]] = {
    "standard": (SERIES[0], "o"),
    "descriptor": (SERIES[2], "D"),
}

#: Colour and marker of each encoding. The standard form's structured encoding takes the form's
#: colour, the paper's published one the faithful variant's, and the descriptor its own.
ENCODING_STYLE: Final[Mapping[str, tuple[str, str]]] = {
    "structured": (SERIES[0], "o"),
    "published": (SERIES[1], "s"),
    "descriptor": (SERIES[2], "D"),
}

#: The Carleman layouts' colours and markers, for the layout-cost figure. The tensor layout takes
#: the third series colour, and the pipeline's symmetric layout, its baseline, secondary ink.
LAYOUT_STYLE: Final[Mapping[str, tuple[str, str]]] = {
    "symmetric": (INK_SECONDARY, "s"),
    "tensor": (SERIES[1], "o"),
}

#: Every series' colour and marker, by name. The forms and the encodings keep theirs, and the
#: descriptor's unscaled ablation is drawn in secondary ink, as context.
STYLES: Final[Mapping[str, tuple[str, str]]] = {
    **REGIME_STYLE,
    **ENCODING_STYLE,
    "unscaled descriptor": (INK_SECONDARY, "v"),
    "chain ablation": (INK_SECONDARY, "^"),
}


def apply_style() -> None:
    """Set matplotlib's defaults to the shared style (call once per notebook)."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK_SECONDARY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "text.color": INK,
            "lines.linewidth": 1.8,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "lines.markersize": 6,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "legend.labelcolor": INK_SECONDARY,
        }
    )


def select(rows: Iterable[Mapping[str, Any]], **match: Any) -> list[Mapping[str, Any]]:
    """Return the summary rows whose columns equal every ``match``, ordered by ``n``."""
    chosen = [row for row in rows if all(row.get(key) == value for key, value in match.items())]
    return sorted(chosen, key=lambda row: int(row["n"]))


def _grid(count: int, columns: int, size: tuple[float, float]) -> tuple[Any, list[Any]]:
    """Return a figure with ``count`` small-multiple axes (spare axes removed)."""
    import matplotlib.pyplot as plt

    rows = math.ceil(count / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(size[0] * columns, size[1] * rows),
        squeeze=False,
        constrained_layout=True,
    )
    flat = list(axes.ravel())
    for spare in flat[count:]:
        spare.remove()
    return figure, flat[:count]


def _title(row: Mapping[str, Any]) -> str:
    """Return a panel title: the paper's figure when there is one, and the rung."""
    figure = str(row.get("figure", ""))
    return f"Fig. {figure} ({row['problem']})" if figure else str(row["problem"])


def _axis_names(root: Path, row: Mapping[str, Any]) -> list[str]:
    """Return the axis names of the problem behind a summary row, from its record's spec."""
    meta = loads_strict((root / str(row["directory"]) / "meta.json").read_text())
    return [str(axis["name"]) for axis in meta["spec"]["problem"]["axes"]]


def _titles(axis: Any, title: str, subtitle: str) -> None:
    """Set a bold panel title with a muted subtitle line beneath it."""
    axis.set_title(title, loc="left", pad=15 if subtitle else 6)
    if subtitle:
        axis.annotate(
            subtitle,
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=8,
            color=INK_SECONDARY,
            va="bottom",
        )


def _legend_above(
    figure: Any, axes: Sequence[Any], untrusted: bool = False, columns: int | None = None
) -> None:
    """Place one legend for the whole figure above its panels (it never covers data).

    Its entries run in ``columns`` columns, by default all on one row.
    """
    from matplotlib.lines import Line2D

    handles, labels = [], []
    for axis in axes:
        for handle, label in zip(*axis.get_legend_handles_labels(), strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if untrusted:
        handles.append(
            Line2D([], [], marker="o", linestyle="none", color=MUTED, markerfacecolor=SURFACE)
        )
        labels.append("not trusted (hollow)")
    figure.legend(handles, labels, loc="outside upper center", ncols=columns or len(labels))


def _finite(value: Any) -> bool:
    """Whether a summary cell holds a finite number."""
    return isinstance(value, (int, float)) and math.isfinite(value)


def markdown_table(
    entries: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str, str]]
) -> str:
    """Return ``entries`` as a Markdown table, for the notebooks and the reproduction report.

    Parameters
    ----------
    entries
        One mapping per row.
    columns
        ``(key, header, format)`` per column; the format is a :func:`format` specification, and
        ``"flag"`` writes ``yes``/``no``.

    Examples
    --------
    >>> print(markdown_table([{"n": 3, "eta": 1.2873}], [("n", "n", "d"), ("eta", "eta", ".3f")]))
    | n | eta |
    |---|---|
    | 3 | 1.287 |
    """
    lines = [
        "| " + " | ".join(header for _, header, _ in columns) + " |",
        "|" + "---|" * len(columns),
    ]
    for entry in entries:
        cells = []
        for key, _, spec in columns:
            value = entry.get(key, "")
            if spec == "flag":
                cells.append("yes" if value else "no")
            elif isinstance(value, (int, float)) and spec:
                cells.append(format(value, spec))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
