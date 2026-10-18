"""The solution viewer: a case chosen from dropdowns, and everything its records show.

:func:`show_case` displays one case: its heading and records, its fields' figures
(:func:`~pihm.plotting.solutions.solution_figures`), the spy plots of its matrices in each form
(:func:`~pihm.plotting.solutions.spy`), and its metrics from every stage. :func:`viewer` wraps it
in ipywidgets: a problem, then one of its cases (n and K, or a paper panel as printed), and for a
time-dependent case a slider through its window (:func:`~pihm.plotting.solutions.snapshot`).

Each figure is displayed and then closed, so that it shows once, in the notebook or in the
viewer's output area alike. ipywidgets belongs to the optional ``notebooks`` extra, as matplotlib
does, and is imported only when the viewer is built.
"""

from __future__ import annotations

from typing import Any

from pihm.pipeline.solutions import (
    CaseKey,
    Catalogue,
    Solution,
    coupling,
    metrics,
    solution,
    structure,
)
from pihm.plotting.solutions import (
    case_markdown,
    metrics_markdown,
    snapshot,
    solution_figures,
    spy,
)

__all__ = ["show_case", "viewer"]


def _display(figure: Any) -> None:
    """Display a figure once, then close it."""
    import matplotlib.pyplot as plt
    from IPython.display import display

    display(figure)
    plt.close(figure)


def show_case(catalogue: Catalogue, key: CaseKey) -> Solution | None:
    """Display a case: its heading and records, its fields, its matrices and its metrics.

    Returns
    -------
    Solution or None
        The case's decoded fields, ``None`` if no record of it holds one.
    """
    from IPython.display import Markdown, display

    records = catalogue.case(key)
    found = solution(records)
    display(Markdown(case_markdown(key, records, catalogue.results.labels, found)))
    for figure in solution_figures(found) if found is not None else []:
        _display(figure)
    if records.build:
        _display(spy({regime: structure(record) for regime, record in records.build.items()}))
    display(Markdown(metrics_markdown(metrics(records))))
    circuit = coupling(records)
    if circuit is not None:
        display(Markdown(metrics_markdown(circuit)))
    return found


def _case_name(key: CaseKey) -> str:
    """Return a case as its dropdown lists it, within its problem."""
    printed = "faithful, " if key.variant == "faithful" else ""
    order = "" if key.carleman is None else f", K = {key.carleman}"
    return f"{printed}n = {key.n}{order}"


def viewer(catalogue: Catalogue, problem: str | None = None) -> Any:
    """Return the viewer: dropdowns for a problem and its case, a time slider, and the case.

    Parameters
    ----------
    catalogue
        The records to view (:meth:`~pihm.pipeline.solutions.Catalogue.read`).
    problem
        The problem shown first, by default the ladder's first; its case is the one with the most
        fields to draw (:meth:`~pihm.pipeline.solutions.Catalogue.richest`).

    Returns
    -------
    ipywidgets.VBox
        The viewer, for the notebook to display.
    """
    import ipywidgets as widgets

    problems = catalogue.problems()
    chooser = widgets.Dropdown(
        options=problems, value=problem or problems[0], description="problem"
    )
    case = widgets.Dropdown(description="case", layout=widgets.Layout(width="22em"))
    time = widgets.FloatSlider(
        value=0.0,
        min=0.0,
        max=1.0,
        step=0.02,
        description="time",
        continuous_update=False,
        readout_format=".2f",
    )
    moment = widgets.Output()
    page = widgets.Output()
    shown: dict[str, Solution | None] = {"solution": None}
    refilling = {"now": False}

    def draw_moment(_: Any = None) -> None:
        current = shown["solution"]
        figure = snapshot(current, time.value) if current is not None else None
        time.layout.visibility = "hidden" if figure is None else "visible"
        with moment:
            moment.clear_output(wait=True)
            if figure is not None:
                _display(figure)

    def draw_case(_: Any = None) -> None:
        if refilling["now"]:
            return
        with page:
            page.clear_output(wait=True)
            shown["solution"] = show_case(catalogue, case.value)
        draw_moment()

    def choose_problem(_: Any = None) -> None:
        # Refilling the cases changes the chosen one more than once; draw only the last.
        refilling["now"] = True
        case.options = [(_case_name(key), key) for key in catalogue.cases(chooser.value)]
        case.value = catalogue.richest(chooser.value)
        refilling["now"] = False
        draw_case()

    chooser.observe(choose_problem, names="value")
    case.observe(draw_case, names="value")
    time.observe(draw_moment, names="value")
    choose_problem()
    return widgets.VBox([widgets.HBox([chooser, case, time]), moment, page])
