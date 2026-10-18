"""The solution viewer's figures and widget (``plotting.solutions``, ``plotting.viewer``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pihm.pipeline.results import Results
from pihm.pipeline.solutions import CaseKey, Catalogue, metrics, solution, structure
from pihm.pipeline.spec import RunSpec
from pihm.pipeline.stages import run
from pihm.pipeline.summary import collect
from pihm.problems.ladder import problem

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")
plotting = pytest.importorskip("pihm.plotting")

#: One case of each family, built in both forms: (problem, n, Carleman order).
CASES = (
    ("R1", (3,), None),
    ("R7", (3,), None),
    ("R5b", (2, 2), None),
    ("R9c-M4", (2,), 2),
    ("R10a-M12", (2,), 2),
)

#: Each family's figures: the fields (and for a time axis, the moments), before the spy plots.
FIGURES = {"R1": 1, "R7": 1, "R5b": 2, "R9c-M4": 2, "R10a-M12": 2}


@pytest.fixture(scope="module")
def catalogue(tmp_path_factory: pytest.TempPathFactory) -> Catalogue:
    """Every family's builds in both forms, and R1's T2 preparation in the standard form."""
    root = tmp_path_factory.mktemp("results")
    for name, n, carleman in CASES:
        for regime in ("standard", "descriptor"):
            run(
                RunSpec(problem=problem(name), n=n, regime=regime, carleman=carleman),
                root,
            )
    run(RunSpec(problem=problem("R1"), n=(3,), stage="stateprep"), root)
    collect(root)
    return Catalogue.read(Results({"c": root}, {}))


@pytest.mark.parametrize(("name", "n", "carleman"), CASES)
def test_every_family_draws_its_fields_moments_and_matrices(
    catalogue: Catalogue, name: str, n: tuple[int, ...], carleman: int | None
) -> None:
    records = catalogue.case(CaseKey(name, max(n), carleman))
    found = solution(records)
    assert found is not None
    figures = plotting.solution_figures(found)
    assert len(figures) == FIGURES[name]
    moment = plotting.snapshot(found, 0.5)
    assert (moment is None) == (found.family in ("curve", "fields"))
    figures.append(plotting.spy({regime: structure(f) for regime, f in records.build.items()}))
    for figure in [*figures, *([moment] if moment is not None else [])]:
        figure.canvas.draw()
        plt.close(figure)


def test_a_prepared_field_is_drawn_dashed_beside_the_ground_state(catalogue: Catalogue) -> None:
    found = solution(catalogue.case(CaseKey("R1", 3)))
    assert found is not None
    (figure,) = plotting.solution_figures(found)
    styles = {line.get_label(): line.get_linestyle() for line in figure.axes[0].get_lines()}
    assert styles["standard, ground state"] == "-"
    assert styles["standard, prepared, T2"] == "--"
    plt.close(figure)


def test_the_tables_hold_the_metrics_and_the_run_keys(catalogue: Catalogue) -> None:
    key = CaseKey("R1", 3)
    records = catalogue.case(key)
    table = plotting.metrics_markdown(metrics(records))
    assert "| | standard | descriptor |" in table
    assert "| **preparation, T2 (exact emulation)** |" in table
    assert "success (post-selection) probability" in table
    heading = plotting.case_markdown(key, records, {}, solution(records))
    assert heading.startswith("### R1, n = 3")
    assert "`stateprep/R1/standard/n3/T2`" in heading


def test_the_viewer_redraws_when_the_problem_changes(catalogue: Catalogue) -> None:
    widgets = pytest.importorskip("ipywidgets")
    view = plotting.viewer(catalogue)
    assert isinstance(view, widgets.VBox)
    chooser, case, time = view.children[0].children
    assert chooser.value == "R1"
    assert case.value == CaseKey("R1", 3)
    assert time.layout.visibility == "hidden"
    chooser.value = "R10a-M12"
    assert case.value == CaseKey("R10a-M12", 2, 2)
    assert time.layout.visibility == "visible"
    time.value = 1.0
    plt.close("all")


def test_a_case_without_a_field_shows_its_metrics(tmp_path: Path) -> None:
    run(RunSpec(problem=problem("R1"), n=(3,), stage="stateprep", tier="T3"), tmp_path)
    collect(tmp_path)
    catalogue = Catalogue.read(Results({"c": tmp_path}, {}))
    assert plotting.show_case(catalogue, CaseKey("R1", 3)) is None
