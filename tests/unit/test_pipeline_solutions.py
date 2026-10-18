"""A case's records across the campaigns: fields, matrices, metrics (``pipeline.solutions``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pihm.assemble import Regime, assemble
from pihm.pipeline.results import Results
from pihm.pipeline.solutions import (
    REGIMES,
    CaseKey,
    Catalogue,
    coupling,
    family,
    metrics,
    solution,
    structure,
)
from pihm.pipeline.spec import RunSpec
from pihm.pipeline.stages import run
from pihm.pipeline.summary import collect
from pihm.problems.ladder import problem
from pihm.qsp.prepare import FilterKind


@pytest.fixture(scope="module")
def catalogue(tmp_path_factory: pytest.TempPathFactory) -> Catalogue:
    """Two campaigns: R1 and R7 built and prepared (T2) in both forms, and an R1 estimate."""
    roots = {name: tmp_path_factory.mktemp(name) for name in ("builds", "preparations")}
    for name, n in (("R1", (3,)), ("R7", (3,))):
        for regime in ("standard", "descriptor"):
            spec = RunSpec(problem=problem(name), n=n, regime=regime)
            run(spec, roots["builds"])
            prepared = RunSpec(
                problem=problem(name), n=n, regime=regime, stage="stateprep", filter="minimax"
            )
            run(prepared, roots["preparations"])
    run(RunSpec(problem=problem("R1"), n=(3,), stage="stateprep", tier="T3"), roots["preparations"])
    for root in roots.values():
        collect(root)
    return Catalogue.read(Results(roots, {}))


def test_a_case_is_named_by_its_problem_resolution_and_order() -> None:
    assert CaseKey.of({"problem": "R8", "variant": "corrected", "n": 4, "carleman": 3}) == (
        CaseKey("R8", 4, 3)
    )
    no_lift = CaseKey("R2a", 3, variant="faithful")
    assert CaseKey.of({"problem": "R2a", "variant": "faithful", "n": 3, "carleman": 0}) == no_lift
    assert CaseKey.of({"problem": "R2a", "variant": "faithful", "n": 3, "carleman": ""}) == no_lift
    assert str(CaseKey("R8", 4, 3)) == "R8, n = 4, K = 3"
    assert str(no_lift) == "R2a (faithful), n = 3"
    ordered = sorted(
        [CaseKey("R10c-M12", 3, 2), CaseKey("R2a", 3), no_lift],
        key=lambda key: key.order,
    )
    assert ordered == [
        no_lift,
        CaseKey("R2a", 3),
        CaseKey("R10c-M12", 3, 2),
    ]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("R2c", "curve"),
        ("R6a", "curve"),
        ("R8", "curve"),
        ("R7", "fields"),
        ("R5a", "surface"),
        ("R5b", "surface"),
        ("R9c-M4", "burgers"),
        ("R10a-M12", "vorticity"),
    ],
)
def test_each_rung_is_drawn_by_its_family(name: str, expected: str) -> None:
    assert family(problem(name)) == expected


def test_a_case_finds_each_stages_record_in_both_forms(catalogue: Catalogue) -> None:
    assert catalogue.problems() == ["R1", "R7"]
    records = catalogue.case(CaseKey("R1", 3))
    assert set(records.build) == set(records.t2) == {"standard", "descriptor"}
    assert set(records.t3) == {"standard"}
    assert not records.t1
    assert not records.lift
    assert coupling(records) is None
    stages = {stage for stage, _, _ in records.run_keys({})}
    assert stages == {"build", "T2", "T3"}


def test_the_pipelines_filter_is_chosen_where_both_filters_ran_a_case(tmp_path: Path) -> None:
    """The reproduction's QITE runs share the standard form's cases with the minimax ones."""
    roots = {name: tmp_path / name for name in ("reproduction", "comparison")}
    filters: tuple[FilterKind, ...] = ("qite", "minimax")
    for root, chosen in zip(roots.values(), filters, strict=True):
        root.mkdir()
        for tier in ("T2", "T1"):
            spec = RunSpec(
                problem=problem("R1"),
                n=(2,),
                stage="stateprep",
                tier=tier,
                filter=chosen,
                geometric_r=(0.5,),
            )
            run(spec, root)
        collect(root)
    records = Catalogue.read(Results(roots)).case(CaseKey("R1", 2))
    assert records.t2["standard"].row["filter"] == "minimax"
    assert records.t1["standard"].row["filter"] == "minimax"


def test_a_solution_holds_the_ground_and_prepared_fields_on_the_builds_grid(
    catalogue: Catalogue,
) -> None:
    found = solution(catalogue.case(CaseKey("R7", 3)))
    assert found is not None
    assert found.family == "fields"
    assert [(layer.regime, layer.source) for layer in found.layers] == [
        ("standard", "ground"),
        ("standard", "T2"),
        ("descriptor", "ground"),
        ("descriptor", "T2"),
    ]
    assert found.reference.shape == (2, found.axes[0].size)
    assert not found.notes
    for layer in found.layers:
        assert layer.field.shape == found.reference.shape
        error = float(np.max(np.abs(layer.field - found.reference)))
        assert error == layer.record.row["field_err_linf"]


def test_a_preparation_without_its_field_is_noted(tmp_path: Path) -> None:
    """A record that kept no arrays: the viewer says so and draws the rest."""
    spec = RunSpec(problem=problem("R1"), n=(3,), stage="stateprep")
    run(RunSpec(problem=problem("R1"), n=(3,)), tmp_path)
    record = run(spec, tmp_path)
    assert record.arrays
    leaf = tmp_path / "stateprep" / "R1"
    (next(leaf.iterdir()) / "arrays.npz").unlink()
    collect(tmp_path)
    found = solution(Catalogue.read(Results({"c": tmp_path})).case(CaseKey("R1", 3)))
    assert found is not None
    assert [layer.source for layer in found.layers] == ["ground"]
    (note,) = found.notes
    assert "kept no field" in note


def test_the_structure_is_the_builds_residual_with_its_blocks_marked(
    catalogue: Catalogue,
) -> None:
    records = catalogue.case(CaseKey("R1", 3))
    for regime in REGIMES:
        shown = structure(records.build[regime])
        residual = assemble(problem("R1"), (3,), regime=regime)
        assert np.array_equal(shown.residual.toarray(), residual.matrix().toarray())
        assert shown.hamiltonian is not None
        assert np.array_equal(shown.hamiltonian.toarray(), residual.hamiltonian().toarray())
        assert shown.rows[-1].stop == shown.residual.shape[0]
        assert shown.columns[-1].stop == shown.residual.shape[1]
    descriptor = structure(records.build["descriptor"])
    assert [band.label for band in descriptor.rows] == ["equation"]  # f(-1) is the regular datum
    assert [band.label for band in descriptor.columns] == ["f"]


def test_the_columns_name_derivatives_fields_and_carleman_degrees(tmp_path: Path) -> None:
    builds: list[tuple[str, tuple[int, ...], Regime, int | None]] = [
        ("R2c", (3,), "descriptor", None),
        ("R5b", (2, 2), "descriptor", None),
        ("R8", (3,), "standard", 3),
        ("R7", (3,), "descriptor", None),
    ]
    for name, n, regime, carleman in builds:
        run(
            RunSpec(problem=problem(name), n=n, regime=regime, carleman=carleman),
            tmp_path,
        )
    collect(tmp_path)
    catalogue = Catalogue.read(Results({"c": tmp_path}))
    labels = {}
    for key in catalogue.cases():
        (record,) = catalogue.case(key).build.values()
        shown = structure(record)
        labels[key.problem] = [band.label for band in shown.columns]
        rows = [band.label for band in shown.rows]
        assert rows[0] == "equation"
    assert labels == {
        "R2c": ["f", "f′", "f″", "padding"],
        "R5b": ["f", "∂t f", "∂x f", "∂x² f"],
        "R7": ["u", "v"],
        "R8": ["degree 1", "degree 2", "degree 3", "padding"],
    }


def test_the_metrics_set_both_forms_side_by_side(catalogue: Catalogue) -> None:
    records = catalogue.case(CaseKey("R1", 3))
    table = metrics(records)
    assert table.columns == ("standard", "descriptor")
    rows = {(section, name): values for section, name, values in table.rows}
    probability = rows[
        ("preparation, T2 (exact emulation)", "success (post-selection) probability")
    ]
    assert probability == tuple(
        records.t2[regime].row["success_probability"] for regime in table.columns
    )
    assert rows[("estimate, T3", "filter degree")][1] is None
    assert rows[("classical build", "state qubits")] == (3, 3)
    assert not any(section.startswith("encoding") for section, _ in rows)
    assert all(any(v is not None for v in values) for values in rows.values())


def test_the_richest_case_has_both_forms_and_their_preparations(catalogue: Catalogue) -> None:
    assert catalogue.richest("R1") == CaseKey("R1", 3)
    with pytest.raises(KeyError, match="R2a"):
        catalogue.richest("R2a")
    coverage = {entry["family"]: entry for entry in catalogue.coverage()}
    assert coverage["curve"]["prepared"] == 2
    assert coverage["fields"]["with_field"] == 1
